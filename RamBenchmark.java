import java.lang.foreign.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;

/**
 * RAM Speed Benchmark v2.1 (Performance Tuning)
 * -------------------------------------------------------------------------
 * JAVA VERSION: 
 * Requires JDK 21 or newer (Long Term Support).
 * * HOW TO COMPILE:
 * javac --enable-preview --release 21 RamBenchmark.java
 * * HOW TO RUN:
 * java --enable-preview --enable-native-access ALL-UNNAMED RamBenchmark
 * -------------------------------------------------------------------------
 * Changes in v2.1:
 * 1. Manual Task Splitting: Replaced simple parallel stream with balanced chunking.
 * 2. Minimized Reduction Overhead: Reduced contention in Read benchmark.
 * 3. Cache Line Padding: Optimized memory layout for modern multi-core CPUs.
 */
public class RamBenchmark {

    private static final long BUFFER_SIZE_MB = 2048;
    private static final long BUFFER_SIZE_BYTES = BUFFER_SIZE_MB * 1024 * 1024;
    private static final long NUM_ELEMENTS = BUFFER_SIZE_BYTES / 8; 
    private static final int ITERATIONS = 5;

    public static void main(String[] args) {
        System.out.println("Java RAM Benchmark v2.1 (Tuned Implementation)");
        int processors = Runtime.getRuntime().availableProcessors();
        int parallelism = ForkJoinPool.getCommonPoolParallelism();
        System.out.println("检测到核心数: " + processors);
        System.out.println("并行线程数: " + parallelism);
        System.out.println("测试数据大小: " + BUFFER_SIZE_MB + " MB");
        System.out.println("---------------------------------------");

        try (Arena arena = Arena.ofShared()) {
            MemorySegment src = arena.allocate(BUFFER_SIZE_BYTES, 64);
            MemorySegment dest = arena.allocate(BUFFER_SIZE_BYTES, 64);

            System.out.println("正在预热内存...");
            warmup(src, dest);

            runWriteBenchmark(src, parallelism);
            runReadBenchmark(src, parallelism);
            runMoveBenchmark(src, dest, parallelism);
            runLatencyBenchmark(src);

        } catch (Exception e) {
            System.err.println("发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void warmup(MemorySegment src, MemorySegment dest) {
        src.fill((byte) 0);
        dest.fill((byte) 0);
        System.out.println("预热完成。\n");
    }

    private static void runWriteBenchmark(MemorySegment segment, int parallelism) {
        System.out.println("--- 顺序写入测试 (" + BUFFER_SIZE_MB + " MB) ---");
        List<Double> results = new ArrayList<>();

        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            
            IntStream.range(0, parallelism + 1).parallel().forEach(t -> {
                long itemsPerThread = (NUM_ELEMENTS + parallelism) / (parallelism + 1);
                long startIdx = t * itemsPerThread;
                long endIdx = Math.min(startIdx + itemsPerThread, NUM_ELEMENTS);
                
                for (long j = startIdx; j < endIdx; j++) {
                    segment.setAtIndex(ValueLayout.JAVA_LONG, j, 0xDEADBEEFL);
                }
            });

            long end = System.nanoTime();
            results.add(calculateGbps(end - start));
        }
        printAvg(results);
    }

    private static void runReadBenchmark(MemorySegment segment, int parallelism) {
        System.out.println("--- 顺序读取测试 (" + BUFFER_SIZE_MB + " MB) ---");
        List<Double> results = new ArrayList<>();

        for (int i = 0; i < ITERATIONS; i++) {
            LongAdder sum = new LongAdder();
            long start = System.nanoTime();

            IntStream.range(0, parallelism + 1).parallel().forEach(t -> {
                long itemsPerThread = (NUM_ELEMENTS + parallelism) / (parallelism + 1);
                long startIdx = t * itemsPerThread;
                long endIdx = Math.min(startIdx + itemsPerThread, NUM_ELEMENTS);
                
                long localSum = 0;
                for (long j = startIdx; j < endIdx; j++) {
                    localSum += segment.getAtIndex(ValueLayout.JAVA_LONG, j);
                }
                sum.add(localSum);
            });

            long end = System.nanoTime();
            if (sum.sum() == 1) System.out.print(" ");
            results.add(calculateGbps(end - start));
        }
        printAvg(results);
    }

    private static void runMoveBenchmark(MemorySegment src, MemorySegment dest, int parallelism) {
        System.out.println("--- 内存拷贝测试 (并行) (" + BUFFER_SIZE_MB + " MB) ---");
        List<Double> results = new ArrayList<>();

        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            
            IntStream.range(0, parallelism + 1).parallel().forEach(t -> {
                long bytesPerThread = (BUFFER_SIZE_BYTES + parallelism) / (parallelism + 1);
                // 对齐到 8 字节边界以保证原子性
                bytesPerThread = (bytesPerThread + 7) & ~7; 
                long offset = t * bytesPerThread;
                if (offset < BUFFER_SIZE_BYTES) {
                    long size = Math.min(bytesPerThread, BUFFER_SIZE_BYTES - offset);
                    MemorySegment.copy(src, offset, dest, offset, size);
                }
            });

            long end = System.nanoTime();
            results.add(calculateGbps(end - start));
        }
        printAvg(results);
    }

    private static void runLatencyBenchmark(MemorySegment segment) {
        System.out.println("--- 随机访问延迟测试 (Pointer Chasing) ---");
        
        // 使用 128MB 范围进行延迟测试，减少 TLB Miss 干扰，专注于 DRAM 响应
        int latencySize = 16 * 1024 * 1024; 
        int[] indices = new int[latencySize];
        for (int i = 0; i < latencySize; i++) indices[i] = i;
        
        Random rand = new Random(42);
        for (int i = latencySize - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        for (int i = 0; i < latencySize - 1; i++) {
            segment.setAtIndex(ValueLayout.JAVA_LONG, indices[i], (long)indices[i + 1]);
        }
        segment.setAtIndex(ValueLayout.JAVA_LONG, indices[latencySize - 1], (long)indices[0]);

        final int latencyIterations = 50_000_000;
        long curr = indices[0];
        
        long start = System.nanoTime();
        for (int i = 0; i < latencyIterations; i++) {
            curr = segment.getAtIndex(ValueLayout.JAVA_LONG, curr);
        }
        long end = System.nanoTime();

        if (curr == 0xBAADF00D) System.out.print(" ");
        
        double totalNs = (double)(end - start);
        System.out.printf("平均延迟: %.2f ns\n\n", (totalNs / latencyIterations));
    }

    private static double calculateGbps(long durationNs) {
        double seconds = durationNs / 1_000_000_000.0;
        return (BUFFER_SIZE_MB / 1024.0) / seconds;
    }

    private static void printAvg(List<Double> results) {
        double avg = results.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        System.out.printf("平均速度: %.2f GB/s\n\n", avg);
    }
}