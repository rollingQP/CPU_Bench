/*
 * Java CPU Benchmark v1.0
 * Compile: javac CPUBenchmark.java
 * Run: java CPUBenchmark
 */

import java.util.*;
import java.util.concurrent.*;

public class CPUBenchmark {
    static final int WARMUP_ITERATIONS = 3;
    static final int BENCHMARK_ITERATIONS = 5;

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║              JAVA CPU BENCHMARK v1.0                         ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝\n");

        printSystemInfo();
        System.out.println();

        // Warmup
        System.out.println("Warming up...");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            runAllBenchmarks(true);
        }
        System.out.println("Warmup complete.\n");

        // Run benchmarks
        System.out.println("Running benchmarks...\n");

        double[][] allScores = new double[8][BENCHMARK_ITERATIONS];

        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            System.out.printf("--- Iteration %d of %d ---%n", i + 1, BENCHMARK_ITERATIONS);
            double[] scores = runAllBenchmarks(false);
            for (int j = 0; j < scores.length; j++) {
                allScores[j][i] = scores[j];
            }
            System.out.println();
        }

        // Calculate averages
        double[] avgScores = new double[8];
        for (int i = 0; i < 8; i++) {
            double sum = 0;
            for (int j = 0; j < BENCHMARK_ITERATIONS; j++) {
                sum += allScores[i][j];
            }
            avgScores[i] = sum / BENCHMARK_ITERATIONS;
        }

        printFinalResults(avgScores);
    }

    static void printSystemInfo() {
        System.out.println("System Information:");
        System.out.println("  OS: " + System.getProperty("os.name") + " " + System.getProperty("os.version"));
        System.out.println("  Architecture: " + System.getProperty("os.arch"));
        System.out.println("  Java Version: " + System.getProperty("java.version"));
        System.out.println("  JVM: " + System.getProperty("java.vm.name"));
        System.out.println("  Available Processors: " + Runtime.getRuntime().availableProcessors());
    }

    static double[] runAllBenchmarks(boolean quiet) {
        double[] scores = new double[8];
        scores[0] = benchmarkIntegerArithmetic(quiet);
        scores[1] = benchmarkFloatingPoint(quiet);
        scores[2] = benchmarkPrimeCalculation(quiet);
        scores[3] = benchmarkMatrixMultiplication(quiet);
        scores[4] = benchmarkSorting(quiet);
        scores[5] = benchmarkMultiThreaded(quiet);
        scores[6] = benchmarkRecursion(quiet);
        scores[7] = benchmarkStringOperations(quiet);
        return scores;
    }

    static double benchmarkIntegerArithmetic(boolean quiet) {
        final long operations = 10_000_000L;

        long start = System.nanoTime();

        long result = 0;
        for (long i = 1; i <= operations; i++) {
            result += i;
            result ^= (i * 31);
            result -= (i / 3);
            result |= (i << 2);
            result &= (i >> 1);
        }

        long end = System.nanoTime();
        double timeMs = (end - start) / 1_000_000.0;
        double opsPerSecond = (operations * 5.0) / (timeMs / 1000.0);

        if (!quiet) {
            System.out.printf("  Integer Arithmetic: %.2f ms (%.2f M ops/sec) [checksum: %d]%n",
                    timeMs, opsPerSecond / 1_000_000.0, result);
        }

        return opsPerSecond;
    }

    static double benchmarkFloatingPoint(boolean quiet) {
        final int operations = 5_000_000;

        long start = System.nanoTime();

        double result = 1.0;
        for (int i = 1; i <= operations; i++) {
            result += Math.sqrt(i);
            result *= Math.sin(i * 0.001);
            result += Math.cos(i * 0.001);
            result /= (1.0 + Math.abs(result) * 0.0001);
        }

        long end = System.nanoTime();
        double timeMs = (end - start) / 1_000_000.0;
        double opsPerSecond = (operations * 4.0) / (timeMs / 1000.0);

        if (!quiet) {
            System.out.printf("  Floating Point: %.2f ms (%.2f M ops/sec) [checksum: %.6f]%n",
                    timeMs, opsPerSecond / 1_000_000.0, result);
        }

        return opsPerSecond;
    }

    static double benchmarkPrimeCalculation(boolean quiet) {
        final int limit = 2_000_000;

        long start = System.nanoTime();

        boolean[] isComposite = new boolean[limit + 1];
        int primeCount = 0;

        for (int i = 2; i <= limit; i++) {
            if (!isComposite[i]) {
                primeCount++;
                for (long j = (long) i * i; j <= limit; j += i) {
                    isComposite[(int) j] = true;
                }
            }
        }

        long end = System.nanoTime();
        double timeMs = (end - start) / 1_000_000.0;
        double score = limit / timeMs * 1000;

        if (!quiet) {
            System.out.printf("  Prime Sieve (up to %,d): %.2f ms (%,d primes found)%n",
                    limit, timeMs, primeCount);
        }

        return score;
    }

    static double benchmarkMatrixMultiplication(boolean quiet) {
        final int size = 256;

        Random random = new Random(42);
        double[][] a = new double[size][size];
        double[][] b = new double[size][size];
        double[][] c = new double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                a[i][j] = random.nextDouble();
                b[i][j] = random.nextDouble();
            }
        }

        long start = System.nanoTime();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double total = 0.0;
                for (int k = 0; k < size; k++) {
                    total += a[i][k] * b[k][j];
                }
                c[i][j] = total;
            }
        }

        long end = System.nanoTime();
        double timeMs = (end - start) / 1_000_000.0;
        double flops = 2.0 * size * size * size / (timeMs / 1000.0);

        if (!quiet) {
            System.out.printf("  Matrix Multiplication (%dx%d): %.2f ms (%.4f GFLOPS) [checksum: %.6f]%n",
                    size, size, timeMs, flops / 1_000_000_000.0, c[size / 2][size / 2]);
        }

        return flops;
    }

    static double benchmarkSorting(boolean quiet) {
        final int size = 1_000_000;

        Random random = new Random(42);
        int[] array = new int[size];
        for (int i = 0; i < size; i++) {
            array[i] = random.nextInt();
        }

        long start = System.nanoTime();

        quicksort(array, 0, size - 1);

        long end = System.nanoTime();
        double timeMs = (end - start) / 1_000_000.0;
        double elementsPerSecond = size / (timeMs / 1000.0);

        if (!quiet) {
            System.out.printf("  QuickSort (%,d elements): %.2f ms (%.2f M elem/sec)%n",
                    size, timeMs, elementsPerSecond / 1_000_000.0);
        }

        return elementsPerSecond;
    }

    static void quicksort(int[] arr, int low, int high) {
        Deque<int[]> stack = new ArrayDeque<>();
        stack.push(new int[]{low, high});

        while (!stack.isEmpty()) {
            int[] range = stack.pop();
            int lo = range[0];
            int hi = range[1];

            if (lo < hi) {
                int pivot = partition(arr, lo, hi);
                stack.push(new int[]{lo, pivot - 1});
                stack.push(new int[]{pivot + 1, hi});
            }
        }
    }

    static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }

        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        return i + 1;
    }

    static double benchmarkMultiThreaded(boolean quiet) {
        int numThreads = Runtime.getRuntime().availableProcessors();
        final int workPerThread = 5000;

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<Long>> futures = new ArrayList<>();

        long start = System.nanoTime();

        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            futures.add(executor.submit(() -> {
                long result = 0;
                for (int i = 0; i < workPerThread; i++) {
                    result += computeIntensive(i + threadId * workPerThread);
                }
                return result;
            }));
        }

        long checksum = 0;
        try {
            for (Future<Long> f : futures) {
                checksum += f.get();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        executor.shutdown();

        long end = System.nanoTime();
        double timeMs = (end - start) / 1_000_000.0;

        double totalOps = (double) numThreads * workPerThread * 1000;
        double opsPerSecond = totalOps / (timeMs / 1000.0);

        if (!quiet) {
            System.out.printf("  Multi-threaded (%d threads): %.2f ms (%.2f M ops/sec) [checksum: %d]%n",
                    numThreads, timeMs, opsPerSecond / 1_000_000.0, checksum);
        }

        return opsPerSecond;
    }

    static long computeIntensive(long seed) {
        long x = seed;
        for (int i = 0; i < 1000; i++) {
            x ^= (x << 21);
            x ^= (x >>> 35);
            x ^= (x << 4);
        }
        return x;
    }

    static double benchmarkRecursion(boolean quiet) {
        final int iterations = 100_000;
        final int fibN = 100;  // Match Python version

        long start = System.nanoTime();

        long result = 0;
        for (int i = 0; i < iterations; i++) {
            Map<Integer, Long> memo = new HashMap<>();
            result = fibMemo(fibN, memo);
        }

        long end = System.nanoTime();
        double timeMs = (end - start) / 1_000_000.0;
        double opsPerSecond = iterations / (timeMs / 1000.0);

        if (!quiet) {
            System.out.printf("  Recursion (Fibonacci): %.2f ms (%.2f K ops/sec) [checksum: %d]%n",
                    timeMs, opsPerSecond / 1000.0, result);
        }

        return opsPerSecond;
    }

    static long fibMemo(int n, Map<Integer, Long> memo) {
        if (n <= 1) return n;
        if (memo.containsKey(n)) return memo.get(n);
        long value = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
        memo.put(n, value);
        return value;
    }

    static double benchmarkStringOperations(boolean quiet) {
        final int iterations = 100_000;

        long start = System.nanoTime();

        String result = "";
        for (int i = 0; i < iterations; i++) {
            String s = "benchmark_string_" + i + "_test";

            // to upper
            s = s.toUpperCase();

            // replace _ with -
            s = s.replace('_', '-');

            // reverse
            s = new StringBuilder(s).reverse().toString();

            // split and join
            String[] parts = s.split("-");
            result = String.join("-", parts);
        }

        long end = System.nanoTime();
        double timeMs = (end - start) / 1_000_000.0;
        double opsPerSecond = iterations / (timeMs / 1000.0);

        if (!quiet) {
            System.out.printf("  String Operations: %.2f ms (%.2f K ops/sec) [checksum: %d]%n",
                    timeMs, opsPerSecond / 1000.0, result.length());
        }

        return opsPerSecond;
    }

    static void printFinalResults(double[] scores) {
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║                    FINAL RESULTS                             ║");
        System.out.println("╠══════════════════════════════════════════════════════════════╣");

        System.out.printf("║  Integer Arithmetic:     %12.2f M ops/sec              ║%n", scores[0] / 1_000_000.0);
        System.out.printf("║  Floating Point:         %12.2f M ops/sec              ║%n", scores[1] / 1_000_000.0);
        System.out.printf("║  Prime Sieve:            %12.2f M numbers/sec          ║%n", scores[2] / 1_000_000.0);
        System.out.printf("║  Matrix Multiplication:  %12.4f GFLOPS                 ║%n", scores[3] / 1_000_000_000.0);
        System.out.printf("║  QuickSort:              %12.2f M elem/sec             ║%n", scores[4] / 1_000_000.0);
        System.out.printf("║  Multi-threaded:         %12.2f M ops/sec              ║%n", scores[5] / 1_000_000.0);
        System.out.printf("║  Recursion:              %12.2f K ops/sec              ║%n", scores[6] / 1000.0);
        System.out.printf("║  String Operations:      %12.2f K ops/sec              ║%n", scores[7] / 1000.0);

        System.out.println("╠══════════════════════════════════════════════════════════════╣");

        double overallScore = Math.pow(
                (scores[0] / 50_000_000.0) *
                (scores[1] / 20_000_000.0) *
                (scores[2] / 2_000_000.0) *
                (scores[3] / 100_000_000.0) *
                (scores[4] / 500_000.0) *
                (scores[5] / 50_000_000.0) *
                (scores[6] / 100_000.0) *
                (scores[7] / 100_000.0),
                1.0 / 8.0
        ) * 1000;

        System.out.printf("║  OVERALL SCORE:          %12.0f                        ║%n", overallScore);
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
    }
}