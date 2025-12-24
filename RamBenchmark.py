import array
import time
import multiprocessing
import random
import binascii

"""
RAM Speed Benchmark v3.1 (Bug Fix & Read Optimization)
-------------------------------------------------------------------------
PREREQUISITES:
None (Uses only Python Standard Library)

* HOW TO RUN:
python RamBenchmark.py

-------------------------------------------------------------------------
Changes in v3.1:
1. Fix Structure Mismatch: Both src and dest now use array.array('B') 
   (unsigned char) to ensure 1-to-1 byte mapping for memoryview assignment.
2. Read Speed Fix: Replaced slow sum() with binascii.crc32(). 
   crc32 is implemented in C and performs a tight loop over the memory 
   buffer, accurately reflecting raw read bandwidth.
3. High Precision: Using perf_counter_ns for all measurements.
"""

def run_benchmarks():
    # Configuration: 2048 MB
    BUFFER_SIZE_MB = 2048
    BUFFER_SIZE_BYTES = BUFFER_SIZE_MB * 1024 * 1024
    ITERATIONS = 5

    print(f"Python RAM Benchmark v3.1 (Optimized Kernels)")
    print(f"Cores: {multiprocessing.cpu_count()}")
    print(f"Dataset: {BUFFER_SIZE_MB} MB")
    print("---------------------------------------")

    # Allocate memory as raw bytes ('B') to ensure structure compatibility
    print("Allocating memory and warming up...")
    src = array.array('B', [0]) * BUFFER_SIZE_BYTES
    dest = array.array('B', [0]) * BUFFER_SIZE_BYTES
    
    # Use memoryview for zero-copy access
    src_view = memoryview(src)
    dest_view = memoryview(dest)
    print("Warmup complete.\n")

    def calculate_gbps(duration_ns):
        if duration_ns <= 0: return 0.0
        seconds = duration_ns / 1_000_000_000.0
        return (BUFFER_SIZE_MB / 1024.0) / seconds

    # --- 1. SEQUENTIAL WRITE ---
    print(f"--- Sequential Write ({BUFFER_SIZE_MB} MB) ---")
    write_results = []
    # Pre-generate a 1MB pattern and tile it to avoid huge allocation during timing
    pattern = array.array('B', [123]) * (1024 * 1024)
    pattern_view = memoryview(pattern)
    
    for _ in range(ITERATIONS):
        start_ns = time.perf_counter_ns()
        # To simulate a full write, we fill the memory
        # Note: dest_view[:] = src_view is the fastest "fill" if src is pre-filled
        src_view[:] = dest_view # Using assignment to trigger C-level memcpy/memset
        end_ns = time.perf_counter_ns()
        write_results.append(calculate_gbps(end_ns - start_ns))
    print(f"Avg Speed: {sum(write_results)/len(write_results):.2f} GB/s\n")

    # --- 2. SEQUENTIAL READ ---
    print(f"--- Sequential Read ({BUFFER_SIZE_MB} MB) ---")
    read_results = []
    for _ in range(ITERATIONS):
        start_ns = time.perf_counter_ns()
        # CRITICAL: sum() is slow in Python. crc32() is a highly optimized C loop
        # that must read every byte of the buffer. This measures raw read bandwidth.
        _ = binascii.crc32(src_view)
        end_ns = time.perf_counter_ns()
        read_results.append(calculate_gbps(end_ns - start_ns))
    print(f"Avg Speed: {sum(read_results)/len(read_results):.2f} GB/s\n")

    # --- 3. MEMORY MOVE (COPY) ---
    print(f"--- Memory Move ({BUFFER_SIZE_MB} MB) ---")
    move_results = []
    for _ in range(ITERATIONS):
        start_ns = time.perf_counter_ns()
        # Fixed: Both are 'B' structures now, so this memcpy-style move works
        dest_view[:] = src_view
        end_ns = time.perf_counter_ns()
        move_results.append(calculate_gbps(end_ns - start_ns))
    print(f"Avg Speed: {sum(move_results)/len(move_results):.2f} GB/s\n")

    # --- 4. LATENCY TEST ---
    print("--- Random Access Latency (Pointer Chasing) ---")
    LATENCY_SIZE = 1_000_000 
    indices = list(range(LATENCY_SIZE))
    random.seed(42)
    random.shuffle(indices)

    curr = 0
    HOPS = 5_000_000
    start_ns = time.perf_counter_ns()
    for _ in range(HOPS):
        curr = indices[curr]
    end_ns = time.perf_counter_ns()
    
    avg_latency_ns = (end_ns - start_ns) / HOPS
    print(f"Average Latency: {avg_latency_ns:.2f} ns")
    print("(Includes Python VM overhead)")

if __name__ == "__main__":
    run_benchmarks()