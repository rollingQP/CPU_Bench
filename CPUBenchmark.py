#!/usr/bin/env python3
"""
Python CPU Benchmark v1.0
Tests various CPU workloads without external dependencies.
"""

import sys
import platform
import time
import random
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

WARMUP_ITERATIONS = 3
BENCHMARK_ITERATIONS = 5


def print_system_info():
    print("System Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python Version: {platform.python_version()}")
    print(f"  Python Implementation: {platform.python_implementation()}")
    print(f"  Available Processors: {multiprocessing.cpu_count()}")


def benchmark_integer_arithmetic(quiet=False):
    """Benchmark 1: Integer Arithmetic"""
    operations = 10_000_000
    
    start_time = time.perf_counter()
    
    result = 0
    for i in range(1, operations + 1):
        result += i
        result ^= (i * 31)
        result -= (i // 3)
        result |= (i << 2)
        result &= (i >> 1)
        result &= 0xFFFFFFFFFFFFFFFF  # Keep it bounded
    
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    ops_per_second = (operations * 5.0) / (time_ms / 1000.0)
    
    if not quiet:
        print(f"  Integer Arithmetic: {time_ms:.2f} ms ({ops_per_second / 1_000_000:.2f} M ops/sec) [checksum: {result}]")
    
    return ops_per_second


def benchmark_floating_point(quiet=False):
    """Benchmark 2: Floating Point Operations"""
    operations = 5_000_000
    
    start_time = time.perf_counter()
    
    result = 1.0
    for i in range(1, operations + 1):
        result += math.sqrt(i)
        result *= math.sin(i * 0.001)
        result += math.cos(i * 0.001)
        result /= (1.0 + abs(result) * 0.0001)
    
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    ops_per_second = (operations * 4.0) / (time_ms / 1000.0)
    
    if not quiet:
        print(f"  Floating Point: {time_ms:.2f} ms ({ops_per_second / 1_000_000:.2f} M ops/sec) [checksum: {result:.6f}]")
    
    return ops_per_second


def benchmark_prime_calculation(quiet=False):
    """Benchmark 3: Prime Number Calculation (Sieve of Eratosthenes)"""
    limit = 2_000_000
    
    start_time = time.perf_counter()
    
    is_composite = [False] * (limit + 1)
    prime_count = 0
    
    for i in range(2, limit + 1):
        if not is_composite[i]:
            prime_count += 1
            for j in range(i * i, limit + 1, i):
                is_composite[j] = True
    
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    score = limit / time_ms * 1000
    
    if not quiet:
        print(f"  Prime Sieve (up to {limit:,}): {time_ms:.2f} ms ({prime_count:,} primes found)")
    
    return score


def benchmark_matrix_multiplication(quiet=False):
    """Benchmark 4: Matrix Multiplication"""
    size = 256
    
    random.seed(42)
    a = [[random.random() for _ in range(size)] for _ in range(size)]
    b = [[random.random() for _ in range(size)] for _ in range(size)]
    c = [[0.0] * size for _ in range(size)]
    
    start_time = time.perf_counter()
    
    for i in range(size):
        for j in range(size):
            total = 0.0
            for k in range(size):
                total += a[i][k] * b[k][j]
            c[i][j] = total
    
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    
    # Calculate FLOPS (2 operations per multiply-add, size^3 iterations)
    flops = 2.0 * size * size * size / (time_ms / 1000.0)
    
    if not quiet:
        print(f"  Matrix Multiplication ({size}x{size}): {time_ms:.2f} ms ({flops / 1_000_000_000:.4f} GFLOPS) [checksum: {c[size//2][size//2]:.6f}]")
    
    return flops


def benchmark_sorting(quiet=False):
    """Benchmark 5: Sorting"""
    size = 1_000_000
    
    random.seed(42)
    array = [random.randint(-2**31, 2**31 - 1) for _ in range(size)]
    
    start_time = time.perf_counter()
    
    quicksort(array, 0, len(array) - 1)
    
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    elements_per_second = size / (time_ms / 1000.0)
    
    if not quiet:
        print(f"  QuickSort ({size:,} elements): {time_ms:.2f} ms ({elements_per_second / 1_000_000:.2f} M elem/sec)")
    
    return elements_per_second


def quicksort(arr, low, high):
    """In-place quicksort implementation"""
    stack = [(low, high)]
    
    while stack:
        low, high = stack.pop()
        if low < high:
            pivot_index = partition(arr, low, high)
            stack.append((low, pivot_index - 1))
            stack.append((pivot_index + 1, high))


def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def compute_intensive(seed):
    """CPU-intensive computation for multi-threaded benchmark"""
    x = seed
    for _ in range(1000):
        x ^= (x << 21) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 35)
        x ^= (x << 4) & 0xFFFFFFFFFFFFFFFF
    return x


def worker_task(args):
    """Worker function for multiprocessing"""
    thread_id, work_per_thread = args
    result = 0
    for i in range(work_per_thread):
        result += compute_intensive(i + thread_id * work_per_thread)
        result &= 0xFFFFFFFFFFFFFFFF
    return result


def benchmark_multi_threaded(quiet=False):
    """Benchmark 6: Multi-threaded/Multi-process Performance"""
    num_processes = multiprocessing.cpu_count()
    work_per_process = 5000
    
    start_time = time.perf_counter()
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        args = [(i, work_per_process) for i in range(num_processes)]
        results = list(executor.map(worker_task, args))
    
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    
    total_ops = num_processes * work_per_process * 1000  # 1000 iterations in compute_intensive
    ops_per_second = total_ops / (time_ms / 1000.0)
    
    checksum = sum(results) & 0xFFFFFFFFFFFFFFFF
    
    if not quiet:
        print(f"  Multi-process ({num_processes} processes): {time_ms:.2f} ms ({ops_per_second / 1_000_000:.2f} M ops/sec) [checksum: {checksum}]")
    
    return ops_per_second


def benchmark_recursion(quiet=False):
    """Benchmark 7: Recursion (Fibonacci with memoization)"""
    
    def fib_memo(n, memo={}):
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
        return memo[n]
    
    iterations = 100_000
    
    start_time = time.perf_counter()
    
    result = 0
    for i in range(iterations):
        memo = {}
        result = fib_memo(100, memo)
    
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    ops_per_second = iterations / (time_ms / 1000.0)
    
    if not quiet:
        print(f"  Recursion (Fibonacci): {time_ms:.2f} ms ({ops_per_second / 1_000:.2f} K ops/sec) [checksum: {result}]")
    
    return ops_per_second


def benchmark_string_operations(quiet=False):
    """Benchmark 8: String Operations"""
    iterations = 100_000
    
    start_time = time.perf_counter()
    
    result = ""
    for i in range(iterations):
        s = f"benchmark_string_{i}_test"
        s = s.upper()
        s = s.replace("_", "-")
        s = s[::-1]
        parts = s.split("-")
        result = "-".join(parts)
    
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    ops_per_second = iterations / (time_ms / 1000.0)
    
    if not quiet:
        print(f"  String Operations: {time_ms:.2f} ms ({ops_per_second / 1_000:.2f} K ops/sec) [checksum: {len(result)}]")
    
    return ops_per_second


def run_all_benchmarks(quiet=False):
    """Run all benchmarks and return scores"""
    scores = []
    
    scores.append(benchmark_integer_arithmetic(quiet))
    scores.append(benchmark_floating_point(quiet))
    scores.append(benchmark_prime_calculation(quiet))
    scores.append(benchmark_matrix_multiplication(quiet))
    scores.append(benchmark_sorting(quiet))
    scores.append(benchmark_multi_threaded(quiet))
    scores.append(benchmark_recursion(quiet))
    scores.append(benchmark_string_operations(quiet))
    
    return scores


def print_final_results(scores):
    """Print final benchmark results"""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    FINAL RESULTS                             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Integer Arithmetic:     {scores[0] / 1_000_000:12.2f} M ops/sec              ║")
    print(f"║  Floating Point:         {scores[1] / 1_000_000:12.2f} M ops/sec              ║")
    print(f"║  Prime Sieve:            {scores[2] / 1_000_000:12.2f} M numbers/sec          ║")
    print(f"║  Matrix Multiplication:  {scores[3] / 1_000_000_000:12.4f} GFLOPS                 ║")
    print(f"║  QuickSort:              {scores[4] / 1_000_000:12.2f} M elem/sec             ║")
    print(f"║  Multi-process:          {scores[5] / 1_000_000:12.2f} M ops/sec              ║")
    print(f"║  Recursion:              {scores[6] / 1_000:12.2f} K ops/sec              ║")
    print(f"║  String Operations:      {scores[7] / 1_000:12.2f} K ops/sec              ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    
    # Calculate overall score (geometric mean normalized to a baseline)
    overall_score = (
        (scores[0] / 50_000_000.0) *
        (scores[1] / 20_000_000.0) *
        (scores[2] / 2_000_000.0) *
        (scores[3] / 100_000_000.0) *
        (scores[4] / 500_000.0) *
        (scores[5] / 50_000_000.0) *
        (scores[6] / 100_000.0) *
        (scores[7] / 100_000.0)
    ) ** (1.0 / 8.0) * 1000
    
    print(f"║  OVERALL SCORE:          {overall_score:12.0f}                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            PYTHON CPU BENCHMARK v1.0                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    print_system_info()
    print()
    
    # Warmup
    print("Warming up...")
    for i in range(WARMUP_ITERATIONS):
        run_all_benchmarks(quiet=True)
    print("Warmup complete.\n")
    
    # Run actual benchmarks
    print("Running benchmarks...\n")
    
    all_scores = [[] for _ in range(8)]
    
    for i in range(BENCHMARK_ITERATIONS):
        print(f"--- Iteration {i + 1} of {BENCHMARK_ITERATIONS} ---")
        scores = run_all_benchmarks(quiet=False)
        for j, score in enumerate(scores):
            all_scores[j].append(score)
        print()
    
    # Calculate averages
    avg_scores = [sum(s) / len(s) for s in all_scores]
    
    # Print final results
    print_final_results(avg_scores)


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()