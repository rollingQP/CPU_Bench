/*
 * C++ CPU Benchmark v1.0
 * Tests various CPU workloads
 * Compile with: g++ -O2 -pthread -o cpu_benchmark cpu_benchmark.cpp
 * Or with maximum optimization: g++ -O3 -march=native -pthread -o cpu_benchmark cpu_benchmark.cpp
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <thread>
#include <future>
#include <algorithm>
#include <sstream>
#include <cstdint>
#include <unordered_map>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
    #include <sys/utsname.h>
#endif

constexpr int WARMUP_ITERATIONS = 3;
constexpr int BENCHMARK_ITERATIONS = 5;

void print_system_info() {
    std::cout << "System Information:\n";
    
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    std::cout << "  OS: Windows\n";
    std::cout << "  Architecture: " << (sysInfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64 ? "x86_64" : "x86") << "\n";
    std::cout << "  Available Processors: " << sysInfo.dwNumberOfProcessors << "\n";
#else
    struct utsname unameData;
    if (uname(&unameData) == 0) {
        std::cout << "  OS: " << unameData.sysname << " " << unameData.release << "\n";
        std::cout << "  Architecture: " << unameData.machine << "\n";
    }
    std::cout << "  Available Processors: " << std::thread::hardware_concurrency() << "\n";
#endif
    
    std::cout << "  C++ Standard: " << __cplusplus << "\n";
}

double benchmark_integer_arithmetic(bool quiet = false) {
    constexpr int64_t operations = 10'000'000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t result = 0;
    for (int64_t i = 1; i <= operations; ++i) {
        result += i;
        result ^= (i * 31);
        result -= (i / 3);
        result |= (i << 2);
        result &= (i >> 1);
        result &= 0xFFFFFFFFFFFFFFFFULL;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ops_per_second = (operations * 5.0) / (time_ms / 1000.0);
    
    if (!quiet) {
        std::cout << "  Integer Arithmetic: " << std::fixed << std::setprecision(2) 
                  << time_ms << " ms (" << (ops_per_second / 1'000'000.0) 
                  << " M ops/sec) [checksum: " << result << "]\n";
    }
    
    return ops_per_second;
}

double benchmark_floating_point(bool quiet = false) {
    constexpr int operations = 5'000'000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    double result = 1.0;
    for (int i = 1; i <= operations; ++i) {
        result += std::sqrt(static_cast<double>(i));
        result *= std::sin(i * 0.001);
        result += std::cos(i * 0.001);
        result /= (1.0 + std::abs(result) * 0.0001);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ops_per_second = (operations * 4.0) / (time_ms / 1000.0);
    
    if (!quiet) {
        std::cout << "  Floating Point: " << std::fixed << std::setprecision(2) 
                  << time_ms << " ms (" << (ops_per_second / 1'000'000.0) 
                  << " M ops/sec) [checksum: " << std::setprecision(6) << result << "]\n";
    }
    
    return ops_per_second;
}

double benchmark_prime_calculation(bool quiet = false) {
    constexpr int limit = 2'000'000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<bool> is_composite(limit + 1, false);
    int prime_count = 0;
    
    for (int i = 2; i <= limit; ++i) {
        if (!is_composite[i]) {
            ++prime_count;
            for (int64_t j = static_cast<int64_t>(i) * i; j <= limit; j += i) {
                is_composite[j] = true;
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double score = limit / time_ms * 1000;
    
    if (!quiet) {
        std::cout << "  Prime Sieve (up to " << limit << "): " << std::fixed 
                  << std::setprecision(2) << time_ms << " ms (" << prime_count << " primes found)\n";
    }
    
    return score;
}

double benchmark_matrix_multiplication(bool quiet = false) {
    constexpr int size = 256;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    std::vector<std::vector<double>> a(size, std::vector<double>(size));
    std::vector<std::vector<double>> b(size, std::vector<double>(size));
    std::vector<std::vector<double>> c(size, std::vector<double>(size, 0.0));
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a[i][j] = dist(gen);
            b[i][j] = dist(gen);
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double total = 0.0;
            for (int k = 0; k < size; ++k) {
                total += a[i][k] * b[k][j];
            }
            c[i][j] = total;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double flops = 2.0 * size * size * size / (time_ms / 1000.0);
    
    if (!quiet) {
        std::cout << "  Matrix Multiplication (" << size << "x" << size << "): " 
                  << std::fixed << std::setprecision(2) << time_ms << " ms (" 
                  << std::setprecision(4) << (flops / 1'000'000'000.0) << " GFLOPS) [checksum: " 
                  << std::setprecision(6) << c[size/2][size/2] << "]\n";
    }
    
    return flops;
}

int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; ++j) {
        if (arr[j] <= pivot) {
            ++i;
            std::swap(arr[i], arr[j]);
        }
    }
    
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quicksort(std::vector<int>& arr, int low, int high) {
    std::vector<std::pair<int, int>> stack;
    stack.emplace_back(low, high);
    
    while (!stack.empty()) {
        auto [lo, hi] = stack.back();
        stack.pop_back();
        
        if (lo < hi) {
            int pivot_index = partition(arr, lo, hi);
            stack.emplace_back(lo, pivot_index - 1);
            stack.emplace_back(pivot_index + 1, hi);
        }
    }
}

double benchmark_sorting(bool quiet = false) {
    constexpr int size = 1'000'000;
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(INT32_MIN, INT32_MAX);
    
    std::vector<int> array(size);
    for (int i = 0; i < size; ++i) {
        array[i] = dist(gen);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    quicksort(array, 0, size - 1);
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double elements_per_second = size / (time_ms / 1000.0);
    
    if (!quiet) {
        std::cout << "  QuickSort (" << size << " elements): " << std::fixed 
                  << std::setprecision(2) << time_ms << " ms (" 
                  << (elements_per_second / 1'000'000.0) << " M elem/sec)\n";
    }
    
    return elements_per_second;
}

uint64_t compute_intensive(uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 1000; ++i) {
        x ^= (x << 21);
        x ^= (x >> 35);
        x ^= (x << 4);
    }
    return x;
}

uint64_t worker_task(int thread_id, int work_per_thread) {
    uint64_t result = 0;
    for (int i = 0; i < work_per_thread; ++i) {
        result += compute_intensive(i + thread_id * work_per_thread);
    }
    return result;
}

double benchmark_multi_threaded(bool quiet = false) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    constexpr int work_per_thread = 5000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::future<uint64_t>> futures;
    for (int i = 0; i < num_threads; ++i) {
        futures.push_back(std::async(std::launch::async, worker_task, i, work_per_thread));
    }
    
    uint64_t checksum = 0;
    for (auto& f : futures) {
        checksum += f.get();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    double total_ops = static_cast<double>(num_threads) * work_per_thread * 1000;
    double ops_per_second = total_ops / (time_ms / 1000.0);
    
    if (!quiet) {
        std::cout << "  Multi-threaded (" << num_threads << " threads): " << std::fixed 
                  << std::setprecision(2) << time_ms << " ms (" 
                  << (ops_per_second / 1'000'000.0) << " M ops/sec) [checksum: " << checksum << "]\n";
    }
    
    return ops_per_second;
}

uint64_t fib_memo(int n, std::unordered_map<int, uint64_t>& memo) {
    if (n <= 1) return n;
    auto it = memo.find(n);
    if (it != memo.end()) return it->second;
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo);
    return memo[n];
}

double benchmark_recursion(bool quiet = false) {
    constexpr int iterations = 100'000;
    constexpr int fib_n = 100;  // Match Python version
    
    auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t result = 0;
    for (int i = 0; i < iterations; ++i) {
        std::unordered_map<int, uint64_t> memo;
        result = fib_memo(fib_n, memo);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ops_per_second = iterations / (time_ms / 1000.0);
    
    if (!quiet) {
        std::cout << "  Recursion (Fibonacci): " << std::fixed << std::setprecision(2) 
                  << time_ms << " ms (" << (ops_per_second / 1000.0) 
                  << " K ops/sec) [checksum: " << result << "]\n";
    }
    
    return ops_per_second;
}

double benchmark_string_operations(bool quiet = false) {
    constexpr int iterations = 100'000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string result;
    for (int i = 0; i < iterations; ++i) {
        std::string s = "benchmark_string_" + std::to_string(i) + "_test";
        
        // to upper
        std::transform(s.begin(), s.end(), s.begin(), ::toupper);
        
        // replace _ with -
        std::replace(s.begin(), s.end(), '_', '-');
        
        // reverse
        std::reverse(s.begin(), s.end());
        
        // split and join
        std::vector<std::string> parts;
        std::stringstream ss(s);
        std::string token;
        while (std::getline(ss, token, '-')) {
            parts.push_back(token);
        }
        
        result.clear();
        for (size_t j = 0; j < parts.size(); ++j) {
            if (j > 0) result += "-";
            result += parts[j];
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ops_per_second = iterations / (time_ms / 1000.0);
    
    if (!quiet) {
        std::cout << "  String Operations: " << std::fixed << std::setprecision(2) 
                  << time_ms << " ms (" << (ops_per_second / 1000.0) 
                  << " K ops/sec) [checksum: " << result.length() << "]\n";
    }
    
    return ops_per_second;
}

std::vector<double> run_all_benchmarks(bool quiet = false) {
    std::vector<double> scores;
    
    scores.push_back(benchmark_integer_arithmetic(quiet));
    scores.push_back(benchmark_floating_point(quiet));
    scores.push_back(benchmark_prime_calculation(quiet));
    scores.push_back(benchmark_matrix_multiplication(quiet));
    scores.push_back(benchmark_sorting(quiet));
    scores.push_back(benchmark_multi_threaded(quiet));
    scores.push_back(benchmark_recursion(quiet));
    scores.push_back(benchmark_string_operations(quiet));
    
    return scores;
}

void print_final_results(const std::vector<double>& scores) {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    FINAL RESULTS                             ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    
    std::cout << "║  Integer Arithmetic:     " << std::fixed << std::setw(12) << std::setprecision(2) 
              << (scores[0] / 1'000'000.0) << " M ops/sec              ║\n";
    std::cout << "║  Floating Point:         " << std::setw(12) << std::setprecision(2) 
              << (scores[1] / 1'000'000.0) << " M ops/sec              ║\n";
    std::cout << "║  Prime Sieve:            " << std::setw(12) << std::setprecision(2) 
              << (scores[2] / 1'000'000.0) << " M numbers/sec          ║\n";
    std::cout << "║  Matrix Multiplication:  " << std::setw(12) << std::setprecision(4) 
              << (scores[3] / 1'000'000'000.0) << " GFLOPS                 ║\n";
    std::cout << "║  QuickSort:              " << std::setw(12) << std::setprecision(2) 
              << (scores[4] / 1'000'000.0) << " M elem/sec             ║\n";
    std::cout << "║  Multi-threaded:         " << std::setw(12) << std::setprecision(2) 
              << (scores[5] / 1'000'000.0) << " M ops/sec              ║\n";
    std::cout << "║  Recursion:              " << std::setw(12) << std::setprecision(2) 
              << (scores[6] / 1000.0) << " K ops/sec              ║\n";
    std::cout << "║  String Operations:      " << std::setw(12) << std::setprecision(2) 
              << (scores[7] / 1000.0) << " K ops/sec              ║\n";
    
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    
    double overall_score = std::pow(
        (scores[0] / 50'000'000.0) *
        (scores[1] / 20'000'000.0) *
        (scores[2] / 2'000'000.0) *
        (scores[3] / 100'000'000.0) *
        (scores[4] / 500'000.0) *
        (scores[5] / 50'000'000.0) *
        (scores[6] / 100'000.0) *
        (scores[7] / 100'000.0),
        1.0 / 8.0
    ) * 1000;
    
    std::cout << "║  OVERALL SCORE:          " << std::setw(12) << std::setprecision(0) 
              << overall_score << "                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              C++ CPU BENCHMARK v1.0                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    print_system_info();
    std::cout << "\n";
    
    // Warmup
    std::cout << "Warming up...\n";
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        run_all_benchmarks(true);
    }
    std::cout << "Warmup complete.\n\n";
    
    // Run actual benchmarks
    std::cout << "Running benchmarks...\n\n";
    
    std::vector<std::vector<double>> all_scores(8);
    
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        std::cout << "--- Iteration " << (i + 1) << " of " << BENCHMARK_ITERATIONS << " ---\n";
        auto scores = run_all_benchmarks(false);
        for (size_t j = 0; j < scores.size(); ++j) {
            all_scores[j].push_back(scores[j]);
        }
        std::cout << "\n";
    }
    
    // Calculate averages
    std::vector<double> avg_scores;
    for (const auto& s : all_scores) {
        double sum = 0;
        for (double v : s) sum += v;
        avg_scores.push_back(sum / s.size());
    }
    
    // Print final results
    print_final_results(avg_scores);
    
    return 0;
}