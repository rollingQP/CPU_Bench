#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <random>
#include <cstring>
#include <omp.h>
#include <immintrin.h> // For AVX/Streaming intrinsics

/**
 * RAM Speed Benchmark Tool (Kernel-Level Optimization + Latency)
 * * Changes in v6.2:
 * 1. Added Pointer-Chasing Latency Test.
 * 2. Maintained Streaming Stores and AVX Read kernels.
 * * Compile:
 * g++ -O3 -mavx2 -march=native -fopenmp RamBenchmark.cpp -o RamBenchmark
 */

using namespace std;
using namespace std::chrono;

const size_t BUFFER_SIZE_MB = 2048; 
const size_t BUFFER_SIZE_BYTES = BUFFER_SIZE_MB * 1024 * 1024;
const size_t NUM_ELEMENTS = BUFFER_SIZE_BYTES / sizeof(uint64_t);
const int ITERATIONS = 5;

// Helper for aligned allocation
void* aligned_alloc_helper(size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, size) != 0) return nullptr;
    return ptr;
}

void run_benchmarks() {
    uint64_t* src = (uint64_t*)aligned_alloc_helper(BUFFER_SIZE_BYTES);
    uint64_t* dest = (uint64_t*)aligned_alloc_helper(BUFFER_SIZE_BYTES);

    if (!src || !dest) {
        cerr << "Failed to allocate aligned memory." << endl;
        return;
    }

    // Warm-up
    #pragma omp parallel for
    for (size_t j = 0; j < NUM_ELEMENTS; ++j) {
        src[j] = j;
        dest[j] = 0;
    }

    // --- 1. STREAMING WRITE TEST ---
    cout << "--- Streaming Write (AVX + Bypass Cache) (" << BUFFER_SIZE_MB << " MB) ---" << endl;
    vector<double> write_results;
    for (int i = 0; i < ITERATIONS; ++i) {
        auto start = high_resolution_clock::now();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            size_t chunk = NUM_ELEMENTS / nthreads;
            size_t start_idx = tid * chunk;
            __m256i* v_dest = (__m256i*)&dest[start_idx];
            __m256i val = _mm256_set1_epi64x(0xDEADBEEF);
            size_t v_count = chunk / 4;
            for (size_t j = 0; j < v_count; ++j) {
                _mm256_stream_si256(&v_dest[j], val);
            }
        }
        _mm_sfence();
        auto end = high_resolution_clock::now();
        write_results.push_back((BUFFER_SIZE_MB / 1024.0) / duration<double>(end - start).count());
    }
    cout << "Avg Write: " << fixed << setprecision(2) << accumulate(write_results.begin(), write_results.end(), 0.0) / ITERATIONS << " GB/s" << endl << endl;

    // --- 2. AVX READ TEST ---
    cout << "--- AVX Parallel Read (" << BUFFER_SIZE_MB << " MB) ---" << endl;
    vector<double> read_results;
    for (int i = 0; i < ITERATIONS; ++i) {
        auto start = high_resolution_clock::now();
        uint64_t global_total = 0;
        #pragma omp parallel reduction(+:global_total)
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            size_t chunk = NUM_ELEMENTS / nthreads;
            size_t start_idx = tid * chunk;
            __m256i* v_src = (__m256i*)&src[start_idx];
            size_t v_count = chunk / 4;
            __m256i local_v_sum = _mm256_setzero_si256();
            for (size_t j = 0; j < v_count; ++j) {
                local_v_sum = _mm256_add_epi64(local_v_sum, _mm256_load_si256(&v_src[j]));
            }
            uint64_t temp[4];
            _mm256_storeu_si256((__m256i*)temp, local_v_sum);
            global_total += (temp[0] + temp[1] + temp[2] + temp[3]);
        }
        auto end = high_resolution_clock::now();
        if (global_total == 1) cout << " ";
        read_results.push_back((BUFFER_SIZE_MB / 1024.0) / duration<double>(end - start).count());
    }
    cout << "Avg Read: " << fixed << setprecision(2) << accumulate(read_results.begin(), read_results.end(), 0.0) / ITERATIONS << " GB/s" << endl << endl;

    // --- 3. MEMORY MOVE ---
    cout << "--- Memory Move (Read + Streaming Write) ---" << endl;
    vector<double> move_results;
    for (int i = 0; i < ITERATIONS; ++i) {
        auto start = high_resolution_clock::now();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            size_t chunk = NUM_ELEMENTS / nthreads;
            size_t start_idx = tid * chunk;
            __m256i* v_src = (__m256i*)&src[start_idx];
            __m256i* v_dest = (__m256i*)&dest[start_idx];
            size_t v_count = chunk / 4;
            for (size_t j = 0; j < v_count; ++j) {
                _mm256_stream_si256(&v_dest[j], _mm256_load_si256(&v_src[j]));
            }
        }
        _mm_sfence();
        auto end = high_resolution_clock::now();
        move_results.push_back((BUFFER_SIZE_MB / 1024.0) / duration<double>(end - start).count());
    }
    cout << "Avg Move: " << fixed << setprecision(2) << accumulate(move_results.begin(), move_results.end(), 0.0) / ITERATIONS << " GB/s" << endl << endl;

    // --- 4. LATENCY TEST (Pointer Chasing) ---
    cout << "--- Random Access Latency (Pointer Chasing) ---" << endl;
    // We use the 'src' buffer as the space for the chain
    // A chain of indices is created such that src[i] = next_i
    vector<size_t> indices(NUM_ELEMENTS);
    iota(indices.begin(), indices.end(), 0);
    
    // Shuffle using a fixed seed for consistency
    mt19937_64 g(1337);
    shuffle(indices.begin(), indices.end(), g);

    // Initialize the pointer chain in the allocated buffer
    for (size_t i = 0; i < NUM_ELEMENTS - 1; ++i) {
        src[indices[i]] = indices[i+1];
    }
    src[indices[NUM_ELEMENTS - 1]] = indices[0];

    const size_t LATENCY_ITERATIONS = 50000000; // 50 million hops
    size_t curr = indices[0];
    
    auto lat_start = high_resolution_clock::now();
    for (size_t i = 0; i < LATENCY_ITERATIONS; ++i) {
        // Volatile to ensure the load isn't optimized out
        curr = *(volatile uint64_t*)&src[curr];
    }
    auto lat_end = high_resolution_clock::now();
    
    if (curr == 0xBAADF00D) cout << " "; // Prevent optimization
    
    double total_ns = duration_cast<nanoseconds>(lat_end - lat_start).count();
    cout << "Average Latency: " << fixed << setprecision(2) << (total_ns / LATENCY_ITERATIONS) << " ns" << endl << endl;

    free(src);
    free(dest);
}

int main() {
    int threads = omp_get_max_threads();
    cout << "RAM Benchmark v6.2 (AVX + Latency)" << endl;
    cout << "Threads: " << threads << " | Dataset: " << BUFFER_SIZE_MB << " MB" << endl;
    cout << "---------------------------------------" << endl;
    run_benchmarks();
    return 0;
}