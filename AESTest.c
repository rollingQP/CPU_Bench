/*
 * AES-NI vs Software AES 性能对比测试 (增强版)
 * 编译命令:
 *   Windows (PowerShell): gcc -O3 -march=native AESTest.c -o AESTest.exe
 *   Linux/Mac:            gcc -O3 -march=native AESTest.c -o AESTest
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifdef _MSC_VER
    #include <intrin.h>
    #include <wmmintrin.h>
#else
    #include <x86intrin.h>
    #include <cpuid.h>
#endif

// ==================== 测试参数配置 ====================
#define DATA_SIZE_MB    500                              // 每轮测试数据量 (MB)
#define DATA_SIZE       (DATA_SIZE_MB * 1024ULL * 1024)  // 转换为字节
#define NUM_BLOCKS      (DATA_SIZE / 16)                 // AES 块数量
#define TEST_ROUNDS     5                                // 测试轮数

// ==================== 检测 AES-NI 支持 ====================
int check_aesni_support() {
    uint32_t eax, ebx, ecx, edx;
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    ecx = cpuInfo[2];
#else
    __cpuid(1, eax, ebx, ecx, edx);
#endif
    return (ecx & (1 << 25)) != 0;
}

// 获取 CPU 型号
void get_cpu_name(char* name, size_t size) {
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 0x80000000);
    if ((unsigned)cpuInfo[0] >= 0x80000004) {
        __cpuid(cpuInfo, 0x80000002);
        memcpy(name, cpuInfo, 16);
        __cpuid(cpuInfo, 0x80000003);
        memcpy(name + 16, cpuInfo, 16);
        __cpuid(cpuInfo, 0x80000004);
        memcpy(name + 32, cpuInfo, 16);
        name[48] = '\0';
    } else {
        strcpy(name, "Unknown CPU");
    }
#else
    uint32_t regs[12];
    __cpuid(0x80000000, regs[0], regs[1], regs[2], regs[3]);
    if (regs[0] >= 0x80000004) {
        __cpuid(0x80000002, regs[0], regs[1], regs[2], regs[3]);
        __cpuid(0x80000003, regs[4], regs[5], regs[6], regs[7]);
        __cpuid(0x80000004, regs[8], regs[9], regs[10], regs[11]);
        memcpy(name, regs, 48);
        name[48] = '\0';
    } else {
        strcpy(name, "Unknown CPU");
    }
#endif
    // 去除前导空格
    char* p = name;
    while (*p == ' ') p++;
    if (p != name) memmove(name, p, strlen(p) + 1);
}

// ==================== 软件 AES 实现 (查表法) ====================

static const uint8_t sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

static const uint8_t rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

static uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) p ^= a;
        uint8_t hi = a & 0x80;
        a <<= 1;
        if (hi) a ^= 0x1b;
        b >>= 1;
    }
    return p;
}

void soft_aes_key_expansion(const uint8_t* key, uint8_t* round_keys) {
    memcpy(round_keys, key, 16);
    
    for (int i = 1; i <= 10; i++) {
        uint8_t* prev = round_keys + (i - 1) * 16;
        uint8_t* curr = round_keys + i * 16;
        
        uint8_t temp[4];
        temp[0] = sbox[prev[13]] ^ rcon[i];
        temp[1] = sbox[prev[14]];
        temp[2] = sbox[prev[15]];
        temp[3] = sbox[prev[12]];
        
        for (int j = 0; j < 4; j++) {
            curr[j] = prev[j] ^ temp[j];
        }
        for (int j = 4; j < 16; j++) {
            curr[j] = prev[j] ^ curr[j - 4];
        }
    }
}

void soft_aes_encrypt_block(const uint8_t* in, uint8_t* out, const uint8_t* round_keys) {
    uint8_t state[16];
    memcpy(state, in, 16);
    
    for (int i = 0; i < 16; i++) {
        state[i] ^= round_keys[i];
    }
    
    for (int round = 1; round <= 9; round++) {
        for (int i = 0; i < 16; i++) {
            state[i] = sbox[state[i]];
        }
        
        uint8_t temp;
        temp = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = temp;
        temp = state[2]; state[2] = state[10]; state[10] = temp;
        temp = state[6]; state[6] = state[14]; state[14] = temp;
        temp = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = state[3]; state[3] = temp;
        
        for (int col = 0; col < 4; col++) {
            uint8_t* c = state + col * 4;
            uint8_t a0 = c[0], a1 = c[1], a2 = c[2], a3 = c[3];
            c[0] = gmul(a0, 2) ^ gmul(a1, 3) ^ a2 ^ a3;
            c[1] = a0 ^ gmul(a1, 2) ^ gmul(a2, 3) ^ a3;
            c[2] = a0 ^ a1 ^ gmul(a2, 2) ^ gmul(a3, 3);
            c[3] = gmul(a0, 3) ^ a1 ^ a2 ^ gmul(a3, 2);
        }
        
        for (int i = 0; i < 16; i++) {
            state[i] ^= round_keys[round * 16 + i];
        }
    }
    
    for (int i = 0; i < 16; i++) {
        state[i] = sbox[state[i]];
    }
    uint8_t temp;
    temp = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = temp;
    temp = state[2]; state[2] = state[10]; state[10] = temp;
    temp = state[6]; state[6] = state[14]; state[14] = temp;
    temp = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = state[3]; state[3] = temp;
    for (int i = 0; i < 16; i++) {
        state[i] ^= round_keys[160 + i];
    }
    
    memcpy(out, state, 16);
}

// ==================== AES-NI 硬件实现 ====================

static __m128i aesni_key_expand(__m128i key, __m128i keygened) {
    keygened = _mm_shuffle_epi32(keygened, 0xFF);
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, keygened);
}

void aesni_key_expansion(const uint8_t* key, __m128i* round_keys) {
    round_keys[0] = _mm_loadu_si128((const __m128i*)key);
    round_keys[1]  = aesni_key_expand(round_keys[0],  _mm_aeskeygenassist_si128(round_keys[0],  0x01));
    round_keys[2]  = aesni_key_expand(round_keys[1],  _mm_aeskeygenassist_si128(round_keys[1],  0x02));
    round_keys[3]  = aesni_key_expand(round_keys[2],  _mm_aeskeygenassist_si128(round_keys[2],  0x04));
    round_keys[4]  = aesni_key_expand(round_keys[3],  _mm_aeskeygenassist_si128(round_keys[3],  0x08));
    round_keys[5]  = aesni_key_expand(round_keys[4],  _mm_aeskeygenassist_si128(round_keys[4],  0x10));
    round_keys[6]  = aesni_key_expand(round_keys[5],  _mm_aeskeygenassist_si128(round_keys[5],  0x20));
    round_keys[7]  = aesni_key_expand(round_keys[6],  _mm_aeskeygenassist_si128(round_keys[6],  0x40));
    round_keys[8]  = aesni_key_expand(round_keys[7],  _mm_aeskeygenassist_si128(round_keys[7],  0x80));
    round_keys[9]  = aesni_key_expand(round_keys[8],  _mm_aeskeygenassist_si128(round_keys[8],  0x1B));
    round_keys[10] = aesni_key_expand(round_keys[9],  _mm_aeskeygenassist_si128(round_keys[9],  0x36));
}

// 单块加密
void aesni_encrypt_block(const uint8_t* in, uint8_t* out, const __m128i* round_keys) {
    __m128i state = _mm_loadu_si128((const __m128i*)in);
    
    state = _mm_xor_si128(state, round_keys[0]);
    state = _mm_aesenc_si128(state, round_keys[1]);
    state = _mm_aesenc_si128(state, round_keys[2]);
    state = _mm_aesenc_si128(state, round_keys[3]);
    state = _mm_aesenc_si128(state, round_keys[4]);
    state = _mm_aesenc_si128(state, round_keys[5]);
    state = _mm_aesenc_si128(state, round_keys[6]);
    state = _mm_aesenc_si128(state, round_keys[7]);
    state = _mm_aesenc_si128(state, round_keys[8]);
    state = _mm_aesenc_si128(state, round_keys[9]);
    state = _mm_aesenclast_si128(state, round_keys[10]);
    
    _mm_storeu_si128((__m128i*)out, state);
}

// 4 块并行加密 (利用 CPU 流水线)
void aesni_encrypt_4blocks(const uint8_t* in, uint8_t* out, const __m128i* round_keys) {
    __m128i s0 = _mm_loadu_si128((const __m128i*)(in));
    __m128i s1 = _mm_loadu_si128((const __m128i*)(in + 16));
    __m128i s2 = _mm_loadu_si128((const __m128i*)(in + 32));
    __m128i s3 = _mm_loadu_si128((const __m128i*)(in + 48));
    
    s0 = _mm_xor_si128(s0, round_keys[0]);
    s1 = _mm_xor_si128(s1, round_keys[0]);
    s2 = _mm_xor_si128(s2, round_keys[0]);
    s3 = _mm_xor_si128(s3, round_keys[0]);
    
    for (int i = 1; i <= 9; i++) {
        s0 = _mm_aesenc_si128(s0, round_keys[i]);
        s1 = _mm_aesenc_si128(s1, round_keys[i]);
        s2 = _mm_aesenc_si128(s2, round_keys[i]);
        s3 = _mm_aesenc_si128(s3, round_keys[i]);
    }
    
    s0 = _mm_aesenclast_si128(s0, round_keys[10]);
    s1 = _mm_aesenclast_si128(s1, round_keys[10]);
    s2 = _mm_aesenclast_si128(s2, round_keys[10]);
    s3 = _mm_aesenclast_si128(s3, round_keys[10]);
    
    _mm_storeu_si128((__m128i*)(out),      s0);
    _mm_storeu_si128((__m128i*)(out + 16), s1);
    _mm_storeu_si128((__m128i*)(out + 32), s2);
    _mm_storeu_si128((__m128i*)(out + 48), s3);
}

// 8 块并行加密 (更充分利用流水线)
void aesni_encrypt_8blocks(const uint8_t* in, uint8_t* out, const __m128i* round_keys) {
    __m128i s0 = _mm_loadu_si128((const __m128i*)(in));
    __m128i s1 = _mm_loadu_si128((const __m128i*)(in + 16));
    __m128i s2 = _mm_loadu_si128((const __m128i*)(in + 32));
    __m128i s3 = _mm_loadu_si128((const __m128i*)(in + 48));
    __m128i s4 = _mm_loadu_si128((const __m128i*)(in + 64));
    __m128i s5 = _mm_loadu_si128((const __m128i*)(in + 80));
    __m128i s6 = _mm_loadu_si128((const __m128i*)(in + 96));
    __m128i s7 = _mm_loadu_si128((const __m128i*)(in + 112));
    
    s0 = _mm_xor_si128(s0, round_keys[0]);
    s1 = _mm_xor_si128(s1, round_keys[0]);
    s2 = _mm_xor_si128(s2, round_keys[0]);
    s3 = _mm_xor_si128(s3, round_keys[0]);
    s4 = _mm_xor_si128(s4, round_keys[0]);
    s5 = _mm_xor_si128(s5, round_keys[0]);
    s6 = _mm_xor_si128(s6, round_keys[0]);
    s7 = _mm_xor_si128(s7, round_keys[0]);
    
    for (int i = 1; i <= 9; i++) {
        s0 = _mm_aesenc_si128(s0, round_keys[i]);
        s1 = _mm_aesenc_si128(s1, round_keys[i]);
        s2 = _mm_aesenc_si128(s2, round_keys[i]);
        s3 = _mm_aesenc_si128(s3, round_keys[i]);
        s4 = _mm_aesenc_si128(s4, round_keys[i]);
        s5 = _mm_aesenc_si128(s5, round_keys[i]);
        s6 = _mm_aesenc_si128(s6, round_keys[i]);
        s7 = _mm_aesenc_si128(s7, round_keys[i]);
    }
    
    s0 = _mm_aesenclast_si128(s0, round_keys[10]);
    s1 = _mm_aesenclast_si128(s1, round_keys[10]);
    s2 = _mm_aesenclast_si128(s2, round_keys[10]);
    s3 = _mm_aesenclast_si128(s3, round_keys[10]);
    s4 = _mm_aesenclast_si128(s4, round_keys[10]);
    s5 = _mm_aesenclast_si128(s5, round_keys[10]);
    s6 = _mm_aesenclast_si128(s6, round_keys[10]);
    s7 = _mm_aesenclast_si128(s7, round_keys[10]);
    
    _mm_storeu_si128((__m128i*)(out),       s0);
    _mm_storeu_si128((__m128i*)(out + 16),  s1);
    _mm_storeu_si128((__m128i*)(out + 32),  s2);
    _mm_storeu_si128((__m128i*)(out + 48),  s3);
    _mm_storeu_si128((__m128i*)(out + 64),  s4);
    _mm_storeu_si128((__m128i*)(out + 80),  s5);
    _mm_storeu_si128((__m128i*)(out + 96),  s6);
    _mm_storeu_si128((__m128i*)(out + 112), s7);
}

// ==================== 性能测试 ====================

double get_time_ms() {
    return (double)clock() / CLOCKS_PER_SEC * 1000.0;
}

void print_progress_bar(int current, int total, const char* label) {
    int bar_width = 30;
    float progress = (float)current / total;
    int filled = (int)(bar_width * progress);
    
    printf("\r%s [", label);
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %d/%d", current, total);
    fflush(stdout);
}

int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     AES-NI vs Software AES 性能对比测试 (增强版)             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // CPU 信息
    char cpu_name[64];
    get_cpu_name(cpu_name, sizeof(cpu_name));
    printf("CPU 型号:       %s\n", cpu_name);
    
    int has_aesni = check_aesni_support();
    printf("AES-NI 支持:    %s\n", has_aesni ? "✓ 是" : "✗ 否");
    
    if (!has_aesni) {
        printf("\n⚠ 警告: 您的 CPU 不支持 AES-NI，只能运行软件测试\n");
    }
    
    printf("\n");
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ 测试配置                                                    │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ 每轮数据量:   %-10d MB                                 │\n", DATA_SIZE_MB);
    printf("│ 测试轮数:     %-10d 轮                                 │\n", TEST_ROUNDS);
    printf("│ 总数据量:     %-10d MB                                 │\n", DATA_SIZE_MB * TEST_ROUNDS);
    printf("│ AES 块数量:   %-10llu 块/轮                             │\n", (unsigned long long)NUM_BLOCKS);
    printf("└─────────────────────────────────────────────────────────────┘\n\n");
    
    // 分配内存
    printf("正在分配 %d MB 内存...\n", DATA_SIZE_MB * 2);
    uint8_t* data = (uint8_t*)malloc(DATA_SIZE);
    uint8_t* result = (uint8_t*)malloc(DATA_SIZE);
    if (!data || !result) {
        printf("✗ 内存分配失败!\n");
        return 1;
    }
    printf("✓ 内存分配成功\n\n");
    
    // 初始化测试数据
    printf("正在初始化测试数据...\n");
    for (size_t i = 0; i < DATA_SIZE; i++) {
        data[i] = (uint8_t)(i * 17 + 31);  // 伪随机填充
    }
    printf("✓ 数据初始化完成\n\n");
    
    // 测试密钥
    uint8_t key[16] = {
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
    };
    
    double soft_times[TEST_ROUNDS];
    double aesni_times[TEST_ROUNDS];
    double aesni_4x_times[TEST_ROUNDS];
    double aesni_8x_times[TEST_ROUNDS];
    
    // ===== 软件 AES 测试 =====
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf(" 测试 1: 软件 AES 加密\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    uint8_t soft_round_keys[176];
    soft_aes_key_expansion(key, soft_round_keys);
    
    for (int r = 0; r < TEST_ROUNDS; r++) {
        print_progress_bar(r + 1, TEST_ROUNDS, "进度");
        
        double start = get_time_ms();
        for (size_t i = 0; i < NUM_BLOCKS; i++) {
            soft_aes_encrypt_block(data + i * 16, result + i * 16, soft_round_keys);
        }
        soft_times[r] = get_time_ms() - start;
    }
    printf("\n");
    
    // 保存软件结果用于验证
    uint8_t soft_result_sample[16];
    memcpy(soft_result_sample, result, 16);
    
    // 计算软件 AES 统计
    double soft_total = 0, soft_min = soft_times[0], soft_max = soft_times[0];
    for (int r = 0; r < TEST_ROUNDS; r++) {
        soft_total += soft_times[r];
        if (soft_times[r] < soft_min) soft_min = soft_times[r];
        if (soft_times[r] > soft_max) soft_max = soft_times[r];
    }
    double soft_avg = soft_total / TEST_ROUNDS;
    double soft_speed = (DATA_SIZE / (1024.0 * 1024.0)) / (soft_avg / 1000.0);
    
    printf("  最小: %.2f ms | 最大: %.2f ms | 平均: %.2f ms\n", soft_min, soft_max, soft_avg);
    printf("  吞吐量: %.2f MB/s\n\n", soft_speed);
    
    // ===== AES-NI 测试 =====
    double aesni_speed = 0, aesni_4x_speed = 0, aesni_8x_speed = 0;
    double aesni_avg = 0, aesni_4x_avg = 0, aesni_8x_avg = 0;
    
    if (has_aesni) {
        __m128i aesni_round_keys[11];
        aesni_key_expansion(key, aesni_round_keys);
        
        // 测试 2: AES-NI 单块
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf(" 测试 2: AES-NI 单块加密\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        for (int r = 0; r < TEST_ROUNDS; r++) {
            print_progress_bar(r + 1, TEST_ROUNDS, "进度");
            
            double start = get_time_ms();
            for (size_t i = 0; i < NUM_BLOCKS; i++) {
                aesni_encrypt_block(data + i * 16, result + i * 16, aesni_round_keys);
            }
            aesni_times[r] = get_time_ms() - start;
        }
        printf("\n");
        
        double aesni_total = 0, aesni_min = aesni_times[0], aesni_max = aesni_times[0];
        for (int r = 0; r < TEST_ROUNDS; r++) {
            aesni_total += aesni_times[r];
            if (aesni_times[r] < aesni_min) aesni_min = aesni_times[r];
            if (aesni_times[r] > aesni_max) aesni_max = aesni_times[r];
        }
        aesni_avg = aesni_total / TEST_ROUNDS;
        aesni_speed = (DATA_SIZE / (1024.0 * 1024.0)) / (aesni_avg / 1000.0);
        
        printf("  最小: %.2f ms | 最大: %.2f ms | 平均: %.2f ms\n", aesni_min, aesni_max, aesni_avg);
        printf("  吞吐量: %.2f MB/s\n\n", aesni_speed);
        
        // 测试 3: AES-NI 4 块并行
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf(" 测试 3: AES-NI 4 块并行加密\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        for (int r = 0; r < TEST_ROUNDS; r++) {
            print_progress_bar(r + 1, TEST_ROUNDS, "进度");
            
            double start = get_time_ms();
            for (size_t i = 0; i < NUM_BLOCKS; i += 4) {
                aesni_encrypt_4blocks(data + i * 16, result + i * 16, aesni_round_keys);
            }
            aesni_4x_times[r] = get_time_ms() - start;
        }
        printf("\n");
        
        double aesni_4x_total = 0, aesni_4x_min = aesni_4x_times[0], aesni_4x_max = aesni_4x_times[0];
        for (int r = 0; r < TEST_ROUNDS; r++) {
            aesni_4x_total += aesni_4x_times[r];
            if (aesni_4x_times[r] < aesni_4x_min) aesni_4x_min = aesni_4x_times[r];
            if (aesni_4x_times[r] > aesni_4x_max) aesni_4x_max = aesni_4x_times[r];
        }
        aesni_4x_avg = aesni_4x_total / TEST_ROUNDS;
        aesni_4x_speed = (DATA_SIZE / (1024.0 * 1024.0)) / (aesni_4x_avg / 1000.0);
        
        printf("  最小: %.2f ms | 最大: %.2f ms | 平均: %.2f ms\n", aesni_4x_min, aesni_4x_max, aesni_4x_avg);
        printf("  吞吐量: %.2f MB/s\n\n", aesni_4x_speed);
        
        // 测试 4: AES-NI 8 块并行
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf(" 测试 4: AES-NI 8 块并行加密\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        for (int r = 0; r < TEST_ROUNDS; r++) {
            print_progress_bar(r + 1, TEST_ROUNDS, "进度");
            
            double start = get_time_ms();
            for (size_t i = 0; i < NUM_BLOCKS; i += 8) {
                aesni_encrypt_8blocks(data + i * 16, result + i * 16, aesni_round_keys);
            }
            aesni_8x_times[r] = get_time_ms() - start;
        }
        printf("\n");
        
        double aesni_8x_total = 0, aesni_8x_min = aesni_8x_times[0], aesni_8x_max = aesni_8x_times[0];
        for (int r = 0; r < TEST_ROUNDS; r++) {
            aesni_8x_total += aesni_8x_times[r];
            if (aesni_8x_times[r] < aesni_8x_min) aesni_8x_min = aesni_8x_times[r];
            if (aesni_8x_times[r] > aesni_8x_max) aesni_8x_max = aesni_8x_times[r];
        }
        aesni_8x_avg = aesni_8x_total / TEST_ROUNDS;
        aesni_8x_speed = (DATA_SIZE / (1024.0 * 1024.0)) / (aesni_8x_avg / 1000.0);
        
        printf("  最小: %.2f ms | 最大: %.2f ms | 平均: %.2f ms\n", aesni_8x_min, aesni_8x_max, aesni_8x_avg);
        printf("  吞吐量: %.2f MB/s\n\n", aesni_8x_speed);
        
        // 验证结果
        aesni_encrypt_block(data, result, aesni_round_keys);
        int match = (memcmp(soft_result_sample, result, 16) == 0);
        printf("结果验证: %s\n\n", match ? "✓ 通过 (软件与硬件结果一致)" : "✗ 失败");
    }
    
    // ===== 最终结果汇总 =====
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                      性能对比结果汇总                        ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  测试项目              │  吞吐量 (MB/s)  │  加速比            ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  软件 AES              │ %12.2f    │     1.00x (基准)   ║\n", soft_speed);
    if (has_aesni) {
        printf("║  AES-NI 单块           │ %12.2f    │ %8.2fx          ║\n", aesni_speed, aesni_speed / soft_speed);
        printf("║  AES-NI 4块并行        │ %12.2f    │ %8.2fx          ║\n", aesni_4x_speed, aesni_4x_speed / soft_speed);
        printf("║  AES-NI 8块并行        │ %12.2f    │ %8.2fx          ║\n", aesni_8x_speed, aesni_8x_speed / soft_speed);
    }
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    if (has_aesni) {
        printf("📊 结论:\n");
        printf("   • AES-NI 比纯软件实现快 %.1f 倍 (单块)\n", aesni_speed / soft_speed);
        printf("   • 使用 8 块并行可进一步提升至 %.1f 倍\n", aesni_8x_speed / soft_speed);
        printf("   • 并行处理利用了 CPU 流水线，减少指令延迟影响\n");
    }
    
    free(data);
    free(result);
    
    printf("\n测试完成!\n");
    return 0;
}