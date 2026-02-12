#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <time.h>

// 定义测试内存大小：1GB
// 1024 MB * 1024 KB * 1024 Bytes
#define MEM_SIZE (1024ULL * 1024 * 1024) 

int main() {
    printf("========================================\n");
    printf("      内存交换测试 (1GB 版本)\n");
    printf("========================================\n\n");

    // 1. 分配 1GB 内存
    printf("[1] 正在尝试分配 1GB 内存...\n");
    
    // 使用 VirtualAlloc 直接从操作系统申请大块内存，比 malloc 更适合这种底层测试
    // MEM_COMMIT: 提交物理内存, MEM_RESERVE: 保留地址空间
    char *large_array = (char *)VirtualAlloc(NULL, MEM_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    if (large_array == NULL) {
        printf("错误：内存分配失败！可能是系统可用内存不足。\n");
        return 1;
    }
    printf(" -> 成功分配 1GB 内存。\n\n");

    // 2. 第一次填充数据 (写入)
    // 这一步是必须的，只有写入了数据，操作系统才会真正分配物理内存页 (缺页中断)
    printf("[2] 正在向 1GB 内存写入数据 (初始化)...\n");
    clock_t start = clock();
    
    // 为了提高效率，我们按页大小 (4KB) 跳跃式写入，或者每隔一段写入
    // 这里为了演示完整性，我们遍历写入，但步长设大一点以加快速度，
    // 只要每个 4KB 页面都被触碰到，它就会被分配物理内存。
    size_t step = 4096; // 4KB
    for (size_t i = 0; i < MEM_SIZE; i += step) {
        large_array[i] = (char)(i % 256);
    }
    
    clock_t end = clock();
    printf(" -> 初始化完成。耗时: %.2f 秒\n", (double)(end - start) / CLOCKS_PER_SEC);
    printf(" -> 此时这 1GB 数据位于【物理内存】中。\n\n");

    // 3. 强制将内存交换到磁盘 (Swap Out)
    printf("[3] 按下回车键，将这 1GB 内存强制移出到磁盘 (Pagefile)...\n");
    getchar(); 

    // 设置工作集大小为最小，强制 Windows 把该进程的内存页写入虚拟内存文件
    if (!SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1)) {
        printf("警告：SetProcessWorkingSetSize 失败，错误代码: %lu\n", GetLastError());
    } else {
        printf(" -> 已请求操作系统将内存换出。\n");
        printf(" -> 请打开任务管理器，观察本程序内存占用瞬间下降。\n");
    }

    printf("\n[4] 暂停中... (此时数据应在硬盘上)\n");
    printf("    请等待几秒钟，确保系统完成换出操作。\n");
    printf("    准备好后，按下回车键读取数据 (触发缺页中断)...\n");
    getchar();

    // 4. 再次访问数据 (触发缺页中断 / Swap In)
    printf("[5] 正在读取这 1GB 数据...\n");
    start = clock();

    // 这里的 volatile 变量是为了防止编译器优化掉读取操作
    volatile char temp; 
    for (size_t i = 0; i < MEM_SIZE; i += step) {
        temp = large_array[i];
    }

    end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf(" -> 读取完成。耗时: %.4f 秒\n", time_taken);
    printf(" -> 速度: %.2f MB/s\n\n", (1024.0) / time_taken); // 1024MB / 时间

    // 5. 释放内存
    VirtualFree(large_array, 0, MEM_RELEASE);
    printf("测试结束。按回车退出。\n");
    getchar();

    return 0;
}