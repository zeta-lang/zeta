#include <immintrin.h>

void __zeta_cpu_relax(void) {
    _mm_pause();
}

long long __zeta_streq(const unsigned char *a, const unsigned char *b) {
    if (!a || !b) return a == b ? 1 : 0;
    unsigned long long a_len = *(unsigned long long *)a;
    unsigned long long b_len = *(unsigned long long *)b;
    if (a_len != b_len) return 0;
    return __builtin_memcmp(a + 8, b + 8, a_len) == 0;
}

void *__zeta_memset(void *dst, int value, unsigned long long size) {
    return __builtin_memset(dst, value, size);
}
