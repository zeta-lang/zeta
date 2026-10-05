#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>

static void __zeta_atomic_bad_order(void) {
    abort();
}

/*
 * Zeta Ordering:
 *
 * 0 = Relaxed
 * 1 = Acquire
 * 2 = Release
 * 3 = AcqRel
 * 4 = SeqCst
 *
 * The order is switched here so that the argument passed to the
 * __atomic builtin is still a compile-time constant.
 */

#define ZETA_ATOMIC_LOAD(T, N)                                      \
T __zeta_atomic_load_##N(const T *ptr, int order) {                 \
    switch (order) {                                                \
        case 0: return __atomic_load_n(ptr, __ATOMIC_RELAXED);      \
        case 1: return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);      \
        case 4: return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);      \
        default: __zeta_atomic_bad_order();                        \
    }                                                               \
    return 0;                                                       \
}

#define ZETA_ATOMIC_STORE(T, N)                                    \
void __zeta_atomic_store_##N(T *ptr, T value, int order) {          \
    switch (order) {                                                \
        case 0: __atomic_store_n(ptr, value, __ATOMIC_RELAXED);    \
                return;                                             \
        case 2: __atomic_store_n(ptr, value, __ATOMIC_RELEASE);    \
                return;                                             \
        case 4: __atomic_store_n(ptr, value, __ATOMIC_SEQ_CST);    \
                return;                                             \
        default: __zeta_atomic_bad_order();                        \
    }                                                               \
}

#define ZETA_ATOMIC_SWAP(T, N)                                     \
T __zeta_atomic_swap_##N(T *ptr, T value, int order) {              \
    switch (order) {                                                \
        case 0: return __atomic_exchange_n(ptr, value, __ATOMIC_RELAXED);  \
        case 1: return __atomic_exchange_n(ptr, value, __ATOMIC_ACQUIRE); \
        case 2: return __atomic_exchange_n(ptr, value, __ATOMIC_RELEASE); \
        case 3: return __atomic_exchange_n(ptr, value, __ATOMIC_ACQ_REL); \
        case 4: return __atomic_exchange_n(ptr, value, __ATOMIC_SEQ_CST); \
        default: __zeta_atomic_bad_order();                        \
    }                                                               \
    return 0;                                                       \
}

#define ZETA_ATOMIC_RMW(T, N, OP, NAME)                             \
T __zeta_atomic_##NAME##_##N(T *ptr, T value, int order) {          \
    switch (order) {                                                \
        case 0: return OP(ptr, value, __ATOMIC_RELAXED);             \
        case 1: return OP(ptr, value, __ATOMIC_ACQUIRE);             \
        case 2: return OP(ptr, value, __ATOMIC_RELEASE);             \
        case 3: return OP(ptr, value, __ATOMIC_ACQ_REL);             \
        case 4: return OP(ptr, value, __ATOMIC_SEQ_CST);             \
        default: __zeta_atomic_bad_order();                        \
    }                                                               \
    return 0;                                                       \
}

/*
 * Returns the value observed before the CAS, matching Zeta's
 * current AtomicCas intrinsic semantics.
 *
 * Valid failure orderings:
 *
 * success Relaxed -> failure Relaxed
 * success Acquire -> failure Relaxed / Acquire
 * success Release -> failure Relaxed
 * success AcqRel -> failure Relaxed / Acquire
 * success SeqCst -> failure Relaxed / Acquire / SeqCst
 */

#define ZETA_ATOMIC_CAS(T, N)                                      \
T __zeta_atomic_cas_##N(                                           \
    T *ptr,                                                        \
    T expected,                                                    \
    T desired,                                                     \
    int success,                                                    \
    int failure                                                       \
) {                                                                \
    switch (success) {                                             \
        case 0:                                                     \
            if (failure != 0) __zeta_atomic_bad_order();           \
            break;                                                  \
        case 1:                                                     \
            if (failure == 0) break;                                \
            if (failure == 1) break;                                \
            __zeta_atomic_bad_order();                              \
            break;                                                  \
        case 2:                                                     \
            if (failure != 0) __zeta_atomic_bad_order();            \
            break;                                                  \
        case 3:                                                     \
            if (failure == 0) break;                                \
            if (failure == 1) break;                                \
            __zeta_atomic_bad_order();                              \
            break;                                                  \
        case 4:                                                     \
            if (failure == 0) break;                                \
            if (failure == 1) break;                                \
            if (failure == 4) break;                                \
            __zeta_atomic_bad_order();                              \
            break;                                                  \
        default:                                                    \
            __zeta_atomic_bad_order();                              \
    }                                                               \
                                                                        \
    switch (success) {                                              \
        case 0:                                                     \
            __atomic_compare_exchange_n(                            \
                ptr, &expected, desired, 0,                         \
                __ATOMIC_RELAXED, __ATOMIC_RELAXED);                \
            break;                                                  \
                                                                        \
        case 1:                                                     \
            switch (failure) {                                      \
                case 0:                                             \
                    __atomic_compare_exchange_n(                    \
                        ptr, &expected, desired, 0,                 \
                        __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);         \
                    break;                                          \
                case 1:                                             \
                    __atomic_compare_exchange_n(                    \
                        ptr, &expected, desired, 0,                 \
                        __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE);         \
                    break;                                          \
                default:                                            \
                    __zeta_atomic_bad_order();                      \
            }                                                       \
            break;                                                  \
                                                                        \
        case 2:                                                     \
            __atomic_compare_exchange_n(                            \
                ptr, &expected, desired, 0,                         \
                __ATOMIC_RELEASE, __ATOMIC_RELAXED);               \
            break;                                                  \
                                                                        \
        case 3:                                                     \
            switch (failure) {                                      \
                case 0:                                             \
                    __atomic_compare_exchange_n(                    \
                        ptr, &expected, desired, 0,                 \
                        __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);         \
                    break;                                          \
                case 1:                                             \
                    __atomic_compare_exchange_n(                    \
                        ptr, &expected, desired, 0,                 \
                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);         \
                    break;                                          \
                default:                                            \
                    __zeta_atomic_bad_order();                      \
            }                                                       \
            break;                                                  \
                                                                        \
        case 4:                                                     \
            switch (failure) {                                      \
                case 0:                                             \
                    __atomic_compare_exchange_n(                    \
                        ptr, &expected, desired, 0,                 \
                        __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);         \
                    break;                                          \
                case 1:                                             \
                    __atomic_compare_exchange_n(                    \
                        ptr, &expected, desired, 0,                 \
                        __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);         \
                    break;                                          \
                case 4:                                             \
                    __atomic_compare_exchange_n(                    \
                        ptr, &expected, desired, 0,                 \
                        __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);         \
                    break;                                          \
                default:                                            \
                    __zeta_atomic_bad_order();                      \
            }                                                       \
            break;                                                  \
                                                                        \
        default:                                                    \
            __zeta_atomic_bad_order();                              \
    }                                                               \
                                                                        \
    return expected;                                                \
}

#define ZETA_ATOMIC_FENCE_CASE(ORDER, C_ORDER)                      \
    case ORDER:                                                     \
        __atomic_thread_fence(C_ORDER);                             \
        return;

void __zeta_atomic_fence(int order) {
    switch (order) {
        ZETA_ATOMIC_FENCE_CASE(0, __ATOMIC_RELAXED)
        ZETA_ATOMIC_FENCE_CASE(1, __ATOMIC_ACQUIRE)
        ZETA_ATOMIC_FENCE_CASE(2, __ATOMIC_RELEASE)
        ZETA_ATOMIC_FENCE_CASE(3, __ATOMIC_ACQ_REL)
        ZETA_ATOMIC_FENCE_CASE(4, __ATOMIC_SEQ_CST)
        default:
            __zeta_atomic_bad_order();
    }
}

/* Generate all supported atomic widths. */
ZETA_ATOMIC_LOAD(uint8_t,  u8)
ZETA_ATOMIC_LOAD(uint16_t, u16)
ZETA_ATOMIC_LOAD(uint32_t, u32)
ZETA_ATOMIC_LOAD(uint64_t, u64)

ZETA_ATOMIC_STORE(uint8_t,  u8)
ZETA_ATOMIC_STORE(uint16_t, u16)
ZETA_ATOMIC_STORE(uint32_t, u32)
ZETA_ATOMIC_STORE(uint64_t, u64)

ZETA_ATOMIC_SWAP(uint8_t,  u8)
ZETA_ATOMIC_SWAP(uint16_t, u16)
ZETA_ATOMIC_SWAP(uint32_t, u32)
ZETA_ATOMIC_SWAP(uint64_t, u64)

ZETA_ATOMIC_CAS(uint8_t,  u8)
ZETA_ATOMIC_CAS(uint16_t, u16)
ZETA_ATOMIC_CAS(uint32_t, u32)
ZETA_ATOMIC_CAS(uint64_t, u64)

ZETA_ATOMIC_RMW(uint8_t,  u8, __atomic_fetch_add, add)
ZETA_ATOMIC_RMW(uint16_t, u16, __atomic_fetch_add, add)
ZETA_ATOMIC_RMW(uint32_t, u32, __atomic_fetch_add, add)
ZETA_ATOMIC_RMW(uint64_t, u64, __atomic_fetch_add, add)

ZETA_ATOMIC_RMW(uint8_t,  u8, __atomic_fetch_sub, sub)
ZETA_ATOMIC_RMW(uint16_t, u16, __atomic_fetch_sub, sub)
ZETA_ATOMIC_RMW(uint32_t, u32, __atomic_fetch_sub, sub)
ZETA_ATOMIC_RMW(uint64_t, u64, __atomic_fetch_sub, sub)

ZETA_ATOMIC_RMW(uint8_t,  u8, __atomic_fetch_and, and)
ZETA_ATOMIC_RMW(uint16_t, u16, __atomic_fetch_and, and)
ZETA_ATOMIC_RMW(uint32_t, u32, __atomic_fetch_and, and)
ZETA_ATOMIC_RMW(uint64_t, u64, __atomic_fetch_and, and)

ZETA_ATOMIC_RMW(uint8_t,  u8, __atomic_fetch_or, or)
ZETA_ATOMIC_RMW(uint16_t, u16, __atomic_fetch_or, or)
ZETA_ATOMIC_RMW(uint32_t, u32, __atomic_fetch_or, or)
ZETA_ATOMIC_RMW(uint64_t, u64, __atomic_fetch_or, or)

ZETA_ATOMIC_RMW(uint8_t,  u8, __atomic_fetch_xor, xor)
ZETA_ATOMIC_RMW(uint16_t, u16, __atomic_fetch_xor, xor)
ZETA_ATOMIC_RMW(uint32_t, u32, __atomic_fetch_xor, xor)
ZETA_ATOMIC_RMW(uint64_t, u64, __atomic_fetch_xor, xor)

void __zeta_cpu_relax(void) {
    _mm_pause();
}

void *__zeta_memset(void *dst, int value, unsigned long long size) {
    return __builtin_memset(dst, value, size);
}


void *__zeta_memcpy(void *dst, void *src, unsigned long long size) {
    return __builtin_memcpy(dst, src, size);
}
