#define SLANG_PRELUDE_EXPORT

#ifdef __CUDACC_RTC__
#define SLANG_CUDA_RTC 1
#else
#define SLANG_CUDA_RTC 0
#endif

#if SLANG_CUDA_RTC

#else

#include <cstdint>
#include <stdio.h>

#endif

// Define SLANG_CUDA_ENABLE_HALF to use the cuda_fp16 include to add half support.
// For this to work NVRTC needs to have the path to the CUDA SDK.
//
// As it stands the includes paths defined for Slang are passed down to NVRTC. Similarly defines
// defined for the Slang compile are passed down.

#ifdef SLANG_CUDA_ENABLE_HALF
// We don't want half2 operators, because it will implement comparison operators that return a
// bool(!). We want to generate those functions. Doing so means that we will have to define all
// the other half2 operators.
#define __CUDA_NO_HALF2_OPERATORS__
#include <cuda_fp16.h>
#endif

#ifdef SLANG_CUDA_ENABLE_OPTIX
#include <optix.h>
#endif

// Define slang offsetof implementation
#ifndef SLANG_OFFSET_OF
#define SLANG_OFFSET_OF(type, member) (size_t)((char*)&(((type*)0)->member) - (char*)0)
#endif

#ifndef SLANG_ALIGN_OF
#define SLANG_ALIGN_OF(type) __alignof__(type)
#endif

// Must be large enough to cause overflow and therefore infinity
#ifndef SLANG_INFINITY
#define SLANG_INFINITY ((float)(1e+300 * 1e+300))
#endif

// For now we'll disable any asserts in this prelude
#define SLANG_PRELUDE_ASSERT(x)

#ifndef SLANG_CUDA_WARP_SIZE
#define SLANG_CUDA_WARP_SIZE 32
#endif

#define SLANG_CUDA_WARP_MASK \
    (SLANG_CUDA_WARP_SIZE - 1) // Used for masking threadIdx.x to the warp lane index
#define SLANG_CUDA_WARP_BITMASK (~int(0))

//
#define SLANG_FORCE_INLINE inline

#define SLANG_CUDA_CALL __device__

#define SLANG_FORCE_INLINE inline
#define SLANG_INLINE inline


// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ZERO_INDEX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

// This macro handles how out-of-range surface coordinates are handled;
// I can equal
// cudaBoundaryModeClamp, in which case out-of-range coordinates are clamped to the valid range
// cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are
// ignored cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to
// fail.

#ifndef SLANG_CUDA_BOUNDARY_MODE
#define SLANG_CUDA_BOUNDARY_MODE cudaBoundaryModeZero

// Can be one of SLANG_CUDA_PTX_BOUNDARY_MODE. Only applies *PTX* emitted CUDA operations
// which currently is just RWTextureRW format writes
//
// .trap         causes an execution trap on out-of-bounds addresses
// .clamp        stores data at the nearest surface location (sized appropriately)
// .zero         drops stores to out-of-bounds addresses

#define SLANG_PTX_BOUNDARY_MODE "zero"
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

// Typically defined in cuda.h, but we can't ship/rely on that, so just define here
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;

// On CUDA sampler state is actually bound up with the texture object. We have a SamplerState type,
// backed as a pointer, to simplify code generation, with the downside that such a binding will take
// up uniform space, even though it will have no effect.
// TODO(JS): Consider ways to strip use of variables of this type so have no binding,
struct SamplerStateUnused;
typedef SamplerStateUnused* SamplerState;


// TODO(JS): Not clear yet if this can be handled on CUDA, by just ignoring.
// For now, just map to the index type.
typedef size_t NonUniformResourceIndex;

// Code generator will generate the specific type
template<typename T, int ROWS, int COLS>
struct Matrix;

typedef int1 bool1;
typedef int2 bool2;
typedef int3 bool3;
typedef int4 bool4;

#if SLANG_CUDA_RTC

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#endif

typedef long long longlong;
typedef unsigned long long ulonglong;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

union Union32
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float make_float(T val)
{
    return (float)val;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float _slang_fmod(float x, float y)
{
    return ::fmodf(x, y);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double _slang_fmod(double x, double y)
{
    return ::fmod(x, y);
}

#if SLANG_CUDA_ENABLE_HALF

// Add the other vector half types
struct __half1
{
    __half x;
};
struct __align__(4) __half3
{
    __half x, y, z;
};
struct __align__(4) __half4
{
    __half x, y, z, w;
};
#endif

#define SLANG_VECTOR_GET_ELEMENT(T)                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##1 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##2 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##3 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##4 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }
SLANG_VECTOR_GET_ELEMENT(int)
SLANG_VECTOR_GET_ELEMENT(uint)
SLANG_VECTOR_GET_ELEMENT(short)
SLANG_VECTOR_GET_ELEMENT(ushort)
SLANG_VECTOR_GET_ELEMENT(char)
SLANG_VECTOR_GET_ELEMENT(uchar)
SLANG_VECTOR_GET_ELEMENT(longlong)
SLANG_VECTOR_GET_ELEMENT(ulonglong)
SLANG_VECTOR_GET_ELEMENT(float)
SLANG_VECTOR_GET_ELEMENT(double)

#define SLANG_VECTOR_GET_ELEMENT_PTR(T)                                                      \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##1 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##2 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##3 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##4 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }
SLANG_VECTOR_GET_ELEMENT_PTR(int)
SLANG_VECTOR_GET_ELEMENT_PTR(uint)
SLANG_VECTOR_GET_ELEMENT_PTR(short)
SLANG_VECTOR_GET_ELEMENT_PTR(ushort)
SLANG_VECTOR_GET_ELEMENT_PTR(char)
SLANG_VECTOR_GET_ELEMENT_PTR(uchar)
SLANG_VECTOR_GET_ELEMENT_PTR(longlong)
SLANG_VECTOR_GET_ELEMENT_PTR(ulonglong)
SLANG_VECTOR_GET_ELEMENT_PTR(float)
SLANG_VECTOR_GET_ELEMENT_PTR(double)

#if SLANG_CUDA_ENABLE_HALF
SLANG_VECTOR_GET_ELEMENT(__half)
SLANG_VECTOR_GET_ELEMENT_PTR(__half)
#endif

#define SLANG_CUDA_VECTOR_BINARY_OP(T, n, op)                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal, T##n other)             \
    {                                                                                         \
        T##n result;                                                                          \
        for (int i = 0; i < n; i++)                                                           \
            *_slang_vector_get_element_ptr(&result, i) =                                      \
                _slang_vector_get_element(thisVal, i) op _slang_vector_get_element(other, i); \
        return result;                                                                        \
    }
#define SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, op)                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool##n operator op(T##n thisVal, T##n other) \
    {                                                                                \
        bool##n result;                                                              \
        for (int i = 0; i < n; i++)                                                  \
            *_slang_vector_get_element_ptr(&result, i) =                             \
                (int)(_slang_vector_get_element(thisVal, i)                          \
                          op _slang_vector_get_element(other, i));                   \
        return result;                                                               \
    }
#define SLANG_CUDA_VECTOR_UNARY_OP(T, n, op)                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal)                              \
    {                                                                                              \
        T##n result;                                                                               \
        for (int i = 0; i < n; i++)                                                                \
            *_slang_vector_get_element_ptr(&result, i) = op _slang_vector_get_element(thisVal, i); \
        return result;                                                                             \
    }

#define SLANG_CUDA_VECTOR_INT_OP(T, n)            \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, %)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ^)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, |)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, >>)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, <<)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, !)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, ~)

#define SLANG_CUDA_VECTOR_INT_OPS(T) \
    SLANG_CUDA_VECTOR_INT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 4)

SLANG_CUDA_VECTOR_INT_OPS(int)
SLANG_CUDA_VECTOR_INT_OPS(uint)
SLANG_CUDA_VECTOR_INT_OPS(ushort)
SLANG_CUDA_VECTOR_INT_OPS(short)
SLANG_CUDA_VECTOR_INT_OPS(char)
SLANG_CUDA_VECTOR_INT_OPS(uchar)
SLANG_CUDA_VECTOR_INT_OPS(longlong)
SLANG_CUDA_VECTOR_INT_OPS(ulonglong)

#define SLANG_CUDA_VECTOR_FLOAT_OP(T, n)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)
#define SLANG_CUDA_VECTOR_FLOAT_OPS(T) \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 4)

SLANG_CUDA_VECTOR_FLOAT_OPS(float)
SLANG_CUDA_VECTOR_FLOAT_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_CUDA_VECTOR_FLOAT_OPS(__half)
#endif
#define SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, n)                                             \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator%(const T##n& left, const T##n& right) \
    {                                                                                      \
        T##n result;                                                                       \
        for (int i = 0; i < n; i++)                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_fmod(                      \
                _slang_vector_get_element(left, i),                                        \
                _slang_vector_get_element(right, i));                                      \
        return result;                                                                     \
    }
#define SLANG_CUDA_FLOAT_VECTOR_MOD(T)     \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 2) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 3) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 4)

SLANG_CUDA_FLOAT_VECTOR_MOD(float)
SLANG_CUDA_FLOAT_VECTOR_MOD(double)

#if SLANG_CUDA_RTC || SLANG_CUDA_ENABLE_HALF
#define SLANG_MAKE_VECTOR(T)                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x, T y)           \
    {                                                                       \
        return T##2 {x, y};                                                 \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x, T y, T z)      \
    {                                                                       \
        return T##3 {x, y, z};                                              \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x, T y, T z, T w) \
    {                                                                       \
        return T##4 {x, y, z, w};                                           \
    }
#endif

#if SLANG_CUDA_RTC
SLANG_MAKE_VECTOR(int)
SLANG_MAKE_VECTOR(uint)
SLANG_MAKE_VECTOR(short)
SLANG_MAKE_VECTOR(ushort)
SLANG_MAKE_VECTOR(char)
SLANG_MAKE_VECTOR(uchar)
SLANG_MAKE_VECTOR(float)
SLANG_MAKE_VECTOR(double)
SLANG_MAKE_VECTOR(longlong)
SLANG_MAKE_VECTOR(ulonglong)
#endif

#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR(__half)
#endif

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool1 make_bool1(bool x)
{
    return bool1{x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x, bool y)
{
    return bool2{x, y};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x, bool y, bool z)
{
    return bool3{x, y, z};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x, bool y, bool z, bool w)
{
    return bool4{x, y, z, w};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x)
{
    return bool2{x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x)
{
    return bool3{x, x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x)
{
    return bool4{x, x, x, x};
}

#if SLANG_CUDA_RTC
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##1 make_##T##1(T x) \
    {                                                        \
        return T##1 {x};                                     \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#else
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#endif
SLANG_MAKE_VECTOR_FROM_SCALAR(int)
SLANG_MAKE_VECTOR_FROM_SCALAR(uint)
SLANG_MAKE_VECTOR_FROM_SCALAR(short)
SLANG_MAKE_VECTOR_FROM_SCALAR(ushort)
SLANG_MAKE_VECTOR_FROM_SCALAR(char)
SLANG_MAKE_VECTOR_FROM_SCALAR(uchar)
SLANG_MAKE_VECTOR_FROM_SCALAR(longlong)
SLANG_MAKE_VECTOR_FROM_SCALAR(ulonglong)
SLANG_MAKE_VECTOR_FROM_SCALAR(float)
SLANG_MAKE_VECTOR_FROM_SCALAR(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR_FROM_SCALAR(__half)
#if !SLANG_CUDA_RTC
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half1 make___half1(__half x)
{
    return __half1{x};
}
#endif
#endif

#define SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(Fn, T, N)                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##N Fn(T##N* address, T##N val)                           \
    {                                                                                             \
        T##N result;                                                                              \
        for (int i = 0; i < N; i++)                                                               \
            *_slang_vector_get_element_ptr(&result, i) =                                          \
                Fn(_slang_vector_get_element_ptr(address, i), _slang_vector_get_element(val, i)); \
        return result;                                                                            \
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 4)
#endif
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 4)

template<typename T, int n>
struct GetVectorTypeImpl
{
};

#define GET_VECTOR_TYPE_IMPL(T, n)                                     \
    template<>                                                         \
    struct GetVectorTypeImpl<T, n>                                     \
    {                                                                  \
        typedef T##n type;                                             \
        static SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n fromScalar(T v) \
        {                                                              \
            return make_##T##n(v);                                     \
        }                                                              \
    };
#define GET_VECTOR_TYPE_IMPL_N(T) \
    GET_VECTOR_TYPE_IMPL(T, 1)    \
    GET_VECTOR_TYPE_IMPL(T, 2)    \
    GET_VECTOR_TYPE_IMPL(T, 3)    \
    GET_VECTOR_TYPE_IMPL(T, 4)

GET_VECTOR_TYPE_IMPL_N(int)
GET_VECTOR_TYPE_IMPL_N(uint)
GET_VECTOR_TYPE_IMPL_N(short)
GET_VECTOR_TYPE_IMPL_N(ushort)
GET_VECTOR_TYPE_IMPL_N(char)
GET_VECTOR_TYPE_IMPL_N(uchar)
GET_VECTOR_TYPE_IMPL_N(longlong)
GET_VECTOR_TYPE_IMPL_N(ulonglong)
GET_VECTOR_TYPE_IMPL_N(float)
GET_VECTOR_TYPE_IMPL_N(double)
#if SLANG_CUDA_ENABLE_HALF
GET_VECTOR_TYPE_IMPL_N(__half)
#endif
template<typename T, int n>
using Vector = typename GetVectorTypeImpl<T, n>::type;

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, COLS>& operator[](size_t index)
    {
        return rows[index];
    }
};


template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T scalar)
{
    Matrix<T, ROWS, COLS> result;
    for (int i = 0; i < ROWS; i++)
        result.rows[i] = GetVectorTypeImpl<T, COLS>::fromScalar(scalar);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2,
    const Vector<T, COLS>& row3)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    result.rows[3] = row3;
    return result;
}

template<typename T, int ROWS, int COLS, typename U, int otherRow, int otherCol>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Matrix<U, otherRow, otherCol>& other)
{
    Matrix<T, ROWS, COLS> result;
    int minRow = ROWS;
    int minCol = COLS;
    if (minRow > otherRow)
        minRow = otherRow;
    if (minCol > otherCol)
        minCol = otherCol;
    for (int i = 0; i < minRow; i++)
        for (int j = 0; j < minCol; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) =
                (T)_slang_vector_get_element(other.rows[i], j);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[1].x = v2;
    rs.rows[1].y = v3;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 3)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v5;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
        rs.rows[3].x = v6;
        rs.rows[3].y = v7;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[1].x = v3;
    rs.rows[1].y = v4;
    rs.rows[1].z = v5;
    rs.rows[2].x = v6;
    rs.rows[2].y = v7;
    rs.rows[2].z = v8;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
        *_slang_vector_get_element_ptr(&rs.rows[2], 0) = v8;
        *_slang_vector_get_element_ptr(&rs.rows[2], 1) = v9;
        *_slang_vector_get_element_ptr(&rs.rows[2], 2) = v10;
        *_slang_vector_get_element_ptr(&rs.rows[2], 3) = v11;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[0].z = v2;
        rs.rows[1].x = v3;
        rs.rows[1].y = v4;
        rs.rows[1].z = v5;
        rs.rows[2].x = v6;
        rs.rows[2].y = v7;
        rs.rows[2].z = v8;
        rs.rows[3].x = v9;
        rs.rows[3].y = v10;
        rs.rows[3].z = v11;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11,
    T v12,
    T v13,
    T v14,
    T v15)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[0].w = v3;
    rs.rows[1].x = v4;
    rs.rows[1].y = v5;
    rs.rows[1].z = v6;
    rs.rows[1].w = v7;
    rs.rows[2].x = v8;
    rs.rows[2].y = v9;
    rs.rows[2].z = v10;
    rs.rows[2].w = v11;
    rs.rows[3].x = v12;
    rs.rows[3].y = v13;
    rs.rows[3].z = v14;
    rs.rows[3].w = v15;
    return rs;
}

#define SLANG_MATRIX_BINARY_OP(T, op)                                   \
    template<int R, int C>                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(     \
        const Matrix<T, R, C>& thisVal,                                 \
        const Matrix<T, R, C>& other)                                   \
    {                                                                   \
        Matrix<T, R, C> result;                                         \
        for (int i = 0; i < R; i++)                                     \
            for (int j = 0; j < C; j++)                                 \
                *_slang_vector_get_element_ptr(result.rows + i, j) =    \
                    _slang_vector_get_element(thisVal.rows[i], j)       \
                        op _slang_vector_get_element(other.rows[i], j); \
        return result;                                                  \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                                                               \
    template<int R, int C>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                                                              \
        Matrix<T, R, C> result;                                                                    \
        for (int i = 0; i < R; i++)                                                                \
            for (int j = 0; j < C; j++)                                                            \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                               \
                    op _slang_vector_get_element(thisVal.rows[i], j);                              \
        return result;                                                                             \
    }
#define SLANG_INT_MATRIX_OPS(T)   \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_BINARY_OP(T, &)  \
    SLANG_MATRIX_BINARY_OP(T, |)  \
    SLANG_MATRIX_BINARY_OP(T, &&) \
    SLANG_MATRIX_BINARY_OP(T, ||) \
    SLANG_MATRIX_BINARY_OP(T, ^)  \
    SLANG_MATRIX_BINARY_OP(T, %)  \
    SLANG_MATRIX_UNARY_OP(T, !)   \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_UNARY_OP(T, -)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(short)
SLANG_INT_MATRIX_OPS(ushort)
SLANG_INT_MATRIX_OPS(char)
SLANG_INT_MATRIX_OPS(uchar)
SLANG_INT_MATRIX_OPS(longlong)
SLANG_INT_MATRIX_OPS(ulonglong)
SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_FLOAT_MATRIX_OPS(__half)
#endif
#define SLANG_MATRIX_INT_NEG_OP(T)                                                        \
    template<int R, int C>                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                                     \
        Matrix<T, R, C> result;                                                           \
        for (int i = 0; i < R; i++)                                                       \
            for (int j = 0; j < C; j++)                                                   \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                      \
                    0 - _slang_vector_get_element(thisVal.rows[i], j);                    \
        return result;                                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(short)
SLANG_MATRIX_INT_NEG_OP(ushort)
SLANG_MATRIX_INT_NEG_OP(char)
SLANG_MATRIX_INT_NEG_OP(uchar)
SLANG_MATRIX_INT_NEG_OP(longlong)
SLANG_MATRIX_INT_NEG_OP(ulonglong)

#define SLANG_FLOAT_MATRIX_MOD(T)                                                 \
    template<int R, int C>                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator%(                 \
        Matrix<T, R, C> left,                                                     \
        Matrix<T, R, C> right)                                                    \
    {                                                                             \
        Matrix<T, R, C> result;                                                   \
        for (int i = 0; i < R; i++)                                               \
            for (int j = 0; j < C; j++)                                           \
                *_slang_vector_get_element_ptr(result.rows + i, j) = _slang_fmod( \
                    _slang_vector_get_element(left.rows[i], j),                   \
                    _slang_vector_get_element(right.rows[i], j));                 \
        return result;                                                            \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#if SLANG_CUDA_ENABLE_HALF
template<int R, int C>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<__half, R, C> operator%(
    Matrix<__half, R, C> left,
    Matrix<__half, R, C> right)
{
    Matrix<__half, R, C> result;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) = __float2half(_slang_fmod(
                __half2float(_slang_vector_get_element(left.rows[i], j)),
                __half2float(_slang_vector_get_element(right.rows[i], j))));
    return result;
}
#endif
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

#define SLANG_SELECT_IMPL(T, N)                                                                  \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, N> _slang_select(                               \
        bool##N condition,                                                                       \
        Vector<T, N> v0,                                                                         \
        Vector<T, N> v1)                                                                         \
    {                                                                                            \
        Vector<T, N> result;                                                                     \
        for (int i = 0; i < N; i++)                                                              \
        {                                                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_vector_get_element(condition, i) \
                                                             ? _slang_vector_get_element(v0, i)  \
                                                             : _slang_vector_get_element(v1, i); \
        }                                                                                        \
        return result;                                                                           \
    }
#define SLANG_SELECT_T(T)   \
    SLANG_SELECT_IMPL(T, 2) \
    SLANG_SELECT_IMPL(T, 3) \
    SLANG_SELECT_IMPL(T, 4)

SLANG_SELECT_T(int)
SLANG_SELECT_T(uint)
SLANG_SELECT_T(short)
SLANG_SELECT_T(ushort)
SLANG_SELECT_T(char)
SLANG_SELECT_T(uchar)
SLANG_SELECT_T(float)
SLANG_SELECT_T(double)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

//
// Half support
//

#if SLANG_CUDA_ENABLE_HALF
SLANG_SELECT_T(__half)

// Convenience functions ushort -> half

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 __ushort_as_half(const ushort2& i)
{
    return __halves2half2(__ushort_as_half(i.x), __ushort_as_half(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half3 __ushort_as_half(const ushort3& i)
{
    return __half3{__ushort_as_half(i.x), __ushort_as_half(i.y), __ushort_as_half(i.z)};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 __ushort_as_half(const ushort4& i)
{
    return __half4{
        __ushort_as_half(i.x),
        __ushort_as_half(i.y),
        __ushort_as_half(i.z),
        __ushort_as_half(i.w)};
}

// Convenience functions half -> ushort

SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort2 __half_as_ushort(const __half2& i)
{
    return make_ushort2(__half_as_ushort(i.x), __half_as_ushort(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort3 __half_as_ushort(const __half3& i)
{
    return make_ushort3(__half_as_ushort(i.x), __half_as_ushort(i.y), __half_as_ushort(i.z));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort4 __half_as_ushort(const __half4& i)
{
    return make_ushort4(
        __half_as_ushort(i.x),
        __half_as_ushort(i.y),
        __half_as_ushort(i.z),
        __half_as_ushort(i.w));
}

// This is a little bit of a hack. Fortunately CUDA has the definitions of the templated types in
// include/surface_indirect_functions.h
// Here we find the template definition requires a specialization of __nv_isurf_trait to allow
// a specialization of the surface write functions.
// This *isn't* a problem on the read functions as they don't have a return type that uses this
// mechanism

template<>
struct __nv_isurf_trait<__half>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half2>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half4>
{
    typedef void type;
};

#define SLANG_DROP_PARENS(...) __VA_ARGS__

#define SLANG_SURFACE_READ(FUNC_NAME, TYPE_ARGS, ARGS)                                             \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half FUNC_NAME<__half>(                                   \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(FUNC_NAME<ushort>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 FUNC_NAME<__half2>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 FUNC_NAME<__half4>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }

SLANG_SURFACE_READ(surf1Dread, (int x), (x))
SLANG_SURFACE_READ(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ(surf3Dread, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_READ(surf1DLayeredread, (int x, int layer), (x, layer))
SLANG_SURFACE_READ(surf2DLayeredread, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_READ(surfCubemapread, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_READ(surfCubemapLayeredread, (int x, int y, int layerFace), (x, y, layerFace))

#define SLANG_SURFACE_WRITE(FUNC_NAME, TYPE_ARGS, ARGS)                                            \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half>(                                     \
        __half data,                                                                               \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode);  \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half2>(                                    \
        __half2 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort2>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half4>(                                    \
        __half4 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort4>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }

SLANG_SURFACE_WRITE(surf1Dwrite, (int x), (x))
SLANG_SURFACE_WRITE(surf2Dwrite, (int x, int y), (x, y))
SLANG_SURFACE_WRITE(surf3Dwrite, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_WRITE(surf1DLayeredwrite, (int x, int layer), (x, layer))
SLANG_SURFACE_WRITE(surf2DLayeredwrite, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_WRITE(surfCubemapwrite, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_WRITE(surfCubemapLayeredwrite, (int x, int y, int layerFace), (x, y, layerFace))

// ! Hack to test out reading !!!
// Only works converting *from* half

// template <typename T>
// SLANG_FORCE_INLINE SLANG_CUDA_CALL T surf2Dread_convert(cudaSurfaceObject_t surfObj, int x, int
// y, cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURFACE_READ_HALF_CONVERT(FUNC_NAME, TYPE_ARGS, ARGS)                              \
                                                                                                 \
    template<typename T>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T FUNC_NAME##_convert(                                    \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode);                                                   \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float FUNC_NAME##_convert<float>(                         \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        return __ushort_as_half(                                                                 \
            FUNC_NAME<uint16_t>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float2 FUNC_NAME##_convert<float2>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half2 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float2{v.x, v.y};                                                                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float4 FUNC_NAME##_convert<float4>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half4 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float4{v.x, v.y, v.z, v.w};                                                       \
    }

SLANG_SURFACE_READ_HALF_CONVERT(surf1Dread, (int x), (x))
SLANG_SURFACE_READ_HALF_CONVERT(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ_HALF_CONVERT(surf3Dread, (int x, int y, int z), (x, y, z))

#endif

// Support for doing format conversion when writing to a surface/RWTexture

// NOTE! For normal surface access x values are *byte* addressed.
// For the _convert versions they are *not*. They don't need to be because sust.p does not require
// it.

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode);

// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust

// Float

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(v));
}

// Float2

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3,%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy));
}

// Float4
template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3,%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2}], {%3,%4,%5,%6};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2,%3}], {%4,%5,%6,%7};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

// ----------------------------- F32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_ceil(float f)
{
    return ::ceilf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_floor(float f)
{
    return ::floorf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_round(float f)
{
    return ::roundf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sin(float f)
{
    return ::sinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cos(float f)
{
    return ::cosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F32_sincos(float f, float* s, float* c)
{
    ::sincosf(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tan(float f)
{
    return ::tanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_asin(float f)
{
    return ::asinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_acos(float f)
{
    return ::acosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan(float f)
{
    return ::atanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sinh(float f)
{
    return ::sinhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cosh(float f)
{
    return ::coshf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tanh(float f)
{
    return ::tanhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log2(float f)
{
    return ::log2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log(float f)
{
    return ::logf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log10(float f)
{
    return ::log10f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp2(float f)
{
    return ::exp2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp(float f)
{
    return ::expf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_abs(float f)
{
    return ::fabsf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_trunc(float f)
{
    return ::truncf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sqrt(float f)
{
    return ::sqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_rsqrt(float f)
{
    return ::rsqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sign(float f)
{
    return (f == 0.0f) ? f : ((f < 0.0f) ? -1.0f : 1.0f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frac(float f)
{
    return f - F32_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isnan(float f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isfinite(float f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isinf(float f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_min(float a, float b)
{
    return ::fminf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_max(float a, float b)
{
    return ::fmaxf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_pow(float a, float b)
{
    return ::powf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fmod(float a, float b)
{
    return ::fmodf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_remainder(float a, float b)
{
    return ::remainderf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan2(float a, float b)
{
    return float(::atan2(a, b));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frexp(float x, int* e)
{
    return frexpf(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t F32_asuint(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t F32_asint(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fma(float a, float b, float c)
{
    return ::fmaf(a, b, c);
}


// ----------------------------- F64 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_ceil(double f)
{
    return ::ceil(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_floor(double f)
{
    return ::floor(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_round(double f)
{
    return ::round(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sin(double f)
{
    return ::sin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cos(double f)
{
    return ::cos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_sincos(double f, double* s, double* c)
{
    ::sincos(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tan(double f)
{
    return ::tan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_asin(double f)
{
    return ::asin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_acos(double f)
{
    return ::acos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan(double f)
{
    return ::atan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sinh(double f)
{
    return ::sinh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cosh(double f)
{
    return ::cosh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tanh(double f)
{
    return ::tanh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log2(double f)
{
    return ::log2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log(double f)
{
    return ::log(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log10(float f)
{
    return ::log10(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp2(double f)
{
    return ::exp2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp(double f)
{
    return ::exp(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_abs(double f)
{
    return ::fabs(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_trunc(double f)
{
    return ::trunc(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sqrt(double f)
{
    return ::sqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_rsqrt(double f)
{
    return ::rsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sign(double f)
{
    return (f == 0.0) ? f : ((f < 0.0) ? -1.0 : 1.0);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frac(double f)
{
    return f - F64_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isnan(double f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isfinite(double f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isinf(double f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_min(double a, double b)
{
    return ::fmin(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_max(double a, double b)
{
    return ::fmax(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_pow(double a, double b)
{
    return ::pow(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fmod(double a, double b)
{
    return ::fmod(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_remainder(double a, double b)
{
    return ::remainder(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan2(double a, double b)
{
    return ::atan2(a, b);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frexp(double x, int* e)
{
    return ::frexp(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

// ----------------------------- I32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_abs(int32_t f)
{
    return (f < 0) ? -f : f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float I32_asfloat(int32_t x)
{
    Union32 u;
    u.i = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_asuint(int32_t x)
{
    return uint32_t(x);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double I32_asdouble(int32_t low, int32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

// ----------------------------- U32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_abs(uint32_t f)
{
    return f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float U32_asfloat(uint32_t x)
{
    Union32 u;
    u.u = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_asint(int32_t x)
{
    return uint32_t(x);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_countbits(uint32_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popc(v);
}


// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_abs(int64_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_abs(uint64_t f)
{
    return f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U64_countbits(uint64_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popcll(v);
}


// ----------------------------- ResourceType -----------------------------------------


// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct StructuredBuffer
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

    SLANG_CUDA_CALL const T& Load(size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }
#endif

    T* data;
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    size_t count;
#endif
};

template<typename T>
struct RWStructuredBuffer : StructuredBuffer<T>
{
    SLANG_CUDA_CALL T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, this->count);
#endif
        return this->data[index];
    }
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }
    template<typename T>
    SLANG_CUDA_CALL StructuredBuffer<T> asStructuredBuffer() const
    {
        StructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Missing support for Atomic operations
// Missing support for Load with status
struct RWByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }

    SLANG_CUDA_CALL void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    SLANG_CUDA_CALL void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    SLANG_CUDA_CALL void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    SLANG_CUDA_CALL void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    SLANG_CUDA_CALL void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        memcpy((char*)data + index, &value, sizeof(T));
    }

    /// Can be used in the core module to gain access
    template<typename T>
    SLANG_CUDA_CALL T* _getPtrAt(size_t index)
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return (T*)(((char*)data) + index);
    }
    template<typename T>
    SLANG_CUDA_CALL RWStructuredBuffer<T> asStructuredBuffer() const
    {
        RWStructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};


// ---------------------- Wave --------------------------------------

// TODO(JS): It appears that cuda does not have a simple way to get a lane index.
//
// Another approach could be...
// laneId = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) &
// SLANG_CUDA_WARP_MASK If that is really true another way to do this, would be for code generator
// to add this function with the [numthreads] baked in.
//
// For now I'll just assume you have a launch that makes the following correct if the kernel uses
// WaveGetLaneIndex()
#ifndef SLANG_USE_ASM_LANE_ID
__forceinline__ __device__ uint32_t _getLaneId()
{
    // If the launch is (or I guess some multiple of the warp size)
    // we try this mechanism, which is apparently faster.
    return threadIdx.x & SLANG_CUDA_WARP_MASK;
}
#else
__forceinline__ __device__ uint32_t _getLaneId()
{
    // https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid#
    // This mechanism is not the fastest way to do it, and that is why the other mechanism
    // is the default. But the other mechanism relies on a launch that makes the assumption
    // true.
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
#endif

typedef int WarpMask;

// It appears that the __activemask() cannot always be used because
// threads need to be converged.
//
// For CUDA the article claims mask has to be used carefully
// https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
// With the Warp intrinsics there is no mask, and it's just the 'active lanes'.
// __activemask() though does not require there is convergence, so that doesn't work.
//
// '__ballot_sync' produces a convergance.
//
// From the CUDA docs:
// ```For __all_sync, __any_sync, and __ballot_sync, a mask must be passed that specifies the
// threads participating in the call. A bit, representing the thread's lane ID, must be set for each
// participating thread to ensure they are properly converged before the intrinsic is executed by
// the hardware. All active threads named in mask must execute the same intrinsic with the same
// mask, or the result is undefined.```
//
// Currently there isn't a mechanism to correctly get the mask without it being passed through.
// Doing so will most likely require some changes to slang code generation to track masks, for now
// then we use _getActiveMask.

// Return mask of all the lanes less than the current lane
__forceinline__ __device__ WarpMask _getLaneLtMask()
{
    return (int(1) << _getLaneId()) - 1;
}

// TODO(JS):
// THIS IS NOT CORRECT! That determining the appropriate active mask requires appropriate
// mask tracking.
__forceinline__ __device__ WarpMask _getActiveMask()
{
    return __ballot_sync(__activemask(), true);
}

// Return a mask suitable for the 'MultiPrefix' style functions
__forceinline__ __device__ WarpMask _getMultiPrefixMask(int mask)
{
    return mask;
}

// Note! Note will return true if mask is 0, but thats okay, because there must be one
// lane active to execute anything
__inline__ __device__ bool _waveIsSingleLane(WarpMask mask)
{
    return (mask & (mask - 1)) == 0;
}

// Returns the power of 2 size of run of set bits. Returns 0 if not a suitable run.
// Examples:
// 0b00000000'00000000'00000000'11111111 -> 8
// 0b11111111'11111111'11111111'11111111 -> 32
// 0b00000000'00000000'00000000'00011111 -> 0 (since 5 is not a power of 2)
// 0b00000000'00000000'00000000'11110000 -> 0 (since the run of bits does not start at the LSB)
// 0b00000000'00000000'00000000'00100111 -> 0 (since it is not a single contiguous run)
__inline__ __device__ int _waveCalcPow2Offset(WarpMask mask)
{
    // This should be the most common case, so fast path it
    if (mask == SLANG_CUDA_WARP_BITMASK)
    {
        return SLANG_CUDA_WARP_SIZE;
    }
    // Is it a contiguous run of bits?
    if ((mask & (mask + 1)) == 0)
    {
        // const int offsetSize = __ffs(mask + 1) - 1;
        const int offset = 32 - __clz(mask);
        // Is it a power of 2 size
        if ((offset & (offset - 1)) == 0)
        {
            return offset;
        }
    }
    return 0;
}

__inline__ __device__ bool _waveIsFirstLane()
{
    const WarpMask mask = __activemask();
    // We special case bit 0, as that most warps are expected to be fully active.

    // mask & -mask, isolates the lowest set bit.
    // return (mask & 1 ) || ((mask & -mask) == (1 << _getLaneId()));

    // This mechanism is most similar to what was in an nVidia post, so assume it is prefered.
    return (mask & 1) || ((__ffs(mask) - 1) == _getLaneId());
}

template<typename T>
struct WaveOpOr
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a | b; }
};

template<typename T>
struct WaveOpAnd
{
    __inline__ __device__ static T getInitial(T a) { return ~T(0); }
    __inline__ __device__ static T doOp(T a, T b) { return a & b; }
};

template<typename T>
struct WaveOpXor
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a ^ b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a ^ b; }
};

template<typename T>
struct WaveOpAdd
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a + b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a - b; }
};

template<typename T>
struct WaveOpMul
{
    __inline__ __device__ static T getInitial(T a) { return T(1); }
    __inline__ __device__ static T doOp(T a, T b) { return a * b; }
    // Using this inverse for int is probably undesirable - because in general it requires T to have
    // more precision There is also a performance aspect to it, where divides are generally
    // significantly slower
    __inline__ __device__ static T doInverse(T a, T b) { return a / b; }
};

template<typename T>
struct WaveOpMax
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a > b ? a : b; }
};

template<typename T>
struct WaveOpMin
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a < b ? a : b; }
};

template<typename T>
struct ElementTypeTrait;

// Scalar
template<>
struct ElementTypeTrait<int>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<uint>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<float>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<double>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<uint64_t>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<int64_t>
{
    typedef int64_t Type;
};

// Vector
template<>
struct ElementTypeTrait<int1>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int2>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int3>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int4>
{
    typedef int Type;
};

template<>
struct ElementTypeTrait<uint1>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint2>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint3>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint4>
{
    typedef uint Type;
};

template<>
struct ElementTypeTrait<float1>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float2>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float3>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float4>
{
    typedef float Type;
};

template<>
struct ElementTypeTrait<double1>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double2>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double3>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double4>
{
    typedef double Type;
};

// Matrix
template<typename T, int ROWS, int COLS>
struct ElementTypeTrait<Matrix<T, ROWS, COLS>>
{
    typedef T Type;
};

// Scalar
template<typename INTF, typename T>
__device__ T _waveReduceScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            val = INTF::doOp(val, __shfl_xor_sync(mask, val, offset));
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        T result = INTF::getInitial(val);
        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            result = INTF::doOp(result, __shfl_sync(mask, val, srcLane));
            remaining &= ~laneBit;
        }
        return result;
    }
    return val;
}


// Multiple values
template<typename INTF, typename T, size_t COUNT>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_xor_sync(mask, val[i], offset));
            }
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        // Copy the original
        T originalVal[COUNT];
        for (size_t i = 0; i < COUNT; ++i)
        {
            const T v = val[i];
            originalVal[i] = v;
            val[i] = INTF::getInitial(v);
        }

        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_sync(mask, originalVal[i], srcLane));
            }
            remaining &= ~laneBit;
        }
    }
}

template<typename INTF, typename T>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<INTF, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)val);
}

template<typename T>
__inline__ __device__ T _waveOr(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveAnd(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAnd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveXor(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveProduct(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveSum(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMin(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMax(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMax<T>, T>(mask, val);
}

// Fast-path specializations when CUDA warp reduce operators are available
#if __CUDA_ARCH__ >= 800 // 8.x or higher
template<>
__inline__ __device__ unsigned _waveOr<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_or_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveAnd<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_and_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveXor<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_xor_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveSum<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ int _waveSum<int>(WarpMask mask, int val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMin<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMin<int>(WarpMask mask, int val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMax<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_max_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMax<int>(WarpMask mask, int val)
{
    return __reduce_max_sync(mask, val);
}
#endif


// Multiple

template<typename T>
__inline__ __device__ T _waveOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpOr<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAnd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpXor<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMul<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAdd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMin<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMax<ElemType>>(mask, &val);
    return val;
}


template<typename T>
__inline__ __device__ bool _waveAllEqual(WarpMask mask, T val)
{
    int pred;
    __match_all_sync(mask, val, &pred);
    return pred != 0;
}

template<typename T>
__inline__ __device__ bool _waveAllEqualMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    for (size_t i = 0; i < count; ++i)
    {
        __match_all_sync(mask, src[i], &pred);
        if (pred == 0)
        {
            return false;
        }
    }
    return true;
}

template<typename T>
__inline__ __device__ T _waveReadFirst(WarpMask mask, T val)
{
    const int lowestLaneId = __ffs(mask) - 1;
    return __shfl_sync(mask, val, lowestLaneId);
}

template<typename T>
__inline__ __device__ T _waveReadFirstMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    const int lowestLaneId = __ffs(mask) - 1;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lowestLaneId);
    }
    return outVal;
}

template<typename T>
__inline__ __device__ T _waveShuffleMultiple(WarpMask mask, T inVal, int lane)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lane);
    }
    return outVal;
}

// Scalar

// Invertable means that when we get to the end of the reduce, we can remove val (to make
// exclusive), using the inverse of the op.
template<typename INTF, typename T>
__device__ T _wavePrefixInvertableScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result;
    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        result = val;
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, result, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
            }
        }
        // Remove val from the result, by applyin inverse
        result = INTF::doInverse(result, val);
    }
    else
    {
        result = INTF::getInitial(val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


// This implementation separately tracks the value to be propogated, and the value
// that is the final result
template<typename INTF, typename T>
__device__ T _wavePrefixScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result = INTF::getInitial(val);
    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra multiply for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, val, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
                val = INTF::doOp(val, readVal);
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpCopy(T* dst, const T* src)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        dst[j] = src[j];
    }
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpDoInverse(T* inOut, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        inOut[j] = INTF::doInverse(inOut[j], val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpSetInitial(T* out, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        out[j] = INTF::getInitial(val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixInvertableMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T originalVal[COUNT];
    _waveOpCopy<INTF, T, COUNT>(originalVal, val);

    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        for (int i = 1; i < offsetSize; i += i)
        {
            // TODO(JS): Note that here I don't split the laneId outside so it's only tested once.
            // This may be better but it would also mean that there would be shfl between lanes
            // that are on different (albeit identical) instructions. So this seems more likely to
            // work as expected with everything in lock step.
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, val[j], i, offsetSize);
                if (laneId >= i)
                {
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
        // Remove originalVal from the result, by applyin inverse
        _waveOpDoInverse<INTF, T, COUNT>(val, originalVal);
    }
    else
    {
        _waveOpSetInitial<INTF, T, COUNT>(val, val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, originalVal[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                    remaining &= ~laneBit;
                }
            }
        }
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();

    T work[COUNT];
    _waveOpCopy<INTF, T, COUNT>(work, val);
    _waveOpSetInitial<INTF, T, COUNT>(val, val);

    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra op for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, work[j], i, offsetSize);
                if (laneId >= i)
                {
                    work[j] = INTF::doOp(work[j], readVal);
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, work[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                }
                remaining &= ~laneBit;
            }
        }
    }
}

template<typename T>
__inline__ __device__ T _wavePrefixProduct(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixSum(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixXor(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixOr(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixAnd(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpAnd<T>, T>(mask, val);
}


template<typename T>
__inline__ __device__ T _wavePrefixProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpMul<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpAdd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpXor<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpOr<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpAnd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ uint4 _waveMatchScalar(WarpMask mask, T val)
{
    int pred;
    return make_uint4(__match_all_sync(mask, val, &pred), 0, 0, 0);
}

template<typename T>
__inline__ __device__ uint4 _waveMatchMultiple(WarpMask mask, const T& inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    uint matchBits = 0xffffffff;
    for (size_t i = 0; i < count && matchBits; ++i)
    {
        matchBits = matchBits & __match_all_sync(mask, src[i], &pred);
    }
    return make_uint4(matchBits, 0, 0, 0);
}

__device__ uint getAt(dim3 a, int b)
{
    SLANG_PRELUDE_ASSERT(b >= 0 && b < 3);
    return (&a.x)[b];
}
__device__ uint3 operator*(uint3 a, dim3 b)
{
    uint3 r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r;
}

template<typename TResult, typename TInput>
__inline__ __device__ TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */


/* Type that defines the uniform entry point params. The actual content of this type is dependent on
the entry point parameters, and can be found via reflection or defined such that it matches the
shader appropriately.
*/
struct UniformEntryPointParams;
struct UniformState;

// ---------------------- OptiX Ray Payload --------------------------------------
#ifdef SLANG_CUDA_ENABLE_OPTIX
struct RayDesc
{
    float3 Origin;
    float TMin;
    float3 Direction;
    float TMax;
};

static __forceinline__ __device__ void* unpackOptiXRayPayloadPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packOptiXRayPayloadPointer(
    void* ptr,
    uint32_t& i0,
    uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* getOptiXRayPayloadPtr()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpackOptiXRayPayloadPointer(u0, u1);
}

template<typename T>
__forceinline__ __device__ void* traceOptiXRay(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T* Payload)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTrace(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f, /* Time for motion blur, currently unsupported in slang */
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0,
        r1);
}

#endif

static const int kSlangTorchTensorMaxDim = 5;

// TensorView
struct TensorView
{
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;

    template<typename T>
    __device__ T* data_ptr()
    {
        return reinterpret_cast<T*>(data);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint32_t index)
    {
        uint64_t offset = strides[0] * index;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint2 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint3 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint4 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
                          strides[3] * index.w;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T, unsigned int N>
    __device__ T* data_ptr_at(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T& load(uint32_t x)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y);
    }
    template<typename T>
    __device__ T& load(uint2 index)
    {
        return *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z);
    }
    template<typename T>
    __device__ T& load(uint3 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w);
    }
    template<typename T>
    __device__ T& load(uint4 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w);
    }
    template<typename T>
    __device__ T& load(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4);
    }

    // Generic version of load
    template<typename T, unsigned int N>
    __device__ T& load(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return *reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ void store(uint32_t x, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y) = val;
    }
    template<typename T>
    __device__ void store(uint2 index, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z) = val;
    }
    template<typename T>
    __device__ void store(uint3 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, uint32_t w, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w) = val;
    }
    template<typename T>
    __device__ void store(uint4 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w) = val;
    }
    template<typename T>
    __device__ void store(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4) = val;
    }

    // Generic version
    template<typename T, unsigned int N>
    __device__ void store(uint index[N], T val)
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        *reinterpret_cast<T*>(data + offset) = val;
    }
};


#line 20 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/spline-machine.slang"
struct SplineState_0
{
    float2  distortion_parts_0;
    float2  cum_sum_0;
    float3  padding_0;
    float t_0;
    float4  drgb_0;
    float logT_0;
    float3  C_0;
};


#line 102 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ SplineState_0 get_state_0(TensorView view_0, uint ind_0)
{
    float _S1 = ((view_0).load<float>((ind_0), (0U)));

#line 104
    float _S2 = ((view_0).load<float>((ind_0), (1U)));

#line 104
    float2  _S3 = make_float2 (_S1, _S2);
    float _S4 = ((view_0).load<float>((ind_0), (2U)));

#line 105
    float _S5 = ((view_0).load<float>((ind_0), (3U)));

#line 105
    float2  _S6 = make_float2 (_S4, _S5);
    float _S7 = ((view_0).load<float>((ind_0), (4U)));

#line 106
    float _S8 = ((view_0).load<float>((ind_0), (5U)));

#line 106
    float _S9 = ((view_0).load<float>((ind_0), (6U)));

#line 106
    float3  _S10 = make_float3 (_S7, _S8, _S9);

#line 103
    float _S11 = ((view_0).load<float>((ind_0), (7U)));

#line 108
    float _S12 = ((view_0).load<float>((ind_0), (8U)));

#line 108
    float _S13 = ((view_0).load<float>((ind_0), (9U)));

#line 108
    float _S14 = ((view_0).load<float>((ind_0), (10U)));

#line 108
    float _S15 = ((view_0).load<float>((ind_0), (11U)));

#line 108
    float4  _S16 = make_float4 (_S12, _S13, _S14, _S15);

#line 103
    float _S17 = ((view_0).load<float>((ind_0), (12U)));

#line 111
    float _S18 = ((view_0).load<float>((ind_0), (13U)));

#line 111
    float _S19 = ((view_0).load<float>((ind_0), (14U)));

#line 111
    float _S20 = ((view_0).load<float>((ind_0), (15U)));

#line 103
    SplineState_0 _S21 = { _S3, _S6, _S10, _S11, _S16, _S17, make_float3 (_S18, _S19, _S20) };

#line 103
    return _S21;
}


#line 23
__device__ float3  get_float3_0(TensorView view_1, uint ind_1)
{

#line 24
    float _S22 = ((view_1).load<float>((ind_1), (0U)));

#line 24
    float _S23 = ((view_1).load<float>((ind_1), (1U)));

#line 24
    float _S24 = ((view_1).load<float>((ind_1), (2U)));

#line 24
    return make_float3 (_S22, _S23, _S24);
}


#line 1445 "diff.meta.slang"
__device__ SplineState_0 SplineState_x24_syn_dzero_0()
{

#line 1445
    SplineState_0 result_0;

#line 1751 "core.meta.slang"
    float2  _S25 = make_float2 (0.0f);

#line 1751
    (&result_0)->distortion_parts_0 = _S25;

#line 1751
    (&result_0)->cum_sum_0 = _S25;

#line 1751
    float3  _S26 = make_float3 (0.0f);

#line 1751
    (&result_0)->padding_0 = _S26;

#line 1751
    (&result_0)->t_0 = 0.0f;

#line 1751
    (&result_0)->drgb_0 = make_float4 (0.0f);

#line 1751
    (&result_0)->logT_0 = 0.0f;

#line 1751
    (&result_0)->C_0 = _S26;

#line 1751
    return result_0;
}


#line 39 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ float4  get_float4_0(TensorView view_2, uint ind_2)
{

#line 40
    float _S27 = ((view_2).load<float>((ind_2), (0U)));

#line 40
    float _S28 = ((view_2).load<float>((ind_2), (1U)));

#line 40
    float _S29 = ((view_2).load<float>((ind_2), (2U)));

#line 40
    float _S30 = ((view_2).load<float>((ind_2), (3U)));

#line 40
    return make_float4 (_S27, _S28, _S29, _S30);
}


#line 10123 "hlsl.meta.slang"
struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};


#line 1 "token paste"
__device__ void _d_exp_0(DiffPair_float_0 * dpx_0, float dOut_0)
{

#line 1719 "diff.meta.slang"
    float _S31 = (F32_exp(((*dpx_0).primal_0))) * dOut_0;

#line 1719
    dpx_0->primal_0 = (*dpx_0).primal_0;

#line 1719
    dpx_0->differential_0 = _S31;



    return;
}


#line 1935
__device__ void _d_max_0(DiffPair_float_0 * dpx_1, DiffPair_float_0 * dpy_0, float dOut_1)
{
    DiffPair_float_0 _S32 = *dpx_1;

#line 1937
    float _S33;

#line 1937
    if((*dpx_1).primal_0 > (*dpy_0).primal_0)
    {

#line 1937
        _S33 = dOut_1;

#line 1937
    }
    else
    {

#line 1937
        _S33 = 0.0f;

#line 1937
    }

#line 1937
    dpx_1->primal_0 = _S32.primal_0;

#line 1937
    dpx_1->differential_0 = _S33;
    DiffPair_float_0 _S34 = *dpy_0;

#line 1938
    if((*dpy_0).primal_0 > _S32.primal_0)
    {

#line 1938
        _S33 = dOut_1;

#line 1938
    }
    else
    {

#line 1938
        _S33 = 0.0f;

#line 1938
    }

#line 1938
    dpy_0->primal_0 = _S34.primal_0;

#line 1938
    dpy_0->differential_0 = _S33;
    return;
}


#line 1 "token paste"
__device__ void _d_sqrt_0(DiffPair_float_0 * dpx_2, float dOut_2)
{

#line 1719 "diff.meta.slang"
    float _S35 = 0.5f / (F32_sqrt(((F32_max((1.00000001168609742e-07f), ((*dpx_2).primal_0)))))) * dOut_2;

#line 1719
    dpx_2->primal_0 = (*dpx_2).primal_0;

#line 1719
    dpx_2->differential_0 = _S35;



    return;
}


#line 187 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
__device__ Matrix<float, 4, 4>  inverse_0(Matrix<float, 4, 4>  m_0)
{

#line 193
    float _S36 = m_0.rows[int(2)].y * m_0.rows[int(3)].z;

#line 193
    float _S37 = m_0.rows[int(3)].y * m_0.rows[int(2)].z;

#line 193
    float _S38 = m_0.rows[int(3)].y * m_0.rows[int(1)].z;

#line 193
    float _S39 = m_0.rows[int(1)].y * m_0.rows[int(3)].z;

#line 193
    float _S40 = m_0.rows[int(2)].y * m_0.rows[int(1)].z;

#line 193
    float _S41 = m_0.rows[int(1)].y * m_0.rows[int(2)].z;

#line 193
    float t11_0 = _S36 * m_0.rows[int(1)].w - _S37 * m_0.rows[int(1)].w + _S38 * m_0.rows[int(2)].w - _S39 * m_0.rows[int(2)].w - _S40 * m_0.rows[int(3)].w + _S41 * m_0.rows[int(3)].w;
    float _S42 = m_0.rows[int(3)].x * m_0.rows[int(2)].z;

#line 194
    float _S43 = m_0.rows[int(2)].x * m_0.rows[int(3)].z;

#line 194
    float _S44 = m_0.rows[int(3)].x * m_0.rows[int(1)].z;

#line 194
    float _S45 = m_0.rows[int(1)].x * m_0.rows[int(3)].z;

#line 194
    float _S46 = m_0.rows[int(2)].x * m_0.rows[int(1)].z;

#line 194
    float _S47 = m_0.rows[int(1)].x * m_0.rows[int(2)].z;

#line 194
    float t12_0 = _S42 * m_0.rows[int(1)].w - _S43 * m_0.rows[int(1)].w - _S44 * m_0.rows[int(2)].w + _S45 * m_0.rows[int(2)].w + _S46 * m_0.rows[int(3)].w - _S47 * m_0.rows[int(3)].w;
    float _S48 = m_0.rows[int(2)].x * m_0.rows[int(3)].y;

#line 195
    float _S49 = m_0.rows[int(3)].x * m_0.rows[int(2)].y;

#line 195
    float _S50 = m_0.rows[int(3)].x * m_0.rows[int(1)].y;

#line 195
    float _S51 = m_0.rows[int(1)].x * m_0.rows[int(3)].y;

#line 195
    float _S52 = m_0.rows[int(2)].x * m_0.rows[int(1)].y;

#line 195
    float _S53 = m_0.rows[int(1)].x * m_0.rows[int(2)].y;

#line 195
    float t13_0 = _S48 * m_0.rows[int(1)].w - _S49 * m_0.rows[int(1)].w + _S50 * m_0.rows[int(2)].w - _S51 * m_0.rows[int(2)].w - _S52 * m_0.rows[int(3)].w + _S53 * m_0.rows[int(3)].w;
    float t14_0 = _S49 * m_0.rows[int(1)].z - _S48 * m_0.rows[int(1)].z - _S50 * m_0.rows[int(2)].z + _S51 * m_0.rows[int(2)].z + _S52 * m_0.rows[int(3)].z - _S53 * m_0.rows[int(3)].z;


    float idet_0 = 1.0f / (m_0.rows[int(0)].x * t11_0 + m_0.rows[int(0)].y * t12_0 + m_0.rows[int(0)].z * t13_0 + m_0.rows[int(0)].w * t14_0);

    Matrix<float, 4, 4>  ret_0;

    *&(((&ret_0)->rows + (int(0)))->x) = t11_0 * idet_0;
    float _S54 = m_0.rows[int(3)].y * m_0.rows[int(0)].z;

#line 204
    float _S55 = m_0.rows[int(0)].y * m_0.rows[int(3)].z;

#line 204
    float _S56 = m_0.rows[int(2)].y * m_0.rows[int(0)].z;

#line 204
    float _S57 = m_0.rows[int(0)].y * m_0.rows[int(2)].z;

#line 204
    *&(((&ret_0)->rows + (int(0)))->y) = (_S37 * m_0.rows[int(0)].w - _S36 * m_0.rows[int(0)].w - _S54 * m_0.rows[int(2)].w + _S55 * m_0.rows[int(2)].w + _S56 * m_0.rows[int(3)].w - _S57 * m_0.rows[int(3)].w) * idet_0;
    float _S58 = m_0.rows[int(1)].y * m_0.rows[int(0)].z;

#line 205
    float _S59 = m_0.rows[int(0)].y * m_0.rows[int(1)].z;

#line 205
    *&(((&ret_0)->rows + (int(0)))->z) = (_S39 * m_0.rows[int(0)].w - _S38 * m_0.rows[int(0)].w + _S54 * m_0.rows[int(1)].w - _S55 * m_0.rows[int(1)].w - _S58 * m_0.rows[int(3)].w + _S59 * m_0.rows[int(3)].w) * idet_0;
    *&(((&ret_0)->rows + (int(0)))->w) = (_S40 * m_0.rows[int(0)].w - _S41 * m_0.rows[int(0)].w - _S56 * m_0.rows[int(1)].w + _S57 * m_0.rows[int(1)].w + _S58 * m_0.rows[int(2)].w - _S59 * m_0.rows[int(2)].w) * idet_0;

    *&(((&ret_0)->rows + (int(1)))->x) = t12_0 * idet_0;
    float _S60 = m_0.rows[int(3)].x * m_0.rows[int(0)].z;

#line 209
    float _S61 = m_0.rows[int(0)].x * m_0.rows[int(3)].z;

#line 209
    float _S62 = m_0.rows[int(2)].x * m_0.rows[int(0)].z;

#line 209
    float _S63 = m_0.rows[int(0)].x * m_0.rows[int(2)].z;

#line 209
    *&(((&ret_0)->rows + (int(1)))->y) = (_S43 * m_0.rows[int(0)].w - _S42 * m_0.rows[int(0)].w + _S60 * m_0.rows[int(2)].w - _S61 * m_0.rows[int(2)].w - _S62 * m_0.rows[int(3)].w + _S63 * m_0.rows[int(3)].w) * idet_0;
    float _S64 = m_0.rows[int(1)].x * m_0.rows[int(0)].z;

#line 210
    float _S65 = m_0.rows[int(0)].x * m_0.rows[int(1)].z;

#line 210
    *&(((&ret_0)->rows + (int(1)))->z) = (_S44 * m_0.rows[int(0)].w - _S45 * m_0.rows[int(0)].w - _S60 * m_0.rows[int(1)].w + _S61 * m_0.rows[int(1)].w + _S64 * m_0.rows[int(3)].w - _S65 * m_0.rows[int(3)].w) * idet_0;
    *&(((&ret_0)->rows + (int(1)))->w) = (_S47 * m_0.rows[int(0)].w - _S46 * m_0.rows[int(0)].w + _S62 * m_0.rows[int(1)].w - _S63 * m_0.rows[int(1)].w - _S64 * m_0.rows[int(2)].w + _S65 * m_0.rows[int(2)].w) * idet_0;

    *&(((&ret_0)->rows + (int(2)))->x) = t13_0 * idet_0;
    float _S66 = m_0.rows[int(3)].x * m_0.rows[int(0)].y;

#line 214
    float _S67 = m_0.rows[int(0)].x * m_0.rows[int(3)].y;

#line 214
    float _S68 = m_0.rows[int(2)].x * m_0.rows[int(0)].y;

#line 214
    float _S69 = m_0.rows[int(0)].x * m_0.rows[int(2)].y;

#line 214
    *&(((&ret_0)->rows + (int(2)))->y) = (_S49 * m_0.rows[int(0)].w - _S48 * m_0.rows[int(0)].w - _S66 * m_0.rows[int(2)].w + _S67 * m_0.rows[int(2)].w + _S68 * m_0.rows[int(3)].w - _S69 * m_0.rows[int(3)].w) * idet_0;
    float _S70 = m_0.rows[int(1)].x * m_0.rows[int(0)].y;

#line 215
    float _S71 = m_0.rows[int(0)].x * m_0.rows[int(1)].y;

#line 215
    *&(((&ret_0)->rows + (int(2)))->z) = (_S51 * m_0.rows[int(0)].w - _S50 * m_0.rows[int(0)].w + _S66 * m_0.rows[int(1)].w - _S67 * m_0.rows[int(1)].w - _S70 * m_0.rows[int(3)].w + _S71 * m_0.rows[int(3)].w) * idet_0;
    *&(((&ret_0)->rows + (int(2)))->w) = (_S52 * m_0.rows[int(0)].w - _S53 * m_0.rows[int(0)].w - _S68 * m_0.rows[int(1)].w + _S69 * m_0.rows[int(1)].w + _S70 * m_0.rows[int(2)].w - _S71 * m_0.rows[int(2)].w) * idet_0;

    *&(((&ret_0)->rows + (int(3)))->x) = t14_0 * idet_0;
    *&(((&ret_0)->rows + (int(3)))->y) = (_S48 * m_0.rows[int(0)].z - _S49 * m_0.rows[int(0)].z + _S66 * m_0.rows[int(2)].z - _S67 * m_0.rows[int(2)].z - _S68 * m_0.rows[int(3)].z + _S69 * m_0.rows[int(3)].z) * idet_0;
    *&(((&ret_0)->rows + (int(3)))->z) = (_S50 * m_0.rows[int(0)].z - _S51 * m_0.rows[int(0)].z - _S66 * m_0.rows[int(1)].z + _S67 * m_0.rows[int(1)].z + _S70 * m_0.rows[int(3)].z - _S71 * m_0.rows[int(3)].z) * idet_0;
    *&(((&ret_0)->rows + (int(3)))->w) = (_S53 * m_0.rows[int(0)].z - _S52 * m_0.rows[int(0)].z + _S68 * m_0.rows[int(1)].z - _S69 * m_0.rows[int(1)].z - _S70 * m_0.rows[int(2)].z + _S71 * m_0.rows[int(2)].z) * idet_0;

    return ret_0;
}


#line 31 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ float3  get_float3_1(TensorView view_3, uint ind_3, uint feat_ind_0)
{

#line 32
    float _S72 = ((view_3).load<float>((ind_3), (feat_ind_0), (0U)));

#line 32
    float _S73 = ((view_3).load<float>((ind_3), (feat_ind_0), (1U)));

#line 32
    float _S74 = ((view_3).load<float>((ind_3), (feat_ind_0), (2U)));

#line 32
    return make_float3 (_S72, _S73, _S74);
}


#line 44 "/home/amai/ever_training/ever/splinetracers/slang/sh.slang"
struct Features_0
{
    float3  f0_0;
    float3  f1_0;
    float3  f2_0;
    float3  f3_0;
    float3  f4_0;
    float3  f5_0;
    float3  f6_0;
    float3  f7_0;
    float3  f8_0;
    float3  f9_0;
    float3  f10_0;
    float3  f11_0;
    float3  f12_0;
    float3  f13_0;
    float3  f14_0;
    float3  f15_0;
};


#line 47 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ Features_0 get_feats_0(TensorView features_0, uint prim_ind_0, uint sh_degree_0)
{

#line 48
    Features_0 feat_0;
    float3  _S75 = get_float3_1(features_0, prim_ind_0, 0U);

#line 49
    (&feat_0)->f0_0 = _S75;
    if(sh_degree_0 > 0U)
    {

#line 51
        float3  _S76 = get_float3_1(features_0, prim_ind_0, 1U);

#line 51
        (&feat_0)->f1_0 = _S76;
        float3  _S77 = get_float3_1(features_0, prim_ind_0, 2U);

#line 52
        (&feat_0)->f2_0 = _S77;
        float3  _S78 = get_float3_1(features_0, prim_ind_0, 3U);

#line 53
        (&feat_0)->f3_0 = _S78;
        if(sh_degree_0 > 1U)
        {

#line 55
            float3  _S79 = get_float3_1(features_0, prim_ind_0, 4U);

#line 55
            (&feat_0)->f4_0 = _S79;
            float3  _S80 = get_float3_1(features_0, prim_ind_0, 5U);

#line 56
            (&feat_0)->f5_0 = _S80;
            float3  _S81 = get_float3_1(features_0, prim_ind_0, 6U);

#line 57
            (&feat_0)->f6_0 = _S81;
            float3  _S82 = get_float3_1(features_0, prim_ind_0, 7U);

#line 58
            (&feat_0)->f7_0 = _S82;
            float3  _S83 = get_float3_1(features_0, prim_ind_0, 8U);

#line 59
            (&feat_0)->f8_0 = _S83;
            if(sh_degree_0 > 2U)
            {

#line 61
                float3  _S84 = get_float3_1(features_0, prim_ind_0, 9U);

#line 61
                (&feat_0)->f9_0 = _S84;
                float3  _S85 = get_float3_1(features_0, prim_ind_0, 10U);

#line 62
                (&feat_0)->f10_0 = _S85;
                float3  _S86 = get_float3_1(features_0, prim_ind_0, 11U);

#line 63
                (&feat_0)->f11_0 = _S86;
                float3  _S87 = get_float3_1(features_0, prim_ind_0, 12U);

#line 64
                (&feat_0)->f12_0 = _S87;
                float3  _S88 = get_float3_1(features_0, prim_ind_0, 13U);

#line 65
                (&feat_0)->f13_0 = _S88;
                float3  _S89 = get_float3_1(features_0, prim_ind_0, 14U);

#line 66
                (&feat_0)->f14_0 = _S89;
                float3  _S90 = get_float3_1(features_0, prim_ind_0, 15U);

#line 67
                (&feat_0)->f15_0 = _S90;

#line 60
            }

#line 54
        }

#line 50
    }

#line 71
    return feat_0;
}


#line 71
__device__ Features_0 Features_x24_syn_dzero_0()
{

#line 71
    Features_0 result_1;

#line 1751 "core.meta.slang"
    float3  _S91 = make_float3 (0.0f);

#line 1751
    (&result_1)->f0_0 = _S91;

#line 1751
    (&result_1)->f1_0 = _S91;

#line 1751
    (&result_1)->f2_0 = _S91;

#line 1751
    (&result_1)->f3_0 = _S91;

#line 1751
    (&result_1)->f4_0 = _S91;

#line 1751
    (&result_1)->f5_0 = _S91;

#line 1751
    (&result_1)->f6_0 = _S91;

#line 1751
    (&result_1)->f7_0 = _S91;

#line 1751
    (&result_1)->f8_0 = _S91;

#line 1751
    (&result_1)->f9_0 = _S91;

#line 1751
    (&result_1)->f10_0 = _S91;

#line 1751
    (&result_1)->f11_0 = _S91;

#line 1751
    (&result_1)->f12_0 = _S91;

#line 1751
    (&result_1)->f13_0 = _S91;

#line 1751
    (&result_1)->f14_0 = _S91;

#line 1751
    (&result_1)->f15_0 = _S91;

#line 1751
    return result_1;
}


#line 1751
__device__ Features_0 Features_x24_syn_dadd_0(Features_0 SLANG_anonymous_0_0, Features_0 SLANG_anonymous_1_0)
{

#line 1751
    Features_0 result_2;

#line 1751
    (&result_2)->f0_0 = SLANG_anonymous_0_0.f0_0 + SLANG_anonymous_1_0.f0_0;

#line 1751
    (&result_2)->f1_0 = SLANG_anonymous_0_0.f1_0 + SLANG_anonymous_1_0.f1_0;

#line 1751
    (&result_2)->f2_0 = SLANG_anonymous_0_0.f2_0 + SLANG_anonymous_1_0.f2_0;

#line 1751
    (&result_2)->f3_0 = SLANG_anonymous_0_0.f3_0 + SLANG_anonymous_1_0.f3_0;

#line 1751
    (&result_2)->f4_0 = SLANG_anonymous_0_0.f4_0 + SLANG_anonymous_1_0.f4_0;

#line 1751
    (&result_2)->f5_0 = SLANG_anonymous_0_0.f5_0 + SLANG_anonymous_1_0.f5_0;

#line 1751
    (&result_2)->f6_0 = SLANG_anonymous_0_0.f6_0 + SLANG_anonymous_1_0.f6_0;

#line 1751
    (&result_2)->f7_0 = SLANG_anonymous_0_0.f7_0 + SLANG_anonymous_1_0.f7_0;

#line 1751
    (&result_2)->f8_0 = SLANG_anonymous_0_0.f8_0 + SLANG_anonymous_1_0.f8_0;

#line 1751
    (&result_2)->f9_0 = SLANG_anonymous_0_0.f9_0 + SLANG_anonymous_1_0.f9_0;

#line 1751
    (&result_2)->f10_0 = SLANG_anonymous_0_0.f10_0 + SLANG_anonymous_1_0.f10_0;

#line 1751
    (&result_2)->f11_0 = SLANG_anonymous_0_0.f11_0 + SLANG_anonymous_1_0.f11_0;

#line 1751
    (&result_2)->f12_0 = SLANG_anonymous_0_0.f12_0 + SLANG_anonymous_1_0.f12_0;

#line 1751
    (&result_2)->f13_0 = SLANG_anonymous_0_0.f13_0 + SLANG_anonymous_1_0.f13_0;

#line 1751
    (&result_2)->f14_0 = SLANG_anonymous_0_0.f14_0 + SLANG_anonymous_1_0.f14_0;

#line 1751
    (&result_2)->f15_0 = SLANG_anonymous_0_0.f15_0 + SLANG_anonymous_1_0.f15_0;

#line 1751
    return result_2;
}


#line 49 "/home/amai/ever_training/ever/splinetracers/slang/sh.slang"
__device__ float3  eval_sh_col0_0(float3  dir_0, Features_0 feat_1)
{

#line 50
    return make_float3 (0.282094806432724f) * feat_1.f0_0 + make_float3 (0.5f);
}


__device__ float3  eval_sh_col1_0(float3  dir_1, Features_0 feat_2)
{



    return make_float3 (-0.48860251903533936f * dir_1.y) * feat_2.f1_0 + make_float3 (0.48860251903533936f * dir_1.z) * feat_2.f2_0 - make_float3 (0.48860251903533936f * dir_1.x) * feat_2.f3_0;
}


__device__ float3  eval_sh_col2_0(float3  dir_2, Features_0 feat_3)
{

#line 64
    float x_0 = dir_2.x;
    float y_0 = dir_2.y;
    float z_0 = dir_2.z;
    float xx_0 = x_0 * x_0;

#line 67
    float yy_0 = y_0 * y_0;

    return make_float3 (1.09254848957061768f * (x_0 * y_0)) * feat_3.f4_0 + make_float3 (-1.09254848957061768f * (y_0 * z_0)) * feat_3.f5_0 + make_float3 (0.31539157032966614f * (2.0f * (z_0 * z_0) - xx_0 - yy_0)) * feat_3.f6_0 + make_float3 (-1.09254848957061768f * (x_0 * z_0)) * feat_3.f7_0 + make_float3 (0.54627424478530884f * (xx_0 - yy_0)) * feat_3.f8_0;
}




__device__ float3  eval_sh_col3_0(float3  dir_3, Features_0 feat_4)
{

#line 76
    float x_1 = dir_3.x;
    float y_1 = dir_3.y;
    float z_1 = dir_3.z;
    float xx_1 = x_1 * x_1;

#line 79
    float yy_1 = y_1 * y_1;

#line 79
    float zz_0 = z_1 * z_1;

    float _S92 = 3.0f * xx_1;

    float _S93 = 4.0f * zz_0 - xx_1 - yy_1;
    float _S94 = 3.0f * yy_1;

#line 81
    return make_float3 (-0.59004360437393188f * y_1 * (_S92 - yy_1)) * feat_4.f9_0 + make_float3 (2.89061141014099121f * (x_1 * y_1) * z_1) * feat_4.f10_0 + make_float3 (-0.4570457935333252f * y_1 * _S93) * feat_4.f11_0 + make_float3 (0.37317633628845215f * z_1 * (2.0f * zz_0 - _S92 - _S94)) * feat_4.f12_0 + make_float3 (-0.4570457935333252f * x_1 * _S93) * feat_4.f13_0 + make_float3 (1.44530570507049561f * z_1 * (xx_1 - yy_1)) * feat_4.f14_0 + make_float3 (-0.59004360437393188f * x_1 * (xx_1 - _S94)) * feat_4.f15_0;
}


#line 91
__device__ float3  eval_color_0(float3  dir_4, Features_0 feat_5, uint sh_degree_1)
{

#line 92
    float3  color_0 = eval_sh_col0_0(dir_4, feat_5);

#line 92
    float3  color_1;
    if(sh_degree_1 > 0U)
    {

#line 94
        float3  color_2 = color_0 + eval_sh_col1_0(dir_4, feat_5);
        if(sh_degree_1 > 1U)
        {

#line 96
            float3  color_3 = color_2 + eval_sh_col2_0(dir_4, feat_5);
            if(sh_degree_1 > 2U)
            {

#line 97
                color_1 = color_3 + eval_sh_col3_0(dir_4, feat_5);

#line 97
            }
            else
            {

#line 97
                color_1 = color_3;

#line 97
            }

#line 95
        }
        else
        {

#line 95
            color_1 = color_2;

#line 95
        }

#line 93
    }
    else
    {

#line 93
        color_1 = color_0;

#line 93
    }

#line 102
    return color_1;
}


#line 47 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/spline-machine.slang"
struct ControlPoint_0
{
    float t_1;
    float4  dirac_0;
};


#line 133 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ ControlPoint_0 ControlPoint_x24_syn_dzero_0()
{

#line 133
    ControlPoint_0 result_3;

#line 133
    (&result_3)->t_1 = 0.0f;

#line 133
    (&result_3)->dirac_0 = make_float4 (0.0f);

#line 133
    return result_3;
}


#line 133
struct DiffPair_vectorx3Cfloatx2C3x3E_0
{
    float3  primal_0;
    float3  differential_0;
};


#line 1490 "diff.meta.slang"
__device__ void _d_cross_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * a_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * b_0, float3  dOut_3)
{

#line 1497
    float _S95 = dOut_3.y;

#line 1497
    float _S96 = dOut_3.z;
    float _S97 = dOut_3.x;

#line 1498
    float _S98 = (*a_0).primal_0.z * _S95 + - (*a_0).primal_0.y * _S96;

#line 1498
    float _S99 = - (*a_0).primal_0.z * _S97 + (*a_0).primal_0.x * _S96;

#line 1498
    float _S100 = (*a_0).primal_0.y * _S97 + - (*a_0).primal_0.x * _S95;

#line 1505
    float3  _S101 = make_float3 (- (*b_0).primal_0.z * _S95 + (*b_0).primal_0.y * _S96, (*b_0).primal_0.z * _S97 + - (*b_0).primal_0.x * _S96, - (*b_0).primal_0.y * _S97 + (*b_0).primal_0.x * _S95);

#line 1505
    a_0->primal_0 = (*a_0).primal_0;

#line 1505
    a_0->differential_0 = _S101;
    float3  _S102 = make_float3 (_S98, _S99, _S100);

#line 1506
    b_0->primal_0 = (*b_0).primal_0;

#line 1506
    b_0->differential_0 = _S102;
    return;
}


#line 7415 "hlsl.meta.slang"
__device__ float3  cross_0(float3  left_0, float3  right_0)
{

#line 7429
    float _S103 = left_0.y;

#line 7429
    float _S104 = right_0.z;

#line 7429
    float _S105 = left_0.z;

#line 7429
    float _S106 = right_0.y;
    float _S107 = right_0.x;

#line 7430
    float _S108 = left_0.x;

#line 7428
    return make_float3 (_S103 * _S104 - _S105 * _S106, _S105 * _S107 - _S108 * _S104, _S108 * _S106 - _S103 * _S107);
}


#line 238 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
__device__ float3  rotate_vector_0(float3  v_0, float4  q_0)
{


    float3  _S109 = - float3 {q_0.y, q_0.z, q_0.w};

#line 242
    float3  _S110 = make_float3 (2.0f) * cross_0(_S109, v_0);
    return v_0 + make_float3 (q_0.x) * _S110 + cross_0(_S109, _S110);
}


#line 1443 "diff.meta.slang"
__device__ void _d_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_3, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_1, float dOut_4)
{
    float3  x_d_result_0;



    *&((&x_d_result_0)->x) = (*dpy_1).primal_0.x * dOut_4;

#line 1445
    float3  y_d_result_0;

#line 1450
    *&((&y_d_result_0)->x) = (*dpx_3).primal_0.x * dOut_4;

#line 1449
    *&((&x_d_result_0)->y) = (*dpy_1).primal_0.y * dOut_4;
    *&((&y_d_result_0)->y) = (*dpx_3).primal_0.y * dOut_4;

#line 1449
    *&((&x_d_result_0)->z) = (*dpy_1).primal_0.z * dOut_4;
    *&((&y_d_result_0)->z) = (*dpx_3).primal_0.z * dOut_4;

#line 1450
    dpx_3->primal_0 = (*dpx_3).primal_0;

#line 1450
    dpx_3->differential_0 = x_d_result_0;

#line 1450
    dpy_1->primal_0 = (*dpy_1).primal_0;

#line 1450
    dpy_1->differential_0 = y_d_result_0;



    return;
}


#line 7891 "hlsl.meta.slang"
__device__ float dot_0(float3  x_2, float3  y_2)
{

#line 7891
    int i_0 = int(0);

#line 7891
    float result_4 = 0.0f;

#line 7904
    for(;;)
    {

#line 7904
        if(i_0 < int(3))
        {
        }
        else
        {

#line 7904
            break;
        }

#line 7905
        float result_5 = result_4 + _slang_vector_get_element(x_2, i_0) * _slang_vector_get_element(y_2, i_0);

#line 7904
        i_0 = i_0 + int(1);

#line 7904
        result_4 = result_5;

#line 7904
    }

    return result_4;
}


#line 7891
__device__ float dot_1(float4  x_3, float4  y_3)
{

#line 7891
    int i_1 = int(0);

#line 7891
    float result_6 = 0.0f;

#line 7904
    for(;;)
    {

#line 7904
        if(i_1 < int(4))
        {
        }
        else
        {

#line 7904
            break;
        }

#line 7905
        float result_7 = result_6 + _slang_vector_get_element(x_3, i_1) * _slang_vector_get_element(y_3, i_1);

#line 7904
        i_1 = i_1 + int(1);

#line 7904
        result_6 = result_7;

#line 7904
    }

    return result_6;
}


#line 1 "token paste"
__device__ void _d_abs_0(DiffPair_float_0 * dpx_4, float dOut_5)
{

#line 1719 "diff.meta.slang"
    float _S111 = _slang_select((*dpx_4).primal_0 > 0.0f, 1.0f,_slang_select((*dpx_4).primal_0 == 0.0f, 0.0f,-1.0f)) * dOut_5;

#line 1719
    dpx_4->primal_0 = (*dpx_4).primal_0;

#line 1719
    dpx_4->differential_0 = _S111;



    return;
}


#line 1960
__device__ void _d_min_0(DiffPair_float_0 * dpx_5, DiffPair_float_0 * dpy_2, float dOut_6)
{
    DiffPair_float_0 _S112 = *dpx_5;

#line 1962
    float _S113;

#line 1962
    if((*dpx_5).primal_0 < (*dpy_2).primal_0)
    {

#line 1962
        _S113 = dOut_6;

#line 1962
    }
    else
    {

#line 1962
        _S113 = 0.0f;

#line 1962
    }

#line 1962
    dpx_5->primal_0 = _S112.primal_0;

#line 1962
    dpx_5->differential_0 = _S113;
    DiffPair_float_0 _S114 = *dpy_2;

#line 1963
    if((*dpy_2).primal_0 < _S112.primal_0)
    {

#line 1963
        _S113 = dOut_6;

#line 1963
    }
    else
    {

#line 1963
        _S113 = 0.0f;

#line 1963
    }

#line 1963
    dpy_2->primal_0 = _S114.primal_0;

#line 1963
    dpy_2->differential_0 = _S113;
    return;
}


#line 45 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
__device__ float clip_0(float v_1, float minv_0, float maxv_0)
{

#line 46
    return (F32_max(((F32_min((v_1), (maxv_0)))), (minv_0)));
}


#line 68
__device__ void bw_safe_div_0(DiffPair_float_0 * a_1, DiffPair_float_0 * b_1, float R_0)
{

#line 69
    if((F32_abs(((*b_1).primal_0))) < 1.07549441632776457e-20f)
    {

#line 70
        float _S115 = clip_0(R_0 / 1.07549441632776457e-20f, -1.00000002004087734e+20f, 1.00000002004087734e+20f);

#line 70
        a_1->primal_0 = (*a_1).primal_0;

#line 70
        a_1->differential_0 = _S115;

#line 69
    }
    else
    {
        float _S116 = clip_0(R_0 / (*b_1).primal_0, -1.00000002004087734e+20f, 1.00000002004087734e+20f);

#line 72
        a_1->primal_0 = (*a_1).primal_0;

#line 72
        a_1->differential_0 = _S116;

#line 69
    }

#line 74
    float _S117 = (*b_1).primal_0 * (*b_1).primal_0;
    if(_S117 < 1.07549441632776457e-20f)
    {

#line 76
        float _S118 = clip_0(- (*a_1).primal_0 / 1.07549441632776457e-20f * R_0, -1.00000002004087734e+20f, 1.00000002004087734e+20f);

#line 76
        b_1->primal_0 = (*b_1).primal_0;

#line 76
        b_1->differential_0 = _S118;

#line 75
    }
    else
    {
        float _S119 = clip_0(- (*a_1).primal_0 / _S117 * R_0, -1.00000002004087734e+20f, 1.00000002004087734e+20f);

#line 78
        b_1->primal_0 = (*b_1).primal_0;

#line 78
        b_1->differential_0 = _S119;

#line 75
    }

#line 80
    return;
}



__device__ float safe_div_0(float a_2, float b_2)
{

#line 86
    if((F32_abs((b_2))) < 1.07549441632776457e-20f)
    {

#line 87
        return clip_0(a_2 / 1.07549441632776457e-20f, -1.00000002004087734e+20f, 1.00000002004087734e+20f);
    }
    else
    {

#line 89
        return clip_0(a_2 / b_2, -1.00000002004087734e+20f, 1.00000002004087734e+20f);
    }

#line 89
}




__device__ float3  safe_div_1(float3  a_3, float b_3)
{

#line 95
    return make_float3 (safe_div_0(a_3.x, b_3), safe_div_0(a_3.y, b_3), safe_div_0(a_3.z, b_3));
}


#line 137
__device__ float3  l2_normalize_0(float3  x_4)
{

#line 138
    return safe_div_1(x_4, (F32_sqrt(((F32_max((dot_0(x_4, x_4)), (1.07549441632776457e-20f)))))));
}


#line 113
__device__ float3  safe_div_2(float3  a_4, float3  b_4)
{

#line 114
    return make_float3 (safe_div_0(a_4.x, b_4.x), safe_div_0(a_4.y, b_4.y), safe_div_0(a_4.z, b_4.z));
}


#line 123
__device__ float safe_sqrt_0(float a_5)
{

#line 124
    if(a_5 < 1.07549441632776457e-20f)
    {

#line 125
        return 0.0f;
    }
    else
    {

#line 127
        return (F32_sqrt((a_5)));
    }

#line 127
}


#line 133 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/tri-intersect.slang"
__device__ float2  safe_eliIntersect_0(float3  ro_0, float3  rd_0, float3  ra_0)
{



    float3  ocn_0 = safe_div_2(ro_0, ra_0);
    float3  rdn_0 = safe_div_2(rd_0, ra_0);
    float a_6 = dot_0(rdn_0, rdn_0);
    float bp_0 = - dot_0(ocn_0, rdn_0);
    float3  l_0 = ocn_0 + make_float3 (safe_div_0(bp_0, a_6)) * rdn_0;
    float h_0 = a_6 * (1.0f - dot_0(l_0, l_0));
    float c_0 = dot_0(ocn_0, ocn_0) - 1.0f;
    if(h_0 < 0.0f)
    {

#line 145
        return make_float2 (-1.0f);
    }

#line 146
    float _S120 = bp_0 + float((F32_sign((bp_0)))) * safe_sqrt_0(h_0);
    return make_float2 (safe_div_0(c_0, _S120), safe_div_0(_S120, a_6));
}


#line 179
__device__ float2  safe_ray_intersect_ellipsoid_0(float3  rayo_0, float3  rayd_0, float3  scales_0, float4  quat_0)
{

#line 188
    float2  fminmaxt_0 = safe_eliIntersect_0(rotate_vector_0(rayo_0, quat_0), l2_normalize_0(rotate_vector_0(rayd_0, quat_0)), scales_0);
    float _S121 = fminmaxt_0.x;

#line 189
    float _S122 = fminmaxt_0.y;

#line 189
    return make_float2 ((F32_min((_S121), (_S122))), (F32_max((_S121), (_S122))));
}


#line 290
__device__ ControlPoint_0 safe_intersect_0(float3  rayo_1, float3  rayd_1, float3  scales_1, float3  mean_0, float4  quat_1, float3  color_4, float density_0, uint face_id_0, bool skip_close_0)
{

    float2  minmaxt_0 = safe_ray_intersect_ellipsoid_0(rayo_1 - mean_0, rayd_1, scales_1, quat_1);

    bool _S123 = face_id_0 == 1U;

#line 295
    float t_2;

#line 295
    if(_S123)
    {

#line 295
        t_2 = minmaxt_0.x;

#line 295
    }
    else
    {

#line 295
        t_2 = minmaxt_0.y;

#line 295
    }

#line 295
    float dirac_multi_0;
    if(_S123)
    {

#line 296
        dirac_multi_0 = density_0;

#line 296
    }
    else
    {

#line 296
        dirac_multi_0 = - density_0;

#line 296
    }

    ControlPoint_0 out_0 = { t_2, make_float4 (dirac_multi_0, dirac_multi_0 * color_4.x, dirac_multi_0 * color_4.y, dirac_multi_0 * color_4.z) };


    return out_0;
}


#line 115 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
struct DualModel_0
{
    TensorView means_0;
    TensorView scales_2;
    TensorView quats_0;
    TensorView densities_0;
    TensorView features_1;
    TensorView dL_dmeans_0;
    TensorView dL_dscales_0;
    TensorView dL_dquats_0;
    TensorView dL_ddensities_0;
    TensorView dL_dfeatures_0;
    TensorView dL_drayos_0;
    TensorView dL_drayds_0;
    TensorView dL_dmeans2D_0;
};


#line 224
__device__ ControlPoint_0 load_ctrl_pt_0(uint older_tri_ind_0, DualModel_0 model_0, float3  origin_0, float3  direction_0, uint sh_degree_2, bool skip_close_1)
{
    uint _S124 = uint((F32_floor((float(older_tri_ind_0 / 2U)))));
    uint _S125 = ((older_tri_ind_0) % (2U));


    float3  _S126 = get_float3_0(model_0.means_0, _S124);

    float3  _S127 = get_float3_0(model_0.scales_2, _S124);
    float4  _S128 = get_float4_0(model_0.quats_0, _S124);

    float _S129 = ((model_0.densities_0).load<float>((_S124)));

    Features_0 older_feat_0 = get_feats_0(model_0.features_1, _S124, sh_degree_2);


    return safe_intersect_0(origin_0, direction_0, _S127, _S126, _S128, eval_color_0(direction_0, older_feat_0, sh_degree_2), _S129, _S125, skip_close_1);
}


#line 142 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
__device__ float safe_exp_0(float v_2)
{

#line 143
    return (F32_exp((clip_0(v_2, -1.00000002004087734e+20f, (F32_log((1.00000002004087734e+20f)))))));
}


#line 1896 "diff.meta.slang"
__device__ void _d_pow_0(DiffPair_float_0 * dpx_6, DiffPair_float_0 * dpy_3, float dOut_7)
{

    if((*dpx_6).primal_0 < 9.99999997475242708e-07f)
    {

#line 1899
        dpx_6->primal_0 = (*dpx_6).primal_0;

#line 1899
        dpx_6->differential_0 = 0.0f;

#line 1899
        dpy_3->primal_0 = (*dpy_3).primal_0;

#line 1899
        dpy_3->differential_0 = 0.0f;

#line 1899
    }
    else
    {

#line 1906
        float val_0 = (F32_pow(((*dpx_6).primal_0), ((*dpy_3).primal_0)));

        DiffPair_float_0 _S130 = *dpx_6;

#line 1908
        float _S131 = val_0 * (*dpy_3).primal_0 / (*dpx_6).primal_0 * dOut_7;

#line 1908
        dpx_6->primal_0 = (*dpx_6).primal_0;

#line 1908
        dpx_6->differential_0 = _S131;

#line 1908
        float _S132 = val_0 * (F32_log((_S130.primal_0))) * dOut_7;

#line 1908
        dpy_3->primal_0 = (*dpy_3).primal_0;

#line 1908
        dpy_3->differential_0 = _S132;

#line 1899
    }

#line 1914
    return;
}


#line 247 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
__device__ float tukey_power_ladder_0(float x_5, float p_0)
{

    float _S133 = (F32_abs((p_0 - 1.0f)));

    return float((F32_sign((x_5)))) * _S133 / p_0 * ((F32_pow(((F32_abs((x_5))) / (F32_max((1.07549441632776457e-20f), (_S133))) + 1.0f), (p_0))) - 1.0f);
}


#line 65 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/spline-machine.slang"
__device__ SplineState_0 inverse_update_dual_0(SplineState_0 new_state_0, ControlPoint_0 new_ctrl_pt_0, ControlPoint_0 ctrl_pt_0, float t_min_0, float t_max_0)
{

#line 73
    float dt_0 = (F32_max((new_state_0.t_0 - ctrl_pt_0.t_1), (0.0f)));

    float2  _S134 = make_float2 (0.0f, 0.0f);

#line 75
    float3  _S135 = make_float3 (0.0f, 0.0f, 0.0f);

#line 75
    float4  _S136 = make_float4 (0.0f, 0.0f, 0.0f, 0.0f);

#line 75
    SplineState_0 state_0;

#line 75
    (&state_0)->distortion_parts_0 = _S134;

#line 75
    (&state_0)->cum_sum_0 = _S134;

#line 75
    (&state_0)->padding_0 = _S135;

#line 75
    (&state_0)->t_0 = 0.0f;

#line 75
    (&state_0)->drgb_0 = _S136;

#line 75
    (&state_0)->logT_0 = 0.0f;

#line 75
    (&state_0)->C_0 = _S135;
    (&state_0)->drgb_0 = new_state_0.drgb_0 - new_ctrl_pt_0.dirac_0;

    (&state_0)->t_0 = ctrl_pt_0.t_1;

#line 83
    float _S137 = (&state_0)->drgb_0.x;

#line 83
    float area_0 = (F32_max((_S137 * dt_0), (0.0f)));
    float3  _S138 = safe_div_1(make_float3 ((&state_0)->drgb_0.y, (&state_0)->drgb_0.z, (&state_0)->drgb_0.w), _S137);

    float _S139 = (F32_max((new_state_0.logT_0 - area_0), (0.0f)));

#line 86
    (&state_0)->logT_0 = _S139;
    float weight_0 = clip_0((1.0f - safe_exp_0(- area_0)) * safe_exp_0(- _S139), 0.0f, 1.0f);

    (&state_0)->C_0 = new_state_0.C_0 - make_float3 (weight_0) * _S138;



    float _S140 = tukey_power_ladder_0((new_state_0.t_0 + (&state_0)->t_0) / 2.0f * 1000.0f, -0.10000000149011612f);
    *&((&(&state_0)->cum_sum_0)->x) = new_state_0.cum_sum_0.x - weight_0;
    *&((&(&state_0)->cum_sum_0)->y) = new_state_0.cum_sum_0.y - weight_0 * _S140;
    float _S141 = 2.0f * weight_0;

#line 96
    *&((&(&state_0)->distortion_parts_0)->x) = new_state_0.distortion_parts_0.x - _S141 * _S140 * (&state_0)->cum_sum_0.x;
    *&((&(&state_0)->distortion_parts_0)->y) = new_state_0.distortion_parts_0.y - _S141 * (&state_0)->cum_sum_0.y;

    return state_0;
}


#line 59
__device__ SplineState_0 from_dual_0(SplineState_0 state_1, ControlPoint_0 ctrl_pt_1)
{

    return state_1;
}


#line 87 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ void atomic_add_float3_0(TensorView view_4, uint ind_4, float3  val_1)
{

#line 88
    float temp_0;
    *((&temp_0)) = atomicAdd((view_4).data_ptr_at<float>((make_uint2 (ind_4, 0U))), (val_1.x));
    *((&temp_0)) = atomicAdd((view_4).data_ptr_at<float>((make_uint2 (ind_4, 1U))), (val_1.y));
    *((&temp_0)) = atomicAdd((view_4).data_ptr_at<float>((make_uint2 (ind_4, 2U))), (val_1.z));
    return;
}


#line 94
__device__ void atomic_add_float4_0(TensorView view_5, uint ind_5, float4  val_2)
{

#line 95
    float temp_1;
    *((&temp_1)) = atomicAdd((view_5).data_ptr_at<float>((make_uint2 (ind_5, 0U))), (val_2.x));
    *((&temp_1)) = atomicAdd((view_5).data_ptr_at<float>((make_uint2 (ind_5, 1U))), (val_2.y));
    *((&temp_1)) = atomicAdd((view_5).data_ptr_at<float>((make_uint2 (ind_5, 2U))), (val_2.z));
    *((&temp_1)) = atomicAdd((view_5).data_ptr_at<float>((make_uint2 (ind_5, 3U))), (val_2.w));
    return;
}


#line 74
__device__ void atomic_add_float3_1(TensorView view_6, uint ind_6, uint feat_ind_1, float3  val_3)
{

#line 75
    float temp_2;
    *((&temp_2)) = atomicAdd((view_6).data_ptr_at<float>((make_uint3 (ind_6, feat_ind_1, 0U))), (val_3.x));
    *((&temp_2)) = atomicAdd((view_6).data_ptr_at<float>((make_uint3 (ind_6, feat_ind_1, 1U))), (val_3.y));
    *((&temp_2)) = atomicAdd((view_6).data_ptr_at<float>((make_uint3 (ind_6, feat_ind_1, 2U))), (val_3.z));
    return;
}


#line 79
struct DiffPair_vectorx3Cfloatx2C4x3E_0
{
    float4  primal_0;
    float4  differential_0;
};


#line 172
struct DiffPair_matrixx3Cfloatx2C4x2C4x3E_0
{
    Matrix<float, 4, 4>  primal_0;
    Matrix<float, 4, 4>  differential_0;
};


#line 1312 "diff.meta.slang"
__device__ void _d_mul_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * left_1, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * right_1, float4  dOut_8)
{

    Matrix<float, 4, 4>  right_d_result_0;

#line 1314
    float4  left_d_result_0;

#line 1314
    float _S142 = (*right_1).primal_0.rows[int(0)].x * dOut_8.x;

#line 1324
    *&(((&right_d_result_0)->rows + (int(0)))->x) = (*left_1).primal_0.x * dOut_8.x;

#line 1323
    float sum_0 = _S142 + (*right_1).primal_0.rows[int(0)].y * dOut_8.y;
    *&(((&right_d_result_0)->rows + (int(0)))->y) = (*left_1).primal_0.x * dOut_8.y;

#line 1323
    float sum_1 = sum_0 + (*right_1).primal_0.rows[int(0)].z * dOut_8.z;
    *&(((&right_d_result_0)->rows + (int(0)))->z) = (*left_1).primal_0.x * dOut_8.z;

#line 1323
    float sum_2 = sum_1 + (*right_1).primal_0.rows[int(0)].w * dOut_8.w;
    *&(((&right_d_result_0)->rows + (int(0)))->w) = (*left_1).primal_0.x * dOut_8.w;

    *&((&left_d_result_0)->x) = sum_2;

#line 1326
    float _S143 = (*right_1).primal_0.rows[int(1)].x * dOut_8.x;

#line 1324
    *&(((&right_d_result_0)->rows + (int(1)))->x) = (*left_1).primal_0.y * dOut_8.x;

#line 1323
    float sum_3 = _S143 + (*right_1).primal_0.rows[int(1)].y * dOut_8.y;
    *&(((&right_d_result_0)->rows + (int(1)))->y) = (*left_1).primal_0.y * dOut_8.y;

#line 1323
    float sum_4 = sum_3 + (*right_1).primal_0.rows[int(1)].z * dOut_8.z;
    *&(((&right_d_result_0)->rows + (int(1)))->z) = (*left_1).primal_0.y * dOut_8.z;

#line 1323
    float sum_5 = sum_4 + (*right_1).primal_0.rows[int(1)].w * dOut_8.w;
    *&(((&right_d_result_0)->rows + (int(1)))->w) = (*left_1).primal_0.y * dOut_8.w;

    *&((&left_d_result_0)->y) = sum_5;

#line 1326
    float _S144 = (*right_1).primal_0.rows[int(2)].x * dOut_8.x;

#line 1324
    *&(((&right_d_result_0)->rows + (int(2)))->x) = (*left_1).primal_0.z * dOut_8.x;

#line 1323
    float sum_6 = _S144 + (*right_1).primal_0.rows[int(2)].y * dOut_8.y;
    *&(((&right_d_result_0)->rows + (int(2)))->y) = (*left_1).primal_0.z * dOut_8.y;

#line 1323
    float sum_7 = sum_6 + (*right_1).primal_0.rows[int(2)].z * dOut_8.z;
    *&(((&right_d_result_0)->rows + (int(2)))->z) = (*left_1).primal_0.z * dOut_8.z;

#line 1323
    float sum_8 = sum_7 + (*right_1).primal_0.rows[int(2)].w * dOut_8.w;
    *&(((&right_d_result_0)->rows + (int(2)))->w) = (*left_1).primal_0.z * dOut_8.w;

    *&((&left_d_result_0)->z) = sum_8;

#line 1326
    float _S145 = (*right_1).primal_0.rows[int(3)].x * dOut_8.x;

#line 1324
    *&(((&right_d_result_0)->rows + (int(3)))->x) = (*left_1).primal_0.w * dOut_8.x;

#line 1323
    float sum_9 = _S145 + (*right_1).primal_0.rows[int(3)].y * dOut_8.y;
    *&(((&right_d_result_0)->rows + (int(3)))->y) = (*left_1).primal_0.w * dOut_8.y;

#line 1323
    float sum_10 = sum_9 + (*right_1).primal_0.rows[int(3)].z * dOut_8.z;
    *&(((&right_d_result_0)->rows + (int(3)))->z) = (*left_1).primal_0.w * dOut_8.z;

#line 1323
    float sum_11 = sum_10 + (*right_1).primal_0.rows[int(3)].w * dOut_8.w;
    *&(((&right_d_result_0)->rows + (int(3)))->w) = (*left_1).primal_0.w * dOut_8.w;

    *&((&left_d_result_0)->w) = sum_11;

#line 1326
    left_1->primal_0 = (*left_1).primal_0;

#line 1326
    left_1->differential_0 = left_d_result_0;

#line 1326
    right_1->primal_0 = (*right_1).primal_0;

#line 1326
    right_1->differential_0 = right_d_result_0;



    return;
}


#line 10858 "hlsl.meta.slang"
__device__ float4  mul_0(float4  left_2, Matrix<float, 4, 4>  right_2)
{

#line 10870
    float4  result_8;

#line 10870
    int j_0 = int(0);
    for(;;)
    {

#line 10871
        if(j_0 < int(4))
        {
        }
        else
        {

#line 10871
            break;
        }

#line 10871
        int i_2 = int(0);

#line 10871
        float sum_12 = 0.0f;


        for(;;)
        {

#line 10874
            if(i_2 < int(4))
            {
            }
            else
            {

#line 10874
                break;
            }
            float sum_13 = sum_12 + _slang_vector_get_element(left_2, i_2) * _slang_vector_get_element(right_2.rows[i_2], j_0);

#line 10874
            i_2 = i_2 + int(1);

#line 10874
            sum_12 = sum_13;

#line 10874
        }



        *_slang_vector_get_element_ptr(&result_8, j_0) = sum_12;

#line 10871
        j_0 = j_0 + int(1);

#line 10871
    }

#line 10880
    return result_8;
}


#line 10858
__device__ float3  mul_1(float3  left_3, Matrix<float, 3, 3>  right_3)
{

#line 10870
    float3  result_9;

#line 10870
    int j_1 = int(0);
    for(;;)
    {

#line 10871
        if(j_1 < int(3))
        {
        }
        else
        {

#line 10871
            break;
        }

#line 10871
        int i_3 = int(0);

#line 10871
        float sum_14 = 0.0f;


        for(;;)
        {

#line 10874
            if(i_3 < int(3))
            {
            }
            else
            {

#line 10874
                break;
            }
            float sum_15 = sum_14 + _slang_vector_get_element(left_3, i_3) * _slang_vector_get_element(right_3.rows[i_3], j_1);

#line 10874
            i_3 = i_3 + int(1);

#line 10874
            sum_14 = sum_15;

#line 10874
        }



        *_slang_vector_get_element_ptr(&result_9, j_1) = sum_14;

#line 10871
        j_1 = j_1 + int(1);

#line 10871
    }

#line 10880
    return result_9;
}


#line 245 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ float3  project_0(float3  xyz_0, Matrix<float, 4, 4>  wct_0)
{
    float4  _S146 = mul_0(make_float4 (xyz_0.x, xyz_0.y, xyz_0.z, 1.0f), wct_0);
    float _S147 = _S146.z;
    return make_float3 (safe_div_0(_S146.x, _S147), safe_div_0(_S146.y, _S147), _S147);
}


#line 81
__device__ void atomic_add_float2_0(TensorView view_7, uint ind_7, float2  val_4)
{

#line 82
    float temp_3;
    *((&temp_3)) = atomicAdd((view_7).data_ptr_at<float>((make_uint2 (ind_7, 0U))), (val_4.x));
    *((&temp_3)) = atomicAdd((view_7).data_ptr_at<float>((make_uint2 (ind_7, 1U))), (val_4.y));
    return;
}


#line 85
struct DiffPair_SplineState_0
{
    SplineState_0 primal_0;
    SplineState_0 differential_0;
};


#line 133
struct DiffPair_ControlPoint_0
{
    ControlPoint_0 primal_0;
    ControlPoint_0 differential_0;
};


#line 154
__device__ float s_primal_ctx_max_0(float _S148, float _S149)
{

#line 154
    return (F32_max((_S148), (_S149)));
}


#line 154
__device__ float s_primal_ctx_safe_div_0(float _S150, float _S151)
{

#line 154
    return safe_div_0(_S150, _S151);
}


#line 154
__device__ float3  s_primal_ctx_safe_div_1(float3  dpa_0, float dpb_0)
{

#line 94 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
    return make_float3 (s_primal_ctx_safe_div_0(dpa_0.x, dpb_0), s_primal_ctx_safe_div_0(dpa_0.y, dpb_0), s_primal_ctx_safe_div_0(dpa_0.z, dpb_0));
}


#line 94
__device__ float s_primal_ctx_log_0(float _S152)
{

#line 94
    return (F32_log((_S152)));
}


#line 94
__device__ float s_primal_ctx_min_0(float _S153, float _S154)
{

#line 94
    return (F32_min((_S153), (_S154)));
}


#line 94
__device__ float s_primal_ctx_clip_0(float dpv_0, float dpminv_0, float dpmaxv_0)
{

#line 45
    return s_primal_ctx_max_0(s_primal_ctx_min_0(dpv_0, dpmaxv_0), dpminv_0);
}


#line 45
__device__ float s_primal_ctx_exp_0(float _S155)
{

#line 45
    return (F32_exp((_S155)));
}


#line 45
__device__ float s_primal_ctx_safe_exp_0(float dpv_1)
{

#line 142
    return s_primal_ctx_exp_0(s_primal_ctx_clip_0(dpv_1, -1.00000002004087734e+20f, s_primal_ctx_log_0(1.00000002004087734e+20f)));
}


#line 142
__device__ float s_primal_ctx_abs_0(float _S156)
{

#line 142
    return (F32_abs((_S156)));
}


#line 142
__device__ float s_primal_ctx_pow_0(float _S157, float _S158)
{

#line 142
    return (F32_pow((_S157), (_S158)));
}


#line 142
__device__ float s_primal_ctx_tukey_power_ladder_0(float dpx_7, float dpp_0)
{

#line 247
    float _S159 = s_primal_ctx_abs_0(dpp_0 - 1.0f);

#line 247
    return float((F32_sign((dpx_7)))) * _S159 / dpp_0 * (s_primal_ctx_pow_0(s_primal_ctx_abs_0(dpx_7) / s_primal_ctx_max_0(1.07549441632776457e-20f, _S159) + 1.0f, dpp_0) - 1.0f);
}


#line 170 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ void s_bwd_prop_pow_0(DiffPair_float_0 * _S160, DiffPair_float_0 * _S161, float _S162)
{

#line 170
    _d_pow_0(_S160, _S161, _S162);

#line 170
    return;
}


#line 170
__device__ void s_bwd_prop_max_0(DiffPair_float_0 * _S163, DiffPair_float_0 * _S164, float _S165)
{

#line 170
    _d_max_0(_S163, _S164, _S165);

#line 170
    return;
}


#line 170
__device__ void s_bwd_prop_abs_0(DiffPair_float_0 * _S166, float _S167)
{

#line 170
    _d_abs_0(_S166, _S167);

#line 170
    return;
}


#line 247 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
__device__ void s_bwd_prop_tukey_power_ladder_0(DiffPair_float_0 * dpx_8, DiffPair_float_0 * dpp_1, float _s_dOut_0)
{

#line 247
    float _S168 = s_primal_ctx_abs_0((*dpx_8).primal_0);


    float _S169 = (*dpp_1).primal_0 - 1.0f;

#line 250
    float _S170 = s_primal_ctx_abs_0(_S169);

#line 250
    float _S171 = s_primal_ctx_max_0(1.07549441632776457e-20f, _S170);

#line 250
    float _S172 = _S171 * _S171;
    float _S173 = float((F32_sign(((*dpx_8).primal_0))));

#line 251
    float _S174 = _S173 * _S170;

#line 251
    float _S175 = (*dpp_1).primal_0 * (*dpp_1).primal_0;

#line 251
    float _S176 = _S168 / _S171 + 1.0f;

#line 251
    float _S177 = _S174 / (*dpp_1).primal_0 * _s_dOut_0;

#line 251
    float _S178 = (s_primal_ctx_pow_0(_S176, (*dpp_1).primal_0) - 1.0f) * _s_dOut_0;

#line 251
    DiffPair_float_0 _S179;

#line 251
    (&_S179)->primal_0 = _S176;

#line 251
    (&_S179)->differential_0 = 0.0f;

#line 251
    DiffPair_float_0 _S180;

#line 251
    (&_S180)->primal_0 = (*dpp_1).primal_0;

#line 251
    (&_S180)->differential_0 = 0.0f;

#line 251
    s_bwd_prop_pow_0(&_S179, &_S180, _S177);

#line 251
    float _S181 = _S178 / _S175;

#line 251
    float _S182 = _S174 * - _S181;

#line 251
    float _S183 = _S173 * ((*dpp_1).primal_0 * _S181);

#line 250
    float _S184 = _S179.differential_0 / _S172;

#line 250
    float _S185 = _S168 * - _S184;

#line 250
    float _S186 = _S171 * _S184;

#line 250
    DiffPair_float_0 _S187;

#line 250
    (&_S187)->primal_0 = 1.07549441632776457e-20f;

#line 250
    (&_S187)->differential_0 = 0.0f;

#line 250
    DiffPair_float_0 _S188;

#line 250
    (&_S188)->primal_0 = _S170;

#line 250
    (&_S188)->differential_0 = 0.0f;

#line 250
    s_bwd_prop_max_0(&_S187, &_S188, _S185);

#line 250
    float _S189 = _S183 + _S188.differential_0;

#line 250
    DiffPair_float_0 _S190;

#line 250
    (&_S190)->primal_0 = _S169;

#line 250
    (&_S190)->differential_0 = 0.0f;

#line 250
    s_bwd_prop_abs_0(&_S190, _S189);

#line 249
    DiffPair_float_0 _S191;

#line 249
    (&_S191)->primal_0 = (*dpx_8).primal_0;

#line 249
    (&_S191)->differential_0 = 0.0f;

#line 249
    s_bwd_prop_abs_0(&_S191, _S186);

#line 958 "core.meta.slang"
    float _S192 = _S180.differential_0 + _S182 + _S190.differential_0;

#line 958
    dpp_1->primal_0 = (*dpp_1).primal_0;

#line 958
    dpp_1->differential_0 = _S192;

#line 958
    dpx_8->primal_0 = (*dpx_8).primal_0;

#line 958
    dpx_8->differential_0 = _S191.differential_0;

#line 247 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
    return;
}


#line 247
__device__ void s_bwd_prop_min_0(DiffPair_float_0 * _S193, DiffPair_float_0 * _S194, float _S195)
{

#line 247
    _d_min_0(_S193, _S194, _S195);

#line 247
    return;
}


#line 45
__device__ void s_bwd_prop_clip_0(DiffPair_float_0 * dpv_2, DiffPair_float_0 * dpminv_1, DiffPair_float_0 * dpmaxv_1, float _s_dOut_1)
{

#line 46
    DiffPair_float_0 _S196;

#line 46
    (&_S196)->primal_0 = s_primal_ctx_min_0((*dpv_2).primal_0, (*dpmaxv_1).primal_0);

#line 46
    (&_S196)->differential_0 = 0.0f;

#line 46
    DiffPair_float_0 _S197;

#line 46
    (&_S197)->primal_0 = (*dpminv_1).primal_0;

#line 46
    (&_S197)->differential_0 = 0.0f;

#line 46
    s_bwd_prop_max_0(&_S196, &_S197, _s_dOut_1);

#line 46
    DiffPair_float_0 _S198;

#line 46
    (&_S198)->primal_0 = (*dpv_2).primal_0;

#line 46
    (&_S198)->differential_0 = 0.0f;

#line 46
    DiffPair_float_0 _S199;

#line 46
    (&_S199)->primal_0 = (*dpmaxv_1).primal_0;

#line 46
    (&_S199)->differential_0 = 0.0f;

#line 46
    s_bwd_prop_min_0(&_S198, &_S199, _S196.differential_0);

#line 46
    dpmaxv_1->primal_0 = (*dpmaxv_1).primal_0;

#line 46
    dpmaxv_1->differential_0 = _S199.differential_0;

#line 46
    dpminv_1->primal_0 = (*dpminv_1).primal_0;

#line 46
    dpminv_1->differential_0 = _S197.differential_0;

#line 46
    dpv_2->primal_0 = (*dpv_2).primal_0;

#line 46
    dpv_2->differential_0 = _S198.differential_0;

#line 45
    return;
}


#line 45
__device__ void s_bwd_prop_exp_0(DiffPair_float_0 * _S200, float _S201)
{

#line 45
    _d_exp_0(_S200, _S201);

#line 45
    return;
}


#line 142
__device__ void s_bwd_prop_safe_exp_0(DiffPair_float_0 * dpv_3, float _s_dOut_2)
{

#line 142
    float _S202 = s_primal_ctx_log_0(1.00000002004087734e+20f);
    DiffPair_float_0 _S203;

#line 143
    (&_S203)->primal_0 = s_primal_ctx_clip_0((*dpv_3).primal_0, -1.00000002004087734e+20f, _S202);

#line 143
    (&_S203)->differential_0 = 0.0f;

#line 143
    s_bwd_prop_exp_0(&_S203, _s_dOut_2);

#line 143
    DiffPair_float_0 _S204;

#line 143
    (&_S204)->primal_0 = (*dpv_3).primal_0;

#line 143
    (&_S204)->differential_0 = 0.0f;

#line 143
    DiffPair_float_0 _S205;

#line 143
    (&_S205)->primal_0 = -1.00000002004087734e+20f;

#line 143
    (&_S205)->differential_0 = 0.0f;

#line 143
    DiffPair_float_0 _S206;

#line 143
    (&_S206)->primal_0 = _S202;

#line 143
    (&_S206)->differential_0 = 0.0f;

#line 143
    s_bwd_prop_clip_0(&_S204, &_S205, &_S206, _S203.differential_0);

#line 143
    dpv_3->primal_0 = (*dpv_3).primal_0;

#line 143
    dpv_3->differential_0 = _S204.differential_0;

#line 142
    return;
}


#line 164 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ void s_bwd_prop_safe_div_0(DiffPair_float_0 * _S207, DiffPair_float_0 * _S208, float _S209)
{

#line 164
    bw_safe_div_0(_S207, _S208, _S209);

#line 164
    return;
}


#line 94 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
__device__ void s_bwd_prop_safe_div_1(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpa_1, DiffPair_float_0 * dpb_1, float3  _s_dOut_3)
{
    float _S210 = (*dpa_1).primal_0.x;
    float _S211 = (*dpa_1).primal_0.y;
    DiffPair_float_0 _S212;

#line 98
    (&_S212)->primal_0 = (*dpa_1).primal_0.z;

#line 98
    (&_S212)->differential_0 = 0.0f;

#line 98
    DiffPair_float_0 _S213;

#line 98
    (&_S213)->primal_0 = (*dpb_1).primal_0;

#line 98
    (&_S213)->differential_0 = 0.0f;

#line 98
    s_bwd_prop_safe_div_0(&_S212, &_S213, _s_dOut_3.z);

#line 97
    DiffPair_float_0 _S214;

#line 97
    (&_S214)->primal_0 = _S211;

#line 97
    (&_S214)->differential_0 = 0.0f;

#line 97
    DiffPair_float_0 _S215;

#line 97
    (&_S215)->primal_0 = (*dpb_1).primal_0;

#line 97
    (&_S215)->differential_0 = 0.0f;

#line 97
    s_bwd_prop_safe_div_0(&_S214, &_S215, _s_dOut_3.y);

#line 96
    DiffPair_float_0 _S216;

#line 96
    (&_S216)->primal_0 = _S210;

#line 96
    (&_S216)->differential_0 = 0.0f;

#line 96
    DiffPair_float_0 _S217;

#line 96
    (&_S217)->primal_0 = (*dpb_1).primal_0;

#line 96
    (&_S217)->differential_0 = 0.0f;

#line 96
    s_bwd_prop_safe_div_0(&_S216, &_S217, _s_dOut_3.x);

#line 958 "core.meta.slang"
    float _S218 = _S213.differential_0 + _S215.differential_0 + _S217.differential_0;

#line 958
    dpb_1->primal_0 = (*dpb_1).primal_0;

#line 958
    dpb_1->differential_0 = _S218;

#line 958
    float3  _S219 = make_float3 (_S216.differential_0, _S214.differential_0, _S212.differential_0);

#line 958
    dpa_1->primal_0 = (*dpa_1).primal_0;

#line 958
    dpa_1->differential_0 = _S219;

#line 94 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
    return;
}


#line 104 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/spline-machine.slang"
__device__ void s_bwd_prop_update_0(DiffPair_SplineState_0 * dpstate_0, DiffPair_ControlPoint_0 * dpctrl_pt_0, float t_min_1, float t_max_1, float max_prim_size_0, SplineState_0 _s_dOut_4)
{

#line 112
    float _S220 = (*dpctrl_pt_0).primal_0.t_1 - (*dpstate_0).primal_0.t_0;

#line 112
    float _S221 = s_primal_ctx_max_0(_S220, 0.0f);

#line 121
    float _S222 = (*dpstate_0).primal_0.drgb_0.x;

#line 121
    float _S223 = _S222 * _S221;

#line 121
    float _S224 = s_primal_ctx_max_0(_S223, 0.0f);

    float3  _S225 = make_float3 ((*dpstate_0).primal_0.drgb_0.y, (*dpstate_0).primal_0.drgb_0.z, (*dpstate_0).primal_0.drgb_0.w);

#line 123
    float3  _S226 = s_primal_ctx_safe_div_1(_S225, _S222);

    float _S227 = _S224 + (*dpstate_0).primal_0.logT_0;
    float _S228 = - _S224;

#line 126
    float _S229 = 1.0f - s_primal_ctx_safe_exp_0(_S228);

#line 126
    float _S230 = - (*dpstate_0).primal_0.logT_0;

#line 126
    float _S231 = s_primal_ctx_safe_exp_0(_S230);

#line 126
    float _S232 = _S229 * _S231;

#line 126
    float _S233 = s_primal_ctx_clip_0(_S232, 0.0f, 1.0f);
    float3  _S234 = make_float3 (_S233);



    float _S235 = ((*dpctrl_pt_0).primal_0.t_1 + (*dpstate_0).primal_0.t_0) / 2.0f * 1000.0f;

#line 131
    float _S236 = s_primal_ctx_tukey_power_ladder_0(_S235, -0.10000000149011612f);
    float _S237 = 2.0f * _S233;

#line 132
    float _S238 = _S237 * _S236;

#line 132
    float _S239 = (*dpstate_0).primal_0.cum_sum_0.x;
    float _S240 = (*dpstate_0).primal_0.cum_sum_0.y;

#line 133
    SplineState_0 _S241 = _s_dOut_4;

#line 133
    *&((&(&_S241)->cum_sum_0)->y) = 0.0f;

    float _S242 = _S233 * _s_dOut_4.cum_sum_0.y;

#line 135
    float _S243 = _S236 * _s_dOut_4.cum_sum_0.y;

#line 135
    *&((&(&_S241)->cum_sum_0)->x) = 0.0f;

#line 135
    *&((&(&_S241)->distortion_parts_0)->y) = 0.0f;

#line 133
    float _S244 = _S240 * _s_dOut_4.distortion_parts_0.y;

#line 133
    float _S245 = _s_dOut_4.cum_sum_0.y + _S237 * _s_dOut_4.distortion_parts_0.y;

#line 133
    *&((&(&_S241)->distortion_parts_0)->x) = 0.0f;

#line 132
    float _S246 = _S239 * _s_dOut_4.distortion_parts_0.x;

#line 132
    float2  _S247 = make_float2 (_s_dOut_4.cum_sum_0.x + _S238 * _s_dOut_4.distortion_parts_0.x, _S245);

#line 132
    float _S248 = 2.0f * (_S244 + _S236 * _S246);

#line 132
    float2  _S249 = make_float2 (_s_dOut_4.distortion_parts_0.x, _s_dOut_4.distortion_parts_0.y);

#line 131
    float _S250 = _S242 + _S237 * _S246;

#line 131
    DiffPair_float_0 _S251;

#line 131
    (&_S251)->primal_0 = _S235;

#line 131
    (&_S251)->differential_0 = 0.0f;

#line 131
    DiffPair_float_0 _S252;

#line 131
    (&_S252)->primal_0 = -0.10000000149011612f;

#line 131
    (&_S252)->differential_0 = 0.0f;

#line 131
    s_bwd_prop_tukey_power_ladder_0(&_S251, &_S252, _S250);

#line 131
    float _S253 = 0.5f * (1000.0f * _S251.differential_0);

#line 127
    float3  _S254 = _S234 * _S241.C_0;

#line 127
    float3  _S255 = _S226 * _S241.C_0;

#line 126
    float _S256 = _S243 + _s_dOut_4.cum_sum_0.x + _S248 + _S255.x + _S255.y + _S255.z;

#line 126
    DiffPair_float_0 _S257;

#line 126
    (&_S257)->primal_0 = _S232;

#line 126
    (&_S257)->differential_0 = 0.0f;

#line 126
    DiffPair_float_0 _S258;

#line 126
    (&_S258)->primal_0 = 0.0f;

#line 126
    (&_S258)->differential_0 = 0.0f;

#line 126
    DiffPair_float_0 _S259;

#line 126
    (&_S259)->primal_0 = 1.0f;

#line 126
    (&_S259)->differential_0 = 0.0f;

#line 126
    s_bwd_prop_clip_0(&_S257, &_S258, &_S259, _S256);

#line 126
    float _S260 = _S229 * _S257.differential_0;

#line 126
    float _S261 = _S231 * _S257.differential_0;

#line 126
    DiffPair_float_0 _S262;

#line 126
    (&_S262)->primal_0 = _S230;

#line 126
    (&_S262)->differential_0 = 0.0f;

#line 126
    s_bwd_prop_safe_exp_0(&_S262, _S260);

#line 126
    float _S263 = - _S262.differential_0;

#line 126
    float _S264 = - _S261;

#line 126
    DiffPair_float_0 _S265;

#line 126
    (&_S265)->primal_0 = _S228;

#line 126
    (&_S265)->differential_0 = 0.0f;

#line 126
    s_bwd_prop_safe_exp_0(&_S265, _S264);

#line 126
    float _S266 = - _S265.differential_0;

#line 125
    DiffPair_float_0 _S267;

#line 125
    (&_S267)->primal_0 = _S227;

#line 125
    (&_S267)->differential_0 = 0.0f;

#line 125
    DiffPair_float_0 _S268;

#line 125
    (&_S268)->primal_0 = 0.0f;

#line 125
    (&_S268)->differential_0 = 0.0f;

#line 125
    s_bwd_prop_max_0(&_S267, &_S268, _S241.logT_0);

#line 958 "core.meta.slang"
    float _S269 = _S263 + _S267.differential_0;

#line 123 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/spline-machine.slang"
    float3  _S270 = make_float3 (0.0f);

#line 123
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S271;

#line 123
    (&_S271)->primal_0 = _S225;

#line 123
    (&_S271)->differential_0 = _S270;

#line 123
    DiffPair_float_0 _S272;

#line 123
    (&_S272)->primal_0 = _S222;

#line 123
    (&_S272)->differential_0 = 0.0f;

#line 123
    s_bwd_prop_safe_div_1(&_S271, &_S272, _S254);

#line 121
    float _S273 = _S266 + _S267.differential_0;

#line 121
    DiffPair_float_0 _S274;

#line 121
    (&_S274)->primal_0 = _S223;

#line 121
    (&_S274)->differential_0 = 0.0f;

#line 121
    DiffPair_float_0 _S275;

#line 121
    (&_S275)->primal_0 = 0.0f;

#line 121
    (&_S275)->differential_0 = 0.0f;

#line 121
    s_bwd_prop_max_0(&_S274, &_S275, _S273);

#line 121
    float _S276 = _S222 * _S274.differential_0;

#line 121
    float4  _S277 = _S241.drgb_0 + make_float4 (_S272.differential_0 + _S221 * _S274.differential_0, _S271.differential_0.x, _S271.differential_0.y, _S271.differential_0.z);

#line 112
    DiffPair_float_0 _S278;

#line 112
    (&_S278)->primal_0 = _S220;

#line 112
    (&_S278)->differential_0 = 0.0f;

#line 112
    DiffPair_float_0 _S279;

#line 112
    (&_S279)->primal_0 = 0.0f;

#line 112
    (&_S279)->differential_0 = 0.0f;

#line 112
    s_bwd_prop_max_0(&_S278, &_S279, _S276);

#line 958 "core.meta.slang"
    float _S280 = _S253 + - _S278.differential_0;

#line 958
    float _S281 = _S253 + _S241.t_0 + _S278.differential_0;

#line 958
    ControlPoint_0 _S282 = ControlPoint_x24_syn_dzero_0();

#line 958
    (&_S282)->dirac_0 = _S241.drgb_0;

#line 958
    (&_S282)->t_1 = _S281;

#line 958
    dpctrl_pt_0->primal_0 = (*dpctrl_pt_0).primal_0;

#line 958
    dpctrl_pt_0->differential_0 = _S282;

#line 958
    SplineState_0 _S283 = SplineState_x24_syn_dzero_0();

#line 958
    (&_S283)->cum_sum_0 = _S247;

#line 958
    (&_S283)->distortion_parts_0 = _S249;

#line 958
    (&_S283)->C_0 = _S241.C_0;

#line 958
    (&_S283)->logT_0 = _S269;

#line 958
    (&_S283)->drgb_0 = _S277;

#line 958
    (&_S283)->t_0 = _S280;

#line 958
    dpstate_0->primal_0 = (*dpstate_0).primal_0;

#line 958
    dpstate_0->differential_0 = _S283;

#line 104 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/spline-machine.slang"
    return;
}


#line 104
__device__ void s_bwd_update_0(DiffPair_SplineState_0 * _S284, DiffPair_ControlPoint_0 * _S285, float _S286, float _S287, float _S288, SplineState_0 _S289)
{



    s_bwd_prop_update_0(_S284, _S285, _S286, _S287, _S288, _S289);

#line 109
    return;
}


#line 168 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ float3  s_primal_ctx_cross_0(float3  _S290, float3  _S291)
{

#line 168
    return cross_0(_S290, _S291);
}


#line 168
__device__ float3  s_primal_ctx_rotate_vector_0(float3  dpv_4, float4  dpq_0)
{

#line 242 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
    float3  _S292 = - float3 {dpq_0.y, dpq_0.z, dpq_0.w};

#line 242
    float3  _S293 = make_float3 (2.0f) * s_primal_ctx_cross_0(_S292, dpv_4);

#line 242
    return dpv_4 + make_float3 (dpq_0.x) * _S293 + s_primal_ctx_cross_0(_S292, _S293);
}


#line 242
__device__ float s_primal_ctx_dot_0(float3  _S294, float3  _S295)
{

#line 242
    return dot_0(_S294, _S295);
}


#line 242
__device__ float s_primal_ctx_sqrt_0(float _S296)
{

#line 242
    return (F32_sqrt((_S296)));
}


#line 242
__device__ float3  s_primal_ctx_l2_normalize_0(float3  dpx_9)
{

#line 137
    return s_primal_ctx_safe_div_1(dpx_9, s_primal_ctx_sqrt_0(s_primal_ctx_max_0(s_primal_ctx_dot_0(dpx_9, dpx_9), 1.07549441632776457e-20f)));
}


#line 137
__device__ float3  s_primal_ctx_safe_div_2(float3  dpa_2, float3  dpb_2)
{

#line 113
    return make_float3 (s_primal_ctx_safe_div_0(dpa_2.x, dpb_2.x), s_primal_ctx_safe_div_0(dpa_2.y, dpb_2.y), s_primal_ctx_safe_div_0(dpa_2.z, dpb_2.z));
}


#line 113
__device__ float s_primal_ctx_safe_sqrt_0(float dpa_3)
{

#line 123
    float _S297;
    if(dpa_3 < 1.07549441632776457e-20f)
    {

#line 124
        _S297 = 0.0f;

#line 124
    }
    else
    {

#line 124
        _S297 = s_primal_ctx_sqrt_0(dpa_3);

#line 124
    }

#line 124
    return _S297;
}


#line 124
__device__ float2  s_primal_ctx_safe_eliIntersect_0(float3  dpro_0, float3  dprd_0, float3  dpra_0)
{

#line 133 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/tri-intersect.slang"
    float3  _S298 = s_primal_ctx_safe_div_2(dpro_0, dpra_0);

#line 133
    float3  _S299 = s_primal_ctx_safe_div_2(dprd_0, dpra_0);

#line 133
    float _S300 = s_primal_ctx_dot_0(_S299, _S299);

#line 141
    float bp_1 = - s_primal_ctx_dot_0(_S298, _S299);
    float3  l_1 = _S298 + make_float3 (s_primal_ctx_safe_div_0(bp_1, _S300)) * _S299;
    float h_1 = _S300 * (1.0f - s_primal_ctx_dot_0(l_1, l_1));
    float c_1 = s_primal_ctx_dot_0(_S298, _S298) - 1.0f;
    bool _S301 = h_1 < 0.0f;

#line 145
    float2  _S302;

#line 145
    if(_S301)
    {

#line 145
        _S302 = make_float2 (-1.0f);

#line 145
    }

#line 145
    bool _S303 = !_S301;

#line 145
    if(_S303)
    {

#line 146
        float _S304 = bp_1 + float((F32_sign((bp_1)))) * s_primal_ctx_safe_sqrt_0(h_1);

#line 146
        _S302 = make_float2 (s_primal_ctx_safe_div_0(c_1, _S304), s_primal_ctx_safe_div_0(_S304, _S300));

#line 146
    }

#line 146
    return _S302;
}


#line 146
__device__ void s_bwd_prop_sqrt_0(DiffPair_float_0 * _S305, float _S306)
{

#line 146
    _d_sqrt_0(_S305, _S306);

#line 146
    return;
}


#line 123 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
__device__ void s_bwd_prop_safe_sqrt_0(DiffPair_float_0 * dpa_4, float _s_dOut_5)
{

#line 123
    DiffPair_float_0 _S307 = *dpa_4;

#line 123
    float _S308;

#line 123
    if((*dpa_4).primal_0 < 1.07549441632776457e-20f)
    {

#line 123
        _S308 = 0.0f;

#line 123
    }
    else
    {

        DiffPair_float_0 _S309;

#line 127
        (&_S309)->primal_0 = _S307.primal_0;

#line 127
        (&_S309)->differential_0 = 0.0f;

#line 127
        s_bwd_prop_sqrt_0(&_S309, _s_dOut_5);

#line 127
        _S308 = _S309.differential_0;

#line 127
    }

#line 127
    dpa_4->primal_0 = (*dpa_4).primal_0;

#line 127
    dpa_4->differential_0 = _S308;

#line 123
    return;
}


#line 123
__device__ void s_bwd_prop_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S310, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S311, float _S312)
{

#line 123
    _d_dot_0(_S310, _S311, _S312);

#line 123
    return;
}


#line 113
__device__ void s_bwd_prop_safe_div_2(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpa_5, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpb_3, float3  _s_dOut_6)
{
    float _S313 = (*dpa_5).primal_0.x;

#line 115
    float _S314 = (*dpb_3).primal_0.x;
    float _S315 = (*dpa_5).primal_0.y;

#line 116
    float _S316 = (*dpb_3).primal_0.y;
    float _S317 = (*dpb_3).primal_0.z;

#line 117
    DiffPair_float_0 _S318;

#line 117
    (&_S318)->primal_0 = (*dpa_5).primal_0.z;

#line 117
    (&_S318)->differential_0 = 0.0f;

#line 117
    DiffPair_float_0 _S319;

#line 117
    (&_S319)->primal_0 = _S317;

#line 117
    (&_S319)->differential_0 = 0.0f;

#line 117
    s_bwd_prop_safe_div_0(&_S318, &_S319, _s_dOut_6.z);

#line 116
    DiffPair_float_0 _S320;

#line 116
    (&_S320)->primal_0 = _S315;

#line 116
    (&_S320)->differential_0 = 0.0f;

#line 116
    DiffPair_float_0 _S321;

#line 116
    (&_S321)->primal_0 = _S316;

#line 116
    (&_S321)->differential_0 = 0.0f;

#line 116
    s_bwd_prop_safe_div_0(&_S320, &_S321, _s_dOut_6.y);

#line 115
    DiffPair_float_0 _S322;

#line 115
    (&_S322)->primal_0 = _S313;

#line 115
    (&_S322)->differential_0 = 0.0f;

#line 115
    DiffPair_float_0 _S323;

#line 115
    (&_S323)->primal_0 = _S314;

#line 115
    (&_S323)->differential_0 = 0.0f;

#line 115
    s_bwd_prop_safe_div_0(&_S322, &_S323, _s_dOut_6.x);

#line 115
    float3  _S324 = make_float3 (_S323.differential_0, _S321.differential_0, _S319.differential_0);

#line 115
    dpb_3->primal_0 = (*dpb_3).primal_0;

#line 115
    dpb_3->differential_0 = _S324;

#line 115
    float3  _S325 = make_float3 (_S322.differential_0, _S320.differential_0, _S318.differential_0);

#line 115
    dpa_5->primal_0 = (*dpa_5).primal_0;

#line 115
    dpa_5->differential_0 = _S325;

#line 113
    return;
}


#line 133 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/tri-intersect.slang"
__device__ void s_bwd_prop_safe_eliIntersect_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpro_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dprd_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpra_1, float2  _s_dOut_7)
{

#line 133
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S326 = *dpro_1;

#line 133
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S327 = *dprd_1;

#line 133
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S328 = *dpra_1;

#line 133
    float3  _S329 = s_primal_ctx_safe_div_2((*dpro_1).primal_0, (*dpra_1).primal_0);

#line 133
    float3  _S330 = s_primal_ctx_safe_div_2((*dprd_1).primal_0, (*dpra_1).primal_0);

#line 133
    float _S331 = s_primal_ctx_dot_0(_S330, _S330);

#line 141
    float bp_2 = - s_primal_ctx_dot_0(_S329, _S330);

#line 141
    float _S332 = s_primal_ctx_safe_div_0(bp_2, _S331);
    float3  _S333 = make_float3 (_S332);

#line 142
    float3  l_2 = _S329 + make_float3 (_S332) * _S330;
    float _S334 = 1.0f - s_primal_ctx_dot_0(l_2, l_2);

#line 143
    float h_2 = _S331 * _S334;
    float c_2 = s_primal_ctx_dot_0(_S329, _S329) - 1.0f;

#line 144
    bool _S335 = !(h_2 < 0.0f);

#line 144
    float _S336;

#line 144
    float _S337;

#line 144
    if(_S335)
    {
        float _S338 = float((F32_sign((bp_2))));

#line 146
        _S336 = bp_2 + _S338 * s_primal_ctx_safe_sqrt_0(h_2);

#line 146
        _S337 = _S338;

#line 146
    }
    else
    {

#line 146
        _S336 = 0.0f;

#line 146
        _S337 = 0.0f;

#line 146
    }

#line 146
    float _S339;

#line 146
    float _S340;

#line 146
    if(_S335)
    {

#line 147
        DiffPair_float_0 _S341;

#line 147
        (&_S341)->primal_0 = _S336;

#line 147
        (&_S341)->differential_0 = 0.0f;

#line 147
        DiffPair_float_0 _S342;

#line 147
        (&_S342)->primal_0 = _S331;

#line 147
        (&_S342)->differential_0 = 0.0f;

#line 147
        s_bwd_prop_safe_div_0(&_S341, &_S342, _s_dOut_7.y);

#line 147
        DiffPair_float_0 _S343;

#line 147
        (&_S343)->primal_0 = c_2;

#line 147
        (&_S343)->differential_0 = 0.0f;

#line 147
        DiffPair_float_0 _S344;

#line 147
        (&_S344)->primal_0 = _S336;

#line 147
        (&_S344)->differential_0 = 0.0f;

#line 147
        s_bwd_prop_safe_div_0(&_S343, &_S344, _s_dOut_7.x);

#line 146
        float _S345 = _S341.differential_0 + _S344.differential_0;

#line 146
        float _S346 = _S337 * _S345;

#line 146
        DiffPair_float_0 _S347;

#line 146
        (&_S347)->primal_0 = h_2;

#line 146
        (&_S347)->differential_0 = 0.0f;

#line 146
        s_bwd_prop_safe_sqrt_0(&_S347, _S346);

#line 146
        _S336 = _S343.differential_0;

#line 146
        _S337 = _S347.differential_0;

#line 146
        _S339 = _S345;

#line 146
        _S340 = _S342.differential_0;

#line 146
    }
    else
    {

#line 146
        _S336 = 0.0f;

#line 146
        _S337 = 0.0f;

#line 146
        _S339 = 0.0f;

#line 146
        _S340 = 0.0f;

#line 146
    }

#line 144
    float3  _S348 = make_float3 (0.0f);

#line 144
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S349;

#line 144
    (&_S349)->primal_0 = _S329;

#line 144
    (&_S349)->differential_0 = _S348;

#line 144
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S350;

#line 144
    (&_S350)->primal_0 = _S329;

#line 144
    (&_S350)->differential_0 = _S348;

#line 144
    s_bwd_prop_dot_0(&_S349, &_S350, _S336);

#line 143
    float _S351 = _S334 * _S337;

#line 143
    float _S352 = - (_S331 * _S337);

#line 143
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S353;

#line 143
    (&_S353)->primal_0 = l_2;

#line 143
    (&_S353)->differential_0 = _S348;

#line 143
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S354;

#line 143
    (&_S354)->primal_0 = l_2;

#line 143
    (&_S354)->differential_0 = _S348;

#line 143
    s_bwd_prop_dot_0(&_S353, &_S354, _S352);

#line 142
    float3  _S355 = _S354.differential_0 + _S353.differential_0;

#line 142
    float3  _S356 = _S333 * _S355;

#line 142
    float3  _S357 = _S330 * _S355;

#line 142
    float _S358 = _S357.x + _S357.y + _S357.z;

#line 142
    DiffPair_float_0 _S359;

#line 142
    (&_S359)->primal_0 = bp_2;

#line 142
    (&_S359)->differential_0 = 0.0f;

#line 142
    DiffPair_float_0 _S360;

#line 142
    (&_S360)->primal_0 = _S331;

#line 142
    (&_S360)->differential_0 = 0.0f;

#line 142
    s_bwd_prop_safe_div_0(&_S359, &_S360, _S358);

#line 141
    float _S361 = - (_S359.differential_0 + _S339);

#line 141
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S362;

#line 141
    (&_S362)->primal_0 = _S329;

#line 141
    (&_S362)->differential_0 = _S348;

#line 141
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S363;

#line 141
    (&_S363)->primal_0 = _S330;

#line 141
    (&_S363)->differential_0 = _S348;

#line 141
    s_bwd_prop_dot_0(&_S362, &_S363, _S361);

#line 140
    float _S364 = _S351 + _S360.differential_0 + _S340;

#line 140
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S365;

#line 140
    (&_S365)->primal_0 = _S330;

#line 140
    (&_S365)->differential_0 = _S348;

#line 140
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S366;

#line 140
    (&_S366)->primal_0 = _S330;

#line 140
    (&_S366)->differential_0 = _S348;

#line 140
    s_bwd_prop_dot_0(&_S365, &_S366, _S364);

#line 139
    float3  _S367 = _S356 + _S363.differential_0 + _S366.differential_0 + _S365.differential_0;

#line 139
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S368;

#line 139
    (&_S368)->primal_0 = _S327.primal_0;

#line 139
    (&_S368)->differential_0 = _S348;

#line 139
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S369;

#line 139
    (&_S369)->primal_0 = _S328.primal_0;

#line 139
    (&_S369)->differential_0 = _S348;

#line 139
    s_bwd_prop_safe_div_2(&_S368, &_S369, _S367);

#line 138
    float3  _S370 = _S350.differential_0 + _S349.differential_0 + _S355 + _S362.differential_0;

#line 138
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S371;

#line 138
    (&_S371)->primal_0 = _S326.primal_0;

#line 138
    (&_S371)->differential_0 = _S348;

#line 138
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S372;

#line 138
    (&_S372)->primal_0 = _S328.primal_0;

#line 138
    (&_S372)->differential_0 = _S348;

#line 138
    s_bwd_prop_safe_div_2(&_S371, &_S372, _S370);

#line 138
    float3  _S373 = _S369.differential_0 + _S372.differential_0;

#line 138
    dpra_1->primal_0 = (*dpra_1).primal_0;

#line 138
    dpra_1->differential_0 = _S373;

#line 138
    dprd_1->primal_0 = (*dprd_1).primal_0;

#line 138
    dprd_1->differential_0 = _S368.differential_0;

#line 138
    dpro_1->primal_0 = (*dpro_1).primal_0;

#line 138
    dpro_1->differential_0 = _S371.differential_0;

#line 133
    return;
}


#line 133
__device__ void s_bwd_prop_cross_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S374, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S375, float3  _S376)
{

#line 133
    _d_cross_0(_S374, _S375, _S376);

#line 133
    return;
}


#line 238 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
__device__ void s_bwd_prop_rotate_vector_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpv_5, DiffPair_vectorx3Cfloatx2C4x3E_0 * dpq_1, float3  _s_dOut_8)
{


    float3  _S377 = - float3 {(*dpq_1).primal_0.y, (*dpq_1).primal_0.z, (*dpq_1).primal_0.w};

#line 242
    float3  _S378 = make_float3 (2.0f) * s_primal_ctx_cross_0(_S377, (*dpv_5).primal_0);
    float3  _S379 = make_float3 ((*dpq_1).primal_0.x);

#line 243
    float3  _S380 = make_float3 (0.0f);

#line 243
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S381;

#line 243
    (&_S381)->primal_0 = _S377;

#line 243
    (&_S381)->differential_0 = _S380;

#line 243
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S382;

#line 243
    (&_S382)->primal_0 = _S378;

#line 243
    (&_S382)->differential_0 = _S380;

#line 243
    s_bwd_prop_cross_0(&_S381, &_S382, _s_dOut_8);

#line 243
    float3  _S383 = _S378 * _s_dOut_8;

#line 243
    float _S384 = _S383.x + _S383.y + _S383.z;

#line 242
    float3  _S385 = make_float3 (2.0f) * (_S382.differential_0 + _S379 * _s_dOut_8);

#line 242
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S386;

#line 242
    (&_S386)->primal_0 = _S377;

#line 242
    (&_S386)->differential_0 = _S380;

#line 242
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S387;

#line 242
    (&_S387)->primal_0 = (*dpv_5).primal_0;

#line 242
    (&_S387)->differential_0 = _S380;

#line 242
    s_bwd_prop_cross_0(&_S386, &_S387, _S385);

#line 242
    float3  _S388 = - (_S381.differential_0 + _S386.differential_0);

#line 242
    float4  _S389 = make_float4 (_S384, _S388.x, _S388.y, _S388.z);

#line 242
    dpq_1->primal_0 = (*dpq_1).primal_0;

#line 242
    dpq_1->differential_0 = _S389;

#line 242
    float3  _S390 = _s_dOut_8 + _S387.differential_0;

#line 242
    dpv_5->primal_0 = (*dpv_5).primal_0;

#line 242
    dpv_5->differential_0 = _S390;

#line 238
    return;
}


#line 137
__device__ void s_bwd_prop_l2_normalize_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_10, float3  _s_dOut_9)
{

#line 137
    float3  _S391 = (*dpx_10).primal_0;

#line 137
    float _S392 = s_primal_ctx_dot_0(_S391, _S391);

#line 137
    float _S393 = s_primal_ctx_max_0(_S392, 1.07549441632776457e-20f);

#line 137
    float _S394 = s_primal_ctx_sqrt_0(_S393);
    float3  _S395 = make_float3 (0.0f);

#line 138
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S396;

#line 138
    (&_S396)->primal_0 = (*dpx_10).primal_0;

#line 138
    (&_S396)->differential_0 = _S395;

#line 138
    DiffPair_float_0 _S397;

#line 138
    (&_S397)->primal_0 = _S394;

#line 138
    (&_S397)->differential_0 = 0.0f;

#line 138
    s_bwd_prop_safe_div_1(&_S396, &_S397, _s_dOut_9);

#line 138
    DiffPair_float_0 _S398;

#line 138
    (&_S398)->primal_0 = _S393;

#line 138
    (&_S398)->differential_0 = 0.0f;

#line 138
    s_bwd_prop_sqrt_0(&_S398, _S397.differential_0);

#line 138
    DiffPair_float_0 _S399;

#line 138
    (&_S399)->primal_0 = _S392;

#line 138
    (&_S399)->differential_0 = 0.0f;

#line 138
    DiffPair_float_0 _S400;

#line 138
    (&_S400)->primal_0 = 1.07549441632776457e-20f;

#line 138
    (&_S400)->differential_0 = 0.0f;

#line 138
    s_bwd_prop_max_0(&_S399, &_S400, _S398.differential_0);

#line 138
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S401;

#line 138
    (&_S401)->primal_0 = (*dpx_10).primal_0;

#line 138
    (&_S401)->differential_0 = _S395;

#line 138
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S402;

#line 138
    (&_S402)->primal_0 = (*dpx_10).primal_0;

#line 138
    (&_S402)->differential_0 = _S395;

#line 138
    s_bwd_prop_dot_0(&_S401, &_S402, _S399.differential_0);

#line 138
    float3  _S403 = _S396.differential_0 + _S402.differential_0 + _S401.differential_0;

#line 138
    dpx_10->primal_0 = (*dpx_10).primal_0;

#line 138
    dpx_10->differential_0 = _S403;

#line 137
    return;
}


#line 179 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/tri-intersect.slang"
__device__ void s_bwd_prop_safe_ray_intersect_ellipsoid_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dprayo_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * dprayd_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpscales_0, DiffPair_vectorx3Cfloatx2C4x3E_0 * dpquat_0, float2  _s_dOut_10)
{


    float3  _S404 = s_primal_ctx_rotate_vector_0((*dprayd_0).primal_0, (*dpquat_0).primal_0);

#line 183
    float3  _S405 = s_primal_ctx_l2_normalize_0(_S404);

#line 183
    float3  _S406 = s_primal_ctx_rotate_vector_0((*dprayo_0).primal_0, (*dpquat_0).primal_0);

#line 183
    float2  _S407 = s_primal_ctx_safe_eliIntersect_0(_S406, _S405, (*dpscales_0).primal_0);

#line 189
    float _S408 = _S407.x;

#line 189
    float _S409 = _S407.y;

#line 189
    DiffPair_float_0 _S410;

#line 189
    (&_S410)->primal_0 = _S408;

#line 189
    (&_S410)->differential_0 = 0.0f;

#line 189
    DiffPair_float_0 _S411;

#line 189
    (&_S411)->primal_0 = _S409;

#line 189
    (&_S411)->differential_0 = 0.0f;

#line 189
    s_bwd_prop_max_0(&_S410, &_S411, _s_dOut_10.y);

#line 189
    DiffPair_float_0 _S412;

#line 189
    (&_S412)->primal_0 = _S408;

#line 189
    (&_S412)->differential_0 = 0.0f;

#line 189
    DiffPair_float_0 _S413;

#line 189
    (&_S413)->primal_0 = _S409;

#line 189
    (&_S413)->differential_0 = 0.0f;

#line 189
    s_bwd_prop_min_0(&_S412, &_S413, _s_dOut_10.x);

#line 188
    float2  _S414 = make_float2 (_S410.differential_0 + _S412.differential_0, _S411.differential_0 + _S413.differential_0);

#line 188
    float3  _S415 = make_float3 (0.0f);

#line 188
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S416;

#line 188
    (&_S416)->primal_0 = _S406;

#line 188
    (&_S416)->differential_0 = _S415;

#line 188
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S417;

#line 188
    (&_S417)->primal_0 = _S405;

#line 188
    (&_S417)->differential_0 = _S415;

#line 188
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S418;

#line 188
    (&_S418)->primal_0 = (*dpscales_0).primal_0;

#line 188
    (&_S418)->differential_0 = _S415;

#line 188
    s_bwd_prop_safe_eliIntersect_0(&_S416, &_S417, &_S418, _S414);

#line 186
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S419;

#line 186
    (&_S419)->primal_0 = (*dprayo_0).primal_0;

#line 186
    (&_S419)->differential_0 = _S415;

#line 186
    float4  _S420 = make_float4 (0.0f);

#line 186
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S421;

#line 186
    (&_S421)->primal_0 = (*dpquat_0).primal_0;

#line 186
    (&_S421)->differential_0 = _S420;

#line 186
    s_bwd_prop_rotate_vector_0(&_S419, &_S421, _S416.differential_0);

#line 185
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S422;

#line 185
    (&_S422)->primal_0 = _S404;

#line 185
    (&_S422)->differential_0 = _S415;

#line 185
    s_bwd_prop_l2_normalize_0(&_S422, _S417.differential_0);

#line 185
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S423;

#line 185
    (&_S423)->primal_0 = (*dprayd_0).primal_0;

#line 185
    (&_S423)->differential_0 = _S415;

#line 185
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S424;

#line 185
    (&_S424)->primal_0 = (*dpquat_0).primal_0;

#line 185
    (&_S424)->differential_0 = _S420;

#line 185
    s_bwd_prop_rotate_vector_0(&_S423, &_S424, _S422.differential_0);

#line 185
    float4  _S425 = _S421.differential_0 + _S424.differential_0;

#line 185
    dpquat_0->primal_0 = (*dpquat_0).primal_0;

#line 185
    dpquat_0->differential_0 = _S425;

#line 185
    dpscales_0->primal_0 = (*dpscales_0).primal_0;

#line 185
    dpscales_0->differential_0 = _S418.differential_0;

#line 185
    dprayd_0->primal_0 = (*dprayd_0).primal_0;

#line 185
    dprayd_0->differential_0 = _S423.differential_0;

#line 185
    dprayo_0->primal_0 = (*dprayo_0).primal_0;

#line 185
    dprayo_0->differential_0 = _S419.differential_0;

#line 179
    return;
}


#line 290
__device__ void s_bwd_prop_safe_intersect_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dprayo_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dprayd_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpscales_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpmean_0, DiffPair_vectorx3Cfloatx2C4x3E_0 * dpquat_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpcolor_0, DiffPair_float_0 * dpdensity_0, uint face_id_1, bool skip_close_2, ControlPoint_0 s_diff_out_T_0)
{

#line 291
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S426 = *dprayd_1;

#line 291
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S427 = *dpscales_1;

#line 291
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S428 = *dpquat_1;

#line 291
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S429 = *dpcolor_0;

#line 291
    DiffPair_float_0 _S430 = *dpdensity_0;

    float3  _S431 = (*dprayo_1).primal_0 - (*dpmean_0).primal_0;

    bool _S432 = face_id_1 == 1U;

#line 295
    float dirac_multi_1;
    if(_S432)
    {

#line 296
        dirac_multi_1 = _S430.primal_0;

#line 296
    }
    else
    {

#line 296
        dirac_multi_1 = - _S430.primal_0;

#line 296
    }


    float _S433 = _S429.primal_0.x;

#line 299
    float _S434 = _S429.primal_0.y;

#line 299
    float _S435 = _S429.primal_0.z;

#line 296
    float _S436 = s_diff_out_T_0.dirac_0.x + _S435 * s_diff_out_T_0.dirac_0.w + _S434 * s_diff_out_T_0.dirac_0.z + _S433 * s_diff_out_T_0.dirac_0.y;

#line 296
    float3  _S437 = make_float3 (dirac_multi_1 * s_diff_out_T_0.dirac_0.y, dirac_multi_1 * s_diff_out_T_0.dirac_0.z, dirac_multi_1 * s_diff_out_T_0.dirac_0.w);

#line 296
    if(_S432)
    {

#line 296
        dirac_multi_1 = _S436;

#line 296
    }
    else
    {

#line 296
        dirac_multi_1 = - _S436;

#line 296
    }

#line 296
    float2  _S438;

#line 296
    if(_S432)
    {

#line 296
        _S438 = make_float2 (s_diff_out_T_0.t_1, 0.0f);

#line 296
    }
    else
    {

#line 296
        _S438 = make_float2 (0.0f, s_diff_out_T_0.t_1);

#line 296
    }

#line 293
    float3  _S439 = make_float3 (0.0f);

#line 293
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S440;

#line 293
    (&_S440)->primal_0 = _S431;

#line 293
    (&_S440)->differential_0 = _S439;

#line 293
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S441;

#line 293
    (&_S441)->primal_0 = _S426.primal_0;

#line 293
    (&_S441)->differential_0 = _S439;

#line 293
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S442;

#line 293
    (&_S442)->primal_0 = _S427.primal_0;

#line 293
    (&_S442)->differential_0 = _S439;

#line 293
    float4  _S443 = make_float4 (0.0f);

#line 293
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S444;

#line 293
    (&_S444)->primal_0 = _S428.primal_0;

#line 293
    (&_S444)->differential_0 = _S443;

#line 293
    s_bwd_prop_safe_ray_intersect_ellipsoid_0(&_S440, &_S441, &_S442, &_S444, _S438);

#line 293
    float3  _S445 = - _S440.differential_0;

#line 293
    dpdensity_0->primal_0 = (*dpdensity_0).primal_0;

#line 293
    dpdensity_0->differential_0 = dirac_multi_1;

#line 293
    dpcolor_0->primal_0 = (*dpcolor_0).primal_0;

#line 293
    dpcolor_0->differential_0 = _S437;

#line 293
    dpquat_1->primal_0 = (*dpquat_1).primal_0;

#line 293
    dpquat_1->differential_0 = _S444.differential_0;

#line 293
    dpmean_0->primal_0 = (*dpmean_0).primal_0;

#line 293
    dpmean_0->differential_0 = _S445;

#line 293
    dpscales_1->primal_0 = (*dpscales_1).primal_0;

#line 293
    dpscales_1->differential_0 = _S442.differential_0;

#line 293
    dprayd_1->primal_0 = (*dprayd_1).primal_0;

#line 293
    dprayd_1->differential_0 = _S441.differential_0;

#line 293
    dprayo_1->primal_0 = (*dprayo_1).primal_0;

#line 293
    dprayo_1->differential_0 = _S440.differential_0;

#line 290
    return;
}


#line 290
__device__ void s_bwd_safe_intersect_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S446, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S447, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S448, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S449, DiffPair_vectorx3Cfloatx2C4x3E_0 * _S450, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S451, DiffPair_float_0 * _S452, uint _S453, bool _S454, ControlPoint_0 _S455)
{

#line 291
    s_bwd_prop_safe_intersect_0(_S446, _S447, _S448, _S449, _S450, _S451, _S452, _S453, _S454, _S455);

#line 291
    return;
}


#line 291
struct DiffPair_Features_0
{
    Features_0 primal_0;
    Features_0 differential_0;
};


#line 75 "/home/amai/ever_training/ever/splinetracers/slang/sh.slang"
__device__ void s_bwd_prop_eval_sh_col3_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdir_0, DiffPair_Features_0 * dpfeat_0, float3  _s_dOut_11)
{

#line 76
    float x_6 = (*dpdir_0).primal_0.x;
    float y_4 = (*dpdir_0).primal_0.y;
    float z_2 = (*dpdir_0).primal_0.z;
    float xx_2 = x_6 * x_6;

#line 79
    float yy_2 = y_4 * y_4;

#line 79
    float zz_1 = z_2 * z_2;

    float _S456 = -0.59004360437393188f * y_4;

#line 81
    float _S457 = 3.0f * xx_2;

#line 81
    float _S458 = _S457 - yy_2;
    float _S459 = 2.89061141014099121f * (x_6 * y_4);
    float _S460 = -0.4570457935333252f * y_4;

#line 83
    float _S461 = 4.0f * zz_1 - xx_2 - yy_2;
    float _S462 = 0.37317633628845215f * z_2;

#line 84
    float _S463 = 3.0f * yy_2;

#line 84
    float _S464 = 2.0f * zz_1 - _S457 - _S463;
    float _S465 = -0.4570457935333252f * x_6;
    float _S466 = 1.44530570507049561f * z_2;

#line 86
    float _S467 = xx_2 - yy_2;
    float _S468 = -0.59004360437393188f * x_6;

#line 87
    float _S469 = xx_2 - _S463;

#line 87
    float3  _S470 = make_float3 (_S468 * _S469) * _s_dOut_11;

#line 87
    float3  _S471 = (*dpfeat_0).primal_0.f15_0 * _s_dOut_11;

#line 87
    float _S472 = _S471.x + _S471.y + _S471.z;

#line 87
    float _S473 = _S468 * _S472;

#line 86
    float3  _S474 = make_float3 (_S466 * _S467) * _s_dOut_11;

#line 86
    float3  _S475 = (*dpfeat_0).primal_0.f14_0 * _s_dOut_11;

#line 86
    float _S476 = _S475.x + _S475.y + _S475.z;

#line 86
    float _S477 = _S466 * _S476;

#line 85
    float3  _S478 = make_float3 (_S465 * _S461) * _s_dOut_11;

#line 85
    float3  _S479 = (*dpfeat_0).primal_0.f13_0 * _s_dOut_11;

#line 85
    float _S480 = _S479.x + _S479.y + _S479.z;

#line 84
    float3  _S481 = make_float3 (_S462 * _S464) * _s_dOut_11;

#line 84
    float3  _S482 = (*dpfeat_0).primal_0.f12_0 * _s_dOut_11;

#line 84
    float _S483 = _S482.x + _S482.y + _S482.z;

#line 84
    float _S484 = _S462 * _S483;

#line 84
    float _S485 = - _S484;

#line 83
    float3  _S486 = make_float3 (_S460 * _S461) * _s_dOut_11;

#line 83
    float3  _S487 = (*dpfeat_0).primal_0.f11_0 * _s_dOut_11;

#line 83
    float _S488 = _S487.x + _S487.y + _S487.z;

#line 83
    float _S489 = _S465 * _S480 + _S460 * _S488;

#line 83
    float _S490 = - _S489;

#line 82
    float3  _S491 = make_float3 (_S459 * z_2) * _s_dOut_11;

#line 82
    float3  _S492 = (*dpfeat_0).primal_0.f10_0 * _s_dOut_11;

#line 82
    float _S493 = _S492.x + _S492.y + _S492.z;

#line 82
    float s_diff_xy_T_0 = 2.89061141014099121f * (z_2 * _S493);

#line 81
    float3  _S494 = make_float3 (_S456 * _S458) * _s_dOut_11;

#line 81
    float3  _S495 = (*dpfeat_0).primal_0.f9_0 * _s_dOut_11;

#line 81
    float _S496 = _S495.x + _S495.y + _S495.z;

#line 81
    float _S497 = _S456 * _S496;

#line 79
    float _S498 = z_2 * (2.0f * _S484 + 4.0f * _S489);

#line 79
    float _S499 = y_4 * (- _S477 + 3.0f * (- _S473 + _S485) + _S490 + - _S497);

#line 79
    float _S500 = x_6 * (_S473 + _S477 + _S490 + 3.0f * (_S485 + _S497));

#line 78
    float _S501 = 1.44530570507049561f * (_S467 * _S476) + 0.37317633628845215f * (_S464 * _S483) + _S459 * _S493 + _S498 + _S498;

#line 77
    float _S502 = -0.4570457935333252f * (_S461 * _S488) + -0.59004360437393188f * (_S458 * _S496) + x_6 * s_diff_xy_T_0 + _S499 + _S499;

#line 76
    float _S503 = -0.59004360437393188f * (_S469 * _S472) + -0.4570457935333252f * (_S461 * _S480) + y_4 * s_diff_xy_T_0 + _S500 + _S500;

#line 76
    Features_0 _S504 = Features_x24_syn_dzero_0();

#line 76
    (&_S504)->f15_0 = _S470;

#line 76
    (&_S504)->f14_0 = _S474;

#line 76
    (&_S504)->f13_0 = _S478;

#line 76
    (&_S504)->f12_0 = _S481;

#line 76
    (&_S504)->f11_0 = _S486;

#line 76
    (&_S504)->f10_0 = _S491;

#line 76
    (&_S504)->f9_0 = _S494;

#line 76
    dpfeat_0->primal_0 = (*dpfeat_0).primal_0;

#line 76
    dpfeat_0->differential_0 = _S504;

#line 76
    float3  _S505 = make_float3 (_S503, _S502, _S501);

#line 76
    dpdir_0->primal_0 = (*dpdir_0).primal_0;

#line 76
    dpdir_0->differential_0 = _S505;

#line 75
    return;
}


#line 63
__device__ void s_bwd_prop_eval_sh_col2_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdir_1, DiffPair_Features_0 * dpfeat_1, float3  _s_dOut_12)
{

#line 64
    float x_7 = (*dpdir_1).primal_0.x;
    float y_5 = (*dpdir_1).primal_0.y;
    float z_3 = (*dpdir_1).primal_0.z;
    float xx_3 = x_7 * x_7;

#line 67
    float yy_3 = y_5 * y_5;



    float3  _S506 = make_float3 (0.54627424478530884f * (xx_3 - yy_3)) * _s_dOut_12;

#line 71
    float3  _S507 = (*dpfeat_1).primal_0.f8_0 * _s_dOut_12;

#line 71
    float _S508 = 0.54627424478530884f * (_S507.x + _S507.y + _S507.z);

#line 71
    float3  _S509 = make_float3 (-1.09254848957061768f * (x_7 * z_3)) * _s_dOut_12;

#line 71
    float3  _S510 = (*dpfeat_1).primal_0.f7_0 * _s_dOut_12;

#line 71
    float s_diff_xz_T_0 = -1.09254848957061768f * (_S510.x + _S510.y + _S510.z);

#line 70
    float3  _S511 = make_float3 (0.31539157032966614f * (2.0f * (z_3 * z_3) - xx_3 - yy_3)) * _s_dOut_12;

#line 70
    float3  _S512 = (*dpfeat_1).primal_0.f6_0 * _s_dOut_12;

#line 70
    float _S513 = 0.31539157032966614f * (_S512.x + _S512.y + _S512.z);

#line 70
    float _S514 = - _S513;

#line 69
    float3  _S515 = make_float3 (-1.09254848957061768f * (y_5 * z_3)) * _s_dOut_12;

#line 69
    float3  _S516 = (*dpfeat_1).primal_0.f5_0 * _s_dOut_12;

#line 69
    float s_diff_yz_T_0 = -1.09254848957061768f * (_S516.x + _S516.y + _S516.z);

#line 69
    float3  _S517 = make_float3 (1.09254848957061768f * (x_7 * y_5)) * _s_dOut_12;

#line 69
    float3  _S518 = (*dpfeat_1).primal_0.f4_0 * _s_dOut_12;

#line 69
    float s_diff_xy_T_1 = 1.09254848957061768f * (_S518.x + _S518.y + _S518.z);

#line 67
    float _S519 = z_3 * (2.0f * _S513);

#line 67
    float _S520 = y_5 * (- _S508 + _S514);

#line 67
    float _S521 = x_7 * (_S508 + _S514);

#line 66
    float _S522 = x_7 * s_diff_xz_T_0 + y_5 * s_diff_yz_T_0 + _S519 + _S519;

#line 65
    float _S523 = z_3 * s_diff_yz_T_0 + x_7 * s_diff_xy_T_1 + _S520 + _S520;

#line 64
    float _S524 = z_3 * s_diff_xz_T_0 + y_5 * s_diff_xy_T_1 + _S521 + _S521;

#line 64
    Features_0 _S525 = Features_x24_syn_dzero_0();

#line 64
    (&_S525)->f8_0 = _S506;

#line 64
    (&_S525)->f7_0 = _S509;

#line 64
    (&_S525)->f6_0 = _S511;

#line 64
    (&_S525)->f5_0 = _S515;

#line 64
    (&_S525)->f4_0 = _S517;

#line 64
    dpfeat_1->primal_0 = (*dpfeat_1).primal_0;

#line 64
    dpfeat_1->differential_0 = _S525;

#line 64
    float3  _S526 = make_float3 (_S524, _S523, _S522);

#line 64
    dpdir_1->primal_0 = (*dpdir_1).primal_0;

#line 64
    dpdir_1->differential_0 = _S526;

#line 63
    return;
}


#line 54
__device__ void s_bwd_prop_eval_sh_col1_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdir_2, DiffPair_Features_0 * dpfeat_2, float3  s_diff_color_T_0)
{


    float3  _S527 = - s_diff_color_T_0;

#line 58
    float3  _S528 = make_float3 (0.48860251903533936f * (*dpdir_2).primal_0.x) * _S527;

#line 58
    float3  _S529 = (*dpfeat_2).primal_0.f3_0 * _S527;

#line 58
    float s_diff_x_T_0 = 0.48860251903533936f * (_S529.x + _S529.y + _S529.z);

#line 58
    float3  _S530 = make_float3 (0.48860251903533936f * (*dpdir_2).primal_0.z) * s_diff_color_T_0;

#line 58
    float3  _S531 = (*dpfeat_2).primal_0.f2_0 * s_diff_color_T_0;

#line 58
    float s_diff_z_T_0 = 0.48860251903533936f * (_S531.x + _S531.y + _S531.z);

#line 58
    float3  _S532 = make_float3 (-0.48860251903533936f * (*dpdir_2).primal_0.y) * s_diff_color_T_0;

#line 58
    float3  _S533 = (*dpfeat_2).primal_0.f1_0 * s_diff_color_T_0;

#line 58
    float s_diff_y_T_0 = -0.48860251903533936f * (_S533.x + _S533.y + _S533.z);

#line 58
    Features_0 _S534 = Features_x24_syn_dzero_0();

#line 58
    (&_S534)->f3_0 = _S528;

#line 58
    (&_S534)->f2_0 = _S530;

#line 58
    (&_S534)->f1_0 = _S532;

#line 58
    dpfeat_2->primal_0 = (*dpfeat_2).primal_0;

#line 58
    dpfeat_2->differential_0 = _S534;

#line 58
    float3  _S535 = make_float3 (s_diff_x_T_0, s_diff_y_T_0, s_diff_z_T_0);

#line 58
    dpdir_2->primal_0 = (*dpdir_2).primal_0;

#line 58
    dpdir_2->differential_0 = _S535;

#line 54
    return;
}


#line 49
__device__ void s_bwd_prop_eval_sh_col0_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdir_3, DiffPair_Features_0 * dpfeat_3, float3  _s_dOut_13)
{

#line 50
    float3  _S536 = make_float3 (0.282094806432724f) * _s_dOut_13;

#line 50
    Features_0 _S537 = Features_x24_syn_dzero_0();

#line 50
    (&_S537)->f0_0 = _S536;

#line 50
    dpfeat_3->primal_0 = (*dpfeat_3).primal_0;

#line 50
    dpfeat_3->differential_0 = _S537;

#line 1751 "core.meta.slang"
    float3  _S538 = make_float3 (0.0f);

#line 1751
    dpdir_3->primal_0 = (*dpdir_3).primal_0;

#line 1751
    dpdir_3->differential_0 = _S538;

#line 49 "/home/amai/ever_training/ever/splinetracers/slang/sh.slang"
    return;
}


#line 91
__device__ void s_bwd_prop_eval_color_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdir_4, DiffPair_Features_0 * dpfeat_4, uint sh_degree_3, float3  _s_dOut_14)
{

#line 91
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S539 = *dpdir_4;

#line 91
    DiffPair_Features_0 _S540 = *dpfeat_4;

    bool _S541 = sh_degree_3 > 0U;

#line 93
    bool _S542;

#line 93
    bool _S543;

#line 93
    if(_S541)
    {
        bool _S544 = sh_degree_3 > 1U;

#line 95
        if(_S544)
        {

#line 95
            _S542 = sh_degree_3 > 2U;

#line 95
        }
        else
        {

#line 95
            _S542 = false;

#line 95
        }

        bool _S545 = _S542;

#line 97
        _S542 = _S544;

#line 97
        _S543 = _S545;

#line 97
    }
    else
    {

#line 97
        _S542 = false;

#line 97
        _S543 = false;

#line 97
    }

#line 1751 "core.meta.slang"
    float3  _S546 = make_float3 (0.0f);

#line 1751
    Features_0 _S547 = Features_x24_syn_dzero_0();

#line 1751
    Features_0 _S548;

#line 1751
    float3  _S549;

#line 1751
    if(_S541)
    {

#line 1751
        if(_S542)
        {

#line 1751
            if(_S543)
            {

#line 98 "/home/amai/ever_training/ever/splinetracers/slang/sh.slang"
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S550;

#line 98
                (&_S550)->primal_0 = _S539.primal_0;

#line 98
                (&_S550)->differential_0 = _S546;

#line 98
                DiffPair_Features_0 _S551;

#line 98
                (&_S551)->primal_0 = _S540.primal_0;

#line 98
                (&_S551)->differential_0 = _S547;

#line 98
                s_bwd_prop_eval_sh_col3_0(&_S550, &_S551, _s_dOut_14);

#line 98
                _S548 = Features_x24_syn_dadd_0(_S551.differential_0, _S547);

#line 98
                _S549 = _S550.differential_0;

#line 98
            }
            else
            {

#line 98
                _S548 = _S547;

#line 98
                _S549 = _S546;

#line 98
            }

#line 96
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S552;

#line 96
            (&_S552)->primal_0 = _S539.primal_0;

#line 96
            (&_S552)->differential_0 = _S546;

#line 96
            DiffPair_Features_0 _S553;

#line 96
            (&_S553)->primal_0 = _S540.primal_0;

#line 96
            (&_S553)->differential_0 = _S547;

#line 96
            s_bwd_prop_eval_sh_col2_0(&_S552, &_S553, _s_dOut_14);

#line 96
            float3  _S554 = _S552.differential_0 + _S549;

#line 96
            _S548 = Features_x24_syn_dadd_0(_S553.differential_0, _S548);

#line 96
            _S549 = _S554;

#line 96
        }
        else
        {

#line 96
            _S548 = _S547;

#line 96
            _S549 = _S546;

#line 96
        }

#line 94
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S555;

#line 94
        (&_S555)->primal_0 = _S539.primal_0;

#line 94
        (&_S555)->differential_0 = _S546;

#line 94
        DiffPair_Features_0 _S556;

#line 94
        (&_S556)->primal_0 = _S540.primal_0;

#line 94
        (&_S556)->differential_0 = _S547;

#line 94
        s_bwd_prop_eval_sh_col1_0(&_S555, &_S556, _s_dOut_14);

#line 94
        float3  _S557 = _S555.differential_0 + _S549;

#line 94
        _S548 = Features_x24_syn_dadd_0(_S556.differential_0, _S548);

#line 94
        _S549 = _S557;

#line 94
    }
    else
    {

#line 94
        _S548 = _S547;

#line 94
        _S549 = _S546;

#line 94
    }

#line 92
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S558;

#line 92
    (&_S558)->primal_0 = _S539.primal_0;

#line 92
    (&_S558)->differential_0 = _S546;

#line 92
    DiffPair_Features_0 _S559;

#line 92
    (&_S559)->primal_0 = _S540.primal_0;

#line 92
    (&_S559)->differential_0 = _S547;

#line 92
    s_bwd_prop_eval_sh_col0_0(&_S558, &_S559, _s_dOut_14);

#line 92
    Features_0 _S560 = Features_x24_syn_dadd_0(_S559.differential_0, _S548);

#line 92
    dpfeat_4->primal_0 = (*dpfeat_4).primal_0;

#line 92
    dpfeat_4->differential_0 = _S560;

#line 92
    float3  _S561 = _S558.differential_0 + _S549;

#line 92
    dpdir_4->primal_0 = (*dpdir_4).primal_0;

#line 92
    dpdir_4->differential_0 = _S561;

#line 91
    return;
}


#line 91
__device__ void s_bwd_eval_color_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S562, DiffPair_Features_0 * _S563, uint _S564, float3  _S565)
{

#line 91
    s_bwd_prop_eval_color_0(_S562, _S563, _S564, _S565);

#line 91
    return;
}


#line 91
struct DiffPair_vectorx3Cfloatx2C2x3E_0
{
    float2  primal_0;
    float2  differential_0;
};


#line 219 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ void s_bwd_prop_mul_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * _S566, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * _S567, float4  _S568)
{

#line 219
    _d_mul_0(_S566, _S567, _S568);

#line 219
    return;
}


#line 253
__device__ void s_bwd_prop_inv_project_0(DiffPair_vectorx3Cfloatx2C2x3E_0 * dpxy_0, DiffPair_float_0 * dpdist_0, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * dpinv_wvt_0, float3  _s_dOut_15)
{

#line 254
    float2  _S569 = make_float2 ((*dpdist_0).primal_0);
    float4  _S570 = make_float4 (_s_dOut_15.x, _s_dOut_15.y, _s_dOut_15.z, 0.0f);

#line 255
    float4  _S571 = make_float4 (0.0f);

#line 255
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S572;

#line 255
    (&_S572)->primal_0 = make_float4 (((*dpxy_0).primal_0 * make_float2 ((*dpdist_0).primal_0)).x, ((*dpxy_0).primal_0 * make_float2 ((*dpdist_0).primal_0)).y, (*dpdist_0).primal_0, 1.0f);

#line 255
    (&_S572)->differential_0 = _S571;

#line 255
    Matrix<float, 4, 4>  _S573 = makeMatrix<float, 4, 4> (0.0f);

#line 255
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S574;

#line 255
    (&_S574)->primal_0 = (*dpinv_wvt_0).primal_0;

#line 255
    (&_S574)->differential_0 = _S573;

#line 255
    s_bwd_prop_mul_0(&_S572, &_S574, _S570);

#line 254
    float2  _S575 = float2 {_S572.differential_0.x, _S572.differential_0.y};

#line 254
    float2  _S576 = (*dpxy_0).primal_0 * _S575;

#line 254
    float2  _S577 = _S569 * _S575;

#line 254
    dpinv_wvt_0->primal_0 = (*dpinv_wvt_0).primal_0;

#line 254
    dpinv_wvt_0->differential_0 = _S574.differential_0;

#line 958 "core.meta.slang"
    float _S578 = _S572.differential_0.z + _S576.x + _S576.y;

#line 958
    dpdist_0->primal_0 = (*dpdist_0).primal_0;

#line 958
    dpdist_0->differential_0 = _S578;

#line 958
    dpxy_0->primal_0 = (*dpxy_0).primal_0;

#line 958
    dpxy_0->differential_0 = _S577;

#line 253 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
    return;
}


#line 253
__device__ void s_bwd_inv_project_0(DiffPair_vectorx3Cfloatx2C2x3E_0 * _S579, DiffPair_float_0 * _S580, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * _S581, float3  _S582)
{

#line 253
    s_bwd_prop_inv_project_0(_S579, _S580, _S581, _S582);

#line 253
    return;
}


#line 133
__device__ DiffPair_SplineState_0 run_update_0(SplineState_0 old_dual_state_0, ControlPoint_0 old_ctrl_pt_0, ControlPoint_0 ctrl_pt_2, uint prim_ind_1, uint face_id_2, uint ray_ind_0, DiffPair_SplineState_0 deriv_state_0, float3  origin_1, float3  direction_1, float tmin_0, float tmax_0, uint sh_degree_4, float max_prim_size_1, Matrix<float, 4, 4>  wct_1, Matrix<float, 4, 4>  inv_wct_0, DualModel_0 * model_1)
{

#line 150
    float2  _S583 = make_float2 (0.0f, 0.0f);

#line 150
    float3  _S584 = make_float3 (0.0f, 0.0f, 0.0f);

#line 150
    float4  _S585 = make_float4 (0.0f, 0.0f, 0.0f, 0.0f);

#line 150
    SplineState_0 _S586 = { _S583, _S583, _S584, 0.0f, _S585, 0.0f, _S584 };

#line 150
    DiffPair_SplineState_0 old_deriv_state_0;

#line 150
    (&old_deriv_state_0)->primal_0 = from_dual_0(old_dual_state_0, old_ctrl_pt_0);

#line 150
    (&old_deriv_state_0)->differential_0 = _S586;
    ControlPoint_0 _S587 = { 0.0f, _S585 };

#line 151
    DiffPair_ControlPoint_0 deriv_ctrl_pt_0;

#line 151
    (&deriv_ctrl_pt_0)->primal_0 = ctrl_pt_2;

#line 151
    (&deriv_ctrl_pt_0)->differential_0 = _S587;


    s_bwd_update_0(&old_deriv_state_0, &deriv_ctrl_pt_0, tmin_0, tmax_0, max_prim_size_1, deriv_state_0.differential_0);


    float3  _S588 = get_float3_0(model_1->means_0, prim_ind_1);
    float3  _S589 = get_float3_0(model_1->scales_2, prim_ind_1);
    float4  _S590 = get_float4_0(model_1->quats_0, prim_ind_1);
    float _S591 = ((model_1->densities_0).load<float>((prim_ind_1)));
    Features_0 feat_6 = get_feats_0(model_1->features_1, prim_ind_1, sh_degree_4);
    float3  color_5 = eval_color_0(direction_1, feat_6, sh_degree_4);

    DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_origin_0;

#line 164
    (&deriv_origin_0)->primal_0 = origin_1;

#line 164
    (&deriv_origin_0)->differential_0 = _S584;
    DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_direction_0;

#line 165
    (&deriv_direction_0)->primal_0 = direction_1;

#line 165
    (&deriv_direction_0)->differential_0 = _S584;
    DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_scales_0;

#line 166
    (&deriv_scales_0)->primal_0 = _S589;

#line 166
    (&deriv_scales_0)->differential_0 = _S584;
    DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_mean_0;

#line 167
    (&deriv_mean_0)->primal_0 = _S588;

#line 167
    (&deriv_mean_0)->differential_0 = _S584;
    DiffPair_vectorx3Cfloatx2C4x3E_0 deriv_quat_0;

#line 168
    (&deriv_quat_0)->primal_0 = _S590;

#line 168
    (&deriv_quat_0)->differential_0 = _S585;
    DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_color_0;

#line 169
    (&deriv_color_0)->primal_0 = color_5;

#line 169
    (&deriv_color_0)->differential_0 = _S584;
    DiffPair_float_0 deriv_density_0;

#line 170
    (&deriv_density_0)->primal_0 = _S591;

#line 170
    (&deriv_density_0)->differential_0 = 0.0f;

    s_bwd_safe_intersect_0(&deriv_origin_0, &deriv_direction_0, &deriv_scales_0, &deriv_mean_0, &deriv_quat_0, &deriv_color_0, &deriv_density_0, face_id_2, false, deriv_ctrl_pt_0.differential_0);


    atomic_add_float3_0(model_1->dL_dmeans_0, prim_ind_1, deriv_mean_0.differential_0);
    atomic_add_float3_0(model_1->dL_dscales_0, prim_ind_1, deriv_scales_0.differential_0);
    atomic_add_float4_0(model_1->dL_dquats_0, prim_ind_1, deriv_quat_0.differential_0);
    float temp_4;
    *((&temp_4)) = atomicAdd((model_1->dL_ddensities_0).data_ptr_at<float>((prim_ind_1)), (deriv_density_0.differential_0));

    DiffPair_vectorx3Cfloatx2C3x3E_0 _S592 = deriv_direction_0;

#line 181
    (&deriv_direction_0)->primal_0 = direction_1;

#line 181
    (&deriv_direction_0)->differential_0 = _S584;

    Features_0 _S593 = { _S584, _S584, _S584, _S584, _S584, _S584, _S584, _S584, _S584, _S584, _S584, _S584, _S584, _S584, _S584, _S584 };

#line 183
    DiffPair_Features_0 d_feat_0;

#line 183
    (&d_feat_0)->primal_0 = feat_6;

#line 183
    (&d_feat_0)->differential_0 = _S593;
    s_bwd_eval_color_0(&deriv_direction_0, &d_feat_0, sh_degree_4, deriv_color_0.differential_0);
    float3  d_rayd_0 = _S592.differential_0 + deriv_direction_0.differential_0;

    atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 0U, d_feat_0.differential_0.f0_0);
    if(sh_degree_4 > 0U)
    {

#line 189
        atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 1U, d_feat_0.differential_0.f1_0);
        atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 2U, d_feat_0.differential_0.f2_0);
        atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 3U, d_feat_0.differential_0.f3_0);
        if(sh_degree_4 > 1U)
        {

#line 193
            atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 4U, d_feat_0.differential_0.f4_0);
            atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 5U, d_feat_0.differential_0.f5_0);
            atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 6U, d_feat_0.differential_0.f6_0);
            atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 7U, d_feat_0.differential_0.f7_0);
            atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 8U, d_feat_0.differential_0.f8_0);
            if(sh_degree_4 > 2U)
            {

#line 199
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 9U, d_feat_0.differential_0.f9_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 10U, d_feat_0.differential_0.f10_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 11U, d_feat_0.differential_0.f11_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 12U, d_feat_0.differential_0.f12_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 13U, d_feat_0.differential_0.f13_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 14U, d_feat_0.differential_0.f14_0);
                atomic_add_float3_1(model_1->dL_dfeatures_0, prim_ind_1, 15U, d_feat_0.differential_0.f15_0);

#line 198
            }

#line 192
        }

#line 188
    }

#line 210
    atomic_add_float3_0(model_1->dL_drayos_0, ray_ind_0, deriv_origin_0.differential_0);
    atomic_add_float3_0(model_1->dL_drayds_0, ray_ind_0, d_rayd_0);

    float3  xyd_0 = project_0(_S588, wct_1);

    DiffPair_vectorx3Cfloatx2C2x3E_0 d_xy_0;

#line 215
    (&d_xy_0)->primal_0 = make_float2 (xyd_0.x, xyd_0.y);

#line 215
    (&d_xy_0)->differential_0 = _S583;
    DiffPair_float_0 d_dist_0;

#line 216
    (&d_dist_0)->primal_0 = xyd_0.z;

#line 216
    (&d_dist_0)->differential_0 = 0.0f;
    Matrix<float, 4, 4>  _S594 = makeMatrix<float, 4, 4> (_S585, _S585, _S585, _S585);

#line 217
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 d_inv_wct_0;

#line 217
    (&d_inv_wct_0)->primal_0 = inv_wct_0;

#line 217
    (&d_inv_wct_0)->differential_0 = _S594;

    s_bwd_inv_project_0(&d_xy_0, &d_dist_0, &d_inv_wct_0, deriv_mean_0.differential_0);
    atomic_add_float2_0(model_1->dL_dmeans2D_0, prim_ind_1, d_xy_0.differential_0);
    return old_deriv_state_0;
}


#line 140 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/spline-machine.slang"
struct SplineOutput_0
{
    float3  C_1;
    float opacity_0;
    float distortion_loss_0;
};

__device__ void s_bwd_prop_extract_color_0(DiffPair_SplineState_0 * dpstate_1, SplineOutput_0 _s_dOut_16)
{

#line 147
    float2  _S595 = make_float2 (_s_dOut_16.distortion_loss_0, - _s_dOut_16.distortion_loss_0);
    float _S596 = - _s_dOut_16.opacity_0;

#line 148
    DiffPair_float_0 _S597;

#line 148
    (&_S597)->primal_0 = - (*dpstate_1).primal_0.logT_0;

#line 148
    (&_S597)->differential_0 = 0.0f;

#line 148
    s_bwd_prop_exp_0(&_S597, _S596);

#line 148
    float _S598 = - _S597.differential_0;

#line 148
    SplineState_0 _S599 = SplineState_x24_syn_dzero_0();

#line 148
    (&_S599)->distortion_parts_0 = _S595;

#line 148
    (&_S599)->C_0 = _s_dOut_16.C_1;

#line 148
    (&_S599)->logT_0 = _S598;

#line 148
    dpstate_1->primal_0 = (*dpstate_1).primal_0;

#line 148
    dpstate_1->differential_0 = _S599;

#line 147
    return;
}


#line 147
__device__ void s_bwd_extract_color_0(DiffPair_SplineState_0 * _S600, SplineOutput_0 _S601)
{

#line 147
    s_bwd_prop_extract_color_0(_S600, _S601);

#line 147
    return;
}


#line 261 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__global__ void __kernel__backwards_kernel(TensorView last_state_0, TensorView last_dirac_0, TensorView iters_0, TensorView tri_collection_0, TensorView ray_origins_0, TensorView ray_directions_0, DualModel_0 model_2, TensorView initial_drgb_0, TensorView dL_dinital_drgb_0, TensorView touch_count_0, TensorView dL_doutputs_0, TensorView wcts_0, float tmin_1, float tmax_1, float max_prim_size_2, uint max_iters_0)
{

#line 261
    DualModel_0 _S602 = model_2;

#line 283
    uint ray_ind_1 = (((threadIdx)) + ((blockIdx)) * ((blockDim))).x;
    uint _S603 = ((ray_origins_0).sizes[(0U)]);

#line 284
    if(ray_ind_1 >= _S603)
    {

#line 285
        return;
    }
    SplineState_0 dual_state_0 = get_state_0(last_state_0, ray_ind_1);
    float3  _S604 = get_float3_0(ray_directions_0, ray_ind_1);
    float3  _S605 = get_float3_0(ray_origins_0, ray_ind_1);

#line 289
    float3  _S606 = _S605 + make_float3 (tmin_1) * _S604;


    float2  _S607 = make_float2 (0.0f, 0.0f);

#line 292
    float3  _S608 = make_float3 (0.0f, 0.0f, 0.0f);

#line 292
    float4  _S609 = make_float4 (0.0f, 0.0f, 0.0f, 0.0f);

#line 292
    SplineState_0 _S610 = { _S607, _S607, _S608, 0.0f, _S609, 0.0f, _S608 };

#line 292
    DiffPair_SplineState_0 deriv_state_1;

#line 292
    (&deriv_state_1)->primal_0 = dual_state_0;

#line 292
    (&deriv_state_1)->differential_0 = _S610;



    float4  _S611 = get_float4_0(dL_doutputs_0, ray_ind_1);
    float _S612 = ((dL_doutputs_0).load<float>((ray_ind_1), (4U)));
    SplineOutput_0 dL_doutput_0;
    (&dL_doutput_0)->C_1 = make_float3 (_S611.x, _S611.y, _S611.z);
    (&dL_doutput_0)->opacity_0 = _S611.w;
    (&dL_doutput_0)->distortion_loss_0 = _S612;

    int _S613 = ((iters_0).load<int>((ray_ind_1)));

#line 303
    uint num_iters_0 = (U32_max(((U32_min((uint(_S613)), (max_iters_0)))), (0U)));
    int _S614 = ((iters_0).load<int>((ray_ind_1)));

#line 304
    if(uint(_S614) >= max_iters_0 - 1U)
    {

#line 304
        return;
    }

    s_bwd_extract_color_0(&deriv_state_1, dL_doutput_0);

    uint _S615 = (((&_S602)->features_1).sizes[(1U)]);
    int _S616 = int((F32_sqrt((float(_S615))))) - int(1);


    uint _S617 = ((wcts_0).sizes[(0U)]);

#line 313
    uint tri_ind_0;

#line 313
    if(ray_ind_1 < _S617)
    {

#line 313
        tri_ind_0 = ray_ind_1;

#line 313
    }
    else
    {

#line 313
        tri_ind_0 = 0U;

#line 313
    }
    float _S618 = ((wcts_0).load<float>((tri_ind_0), (0U), (0U)));

#line 314
    float _S619 = ((wcts_0).load<float>((tri_ind_0), (0U), (1U)));

#line 314
    float _S620 = ((wcts_0).load<float>((tri_ind_0), (0U), (2U)));

#line 314
    float _S621 = ((wcts_0).load<float>((tri_ind_0), (0U), (3U)));

#line 314
    float4  _S622 = make_float4 (_S618, _S619, _S620, _S621);

#line 314
    float _S623 = ((wcts_0).load<float>((tri_ind_0), (1U), (0U)));

#line 314
    float _S624 = ((wcts_0).load<float>((tri_ind_0), (1U), (1U)));

#line 314
    float _S625 = ((wcts_0).load<float>((tri_ind_0), (1U), (2U)));

#line 314
    float _S626 = ((wcts_0).load<float>((tri_ind_0), (1U), (3U)));

#line 314
    float4  _S627 = make_float4 (_S623, _S624, _S625, _S626);

#line 314
    float _S628 = ((wcts_0).load<float>((tri_ind_0), (2U), (0U)));

#line 314
    float _S629 = ((wcts_0).load<float>((tri_ind_0), (2U), (1U)));

#line 314
    float _S630 = ((wcts_0).load<float>((tri_ind_0), (2U), (2U)));

#line 314
    float _S631 = ((wcts_0).load<float>((tri_ind_0), (2U), (3U)));

#line 314
    float4  _S632 = make_float4 (_S628, _S629, _S630, _S631);

#line 314
    float _S633 = ((wcts_0).load<float>((tri_ind_0), (3U), (0U)));

#line 314
    float _S634 = ((wcts_0).load<float>((tri_ind_0), (3U), (1U)));

#line 314
    float _S635 = ((wcts_0).load<float>((tri_ind_0), (3U), (2U)));

#line 314
    float _S636 = ((wcts_0).load<float>((tri_ind_0), (3U), (3U)));

#line 314
    Matrix<float, 4, 4>  wct_2 = makeMatrix<float, 4, 4> (_S622, _S627, _S632, make_float4 (_S633, _S634, _S635, _S636));

#line 321
    Matrix<float, 4, 4>  _S637 = inverse_0(wct_2);

    uint _S638 = (U32_max((num_iters_0 - 1U), (0U)));

#line 323
    uint _S639 = ((ray_origins_0).sizes[(0U)]);

#line 323
    int _S640 = ((tri_collection_0).load<int>((ray_ind_1 + _S638 * _S639)));

#line 323
    uint tri_ind_1 = uint(_S640);
    uint _S641 = uint(_S616);

#line 324
    ControlPoint_0 _S642 = load_ctrl_pt_0(tri_ind_1, _S602, _S606, _S604, _S641, false);


    int _S643 = int(num_iters_0);

#line 327
    SplineState_0 dual_state_1 = dual_state_0;

#line 327
    ControlPoint_0 ctrl_pt_3 = _S642;

#line 327
    tri_ind_0 = tri_ind_1;

#line 327
    int i_4 = _S643;

#line 327
    for(;;)
    {

#line 327
        int i_5 = i_4 - int(1);

#line 327
        if(i_4 > int(0))
        {
        }
        else
        {

#line 327
            break;
        }

        ControlPoint_0 old_ctrl_pt_1;
        int _S644 = i_5 - int(1);

#line 331
        uint old_tri_ind_0;

#line 331
        if(_S644 >= int(0))
        {

#line 332
            uint _S645 = uint(_S644);

#line 332
            uint _S646 = ((ray_origins_0).sizes[(0U)]);

#line 332
            int _S647 = ((tri_collection_0).load<int>((ray_ind_1 + _S645 * _S646)));

#line 332
            uint old_tri_ind_1 = uint(_S647);
            ControlPoint_0 _S648 = load_ctrl_pt_0(old_tri_ind_1, _S602, _S606, _S604, _S641, false);

#line 333
            old_ctrl_pt_1 = _S648;

#line 333
            old_tri_ind_0 = old_tri_ind_1;

#line 331
        }
        else
        {

            (&old_ctrl_pt_1)->t_1 = 0.0f;
            (&old_ctrl_pt_1)->dirac_0 = _S609;

#line 331
        }

#line 340
        SplineState_0 old_dual_state_1 = inverse_update_dual_0(dual_state_1, ctrl_pt_3, old_ctrl_pt_1, tmin_1, tmax_1);

#line 346
        uint _S649 = uint((F32_floor((float(tri_ind_0 / 2U)))));
        uint _S650 = ((tri_ind_0) % (2U));

#line 342
        DiffPair_SplineState_0 _S651 = run_update_0(old_dual_state_1, old_ctrl_pt_1, ctrl_pt_3, _S649, _S650, ray_ind_1, deriv_state_1, _S606, _S604, tmin_1, tmax_1, _S641, max_prim_size_2, wct_2, _S637, &_S602);

#line 352
        int itemp_0;
        *((&itemp_0)) = atomicAdd((touch_count_0).data_ptr_at<int>((uint((F32_floor((float(tri_ind_0 / 2U))))))), (int(1)));



        ControlPoint_0 _S652 = old_ctrl_pt_1;

#line 357
        (&deriv_state_1)->primal_0 = old_dual_state_1;

#line 357
        (&deriv_state_1)->differential_0 = _S651.differential_0;

#line 327
        dual_state_1 = old_dual_state_1;

#line 327
        ctrl_pt_3 = _S652;

#line 327
        tri_ind_0 = old_tri_ind_0;

#line 327
        i_4 = i_5;

#line 327
    }

#line 363
    (dL_dinital_drgb_0).store<float>((ray_ind_1), (0U), (deriv_state_1.differential_0.drgb_0.x));
    (dL_dinital_drgb_0).store<float>((ray_ind_1), (1U), (deriv_state_1.differential_0.drgb_0.y));
    (dL_dinital_drgb_0).store<float>((ray_ind_1), (2U), (deriv_state_1.differential_0.drgb_0.z));
    (dL_dinital_drgb_0).store<float>((ray_ind_1), (3U), (deriv_state_1.differential_0.drgb_0.w));
    return;
}


#line 10224 "hlsl.meta.slang"
__device__ float3  max_0(float3  x_8, float3  y_6)
{

#line 5510
    float3  result_10;

#line 5510
    int i_6 = int(0);

#line 5510
    for(;;)
    {

#line 5510
        if(i_6 < int(3))
        {
        }
        else
        {

#line 5510
            break;
        }

#line 5510
        *_slang_vector_get_element_ptr(&result_10, i_6) = (F32_max((_slang_vector_get_element(x_8, i_6)), (_slang_vector_get_element(y_6, i_6))));

#line 5510
        i_6 = i_6 + int(1);

#line 5510
    }

#line 5510
    return result_10;
}


#line 9729
__device__ float length_0(float4  x_9)
{

#line 9741
    return (F32_sqrt((dot_1(x_9, x_9))));
}


#line 9729
__device__ float length_1(float3  x_10)
{

#line 9741
    return (F32_sqrt((dot_0(x_10, x_10))));
}


#line 103 "/home/amai/ever_training/ever/splinetracers/slang/safe-math.slang"
__device__ float4  safe_div_3(float4  a_7, float b_5)
{

#line 104
    return make_float4 (safe_div_0(a_7.x, b_5), safe_div_0(a_7.y, b_5), safe_div_0(a_7.z, b_5), safe_div_0(a_7.w, b_5));
}


#line 22 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/tri-intersect.slang"
__device__ Matrix<float, 3, 3>  quat2mat_0(float4  quat_2)
{

#line 22
    float _S653 = quat_2.z;

#line 29
    float _S654 = _S653 * _S653;

#line 29
    float _S655 = quat_2.w * quat_2.w;
    float _S656 = quat_2.y * quat_2.z;

#line 30
    float _S657 = quat_2.x * quat_2.w;
    float _S658 = quat_2.y * quat_2.w;

#line 31
    float _S659 = quat_2.x * quat_2.z;


    float _S660 = quat_2.y * quat_2.y;
    float _S661 = quat_2.z * quat_2.w;

#line 35
    float _S662 = quat_2.x * quat_2.y;

#line 41
    return makeMatrix<float, 3, 3> (make_float3 (1.0f - 2.0f * (_S654 + _S655), 2.0f * (_S656 - _S657), 2.0f * (_S658 + _S659)), make_float3 (2.0f * (_S656 + _S657), 1.0f - 2.0f * (_S660 + _S655), 2.0f * (_S661 - _S662)), make_float3 (2.0f * (_S658 - _S659), 2.0f * (_S661 + _S662), 1.0f - 2.0f * (_S660 + _S654)));
}


#line 370 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
__device__ void s_bwd_prop_mix_drgb_0(DiffPair_float_0 * dpdensity_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpcolor_1, float4  _s_dOut_17)
{

#line 371
    float _S663 = (*dpcolor_1).primal_0.z * _s_dOut_17.w;

#line 371
    float _S664 = (*dpcolor_1).primal_0.y * _s_dOut_17.z;

#line 371
    float _S665 = (*dpcolor_1).primal_0.x * _s_dOut_17.y;

#line 371
    float3  _S666 = make_float3 ((*dpdensity_1).primal_0 * _s_dOut_17.y, (*dpdensity_1).primal_0 * _s_dOut_17.z, (*dpdensity_1).primal_0 * _s_dOut_17.w);

#line 371
    dpcolor_1->primal_0 = (*dpcolor_1).primal_0;

#line 371
    dpcolor_1->differential_0 = _S666;

#line 958 "core.meta.slang"
    float _S667 = _s_dOut_17.x + _S663 + _S664 + _S665;

#line 958
    dpdensity_1->primal_0 = (*dpdensity_1).primal_0;

#line 958
    dpdensity_1->differential_0 = _S667;

#line 370 "/home/amai/ever_training/ever/splinetracers/fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"
    return;
}


#line 370
__device__ void s_bwd_mix_drgb_0(DiffPair_float_0 * _S668, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S669, float4  _S670)
{

#line 370
    s_bwd_prop_mix_drgb_0(_S668, _S669, _S670);

#line 370
    return;
}




__global__ void __kernel__backwards_initial_drgb_kernel(TensorView ray_origins_1, TensorView ray_directions_1, DualModel_0 model_3, TensorView initial_drgb_1, TensorView initial_inds_0, TensorView dL_dinital_drgb_1, TensorView touch_count_1, float tmin_2)
{

#line 387
    uint3  _S671 = ((threadIdx));

#line 387
    uint3  _S672 = ((blockIdx));

#line 387
    uint3  _S673 = ((blockDim));

#line 387
    uint thread_j_0 = _S671.x + _S672.x * _S673.x;
    uint thread_i_0 = _S671.y + _S672.y * _S673.y;
    uint _S674 = ((initial_inds_0).sizes[(0U)]);

#line 389
    bool _S675;

#line 389
    if(thread_i_0 >= _S674)
    {

#line 389
        _S675 = true;

#line 389
    }
    else
    {

#line 389
        uint _S676 = ((ray_directions_1).sizes[(0U)]);

#line 389
        _S675 = thread_j_0 >= _S676;

#line 389
    }

#line 389
    if(_S675)
    {

#line 390
        return;
    }
    int _S677 = ((initial_inds_0).load<int>((thread_i_0)));

#line 392
    uint prim_ind_2 = uint(_S677);


    float3  mean_1 = get_float3_0(model_3.means_0, prim_ind_2);
    float4  quat_3 = get_float4_0(model_3.quats_0, prim_ind_2);
    float3  scales_3 = get_float3_0(model_3.scales_2, prim_ind_2);
    float3  rayd_2 = get_float3_0(ray_directions_1, thread_j_0);
    float3  _S678 = get_float3_0(ray_origins_1, 0U);

#line 405
    if(length_1(safe_div_2(mul_1(_S678 + make_float3 (tmin_2) * rayd_2 - mean_1, quat2mat_0(safe_div_3(quat_3, length_0(quat_3)))), max_0(scales_3, make_float3 (9.99999993922529029e-09f)))) <= 1.0f)
    {
        float _S679 = ((model_3.densities_0).load<float>((prim_ind_2)));

        Features_0 feat_7 = get_feats_0(model_3.features_1, prim_ind_2, 0U);

        float3  _S680 = make_float3 (0.0f, 0.0f, 0.0f);

#line 411
        DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_color_1;

#line 411
        (&deriv_color_1)->primal_0 = eval_color_0(rayd_2, feat_7, 0U);

#line 411
        (&deriv_color_1)->differential_0 = _S680;
        DiffPair_float_0 deriv_density_1;

#line 412
        (&deriv_density_1)->primal_0 = _S679;

#line 412
        (&deriv_density_1)->differential_0 = 0.0f;
        float4  vdL_dinital_drgb_0 = get_float4_0(dL_dinital_drgb_1, thread_j_0);
        s_bwd_mix_drgb_0(&deriv_density_1, &deriv_color_1, vdL_dinital_drgb_0);

#line 406
        float temp_5;

#line 416
        *((&temp_5)) = atomicAdd((model_3.dL_ddensities_0).data_ptr_at<float>((prim_ind_2)), (deriv_density_1.differential_0));

        DiffPair_vectorx3Cfloatx2C3x3E_0 deriv_direction_1;

#line 418
        (&deriv_direction_1)->primal_0 = rayd_2;

#line 418
        (&deriv_direction_1)->differential_0 = _S680;
        Features_0 _S681 = { _S680, _S680, _S680, _S680, _S680, _S680, _S680, _S680, _S680, _S680, _S680, _S680, _S680, _S680, _S680, _S680 };

#line 419
        DiffPair_Features_0 d_feat_1;

#line 419
        (&d_feat_1)->primal_0 = feat_7;

#line 419
        (&d_feat_1)->differential_0 = _S681;
        s_bwd_eval_color_0(&deriv_direction_1, &d_feat_1, 0U, deriv_color_1.differential_0);

#line 426
        atomic_add_float3_1(model_3.dL_dfeatures_0, prim_ind_2, 0U, make_float3 (d_feat_1.differential_0.f0_0.x, d_feat_1.differential_0.f0_0.y, d_feat_1.differential_0.f0_0.z));

#line 405
    }

#line 428
    return;
}

