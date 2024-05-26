#include <immintrin.h>
#include <numbers>

#if !defined(__clang__) && defined(__GNUC__)
#define EXPAVX512FULLUNROLL _Pragma("GCC unroll 16")
#elif defined(__clang__)
#define EXPAVX512FULLUNROLL _Pragma("unroll")
#else
#define EXPAVX512FULLUNROLL
#endif

namespace expavx512 {

template <int W, typename T, int U> struct Vec;

template <int U> struct Vec<4, float, U> {
  __m128 data[U];
  constexpr auto operator[](int i) -> __m128 & { return data[i]; }
};
template <int U> struct Vec<8, float, U> {
  __m256 data[U];
  constexpr auto operator[](int i) -> __m256 & { return data[i]; }
};
template <int U> struct Vec<16, float, U> {
  __m512 data[U];
  constexpr auto operator[](int i) -> __m512 & { return data[i]; }
};
template <> struct Vec<4, float, 1> {
  __m128 data;
  constexpr auto operator[](int) -> __m128 & { return data; }
};
template <> struct Vec<8, float, 1> {
  __m256 data;
  constexpr auto operator[](int) -> __m256 & { return data; }
};
template <> struct Vec<16, float, 1> {
  __m512 data;
  constexpr auto operator[](int) -> __m512 & { return data; }
};

template <int W> inline auto vbroadcast(float a) {
  if constexpr (W == 4) {
    return _mm_set1_ps(a);
  } else if constexpr (W == 8) {
    return _mm256_set1_ps(a);
  } else {
    static_assert(W == 16);
    return _mm512_set1_ps(a);
  }
}
template <int W, int U> inline auto broadcast(float a) -> Vec<W, float, U> {
  if constexpr (U != 1) {
    auto v = vbroadcast<W>(a);
    Vec<W, float, U> ret;
    EXPAVX512FULLUNROLL
    for (int u = 0; u < U; ++u)
      ret[u] = v;
    return ret;
  } else
    return {vbroadcast<W>(a)};
}

template <int W, int U>
[[gnu::always_inline]] inline auto
interleave(Vec<W, float, U> a, const auto &f) -> Vec<W, float, U> {
  if constexpr (U != 1) {
    Vec<W, float, U> ret;
    EXPAVX512FULLUNROLL
    for (int u = 0; u < U; ++u)
      ret[u] = f(a[u]);
    return ret;
  } else
    return {f(a.data)};
}
template <int W, int U>
[[gnu::always_inline]] inline auto
interleave(Vec<W, float, U> a, Vec<W, float, U> b,
           const auto &f) -> Vec<W, float, U> {
  if constexpr (U != 1) {
    Vec<W, float, U> ret;
    EXPAVX512FULLUNROLL
    for (int u = 0; u < U; ++u)
      ret[u] = f(a[u], b[u]);
    return ret;
  } else
    return {f(a.data, b.data)};
}
template <int W, int U>
[[gnu::always_inline]] inline auto
interleave(Vec<W, float, U> a, Vec<W, float, U> b, Vec<W, float, U> c,
           const auto &f) -> Vec<W, float, U> {
  if constexpr (U != 1) {
    Vec<W, float, U> ret;
    EXPAVX512FULLUNROLL
    for (int u = 0; u < U; ++u)
      ret[u] = f(a[u], b[u], c[u]);
    return ret;
  } else
    return {f(a.data, b.data, c.data)};
}

[[gnu::always_inline]] inline auto scale(__m128 a, __m128 b) -> __m128 {
  return _mm_scalef_ps(a, b);
}
[[gnu::always_inline]] inline auto scale(__m256 a, __m256 b) -> __m256 {
  return _mm256_scalef_ps(a, b);
}
[[gnu::always_inline]] inline auto scale(__m512 a, __m512 b) -> __m512 {
  return _mm512_scalef_ps(a, b);
}
/// computes x * pow(2, y), for integer-valued `y`
template <int W, int U>
[[gnu::always_inline]] inline auto
scale(Vec<W, float, U> a, Vec<W, float, U> b) -> Vec<W, float, U> {
  return interleave(a, b, [](auto x, auto y) { return scale(x, y); });
}

template <int R = 0>
[[gnu::always_inline]] inline auto reduce(__m128 a) -> __m128 {
  return _mm_reduce_ps(a, R);
}
template <int R = 0>
[[gnu::always_inline]] inline auto reduce(__m256 a) -> __m256 {
  return _mm256_reduce_ps(a, R);
}
template <int R = 0>
[[gnu::always_inline]] inline auto reduce(__m512 a) -> __m512 {
  return _mm512_reduce_ps(a, R);
}
/// computes x * pow(2, y), for integer-valued `y`
template <int R = 0, int W, int U>
[[gnu::always_inline]] inline auto
reduce(Vec<W, float, U> a) -> Vec<W, float, U> {
  return interleave(a, [](auto x) { return reduce<R>(x); });
}

[[gnu::always_inline]] inline auto fmadd(__m128 a, __m128 b,
                                         __m128 c) -> __m128 {
  return _mm_fmadd_ps(a, b, c);
}
[[gnu::always_inline]] inline auto fmadd(__m256 a, __m256 b,
                                         __m256 c) -> __m256 {
  return _mm256_fmadd_ps(a, b, c);
}
[[gnu::always_inline]] inline auto fmadd(__m512 a, __m512 b,
                                         __m512 c) -> __m512 {
  return _mm512_fmadd_ps(a, b, c);
}
template <int W, int U>
[[gnu::always_inline]] inline auto
fmadd(Vec<W, float, U> a, Vec<W, float, U> b,
      Vec<W, float, U> c) -> Vec<W, float, U> {
  return interleave(a, b, c,
                    [](auto x, auto y, auto z) { return fmadd(x, y, z); });
}
template <int W, int U>
[[gnu::always_inline]] inline auto fmadd(Vec<W, float, U> a, Vec<W, float, U> b,
                                         float c) -> Vec<W, float, U> {
  return fmadd(a, b, broadcast<W, U>(c));
}
template <int W, int U>
[[gnu::always_inline]] inline auto
fmadd(float a, Vec<W, float, U> b, Vec<W, float, U> c) -> Vec<W, float, U> {
  return fmadd(broadcast<W, U>(a), b, c);
}
template <int W, int U>
[[gnu::always_inline]] inline auto fmadd(float a, Vec<W, float, U> b,
                                         float c) -> Vec<W, float, U> {
  return fmadd(broadcast<W, U>(a), b, broadcast<W, U>(c));
}

[[gnu::always_inline]] inline auto fsub(__m128 a, __m128 b) -> __m128 {
  return _mm_sub_ps(a, b);
}
[[gnu::always_inline]] inline auto fsub(__m256 a, __m256 b) -> __m256 {
  return _mm256_sub_ps(a, b);
}
[[gnu::always_inline]] inline auto fsub(__m512 a, __m512 b) -> __m512 {
  return _mm512_sub_ps(a, b);
}

template <int W, int U>
[[gnu::always_inline]] inline auto
operator-(Vec<W, float, U> a, Vec<W, float, U> b) -> Vec<W, float, U> {
  return interleave(a, b, [](auto x, auto y) { return fsub(x, y); });
}

template <int B> [[gnu::always_inline]] inline auto kernel(auto x) {
  if constexpr (B == 2) {
    return fmadd(
        fmadd(fmadd(fmadd(fmadd(fmadd(fmadd(1.5316464e-5F, x, 0.00015478022F),
                                      x, 0.0013400431F),
                                x, 0.009617995F),
                          x, 0.05550327F),
                    x, 0.24022652F),
              x, 0.6931472F),
        x, 1.0F);
  } else if constexpr (B == 10) {
    return fmadd(
        fmadd(fmadd(fmadd(fmadd(fmadd(fmadd(0.06837386F, x, 0.20799689F), x,
                                      0.54208815F),
                                x, 1.1712388F),
                          x, 2.034648F),
                    x, 2.6509492F),
              x, 2.3025851F),
        x, 1.0F);
  } else { // b == e assumed
    return fmadd(
        fmadd(fmadd(fmadd(fmadd(fmadd(fmadd(0.00019924171F, x, 0.0013956056F),
                                      x, 0.008375129F),
                                x, 0.041666083F),
                          x, 0.16666415F),
                    x, 0.5F),
              x, 1.0F),
        x, 1.0F);
  }
}

// constants

constexpr float round_float = 1.2582912e7;

// int of 2/10 means exp2/exp10, otherwise it is `e`
template <int>
static inline constexpr float log2b = std::numbers::log2e_v<float>;
template <> static inline constexpr float log2b<2> = 1.0F;
template <> static inline constexpr float log2b<10> = 3.321928F;

template <int>
static inline constexpr float nlogb2_hi = -std::numbers::ln2_v<float>;
template <> static inline constexpr float nlogb2_hi<2> = -1.0F;
template <> static inline constexpr float nlogb2_hi<10> = -0.30103F;

template <int>
static inline constexpr float nlogb2_lo =
    -float(std::numbers::ln2_v<double> + nlogb2_hi<0>);
template <> static inline constexpr float nlogb2_lo<2> = 0.0F;
// float(double(float(log10(2.0))) - log10(2.0)))
template <> static inline constexpr float nlogb2_lo<10> = 1.4320989e-8F;

// we want this to inline into the loop so that the constants can be hoisted out
template <int B = 0, int W, int U>
[[gnu::always_inline]] inline auto exp(Vec<W, float, U> x) -> Vec<W, float, U> {
  if constexpr (B == 2) {
    Vec<W, float, U> r = reduce<0>(x);
    Vec<W, float, U> n_float = x - r;
    return scale(kernel<B>(r), n_float);
  } else {
    Vec<W, float, U> vround = broadcast<W, U>(round_float);
    // `n_float` is integer-valued
    Vec<W, float, U> n_float =
        fmadd(x, broadcast<W, U>(log2b<B>), vround) - vround;
    // reduce, r = x - n_float/log2(base)
    Vec<W, float, U> r =
        fmadd(nlogb2_lo<B>, n_float, fmadd(nlogb2_hi<B>, n_float, x));
    return scale(kernel<B>(r), n_float);
  }
}

// load and store for vectorized function
template <int W> [[gnu::always_inline]] inline auto mm_load(const float *p) {
  if constexpr (W == 4) {
    return _mm_loadu_ps(p);
  } else if constexpr (W == 8) {
    return _mm256_loadu_ps(p);
  } else {
    static_assert(W == 16);
    return _mm512_loadu_ps(p);
  }
}
template <int W>
[[gnu::always_inline]] inline auto mm_load(const float *p, auto k) {
  if constexpr (W == 4) {
    return _mm_maskz_loadu_ps(__mmask8(k), p);
  } else if constexpr (W == 8) {
    return _mm256_maskz_loadu_ps(__mmask8(k), p);
  } else {
    static_assert(W == 16);
    return _mm512_maskz_loadu_ps(__mmask16(k), p);
  }
}
template <int W>
[[gnu::always_inline]] inline auto load(const float *p,
                                        auto k) -> Vec<W, float, 1> {
  return {mm_load<W>(p, k)};
}
template <int W, int U>
[[gnu::always_inline]] inline auto load(const float *p) -> Vec<W, float, U> {
  if constexpr (U != 1) {
    Vec<W, float, U> ret;
    EXPAVX512FULLUNROLL
    for (long u = 0; u < U; ++u)
      ret.data[u] = mm_load<W>(p + W * u);
    return ret;
  } else
    return {mm_load<W>(p)};
}
[[gnu::always_inline]] inline void mm_store(float *p, __m128 x) {
  _mm_storeu_ps(p, x);
}
[[gnu::always_inline]] inline void mm_store(float *p, __m256 x) {
  _mm256_storeu_ps(p, x);
}
[[gnu::always_inline]] inline void mm_store(float *p, __m512 x) {
  _mm512_storeu_ps(p, x);
}
[[gnu::always_inline]] inline void mm_store(float *p, __m128 x, __mmask8 k) {
  _mm_mask_storeu_ps(p, k, x);
}
[[gnu::always_inline]] inline void mm_store(float *p, __m256 x, __mmask8 k) {
  _mm256_mask_storeu_ps(p, k, x);
}
[[gnu::always_inline]] inline void mm_store(float *p, __m512 x, __mmask16 k) {
  _mm512_mask_storeu_ps(p, k, x);
}
template <int W, int U>
[[gnu::always_inline]] inline void store(float *p, Vec<W, float, U> x) {
  if constexpr (U != 1) {
    EXPAVX512FULLUNROLL
    for (long u = 0; u < U; ++u)
      mm_store(p + W * u, x.data[u]);
  } else
    mm_store(p, x.data);
}
template <int W>
[[gnu::always_inline]] inline void store(float *p, Vec<W, float, 1> x, auto k) {
  mm_store(p, x.data, k);
}

template <int W, int U>
[[gnu::flatten]] inline void simd_transform_n(float *p, long N, const auto &f) {
  static constexpr long WU = long(W) * U;
  long n = 0;
  for (; n <= N - WU; n += WU) {
    float *op = p + n;
    store(op, f(load<W, U>(op)));
  }
  if constexpr (U > 1) {
    for (; n <= N - W; n += W) {
      float *op = p + n;
      store(op, f(load<W, 1>(op)));
    }
  }
  if (n < N) {
    unsigned int k = _bzhi_u32(0xffffffff, N - n);
    float *op = p + n;
    store(op, f(load<W>(op, k)), k);
  }
}
template <int W, int U>
[[gnu::flatten]] inline void simd_transform_n(const float *src, long N,
                                              float *dst, const auto &f) {
  static constexpr long WU = long(W) * U;
  long n = 0;
  for (; n <= N - WU; n += WU) {
    store(dst + n, f(load<W, U>(src + n)));
  }
  if constexpr (U > 1) {
    for (; n <= N - W; n += W) {
      store(dst + n, f(load<W, 1>(src + n)));
    }
  }
  if (n < N) {
    unsigned int k = _bzhi_u32(0xffffffff, N - n);
    store(dst + n, f(load<W>(src + n, k)), k);
  }
}
template <int W, int U, int B = 0>
[[gnu::flatten]] inline void simd_exp_n(float *p, long N) {
  simd_transform_n<W, U>(p, N, [](auto x) { return exp<B>(x); });
}

template <int W, int U, int B = 0>
[[gnu::flatten]] inline void simd_exp_n(const float *src, long N, float *dst) {
  simd_transform_n<W, U>(src, N, dst, [](auto x) { return exp<B>(x); });
}

} // namespace expavx512
#undef EXPAVX512FULLUNROLL
