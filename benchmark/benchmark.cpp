#include "ExpAVX512.hpp"
#include <benchmark/benchmark.h>
#include <vector>

template <int W, int U, int B = 0>
static void BM_simd_exp_run(benchmark::State &state) {
  size_t N = state.range(0);

  std::vector<float> src(N), dst(N);
  static constexpr float expmin = -87.33655F;
  static constexpr float expmax = 88.72284F;
  float step_size = (expmax - expmin) / (N - 1);
  for (size_t n = 0; n < N; ++n)
    src[n] = expmin + n * step_size;

  for (auto b : state) {
    expavx512::simd_exp_n<W, U, B>(src.data(), N, dst.data());
  }
}

BENCHMARK(BM_simd_exp_run<4, 1>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<4, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<4, 4>)->RangeMultiplier(2)->Range(128, 1 << 20);

BENCHMARK(BM_simd_exp_run<8, 1>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<8, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<8, 4>)->RangeMultiplier(2)->Range(128, 1 << 20);

BENCHMARK(BM_simd_exp_run<16, 1>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<16, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<16, 4>)->RangeMultiplier(2)->Range(128, 1 << 20);

BENCHMARK(BM_simd_exp_run<4, 1, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<4, 2, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<4, 4, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);

BENCHMARK(BM_simd_exp_run<8, 1, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<8, 2, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<8, 4, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);

BENCHMARK(BM_simd_exp_run<16, 1, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<16, 2, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);
BENCHMARK(BM_simd_exp_run<16, 4, 2>)->RangeMultiplier(2)->Range(128, 1 << 20);

