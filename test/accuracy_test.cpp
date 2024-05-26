
#include "ExpAVX512.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

TEST(AccuracyExp, BasicAssertions) {
  // EXPECT_FLOAT_EQ uses 4ULP, we should be within 3

  static constexpr float expmin = -87.33655F;
  static constexpr float expmax = 88.72284F;
  static constexpr int batch_size = 1024;
  static constexpr int num_batches = 10;
  static constexpr float step_size =
      (expmax - expmin) / (num_batches * batch_size - 1);

  std::vector<float> x(batch_size), y(batch_size);
  for (int j = 0; j < num_batches; ++j) {
    // test in batches to be more cache friendly
    // 1024 `float` is 4k bytes; L1D should be at least 32k
    for (int i = 0; i < batch_size; ++i) {
      int k = i + batch_size * j;
      float a = k * step_size + expmin;
      x[i] = a;
      y[i] = std::exp(double(a));
    }
    expavx512::simd_exp_n<16, 2>(x.data(), x.size());
    for (int i = 0; i < batch_size; ++i) {
      EXPECT_FLOAT_EQ(x[i], y[i]);
    }
  }
}

TEST(AccuracyExp2, BasicAssertions) {
  // EXPECT_FLOAT_EQ uses 4ULP, we should be within 3

  static constexpr float expmin = -87.33655F;
  static constexpr float expmax = 88.72284F;
  static constexpr int batch_size = 1024;
  static constexpr int num_batches = 10;
  static constexpr float step_size =
      (expmax - expmin) / (num_batches * batch_size - 1);

  std::vector<float> x(batch_size), y(batch_size);
  for (int j = 0; j < num_batches; ++j) {
    // test in batches to be more cache friendly
    // 1024 `float` is 4k bytes; L1D should be at least 32k
    for (int i = 0; i < batch_size; ++i) {
      int k = i + batch_size * j;
      float a = k * step_size + expmin;
      x[i] = a;
      y[i] = std::exp2(double(a));
    }
    expavx512::simd_exp_n<16, 2, 2>(x.data(), x.size());
    for (int i = 0; i < batch_size; ++i) {
      EXPECT_FLOAT_EQ(x[i], y[i]);
    }
  }
}

TEST(AccuracyExp10, BasicAssertions) {
  // EXPECT_FLOAT_EQ uses 4ULP, we should be within 3

  static constexpr float expmin = -87.33655F;
  static constexpr float expmax = 88.72284F;
  static constexpr int batch_size = 1024;
  static constexpr int num_batches = 10;
  static constexpr float step_size =
      (expmax - expmin) / (num_batches * batch_size - 1);

  std::vector<float> x(batch_size), y(batch_size);
  for (int j = 0; j < num_batches; ++j) {
    // test in batches to be more cache friendly
    // 1024 `float` is 4k bytes; L1D should be at least 32k
    for (int i = 0; i < batch_size; ++i) {
      int k = i + batch_size * j;
      float a = k * step_size + expmin;
      x[i] = a;
      y[i] = std::pow(10.0, double(a));
    }
    expavx512::simd_exp_n<16, 2, 10>(x.data(), x.size());
    for (int i = 0; i < batch_size; ++i) {
      EXPECT_FLOAT_EQ(x[i], y[i]);
    }
  }
}

