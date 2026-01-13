#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#ifdef _MSC_VER
#include <intrin.h>
#endif

uint64_t umulh(uint64_t x, uint64_t y) {
#ifdef _MSC_VER
  return __umulh(x, y);
#else
  return static_cast<uint64_t>((static_cast<__uint128_t>(x) * y) >> 64ULL);
#endif
}

constexpr size_t T = 12U;

template <uint64_t B> uint64_t get_k(uint64_t d) {
  assert(d > 0U);
  if (d == 1) {
    return B;
  } else if (d == 2) {
    return B + 1;
  }
  return B + 64ULL - __builtin_clzll(d - 1);
}

uint64_t get_m(uint64_t k, uint64_t d) {
  if (k == 64) {
    // 2^64 doesn't fit in uint64_t, but ceil(2^64 / d) = floor((2^64 - 1) / d) + 1
    uint64_t const max_64 = static_cast<uint64_t>(-1);
    return (max_64 / d) + 1;
  }

  uint64_t const pow_k = static_cast<uint64_t>(1) << k;
  uint64_t const v = pow_k / d;
  if (v * d == pow_k) {
    return v;
  } else {
    return v + 1;
  }
}

// 128-bit version for u64div where k can be >= 64
__uint128_t get_m_128(uint64_t k, uint64_t d) {
  if (k == 128) {
    // 2^128 doesn't fit in __uint128_t, but ceil(2^128 / d) = floor((2^128 - 1) / d) + 1
    __uint128_t const max_128 = static_cast<__uint128_t>(-1);
    return (max_128 / d) + 1;
  }

  __uint128_t const pow_k = static_cast<__uint128_t>(1) << k;
  __uint128_t const v = pow_k / d;
  if (v * d == pow_k) {
    return v;
  } else {
    return v + 1;
  }
}

namespace u32div {

uint64_t opt_cal(uint64_t dividend, uint64_t divisor) {
  constexpr uint64_t B = 32U;
  uint64_t const k = get_k<B>(divisor);
  uint64_t const m = get_m(k, divisor);
  // return (dividend * m) >> k;

  uint64_t const low_mask = (static_cast<uint64_t>(1) << B) - 1;
  uint64_t const m_low = m & low_mask;
  return (dividend + ((dividend * m_low) >> B)) >> (k - B);
}

uint64_t normal_cal(uint64_t dividend, uint64_t divisor) {
  return dividend / divisor;
}

uint64_t opt_rem(uint64_t dividend, uint64_t divisor) {
  uint64_t quotient = opt_cal(dividend, divisor);
  return dividend - divisor * quotient;
}

uint64_t normal_rem(uint64_t dividend, uint64_t divisor) {
  return dividend % divisor;
}

void test_div() {
  for (uint64_t dividend = 0; dividend <= static_cast<uint64_t>((1ULL << T) - 1); ++dividend) {
    if (dividend % 1024 == 0) {
      std::cout << "Processing dividend: " << dividend << std::endl;
    }
    for (uint64_t divisor = 1; divisor <= static_cast<uint64_t>((1ULL << T) - 1); ++divisor) {
      uint64_t const result = u32div::opt_cal(dividend, divisor);
      uint64_t const expected = u32div::normal_cal(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " / " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}

void test_rem() {
  for (uint64_t dividend = 0; dividend <= static_cast<uint64_t>((1ULL << T) - 1); ++dividend) {
    if (dividend % 1024 == 0) {
      std::cout << "Processing rem dividend: " << dividend << std::endl;
    }
    for (uint64_t divisor = 1; divisor <= static_cast<uint64_t>((1ULL << T) - 1); ++divisor) {
      uint64_t const result = u32div::opt_rem(dividend, divisor);
      uint64_t const expected = u32div::normal_rem(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " % " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}

void test_large_divisor() {
  uint64_t const max_32 = (1ULL << 32) - 1;
  uint64_t const test_dividends[] = {0, 1, 100, max_32, max_32 - 1, max_32 / 2};
  uint64_t const test_divisors[] = {max_32, max_32 - 1, max_32 / 2, (1ULL << 31), (1ULL << 31) + 1};

  for (uint64_t dividend : test_dividends) {
    for (uint64_t divisor : test_divisors) {
      uint64_t const result = u32div::opt_cal(dividend, divisor);
      uint64_t const expected = u32div::normal_cal(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " / " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }

  std::cout << "u32div large divisor tests passed!" << std::endl;
}
} // namespace u32div

namespace i32div {
int64_t opt_cal_signed(int64_t dividend, int64_t divisor) {
  assert(divisor != 0);
  uint64_t constexpr B = 32U;

  // Handle special case: INT64_MIN / -1 would overflow
  if (dividend == INT64_MIN && divisor == -1) {
    return INT64_MIN; // Undefined behavior in C++, but this is common behavior
  }

  // Determine result sign
  bool const negative = (dividend < 0) ^ (divisor < 0);

  // Work with absolute values
  uint64_t const abs_dividend = dividend < 0 ? -static_cast<uint64_t>(dividend) : static_cast<uint64_t>(dividend);
  uint64_t const abs_divisor = divisor < 0 ? -static_cast<uint64_t>(divisor) : static_cast<uint64_t>(divisor);

  // Use unsigned algorithm
  uint64_t const result = u32div::opt_cal(abs_dividend, abs_divisor);

  // Apply sign
  return negative ? -static_cast<int64_t>(result) : static_cast<int64_t>(result);
}

int64_t normal_cal(int64_t dividend, int64_t divisor) {
  return dividend / divisor;
}

int64_t opt_rem_signed(int64_t dividend, int64_t divisor) {
  int64_t quotient = opt_cal_signed(dividend, divisor);
  return dividend - divisor * quotient;
}

int64_t normal_rem(int64_t dividend, int64_t divisor) {
  return dividend % divisor;
}

void test_div() {
  int64_t const min_val = -(1LL << (T - 1));
  int64_t const max_val = (1LL << (T - 1)) - 1;

  for (int64_t dividend = min_val; dividend <= max_val; ++dividend) {
    if ((dividend - min_val) % 1024 == 0) {
      std::cout << "Processing signed dividend: " << dividend << std::endl;
    }
    for (int64_t divisor = min_val; divisor <= max_val; ++divisor) {
      if (divisor == 0)
        continue; // Skip division by zero

      int64_t const result = i32div::opt_cal_signed(dividend, divisor);
      int64_t const expected = i32div::normal_cal(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " / " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}

void test_rem() {
  int64_t const min_val = -(1LL << (T - 1));
  int64_t const max_val = (1LL << (T - 1)) - 1;

  for (int64_t dividend = min_val; dividend <= max_val; ++dividend) {
    if ((dividend - min_val) % 1024 == 0) {
      std::cout << "Processing signed rem dividend: " << dividend << std::endl;
    }
    for (int64_t divisor = min_val; divisor <= max_val; ++divisor) {
      if (divisor == 0)
        continue; // Skip division by zero

      int64_t const result = i32div::opt_rem_signed(dividend, divisor);
      int64_t const expected = i32div::normal_rem(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " % " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}
} // namespace i32div

namespace u64div {

uint64_t opt_cal(uint64_t dividend, uint64_t divisor) {
  constexpr uint64_t B = 64U;
  uint64_t const k = get_k<B>(divisor);
  __uint128_t const m = get_m_128(k, divisor);

  uint64_t const m_low = static_cast<uint64_t>(m);
  uint64_t const m_high = static_cast<uint64_t>(m >> 64);

  uint64_t const prod_low_high = umulh(dividend, m_low);
  __uint128_t const high_128 = static_cast<__uint128_t>(m_high) * dividend + prod_low_high;

  return static_cast<uint64_t>(high_128 >> (k - B));
}

uint64_t normal_cal(uint64_t dividend, uint64_t divisor) {
  return dividend / divisor;
}

uint64_t opt_rem(uint64_t dividend, uint64_t divisor) {
  uint64_t quotient = opt_cal(dividend, divisor);
  return dividend - divisor * quotient;
}

uint64_t normal_rem(uint64_t dividend, uint64_t divisor) {
  return dividend % divisor;
}

void test_div() {
  for (uint64_t dividend = 0; dividend <= static_cast<uint64_t>((1ULL << T) - 1); ++dividend) {
    if (dividend % 1024 == 0) {
      std::cout << "Processing u64div dividend: " << dividend << std::endl;
    }
    for (uint64_t divisor = 1; divisor <= static_cast<uint64_t>((1ULL << T) - 1); ++divisor) {
      uint64_t const result = u64div::opt_cal(dividend, divisor);
      uint64_t const expected = u64div::normal_cal(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " / " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}

void test_rem() {
  for (uint64_t dividend = 0; dividend <= static_cast<uint64_t>((1ULL << T) - 1); ++dividend) {
    if (dividend % 1024 == 0) {
      std::cout << "Processing u64div rem dividend: " << dividend << std::endl;
    }
    for (uint64_t divisor = 1; divisor <= static_cast<uint64_t>((1ULL << T) - 1); ++divisor) {
      uint64_t const result = u64div::opt_rem(dividend, divisor);
      uint64_t const expected = u64div::normal_rem(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " % " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}

void test_overflow_cases() {
  uint64_t const max_val = static_cast<uint64_t>(-1);

  std::cout << "Testing u64div overflow cases..." << std::endl;

  uint64_t const test_dividends[] = {
      max_val, max_val - 1, max_val / 2, max_val / 3, 1ULL << 63, (1ULL << 63) - 1, 1ULL << 62, 1ULL << 48, 1ULL << 32,
  };

  uint64_t const test_divisors[] = {
      max_val, max_val - 1, max_val / 2, 1ULL << 63, (1ULL << 63) - 1, 1ULL << 32, (1ULL << 32) + 1, 3, 7, 1,
  };

  for (uint64_t dividend : test_dividends) {
    for (uint64_t divisor : test_divisors) {
      uint64_t const result = opt_cal(dividend, divisor);
      uint64_t const expected = normal_cal(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " / " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }

  std::cout << "u64div overflow tests passed!" << std::endl;
}

} // namespace u64div

namespace i64div {

int64_t opt_cal_signed(int64_t dividend, int64_t divisor) {
  assert(divisor != 0);

  // Handle special case: INT64_MIN / -1 would overflow
  if (dividend == INT64_MIN && divisor == -1) {
    return INT64_MIN;
  }

  // Determine result sign
  bool const negative = (dividend < 0) ^ (divisor < 0);

  // Work with absolute values
  uint64_t const abs_dividend = dividend < 0 ? -static_cast<uint64_t>(dividend) : static_cast<uint64_t>(dividend);
  uint64_t const abs_divisor = divisor < 0 ? -static_cast<uint64_t>(divisor) : static_cast<uint64_t>(divisor);

  // Use unsigned algorithm
  uint64_t const result = u64div::opt_cal(abs_dividend, abs_divisor);

  // Apply sign
  return negative ? -static_cast<int64_t>(result) : static_cast<int64_t>(result);
}

int64_t normal_cal(int64_t dividend, int64_t divisor) {
  return dividend / divisor;
}

int64_t opt_rem_signed(int64_t dividend, int64_t divisor) {
  int64_t quotient = opt_cal_signed(dividend, divisor);
  return dividend - divisor * quotient;
}

int64_t normal_rem(int64_t dividend, int64_t divisor) {
  return dividend % divisor;
}

void test_div() {
  int64_t const min_val = -(1LL << (T - 1));
  int64_t const max_val = (1LL << (T - 1)) - 1;

  for (int64_t dividend = min_val; dividend <= max_val; ++dividend) {
    if ((dividend - min_val) % 1024 == 0) {
      std::cout << "Processing i64div dividend: " << dividend << std::endl;
    }
    for (int64_t divisor = min_val; divisor <= max_val; ++divisor) {
      if (divisor == 0)
        continue;

      int64_t const result = i64div::opt_cal_signed(dividend, divisor);
      int64_t const expected = i64div::normal_cal(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " / " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}

void test_rem() {
  int64_t const min_val = -(1LL << (T - 1));
  int64_t const max_val = (1LL << (T - 1)) - 1;

  for (int64_t dividend = min_val; dividend <= max_val; ++dividend) {
    if ((dividend - min_val) % 1024 == 0) {
      std::cout << "Processing i64div rem dividend: " << dividend << std::endl;
    }
    for (int64_t divisor = min_val; divisor <= max_val; ++divisor) {
      if (divisor == 0)
        continue;

      int64_t const result = i64div::opt_rem_signed(dividend, divisor);
      int64_t const expected = i64div::normal_rem(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " % " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}

void test_overflow_cases() {
  int64_t const max_val = INT64_MAX;
  int64_t const min_val = INT64_MIN;

  std::cout << "Testing i64div overflow cases..." << std::endl;

  int64_t const test_dividends[] = {
      max_val, max_val - 1, min_val, min_val + 1, 0, 1, -1, max_val / 2, min_val / 2,
  };

  int64_t const test_divisors[] = {
      max_val, max_val - 1, min_val, min_val + 1, 1, -1, 3, -3, 7, -7,
  };

  for (int64_t dividend : test_dividends) {
    for (int64_t divisor : test_divisors) {
      if (divisor == 0)
        continue;
      // Skip INT64_MIN / -1 as it's UB
      if (dividend == min_val && divisor == -1)
        continue;

      int64_t const result = opt_cal_signed(dividend, divisor);
      int64_t const expected = normal_cal(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " / " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }

      int64_t const rem_result = opt_rem_signed(dividend, divisor);
      int64_t const rem_expected = normal_rem(dividend, divisor);
      if (rem_result != rem_expected) {
        std::cout << "Error: " << dividend << " % " << divisor << " = " << rem_result << " instead " << rem_expected << std::endl;
        std::terminate();
      }
    }
  }

  std::cout << "i64div overflow tests passed!" << std::endl;
}

} // namespace i64div

int main() {
  u32div::test_div();
  u32div::test_rem();
  u32div::test_large_divisor();
  i32div::test_div();
  i32div::test_rem();
  u64div::test_div();
  u64div::test_rem();
  u64div::test_overflow_cases();
  i64div::test_div();
  i64div::test_rem();
  i64div::test_overflow_cases();
  return 0;
}