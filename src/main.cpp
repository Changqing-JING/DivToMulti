#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>

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
  uint64_t const pow_k = static_cast<uint64_t>(1) << k;
  uint64_t const v = pow_k / d;
  if (v * d == pow_k) {
    return v;
  } else {
    return v + 1;
  }
}

namespace u32div {

template <uint64_t B> uint64_t opt_cal(uint64_t dividend, uint64_t divisor) {
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

template <uint64_t B> uint64_t opt_rem(uint64_t dividend, uint64_t divisor) {
  uint64_t quotient = opt_cal<B>(dividend, divisor);
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
      uint64_t const result = u32div::opt_cal<32>(dividend, divisor);
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
      uint64_t const result = u32div::opt_rem<32>(dividend, divisor);
      uint64_t const expected = u32div::normal_rem(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " % " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}
} // namespace u32div

namespace i32div {
template <uint64_t B> int64_t opt_cal_signed(int64_t dividend, int64_t divisor) {
  assert(divisor != 0);

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
  uint64_t const result = u32div::opt_cal<B>(abs_dividend, abs_divisor);

  // Apply sign
  return negative ? -static_cast<int64_t>(result) : static_cast<int64_t>(result);
}

int64_t normal_cal(int64_t dividend, int64_t divisor) {
  return dividend / divisor;
}

template <uint64_t B> int64_t opt_rem_signed(int64_t dividend, int64_t divisor) {
  int64_t quotient = opt_cal_signed<B>(dividend, divisor);
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

      int64_t const result = i32div::opt_cal_signed<32>(dividend, divisor);
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

      int64_t const result = i32div::opt_rem_signed<32>(dividend, divisor);
      int64_t const expected = i32div::normal_rem(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " % " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}
} // namespace i32div

int main() {
  u32div::test_div();
  u32div::test_rem();
  i32div::test_div();
  i32div::test_rem();
  return 0;
}