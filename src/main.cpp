#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#ifdef _MSC_VER
#include <__msvc_int128.hpp>
#include <intrin.h>
using uint128 = std::_Unsigned128;
#else
using uint128 = __uint128_t;
#endif

uint64_t umulh(uint64_t x, uint64_t y) {
#ifdef _MSC_VER
  return __umulh(x, y);
#else
  return static_cast<uint64_t>((static_cast<uint128>(x) * y) >> 64ULL);
#endif
}

int64_t smulh(int64_t x, int64_t y) {
#ifdef _MSC_VER
  return __mulh(x, y);
#else
  using int128 = __int128_t;
  return static_cast<int64_t>((static_cast<int128>(x) * y) >> 64);
#endif
}

uint64_t clzll(uint64_t x) {
#ifdef _MSC_VER
  unsigned long index;
  if (_BitScanReverse64(&index, x)) {
    return 63ULL - index;
  }
  return 64ULL;
#else
  return static_cast<uint64_t>(__builtin_clzll(x));
#endif
}

uint32_t ctz(uint32_t x) {
#ifdef _MSC_VER
  unsigned long index;
  if (_BitScanForward(&index, x)) {
    return index;
  }
  return 32U;
#else
  return static_cast<uint32_t>(__builtin_ctz(x));
#endif
}

uint64_t ctzll(uint64_t x) {
#ifdef _MSC_VER
  unsigned long index;
  if (_BitScanForward64(&index, x)) {
    return index;
  }
  return 64ULL;
#else
  return static_cast<uint64_t>(__builtin_ctzll(x));
#endif
}

constexpr size_t T = 12U;

// =============================================================================
// Common magic number calculation templates (similar to LLVM's approach)
// =============================================================================

// Type traits for selecting wider type for intermediate calculations
template <typename T> struct WiderType;
template <> struct WiderType<uint32_t> {
  using type = uint64_t;
};
template <> struct WiderType<uint64_t> {
  using type = uint128;
};
template <> struct WiderType<int32_t> {
  using type = int64_t;
};
template <> struct WiderType<int64_t> {
  using type = uint128;
}; // Use uint128 for intermediate

// Unsigned division magic number result
template <typename UIntType> struct UnsignedDivMagic {
  UIntType magic;
  unsigned shift;
  bool is_add;
};

// LLVM-style magic number calculation for unsigned division
// Based on "Hacker's Delight" chapter 10 and LLVM's UnsignedDivisionByConstantInfo
template <typename UIntType> UnsignedDivMagic<UIntType> get_unsigned_magic(UIntType d) {
  static_assert(std::is_unsigned<UIntType>::value, "UIntType must be unsigned");
  assert(d > 1 && "Divisor must be > 1");

  using WideType = typename WiderType<UIntType>::type;
  constexpr unsigned B = sizeof(UIntType) * 8;

  WideType const all_ones = static_cast<WideType>(static_cast<UIntType>(-1)); // 2^B - 1
  WideType const signed_min = static_cast<WideType>(1) << (B - 1);            // 2^(B-1)
  WideType const signed_max = signed_min - 1;                                 // 2^(B-1) - 1

  // NC = largest dividend such that NC % d == d - 1
  WideType const nc = all_ones - (all_ones + 1 - d) % d;

  unsigned p = B - 1;

  // Initialize Q1 = 2^p / NC, R1 = 2^p % NC
  WideType q1 = signed_min / nc;
  WideType r1 = signed_min % nc;

  // Initialize Q2 = (2^p - 1) / D, R2 = (2^p - 1) % D
  WideType q2 = signed_max / d;
  WideType r2 = signed_max % d;

  bool is_add = false;

  WideType delta;
  do {
    p = p + 1;

    if (r1 >= nc - r1) {
      q1 = (q1 << 1) + 1;
      r1 = (r1 << 1) - nc;
    } else {
      q1 = q1 << 1;
      r1 = r1 << 1;
    }

    if (r2 + 1 >= d - r2) {
      if (q2 >= signed_max) {
        is_add = true;
      }
      q2 = (q2 << 1) + 1;
      r2 = (r2 << 1) + 1 - d;
    } else {
      if (q2 >= signed_min) {
        is_add = true;
      }
      q2 = q2 << 1;
      r2 = (r2 << 1) + 1;
    }

    delta = d - 1 - r2;
  } while (p < B * 2 && (q1 < delta || (q1 == delta && r1 == 0)));

  UIntType magic = static_cast<UIntType>(q2 + 1);
  unsigned shift = p - B;

  // When is_add is true, reduce shift by 1 (correction is done in computation)
  if (is_add) {
    assert(shift > 0 && "Unexpected shift");
    shift -= 1;
  }

  return {magic, shift, is_add};
}

// Signed division magic number result
template <typename SIntType> struct SignedDivMagic {
  SIntType magic;
  unsigned shift;
};

// LLVM-style magic number calculation for signed division
// Based on "Hacker's Delight" chapter 10 and LLVM's SignedDivisionByConstantInfo
template <typename SIntType> SignedDivMagic<SIntType> get_signed_magic(SIntType d) {
  static_assert(std::is_signed<SIntType>::value, "SIntType must be signed");
  assert(d != 0 && d != 1 && d != -1 && "Divisor must not be 0, 1, or -1");

  using UIntType = typename std::make_unsigned<SIntType>::type;
  using WideType = typename WiderType<UIntType>::type;
  constexpr unsigned B = sizeof(SIntType) * 8;

  WideType const signed_min = static_cast<WideType>(1) << (B - 1); // 2^(B-1)

  UIntType const ad = d < 0 ? static_cast<UIntType>(-d) : static_cast<UIntType>(d);

  // T = 2^(B-1) + sign_bit
  WideType const t = signed_min + (static_cast<UIntType>(d) >> (B - 1));
  // ANC = T - 1 - T % |D|
  WideType const anc = t - 1 - t % ad;

  unsigned p = B - 1;
  WideType q1 = signed_min / anc;
  WideType r1 = signed_min % anc;
  WideType q2 = signed_min / ad;
  WideType r2 = signed_min % ad;

  WideType delta;
  do {
    p = p + 1;
    q1 = q1 << 1;
    r1 = r1 << 1;
    if (r1 >= anc) {
      ++q1;
      r1 -= anc;
    }
    q2 = q2 << 1;
    r2 = r2 << 1;
    if (r2 >= ad) {
      ++q2;
      r2 -= ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));

  SIntType magic = static_cast<SIntType>(q2 + 1);
  if (d < 0) {
    magic = -magic;
  }

  return {magic, p - B};
}

// =============================================================================
// u32div namespace
// =============================================================================

namespace u32div {

// 32-bit division using umull-style: (dividend * magic) >> (32 + shift)
uint32_t opt_cal(uint32_t dividend, uint32_t divisor) {
  if (divisor == 1) {
    return dividend;
  }

  // For large divisors (> UINT32_MAX/2), quotient can only be 0 or 1
  if (divisor > (static_cast<uint32_t>(-1) >> 1)) {
    return dividend >= divisor ? 1 : 0;
  }

  auto const dm = get_unsigned_magic(divisor);

  // umull: 32x32 -> 64, then shift
  uint64_t const product = static_cast<uint64_t>(dividend) * dm.magic;
  uint32_t const high = static_cast<uint32_t>(product >> 32);

  if (!dm.is_add) {
    // Simple case: just shift the high part
    return high >> dm.shift;
  } else {
    // Correction case: (high + ((dividend - high) >> 1)) >> shift
    uint32_t const t = dividend - high;
    return (high + (t >> 1)) >> dm.shift;
  }
}

uint32_t normal_cal(uint32_t dividend, uint32_t divisor) {
  return dividend / divisor;
}

uint32_t opt_rem(uint32_t dividend, uint32_t divisor) {
  uint32_t quotient = opt_cal(dividend, divisor);
  return dividend - divisor * quotient;
}

uint32_t normal_rem(uint32_t dividend, uint32_t divisor) {
  return dividend % divisor;
}

void test_div() {
  for (uint32_t dividend = 0; dividend <= static_cast<uint32_t>((1ULL << T) - 1); ++dividend) {
    if (dividend % 1024 == 0) {
      std::cout << "Processing dividend: " << dividend << std::endl;
    }
    for (uint32_t divisor = 1; divisor <= static_cast<uint32_t>((1ULL << T) - 1); ++divisor) {
      uint32_t const result = u32div::opt_cal(dividend, divisor);
      uint32_t const expected = u32div::normal_cal(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " / " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}

void test_rem() {
  for (uint32_t dividend = 0; dividend <= static_cast<uint32_t>((1ULL << T) - 1); ++dividend) {
    if (dividend % 1024 == 0) {
      std::cout << "Processing rem dividend: " << dividend << std::endl;
    }
    for (uint32_t divisor = 1; divisor <= static_cast<uint32_t>((1ULL << T) - 1); ++divisor) {
      uint32_t const result = u32div::opt_rem(dividend, divisor);
      uint32_t const expected = u32div::normal_rem(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " % " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}

void test_large_divisor() {
  uint32_t const max_32 = static_cast<uint32_t>(-1);
  uint32_t const test_dividends[] = {0, 1, 100, max_32, max_32 - 1, max_32 / 2};
  uint32_t const test_divisors[] = {max_32, max_32 - 1, max_32 / 2, (1U << 31), (1U << 31) + 1};

  for (uint32_t dividend : test_dividends) {
    for (uint32_t divisor : test_divisors) {
      uint32_t const result = u32div::opt_cal(dividend, divisor);
      uint32_t const expected = u32div::normal_cal(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " / " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }

  std::cout << "u32div large divisor tests passed!" << std::endl;
}
} // namespace u32div

// =============================================================================
// i32div namespace
// =============================================================================

namespace i32div {

// 32-bit signed division using smull-style: 32x32->64 signed multiply
int32_t opt_cal_signed(int32_t dividend, int32_t divisor) {
  assert(divisor != 0);

  // Handle special case: INT32_MIN / -1 would overflow
  if (dividend == INT32_MIN && divisor == -1) {
    return INT32_MIN;
  }

  if (divisor == 1) {
    return dividend;
  }

  if (divisor == -1) {
    return -dividend;
  }

  int32_t const abs_divisor = divisor < 0 ? -divisor : divisor;

  // Check if divisor is power of 2
  if ((abs_divisor & (abs_divisor - 1)) == 0) {
    uint32_t const shift = ctz(static_cast<uint32_t>(abs_divisor));
    int32_t const sign_correction = (dividend >> 31) & (abs_divisor - 1);
    int32_t q = (dividend + sign_correction) >> shift;
    if (divisor < 0) {
      q = -q;
    }
    return q;
  }

  // For large divisors (absolute value > INT32_MAX/2), quotient is -1, 0, or 1
  if (abs_divisor > (INT32_MAX >> 1)) {
    uint32_t const u_dividend = static_cast<uint32_t>(dividend);
    uint32_t const u_abs_dividend = dividend < 0 ? -u_dividend : u_dividend;
    uint32_t const u_abs_divisor = static_cast<uint32_t>(abs_divisor);

    bool const same_sign = (dividend >= 0) == (divisor >= 0);
    if (u_abs_dividend >= u_abs_divisor) {
      return same_sign ? 1 : -1;
    }
    return 0;
  }

  auto const dm = get_signed_magic(divisor);

  // smull: 32x32 -> 64 signed multiply, take high 32 bits
  int64_t const product = static_cast<int64_t>(dividend) * dm.magic;
  int32_t q = static_cast<int32_t>(product >> 32);

  // Correction for magic overflow
  if (divisor > 0 && dm.magic < 0) {
    q += dividend;
  } else if (divisor < 0 && dm.magic > 0) {
    q -= dividend;
  }

  // Arithmetic shift right
  q >>= dm.shift;

  // Round toward zero correction
  q += static_cast<uint32_t>(q) >> 31;

  return q;
}

int32_t normal_cal(int32_t dividend, int32_t divisor) {
  return dividend / divisor;
}

int32_t opt_rem_signed(int32_t dividend, int32_t divisor) {
  int32_t quotient = opt_cal_signed(dividend, divisor);
  return dividend - divisor * quotient;
}

int32_t normal_rem(int32_t dividend, int32_t divisor) {
  return dividend % divisor;
}

void test_div() {
  int32_t const min_val = -(1 << (T - 1));
  int32_t const max_val = (1 << (T - 1)) - 1;

  for (int32_t dividend = min_val; dividend <= max_val; ++dividend) {
    if ((dividend - min_val) % 1024 == 0) {
      std::cout << "Processing signed dividend: " << dividend << std::endl;
    }
    for (int32_t divisor = min_val; divisor <= max_val; ++divisor) {
      if (divisor == 0)
        continue;

      int32_t const result = i32div::opt_cal_signed(dividend, divisor);
      int32_t const expected = i32div::normal_cal(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " / " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}

void test_rem() {
  int32_t const min_val = -(1 << (T - 1));
  int32_t const max_val = (1 << (T - 1)) - 1;

  for (int32_t dividend = min_val; dividend <= max_val; ++dividend) {
    if ((dividend - min_val) % 1024 == 0) {
      std::cout << "Processing signed rem dividend: " << dividend << std::endl;
    }
    for (int32_t divisor = min_val; divisor <= max_val; ++divisor) {
      if (divisor == 0)
        continue;

      int32_t const result = i32div::opt_rem_signed(dividend, divisor);
      int32_t const expected = i32div::normal_rem(dividend, divisor);
      if (result != expected) {
        std::cout << "Error: " << dividend << " % " << divisor << " = " << result << " instead " << expected << std::endl;
        std::terminate();
      }
    }
  }
}
} // namespace i32div

// =============================================================================
// u64div namespace
// =============================================================================

namespace u64div {

uint64_t opt_cal(uint64_t dividend, uint64_t divisor) {
  if (divisor == 1) {
    return dividend;
  }

  // For large divisors (> UINT64_MAX/2), quotient can only be 0 or 1
  if (divisor > (static_cast<uint64_t>(-1) >> 1)) {
    return dividend >= divisor ? 1 : 0;
  }

  auto const dm = get_unsigned_magic(divisor);

  uint64_t const high = umulh(dividend, dm.magic);

  if (!dm.is_add) {
    // Simple case: just shift
    return high >> dm.shift;
  } else {
    // Correction case: (high + ((dividend - high) >> 1)) >> shift
    uint64_t const t = dividend - high;
    return (high + (t >> 1)) >> dm.shift;
  }
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
}

} // namespace u64div

// =============================================================================
// i64div namespace
// =============================================================================

namespace i64div {

int64_t opt_cal_signed(int64_t dividend, int64_t divisor) {
  assert(divisor != 0);

  // Handle special case: INT64_MIN / -1 would overflow
  if (dividend == INT64_MIN && divisor == -1) {
    return INT64_MIN;
  }

  if (divisor == 1) {
    return dividend;
  }

  if (divisor == -1) {
    return -dividend;
  }

  int64_t const abs_divisor = divisor < 0 ? -divisor : divisor;

  // Check if divisor is power of 2
  if ((abs_divisor & (abs_divisor - 1)) == 0) {
    uint64_t const shift = ctzll(static_cast<uint64_t>(abs_divisor));
    int64_t const sign_correction = (dividend >> 63) & (abs_divisor - 1);
    int64_t q = (dividend + sign_correction) >> shift;
    if (divisor < 0) {
      q = -q;
    }
    return q;
  }

  // For large divisors (absolute value > INT64_MAX/2), quotient is -1, 0, or 1
  if (abs_divisor > (INT64_MAX >> 1)) {
    uint64_t const u_dividend = static_cast<uint64_t>(dividend);
    uint64_t const u_abs_dividend = dividend < 0 ? -u_dividend : u_dividend;
    uint64_t const u_abs_divisor = static_cast<uint64_t>(abs_divisor);

    bool const same_sign = (dividend >= 0) == (divisor >= 0);
    if (u_abs_dividend >= u_abs_divisor) {
      return same_sign ? 1 : -1;
    }
    return 0;
  }

  auto const dm = get_signed_magic(divisor);

  // q = smulh(dividend, magic)
  int64_t q = smulh(dividend, dm.magic);

  // If magic is negative (for positive divisor), add dividend
  // If magic is positive (for negative divisor), subtract dividend
  if (divisor > 0 && dm.magic < 0) {
    q += dividend;
  } else if (divisor < 0 && dm.magic > 0) {
    q -= dividend;
  }

  // Arithmetic shift right
  q >>= dm.shift;

  // Round toward zero correction
  q += static_cast<uint64_t>(q) >> 63;

  return q;
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