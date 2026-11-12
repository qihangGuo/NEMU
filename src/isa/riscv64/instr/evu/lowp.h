#ifndef NEMU_EVU_LOWP_H
#define NEMU_EVU_LOWP_H

#include <stdbool.h>
#include <stdint.h>

typedef struct { uint32_t bits; uint8_t flags; } EvuLowpResult;

static inline uint32_t evu_lowp_round(uint32_t significand, int shift, bool *inexact) {
  if (shift <= 0) { *inexact = false; return significand << -shift; }
  if (shift > 24) { *inexact = significand != 0; return 0; }
  const uint32_t discarded = significand & ((UINT32_C(1) << shift) - 1);
  const uint32_t halfway = UINT32_C(1) << (shift - 1);
  const uint32_t retained = significand >> shift;
  *inexact = discarded != 0;
  return retained + (discarded > halfway || (discarded == halfway && (retained & 1)));
}

static inline EvuLowpResult evu_lowp_decode(uint8_t code) {
  const uint32_t sign = (uint32_t)(code >> 7) << 31;
  const unsigned exponent = (code >> 3) & 15;
  const unsigned fraction = code & 7;
  if (exponent == 15 && fraction == 7) return (EvuLowpResult){0x7fc00000, 0};
  if (exponent) return (EvuLowpResult){sign | ((exponent + 120) << 23) | (fraction << 20), 0};
  if (!fraction) return (EvuLowpResult){sign, 0};
  const unsigned leading = fraction >= 4 ? 2 : fraction >= 2 ? 1 : 0;
  return (EvuLowpResult){sign | ((118 + leading) << 23) |
    ((fraction - (1u << leading)) << (23 - leading)), 0};
}

static inline EvuLowpResult evu_lowp_convert(uint32_t source, unsigned opcode, uint8_t factor) {
  if (opcode == 0x73) return evu_lowp_decode((uint8_t)source);
  const bool sign = source >> 31;
  const unsigned field = (source >> 23) & 255;
  const uint32_t fraction = source & 0x7fffff;
  const bool nan = field == 255 && fraction;
  const bool signaling = nan && !(fraction & 0x400000);
  const bool infinity = field == 255 && !fraction;
  const bool zero = field == 0 && !fraction;
  int exponent = (int)field - 127;
  uint32_t significand = fraction | 0x800000;
  if (!field && fraction) {
    unsigned leading = 0;
    for (uint32_t remaining = fraction; remaining >>= 1;) ++leading;
    exponent = (int)leading - 149;
    significand = fraction << (23 - leading);
  }
  if (opcode == 0x70) {
    if (nan || infinity || exponent > 7)
      return (EvuLowpResult){sign && !nan ? 0x80 : 0x7f, 16};
    if (zero) return (EvuLowpResult){0, 0};
    bool inexact;
    const uint32_t magnitude = evu_lowp_round(significand, 23 - exponent, &inexact);
    if (magnitude > (sign ? 128u : 127u)) return (EvuLowpResult){sign ? 0x80 : 0x7f, 16};
    return (EvuLowpResult){(uint8_t)(sign ? -magnitude : magnitude), inexact ? 1 : 0};
  }
  const bool scaled = opcode == 0x74 || opcode == 0x75;
  const bool fp4 = opcode == 0x75;
  const bool e5m2 = opcode == 0x72;
  const unsigned frac_bits = fp4 ? 1 : e5m2 ? 2 : 3;
  const unsigned exp_bits = fp4 ? 2 : e5m2 ? 5 : 4;
  const int bias = fp4 ? 1 : e5m2 ? 15 : 7;
  const unsigned max_exp = e5m2 ? 30 : (1u << exp_bits) - 1;
  const unsigned max_frac = fp4 ? 1 : e5m2 ? 3 : 6;
  const uint32_t sign_bit = (uint32_t)sign << (exp_bits + frac_bits);
  const uint32_t maximum = sign_bit | (max_exp << frac_bits) | max_frac;
  if (nan || (scaled && factor == 255))
    return (EvuLowpResult){fp4 ? 7 : e5m2 ? 0x7e : 0x7f, fp4 || signaling ? 16 : 0};
  if (infinity) return (EvuLowpResult){e5m2 ? sign_bit | 0x7c : maximum, e5m2 ? 0 : 5};
  if (zero) return (EvuLowpResult){sign_bit, 0};
  if (scaled) exponent -= (int)factor - 127;
  bool inexact;
  if (exponent >= 1 - bias) {
    const uint32_t rounded = evu_lowp_round(significand, 23 - (int)frac_bits, &inexact);
    const unsigned carry = rounded >> (frac_bits + 1);
    const unsigned rounded_frac = rounded & ((1u << frac_bits) - 1);
    const int rounded_exp = exponent + bias + (int)carry;
    if (rounded_exp > (int)max_exp || (rounded_exp == (int)max_exp && rounded_frac > max_frac))
      return (EvuLowpResult){e5m2 ? sign_bit | 0x7c : maximum, 5};
    return (EvuLowpResult){sign_bit | ((uint32_t)rounded_exp << frac_bits) | rounded_frac, inexact ? 1 : 0};
  }
  const uint32_t rounded = evu_lowp_round(significand, 24 - bias - (int)frac_bits - exponent, &inexact);
  const bool normal = rounded >= (1u << frac_bits);
  return (EvuLowpResult){sign_bit | rounded, inexact ? (normal ? 1 : 3) : 0};
}

#endif
