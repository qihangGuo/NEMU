#ifndef NEMU_EVU_LOWP_H
#define NEMU_EVU_LOWP_H

#include <stdbool.h>
#include <stdint.h>

typedef struct { uint32_t bits; uint8_t flags; } EvuLowpResult;

enum {
  EVU_CVT_U8_F32 = 0x27, EVU_CVT_F32_I8, EVU_CVT_F32_U8, EVU_CVT_E2M1_F32,
  EVU_CVT_F32_E5M2, EVU_CVT_F32_E2M1, EVU_CVT_BF16_F32, EVU_CVT_F32_BF16,
  EVU_CVT_I8_F32 = 0x70, EVU_CVT_E4M3_F32, EVU_CVT_E5M2_F32, EVU_CVT_F32_E4M3,
  EVU_CVT_MXFP8_F32, EVU_CVT_MXFP4_F32,
  EVU_CVT_MXE5M2_F32 = 0x78, EVU_CVT_F32_MXE4M3, EVU_CVT_F32_MXE5M2, EVU_CVT_F32_MXFP4
};
enum { EVLP_NONE, EVLP_I8, EVLP_U8, EVLP_E4M3, EVLP_E5M2, EVLP_E2M1, EVLP_BF16 };
typedef struct { uint8_t format; bool decode, scaled; } EvuLowpOp;

static inline EvuLowpOp evu_lowp_op(unsigned opcode) {
  switch (opcode) {
  case EVU_CVT_I8_F32:     return (EvuLowpOp){EVLP_I8, false, false};
  case EVU_CVT_U8_F32:     return (EvuLowpOp){EVLP_U8, false, false};
  case EVU_CVT_E4M3_F32:   return (EvuLowpOp){EVLP_E4M3, false, false};
  case EVU_CVT_E5M2_F32:   return (EvuLowpOp){EVLP_E5M2, false, false};
  case EVU_CVT_E2M1_F32:   return (EvuLowpOp){EVLP_E2M1, false, false};
  case EVU_CVT_BF16_F32:   return (EvuLowpOp){EVLP_BF16, false, false};
  case EVU_CVT_F32_I8:     return (EvuLowpOp){EVLP_I8, true, false};
  case EVU_CVT_F32_U8:     return (EvuLowpOp){EVLP_U8, true, false};
  case EVU_CVT_F32_E4M3:   return (EvuLowpOp){EVLP_E4M3, true, false};
  case EVU_CVT_F32_E5M2:   return (EvuLowpOp){EVLP_E5M2, true, false};
  case EVU_CVT_F32_E2M1:   return (EvuLowpOp){EVLP_E2M1, true, false};
  case EVU_CVT_F32_BF16:   return (EvuLowpOp){EVLP_BF16, true, false};
  case EVU_CVT_MXFP8_F32:  return (EvuLowpOp){EVLP_E4M3, false, true};
  case EVU_CVT_MXE5M2_F32: return (EvuLowpOp){EVLP_E5M2, false, true};
  case EVU_CVT_MXFP4_F32:  return (EvuLowpOp){EVLP_E2M1, false, true};
  case EVU_CVT_F32_MXE4M3: return (EvuLowpOp){EVLP_E4M3, true, true};
  case EVU_CVT_F32_MXE5M2: return (EvuLowpOp){EVLP_E5M2, true, true};
  case EVU_CVT_F32_MXFP4:  return (EvuLowpOp){EVLP_E2M1, true, true};
  default: return (EvuLowpOp){0};
  }
}

static inline bool evu_low_precision(unsigned opcode) { return evu_lowp_op(opcode).format != EVLP_NONE; }
static inline bool evu_lowp_scaled(unsigned opcode) { return evu_lowp_op(opcode).scaled; }

static inline uint32_t evu_lowp_round(uint32_t significand, int shift, bool *inexact) {
  if (shift <= 0) { *inexact = false; return significand << -shift; }
  if (shift > 24) { *inexact = significand != 0; return 0; }
  const uint32_t discarded = significand & ((UINT32_C(1) << shift) - 1);
  const uint32_t halfway = UINT32_C(1) << (shift - 1);
  const uint32_t retained = significand >> shift;
  *inexact = discarded != 0;
  return retained + (discarded > halfway || (discarded == halfway && (retained & 1)));
}

/* Convert sign * magnitude * 2^power. Low inputs have at most eight significant bits. */
static inline EvuLowpResult evu_lowp_expand(uint32_t sign, uint32_t magnitude, int power) {
  if (!magnitude) return (EvuLowpResult){sign, 0};
  const int leading = 31 - __builtin_clz(magnitude);
  const int exponent = leading + power;
  if (exponent > 127) return (EvuLowpResult){sign | 0x7f800000, 5};
  if (exponent >= -126) return (EvuLowpResult){sign | ((uint32_t)(exponent + 127) << 23) |
    ((magnitude << (23 - leading)) & 0x7fffff), 0};
  bool inexact;
  const uint32_t rounded = evu_lowp_round(magnitude, -149 - power, &inexact);
  return (EvuLowpResult){sign | rounded, inexact ? 3 : 0};
}

static inline EvuLowpResult evu_lowp_widen(uint32_t source, EvuLowpOp op, uint8_t factor) {
  if (op.format == EVLP_BF16) {
    const uint32_t bits = (source & 0xffff) << 16;
    if ((bits & 0x7fffffff) > 0x7f800000)
      return (EvuLowpResult){0x7fc00000, (bits & 0x400000) ? 0 : 16};
    return (EvuLowpResult){bits, 0};
  }
  if (op.format == EVLP_I8 || op.format == EVLP_U8) {
    const unsigned byte = source & 255;
    const bool negative = op.format == EVLP_I8 && byte >= 128;
    return evu_lowp_expand((uint32_t)negative << 31, negative ? 256 - byte : byte, 0);
  }
  const bool fp4 = op.format == EVLP_E2M1, e5m2 = op.format == EVLP_E5M2;
  const int frac = fp4 ? 1 : e5m2 ? 2 : 3, bias = fp4 ? 1 : e5m2 ? 15 : 7;
  const uint32_t code = source & (fp4 ? 7 : 127);
  const uint32_t sign = ((source >> (fp4 ? 3 : 7)) & 1) << 31;
  const bool nan = e5m2 ? code > 0x7c : !fp4 && code == 0x7f;
  if (nan || (op.scaled && factor == 255)) return (EvuLowpResult){0x7fc00000,
    ((e5m2 && nan && !(code & 2)) || fp4) ? 16 : 0};
  if (e5m2 && code == 0x7c) return (EvuLowpResult){sign | 0x7f800000, 0};
  const int exp = (int)(code >> frac);
  const uint32_t magnitude = (code & ((1u << frac) - 1)) + (exp ? 1u << frac : 0);
  return evu_lowp_expand(sign, magnitude, (exp ? exp : 1) - bias - frac +
    (op.scaled ? (int)factor - 127 : 0));
}

static inline EvuLowpResult evu_lowp_convert(uint32_t source, unsigned opcode, uint8_t factor) {
  const EvuLowpOp op = evu_lowp_op(opcode);
  if (op.decode) return evu_lowp_widen(source, op, factor);
  const bool sign = source >> 31;
  const unsigned field = (source >> 23) & 255;
  const uint32_t fraction = source & 0x7fffff;
  const bool nan = field == 255 && fraction;
  const bool signaling = nan && !(fraction & 0x400000);
  const bool infinity = field == 255 && !fraction;
  const bool zero = field == 0 && !fraction;
  if (op.format == EVLP_BF16) {
    if (nan) return (EvuLowpResult){0x7fc0, signaling ? 16 : 0};
    const uint32_t mag = source & 0x7fffffff;
    const uint32_t rounded = (mag + 0x7fff + ((mag >> 16) & 1)) >> 16;
    /* At BF16 precision with an unbounded exponent, this midpoint rounds to minimum normal. */
    const bool tiny = mag < 0x007fc000;
    return (EvuLowpResult){((uint32_t)sign << 15) | rounded,
      !(mag & 0xffff) ? 0 : rounded == 0x7f80 ? 5 : tiny ? 3 : 1};
  }
  int exponent = (int)field - 127;
  uint32_t significand = fraction | 0x800000;
  if (!field && fraction) {
    unsigned leading = 0;
    for (uint32_t remaining = fraction; remaining >>= 1;) ++leading;
    exponent = (int)leading - 149;
    significand = fraction << (23 - leading);
  }
  if (op.format == EVLP_I8 || op.format == EVLP_U8) {
    const bool is_signed = op.format == EVLP_I8;
    const unsigned lower = is_signed ? 128 : 0, upper = is_signed ? 127 : 255;
    if (nan || infinity || exponent > 7)
      return (EvuLowpResult){sign && !nan ? lower : upper, 16};
    if (zero) return (EvuLowpResult){0, 0};
    bool inexact;
    const uint32_t magnitude = evu_lowp_round(significand, 23 - exponent, &inexact);
    if (magnitude > (sign ? lower : upper)) return (EvuLowpResult){sign ? lower : upper, 16};
    return (EvuLowpResult){(uint8_t)(sign ? -magnitude : magnitude), inexact ? 1 : 0};
  }
  const bool scaled = op.scaled;
  const bool fp4 = op.format == EVLP_E2M1;
  const bool e5m2 = op.format == EVLP_E5M2;
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
