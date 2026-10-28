#include <common.h>

#ifdef CONFIG_CUSTOM_EVU
#include "../../local-include/intr.h"
#include "../rvv/vfexp2.h"
#include "evu-internal.h"
#include <cpu/cpu.h>
#include <rtl/fp.h>
#include <rtl/rtl.h>
#include <softfloat.h>
#include <stdint.h>
#include <string.h>

void fp_set_dirty(void);
void vp_set_dirty(void);

typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;

static inline uint64_t width_mask(unsigned bits) { return bits == 64 ? UINT64_MAX : ((UINT64_C(1) << bits) - 1); }

static inline int64_t sign_extend(uint64_t value, unsigned bits) {
  if (bits == 64)
    return (int64_t)value;
  const uint64_t sign = UINT64_C(1) << (bits - 1);
  const uint64_t mask = width_mask(bits);
  value &= mask;
  return (int64_t)((value ^ sign) - sign);
}

static inline uint64_t get_element(const uint8_t *group, unsigned idx, unsigned sew) {
  const unsigned bytes = 1u << sew;
  uint64_t value = 0;
  Assert(bytes <= sizeof(value), "EVU element exceeds 64 bits");
  memcpy(&value, group + idx * bytes, bytes);
  return value;
}

static inline void set_element(uint8_t *group, unsigned idx, unsigned sew, uint64_t value) {
  const unsigned bytes = 1u << sew;
  Assert(bytes <= sizeof(value), "EVU element exceeds 64 bits");
  switch (bytes) {
  case 1:
    group[idx] = (uint8_t)value;
    break;
  case 2: {
    uint16_t tmp = (uint16_t)value;
    memcpy(group + idx * 2, &tmp, sizeof(tmp));
    break;
  }
  case 4: {
    uint32_t tmp = (uint32_t)value;
    memcpy(group + idx * 4, &tmp, sizeof(tmp));
    break;
  }
  case 8:
    memcpy(group + idx * 8, &value, sizeof(value));
    break;
  default:
    Assert(0, "invalid EVU element width");
  }
}

EvuVType evu_current_vtype(void) {
  EvuVType result;
  result.sew = (cpu.evu.vtype >> 3) & 0x7;
  result.lmul = cpu.evu.vtype & 0x7;
  result.vta = (cpu.evu.vtype >> 6) & 0x1;
  result.vma = (cpu.evu.vtype >> 7) & 0x1;
  result.vill = (cpu.evu.vtype >> 63) & 0x1;
  return result;
}

uint32_t evu_vlmax(uint32_t sew, uint32_t lmul) {
  if (sew > 3 || lmul == 4)
    return 0;
  const uint32_t elems_per_reg = EVU_VLEN_BITS / (8u << sew);
  if (lmul <= 3)
    return elems_per_reg * (1u << lmul);
  return elems_per_reg >> (8 - lmul);
}

uint8_t evu_lmul_regs(uint32_t lmul) { return lmul <= 3 ? (uint8_t)(1u << lmul) : 1; }

uint32_t evu_current_vl(uint32_t sew) {
  const EvuVType vt = evu_current_vtype();
  const uint32_t vlmax = evu_vlmax(sew, vt.lmul);
  if (!cpu.evu.configured)
    return vlmax;
  return cpu.evu.vl < vlmax ? (uint32_t)cpu.evu.vl : vlmax;
}

static unsigned source_regs(const EvuJob *job) { return evu_lmul_regs(evu_current_vtype().lmul); }

static unsigned widening_dest_regs(uint32_t lmul) {
  if (lmul > 3)
    return 1;
  return 2u * (1u << lmul);
}

static void read_group(uint8_t *dst, uint8_t reg, unsigned regs) {
  if (reg + regs > 32 || (regs > 1 && (reg % regs) != 0))
    longjmp_exception(EX_II);
  memset(dst, 0, EVU_VLEN_BYTES * regs);
  for (unsigned i = 0; i < regs; ++i)
    evu_read_register(reg + i, dst + i * EVU_VLEN_BYTES);
}

static void set_write_group(EvuJob *job, uint8_t reg, unsigned regs) {
  if (reg + regs > 32 || (regs > 1 && (reg % regs) != 0))
    longjmp_exception(EX_II);
  job->writes_vrf = true;
  job->write_reg = reg;
  job->write_regs = regs;
  job->output_bytes = regs * EVU_VLEN_BYTES;
}

static void fill_tail(uint8_t *output, uint32_t start, uint32_t vlmax, unsigned sew, bool agnostic) {
  // Tail-agnostic permits preserving the old destination. The current EVU
  // sequencers inhibit writes beyond VL, so preserve those bytes here.
  (void)output;
  (void)start;
  (void)vlmax;
  (void)sew;
  (void)agnostic;
}

static int sat_signed(__int128_t value, unsigned bits, uint64_t *result) {
  const int128_t min = -((int128_t)1 << (bits - 1));
  const int128_t max = ((int128_t)1 << (bits - 1)) - 1;
  if (value > max) {
    *result = width_mask(bits) >> 1;
    return 1;
  }
  if (value < min) {
    *result = UINT64_C(1) << (bits - 1);
    return 1;
  }
  *result = (uint64_t)(int64_t)value & width_mask(bits);
  return 0;
}

static rtlreg_t fp_binary(Decode *s, int op, int fp_type, rtlreg_t a, rtlreg_t b) {
  (void)s;
  // Pure SoftFloat operations: EVU fixed RNE never checks or changes core FS/FCSR.
#define FP_BINARY(width) do { \
    float##width##_t x = { .v = a }, y = { .v = b }; \
    switch (op) { \
    case FPCALL_ADD: return f##width##_add(x, y).v; \
    case FPCALL_SUB: return f##width##_sub(x, y).v; \
    case FPCALL_MUL: return f##width##_mul(x, y).v; \
    case FPCALL_DIV: return f##width##_div(x, y).v; \
    case FPCALL_MIN: case FPCALL_MAX: { \
      const unsigned bits = width; \
      const uint64_t magnitude = (UINT64_C(1) << (bits - 1)) - 1; \
      const uint64_t infinity = width == 16 ? 0x7c00 : 0x7f800000; \
      const bool nx = (a & magnitude) > infinity, ny = (b & magnitude) > infinity; \
      const bool less = f##width##_lt_quiet(x, y); \
      const bool equal = f##width##_eq(x, y); \
      if (nx && ny) return width == 16 ? 0x7e00 : 0x7fc00000; \
      if (nx) return b; \
      if (ny) return a; \
      if (equal) return op == FPCALL_MIN ? (a | b) : (a & b); \
      return (op == FPCALL_MIN ? less : !less) ? a : b; \
    } \
    default: Assert(0, "unsupported EVU FP operation"); \
    } \
  } while (0)
  if (fp_type == FPCALL_W16) FP_BINARY(16);
  else FP_BINARY(32);
#undef FP_BINARY
  return 0;
}

static rtlreg_t fp_macc(Decode *s, int fp_type, rtlreg_t a, rtlreg_t b, rtlreg_t acc) {
  (void)s;
  if (fp_type == FPCALL_W16) return f16_mulAdd((float16_t){a}, (float16_t){b}, (float16_t){acc}).v;
  return f32_mulAdd((float32_t){a}, (float32_t){b}, (float32_t){acc}).v;
}

static void execute_integer_ew(Decode *s, EvuJob *job, const uint8_t *vs2, const uint8_t *vs1, uint8_t *output,
                               uint32_t vl, uint32_t vlmax) {
  (void)s;
  const unsigned bits = 8u << job->sew;
  const uint64_t mask = width_mask(bits);
  for (uint32_t idx = job->vstart; idx < vl; ++idx) {
    const uint64_t a = get_element(vs2, idx, job->sew);
    const uint64_t b = get_element(vs1, idx, job->sew);
    const int64_t sa = sign_extend(a, bits);
    const int64_t sb = sign_extend(b, bits);
    uint64_t result = 0;
    switch (job->opcode) {
    case EVU_ADD:
      result = a + b;
      break;
    case EVU_SUB:
      result = a - b;
      break;
    case EVU_AND:
      result = a & b;
      break;
    case EVU_OR:
      result = a | b;
      break;
    case EVU_XOR:
      result = a ^ b;
      break;
    case EVU_MUL:
      result = a * b;
      break;
    case EVU_MAC:
      result = a * b + get_element(output, idx, job->sew);
      break;
    case EVU_SLL:
      result = a << (b & (bits - 1));
      break;
    case EVU_SRL:
      result = a >> (b & (bits - 1));
      break;
    case EVU_SRA:
      result = (uint64_t)(sa >> (b & (bits - 1)));
      break;
    case EVU_MIN:
      result = sa < sb ? a : b;
      break;
    case EVU_MAX:
      result = sa > sb ? a : b;
      break;
    case EVU_MINU:
      result = a < b ? a : b;
      break;
    case EVU_MAXU:
      result = a > b ? a : b;
      break;
    case EVU_SATADD:
      job->vxsat |= sat_signed((int128_t)sa + sb, bits, &result);
      break;
    case EVU_SATSUB:
      job->vxsat |= sat_signed((int128_t)sa - sb, bits, &result);
      break;
    default:
      Assert(0, "unexpected EVU integer operation 0x%x", job->opcode);
    }
    set_element(output, idx, job->sew, result & mask);
  }
  fill_tail(output, vl, vlmax, job->sew, evu_current_vtype().vta);
}

static void execute_widening(EvuJob *job, const uint8_t *vs2, const uint8_t *vs1, uint8_t *output, uint32_t vl,
                             uint32_t vlmax) {
  const unsigned src_bits = 8u << job->sew;
  const unsigned dst_sew = job->sew + 1;
  const unsigned dst_bits = src_bits * 2;
  for (uint32_t idx = job->vstart; idx < vl; ++idx) {
    const int64_t a = sign_extend(get_element(vs2, idx, job->sew), src_bits);
    const int64_t b = sign_extend(get_element(vs1, idx, job->sew), src_bits);
    int128_t result = 0;
    switch (job->opcode) {
    case EVU_WADD:
      result = (int128_t)a + b;
      break;
    case EVU_WSUB:
      result = (int128_t)a - b;
      break;
    case EVU_WMUL:
      result = (int128_t)a * b;
      break;
    default:
      Assert(0, "unexpected EVU widening operation 0x%x", job->opcode);
    }
    set_element(output, idx, dst_sew, (uint64_t)result & width_mask(dst_bits));
  }
  fill_tail(output, vl, vlmax, dst_sew, evu_current_vtype().vta);
}

static void execute_fp_ew(Decode *s, EvuJob *job, const uint8_t *vs2, const uint8_t *vs1, uint8_t *output, uint32_t vl,
                          uint32_t vlmax) {
  const int fp_type = job->sew == 1 ? FPCALL_W16 : FPCALL_W32;
  for (uint32_t idx = job->vstart; idx < vl; ++idx) {
    const rtlreg_t rhs = get_element(vs1, (job->aux & 0x20) ? 0 : idx, job->sew);
    const rtlreg_t lhs = get_element(vs2, idx, job->sew);
    rtlreg_t result = 0;
    switch (job->opcode) {
    case EVU_FADD:
      result = fp_binary(s, FPCALL_ADD, fp_type, lhs, rhs);
      break;
    case EVU_FSUB:
      result = fp_binary(s, FPCALL_SUB, fp_type, lhs, rhs);
      break;
    case EVU_FMUL:
      result = fp_binary(s, FPCALL_MUL, fp_type, lhs, rhs);
      break;
    case EVU_FDIV:
      result = fp_binary(s, FPCALL_DIV, fp_type, lhs, rhs);
      break;
    case EVU_FMIN:
      result = fp_binary(s, FPCALL_MIN, fp_type, lhs, rhs);
      break;
    case EVU_FMAX:
      result = fp_binary(s, FPCALL_MAX, fp_type, lhs, rhs);
      break;
    case EVU_FMACC:
      result = fp_macc(s, fp_type, lhs, rhs, get_element(output, idx, job->sew));
      break;
    default:
      Assert(0, "unexpected EVU FP operation 0x%x", job->opcode);
    }
    set_element(output, idx, job->sew, result);
  }
  fill_tail(output, vl, vlmax, job->sew, evu_current_vtype().vta);
}

static void execute_reduce(EvuJob *job, const uint8_t *vs2, const uint8_t *vs1, uint8_t *output, uint32_t vl) {
  if (vl == 0)
    return;
  const unsigned bits = 8u << job->sew;
  const uint64_t mask = width_mask(bits);
  uint64_t result = get_element(vs1, 0, job->sew);
  for (uint32_t idx = job->vstart; idx < vl; ++idx) {
    const uint64_t value = get_element(vs2, idx, job->sew);
    switch (job->opcode) {
    case EVU_REDUCE_SUM:
      result += value;
      break;
    case EVU_REDUCE_AND:
      result &= value;
      break;
    case EVU_REDUCE_OR:
      result |= value;
      break;
    case EVU_REDUCE_XOR:
      result ^= value;
      break;
    case EVU_REDUCE_MIN:
      result = sign_extend(value, bits) < sign_extend(result, bits) ? value : result;
      break;
    case EVU_REDUCE_MAX:
      result = sign_extend(value, bits) > sign_extend(result, bits) ? value : result;
      break;
    default:
      Assert(0, "unexpected EVU reduction operation 0x%x", job->opcode);
    }
  }
  set_element(output, 0, job->sew, result & mask);
}

static rtlreg_t fp_unordered_sum(Decode *s, const EvuJob *job, const uint8_t *vs2, rtlreg_t seed, uint32_t vl) {
  const int fp_type = job->sew == 1 ? FPCALL_W16 : FPCALL_W32;
  const unsigned lanes = CONFIG_CUSTOM_EVU_DLEN / (8u << job->sew);
  enum { ACC_SLOTS = 4 };
  rtlreg_t acc[ACC_SLOTS][CONFIG_CUSTOM_EVU_DLEN / 16] = {{0}};
  Assert(lanes && !(lanes & (lanes - 1)), "EVU FP reduction requires a power-of-two lane count");
  acc[0][0] = seed;
  // Match Saturn's four round-robin EG accumulators, then slot merges and
  // upper-half folds. For DLEN=256/FP32, source i goes to slot (i/8)%4,
  // lane i%8. Merge slots 1,2,3 into slot 0, then fold lanes by 4,2,1.
  // Use SoftFloat at the element width for EVERY add, preserving rounding
  // and exception flags. A serial or host-double sum is not bit equivalent.
  for (uint32_t idx = job->vstart; idx < vl; ++idx) {
    const unsigned slot = (idx / lanes) % ACC_SLOTS;
    const unsigned lane = idx % lanes;
    acc[slot][lane] = fp_binary(s, FPCALL_ADD, fp_type, get_element(vs2, idx, job->sew), acc[slot][lane]);
  }
  for (unsigned slot = 1; slot < ACC_SLOTS; ++slot)
    for (unsigned lane = 0; lane < lanes; ++lane)
      acc[0][lane] = fp_binary(s, FPCALL_ADD, fp_type, acc[slot][lane], acc[0][lane]);
  for (unsigned half = lanes / 2; half; half /= 2)
    for (unsigned lane = 0; lane < half; ++lane)
      acc[0][lane] = fp_binary(s, FPCALL_ADD, fp_type, acc[0][lane], acc[0][lane + half]);
  return acc[0][0];
}

static void execute_fp_reduce(Decode *s, EvuJob *job, const uint8_t *vs2, const uint8_t *vs1, uint8_t *output,
                              uint32_t vl) {
  if (vl == 0)
    return;
  const int fp_type = job->sew == 1 ? FPCALL_W16 : FPCALL_W32;
  rtlreg_t result = get_element(vs1, 0, job->sew);
  if (job->opcode == EVU_FREDUCE_SUM) {
    set_element(output, 0, job->sew, fp_unordered_sum(s, job, vs2, result, vl));
    return;
  }
  for (uint32_t idx = job->vstart; idx < vl; ++idx) {
    const rtlreg_t value = get_element(vs2, idx, job->sew);
    switch (job->opcode) {
    case EVU_FREDUCE_MIN:
      result = fp_binary(s, FPCALL_MIN, fp_type, result, value);
      break;
    case EVU_FREDUCE_MAX:
      result = fp_binary(s, FPCALL_MAX, fp_type, result, value);
      break;
    default:
      Assert(0, "unexpected EVU FP reduction operation 0x%x", job->opcode);
    }
  }
  set_element(output, 0, job->sew, result);
}

static void execute_permute(EvuJob *job, const uint8_t *vs2, const uint8_t *vs1, uint8_t *output, uint32_t vl,
                            uint32_t vlmax) {
  const uint64_t offset = job->opcode == EVU_GATHER_IMM ? job->vs1 : get_element(vs1, 0, job->sew);
  for (uint32_t idx = job->vstart; idx < vl; ++idx) {
    uint64_t result = 0;
    if (job->opcode == EVU_GATHER || job->opcode == EVU_GATHER_IMM) {
      const uint64_t index = job->opcode == EVU_GATHER_IMM ? offset : get_element(vs1, idx, job->sew);
      if (index < vlmax)
        result = get_element(vs2, (uint32_t)index, job->sew);
    } else if (job->opcode == EVU_SLIDE_UP) {
      result = idx >= offset ? get_element(vs2, idx - offset, job->sew) : get_element(output, idx, job->sew);
    } else if (job->opcode == EVU_SLIDE_DOWN) {
      result = (uint64_t)idx + offset < vlmax ? get_element(vs2, idx + offset, job->sew) : 0;
    } else {
      Assert(0, "unexpected EVU permute operation 0x%x", job->opcode);
    }
    set_element(output, idx, job->sew, result);
  }
  fill_tail(output, vl, vlmax, job->sew, evu_current_vtype().vta);
}

static void execute_exp2(Decode *s, EvuJob *job, const uint8_t *vs2, uint8_t *output, uint32_t vl, uint32_t vlmax) {
  (void)s;
  for (uint32_t idx = job->vstart; idx < vl; ++idx) {
    vfexp2_result_t result = vfexp2_compute(get_element(vs2, idx, job->sew), job->sew, false, 0);
    set_element(output, idx, job->sew, result.result);
    job->flags |= result.fflags;
  }
  fill_tail(output, vl, vlmax, job->sew, evu_current_vtype().vta);
}

/* FP probability quantization deliberately rounds multiplication before the
 * add, matching RVV's separate vfmul/vfadd, then minNum/clamp and RTZ. */
static void execute_fp_quant_to_cute(EvuJob *job, const uint8_t *input, const uint8_t *scale, uint32_t vl) {
  const uint_fast8_t saved_flags = softfloat_exceptionFlags;
  const uint_fast8_t saved_rounding = softfloat_roundingMode;
  const uint_fast8_t saved_tininess = softfloat_detectTininess;
  softfloat_exceptionFlags = 0;
  softfloat_roundingMode = softfloat_round_near_even;
  softfloat_detectTininess = softfloat_tininess_afterRounding;
  const float32_t zero = { .v = 0 }, half = { .v = 0x3f000000 }, limit = { .v = 0x42fe0000 };
  for (unsigned i = 0; i < vl; ++i) {
    const float32_t a = { .v = (uint32_t)get_element(input, i, 2) };
    const float32_t b = { .v = (uint32_t)get_element(scale, (job->aux & 0x10) ? 0 : i, 2) };
    float32_t value = f32_add(f32_mul(a, b), half);
    // lt_quiet agrees with HardFloat signaling=false (sNaN alone sets NV).
    if (!(f32_lt_quiet(value, limit) || f32_eq(value, limit))) value = limit;
    if (f32_lt_quiet(value, zero)) value = zero;
    job->output[i] = (uint8_t)f32_to_ui32(value, softfloat_round_minMag, true);
  }
  job->output_bytes = vl;
  job->flags = (uint8_t)softfloat_exceptionFlags;
  job->uses_fp = true;
  job->writes_cute = true;
  softfloat_exceptionFlags = saved_flags;
  softfloat_roundingMode = saved_rounding;
  softfloat_detectTininess = saved_tininess;
}

/* Reserved descriptor context slots have no execution consumer. */
void evu_compute_reset(void) { memset(cpu.evu.descriptors, 0, sizeof(cpu.evu.descriptors)); }

void evu_compute_execute(Decode *s, EvuJob *job) {
  const EvuVType vt = evu_current_vtype();
  if (job->opcode == EVU_FENCE || job->opcode == EVU_WAIT || job->opcode == EVU_LAUNCH)
    return;
  const unsigned regs = source_regs(job);
  const uint32_t vlmax = evu_vlmax(job->sew, vt.lmul);
  const uint32_t vl = job->vl < vlmax ? job->vl : vlmax;
  const bool reduction = job->opcode == EVU_REDUCE_SUM || job->opcode == EVU_REDUCE_AND ||
    job->opcode == EVU_REDUCE_OR || job->opcode == EVU_REDUCE_XOR ||
    job->opcode == EVU_REDUCE_MIN || job->opcode == EVU_REDUCE_MAX ||
    job->opcode == EVU_FREDUCE_SUM || job->opcode == EVU_FREDUCE_MIN || job->opcode == EVU_FREDUCE_MAX;
  uint8_t vs2[EVU_MAX_GROUP_BYTES];
  uint8_t vs1[EVU_MAX_GROUP_BYTES];
  read_group(vs2, job->vs2, regs);
  if (job->opcode != EVU_EXP2 && job->opcode != EVU_GATHER_IMM)
    read_group(vs1, job->vs1, reduction ? 1 : regs);

  if (job->opcode == EVU_FP_QUANT_TO_CUTE) {
    uint8_t inverse_scale[EVU_MAX_GROUP_BYTES];
    read_group(inverse_scale, job->vd, 1);
    if (vl != 0) execute_fp_quant_to_cute(job, vs2, inverse_scale, vl);
    return;
  }

  unsigned dest_regs = regs;
  if (job->opcode == EVU_WADD || job->opcode == EVU_WSUB || job->opcode == EVU_WMUL)
    dest_regs = widening_dest_regs(vt.lmul);
  if (dest_regs > 8) longjmp_exception(EX_II);
  if (reduction)
    dest_regs = 1;
  set_write_group(job, job->vd, dest_regs);
  read_group(job->output, job->vd, dest_regs);
  const int64_t saved_flags = softfloat_exceptionFlags;
  const uint_fast8_t saved_rounding = softfloat_roundingMode;
  const uint_fast8_t saved_tininess = softfloat_detectTininess;
  softfloat_exceptionFlags = 0;
  softfloat_roundingMode = softfloat_round_near_even;
  softfloat_detectTininess = softfloat_tininess_afterRounding;

  switch (job->opcode) {
  case EVU_ADD:
  case EVU_SUB:
  case EVU_AND:
  case EVU_OR:
  case EVU_XOR:
  case EVU_MUL:
  case EVU_MAC:
  case EVU_SLL:
  case EVU_SRL:
  case EVU_SRA:
  case EVU_MIN:
  case EVU_MAX:
  case EVU_MINU:
  case EVU_MAXU:
  case EVU_SATADD:
  case EVU_SATSUB:
    execute_integer_ew(s, job, vs2, vs1, job->output, vl, vlmax);
    break;
  case EVU_WADD:
  case EVU_WSUB:
  case EVU_WMUL:
    execute_widening(job, vs2, vs1, job->output, vl, vlmax);
    break;
  case EVU_FADD:
  case EVU_FSUB:
  case EVU_FMUL:
  case EVU_FMACC:
  case EVU_FDIV:
  case EVU_FMIN:
  case EVU_FMAX:
    execute_fp_ew(s, job, vs2, vs1, job->output, vl, vlmax);
    job->uses_fp = true;
    break;
  case EVU_EXP2:
    execute_exp2(s, job, vs2, job->output, vl, vlmax);
    job->uses_fp = true;
    break;
  case EVU_REDUCE_SUM:
  case EVU_REDUCE_AND:
  case EVU_REDUCE_OR:
  case EVU_REDUCE_XOR:
  case EVU_REDUCE_MIN:
  case EVU_REDUCE_MAX:
    execute_reduce(job, vs2, vs1, job->output, vl);
    break;
  case EVU_FREDUCE_SUM:
  case EVU_FREDUCE_MIN:
  case EVU_FREDUCE_MAX:
    execute_fp_reduce(s, job, vs2, vs1, job->output, vl);
    job->uses_fp = true;
    break;
  case EVU_GATHER:
  case EVU_GATHER_IMM:
  case EVU_SLIDE_UP:
  case EVU_SLIDE_DOWN:
    execute_permute(job, vs2, vs1, job->output, vl, vlmax);
    break;
  default:
    Assert(0, "EVU reference implementation missing opcode 0x%x", job->opcode);
  }

  job->flags |= (uint8_t)softfloat_exceptionFlags;
  softfloat_exceptionFlags = saved_flags;
  softfloat_roundingMode = saved_rounding;
  softfloat_detectTininess = saved_tininess;
}

void evu_publish_completion_flags(EvuJob *job) {
  if (job->cause && !(cpu.evu.status & 2)) {
    cpu.evu.status = 2 | (job->fault_address_valid ? 4 : 0);
    cpu.evu.cause = job->cause;
    cpu.evu.epc = job->pc;
    cpu.evu.tval = job->fault_address_valid ? job->fault_address : 0;
  }
  cpu.evu.fflags |= job->flags;
  cpu.evu.vxsat |= job->vxsat;
  if (job->cause || job->flags || job->vxsat || job->writes_vrf || job->updates_vector_state)
    evu_context_dirty();
}

#endif
