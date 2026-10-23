#include <common.h>

#ifdef CONFIG_CUSTOM_EVU
#include <cpu/cpu.h>
#include <cpu/difftest/evu.h>
#include <memory/host.h>
#include <memory/paddr.h>
#include <softfloat.h>
#include "../../local-include/csr.h"
#include "matrix.h"

enum { MATRIX_DIM = CONFIG_CUSTOM_EVU_VLEN / 32, MATRIX_K = 64,
       MATRIX_ELEMENTS = MATRIX_DIM * MATRIX_DIM, MATRIX_BYTES = MATRIX_ELEMENTS * 4 };

static struct {
  bool active, ready, asynchronous, last;
  uint64_t pc, base;
  uint32_t scale_a[MATRIX_DIM], scale_b[MATRIX_DIM];
  int8_t ab[4][MATRIX_DIM][MATRIX_K];
  uint32_t acc[4][MATRIX_DIM][MATRIX_DIM];
  unsigned ab_rows[4], ab_cols[4];
  bool acc_valid[4];
  uint8_t zero_mask, load_mask;
  unsigned mma_count;
  uint32_t output[MATRIX_ELEMENTS];
  uint8_t flags[MATRIX_ELEMENTS], processed[MATRIX_ELEMENTS], observed[MATRIX_BYTES];
  unsigned processed_count, written;
  uint8_t sticky_flags;
} matrix;

void evu_matrix_reset(void) { memset(&matrix, 0, sizeof(matrix)); }

bool evu_matrix_physical_mode(void) {
  bool physical = cpu.mode == MODE_M && satp->mode == 0 &&
    !(mstatus->mprv && mstatus->mpp != MODE_M);
#ifdef CONFIG_RVH
  physical = physical && !cpu.v;
#endif
  return physical;
}

bool evu_matrix_active(void) { return matrix.active; }
uint8_t evu_matrix_flags(void) { return matrix.sticky_flags; }
uint8_t evu_matrix_take_flags(void) { uint8_t flags = matrix.sticky_flags; matrix.sticky_flags = 0; return flags; }

void evu_matrix_launch(uint64_t pc, uint64_t base, const void *scale_a,
                       const void *scale_b, bool asynchronous) {
  Assert(!matrix.active, "xv.launch requires xv.fence before reusing the matrix stream");
  memset(&matrix, 0, sizeof(matrix));
  matrix.active = true;
  matrix.asynchronous = asynchronous;
  matrix.pc = pc;
  matrix.base = base;
  memcpy(matrix.scale_a, scale_a, sizeof(matrix.scale_a));
  memcpy(matrix.scale_b, scale_b, sizeof(matrix.scale_b));
}

void evu_matrix_zero(uint32_t reg) {
  if (!matrix.active) return;
  Assert(reg >= 4 && reg < 8, "EVU matrix reference currently requires accumulator mzero");
  memset(matrix.acc[reg - 4], 0, sizeof(matrix.acc[reg - 4]));
  matrix.acc_valid[reg - 4] = true;
  matrix.zero_mask |= 1u << reg;
}

void evu_matrix_load(uint32_t reg, uint64_t base, uint64_t stride,
                     unsigned rows, unsigned cols, unsigned size,
                     bool transpose, bool signed_int8) {
  if (!matrix.active) return;
  Assert(evu_matrix_physical_mode() && reg < 4 && size == 0 && signed_int8 &&
         rows <= MATRIX_DIM && cols <= MATRIX_K,
         "EVU matrix reference supports M/Bare signed INT8 tile loads after launch");
  matrix.ab_rows[reg] = rows;
  matrix.ab_cols[reg] = cols;
  matrix.load_mask |= 1u << reg;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      uint64_t address = base + (transpose ? c * stride + r : r * stride + c);
      Assert(address >= base && in_pmem(address), "EVU matrix input is outside physical RAM");
      memcpy(&matrix.ab[reg][r][c], guest_to_host(address), 1);
    }
  }
}

static int32_t signed_word(uint32_t raw) {
  int32_t value;
  memcpy(&value, &raw, sizeof(value));
  return value;
}

void evu_matrix_mma(uint32_t dest, uint32_t src1, uint32_t src2,
                    unsigned rows, unsigned cols, unsigned depth,
                    bool signed_int8_int32, bool saturate) {
  if (!matrix.active) return;
  Assert(!matrix.ready && evu_matrix_physical_mode() && signed_int8_int32 &&
         dest >= 4 && dest < 8 && src1 < 4 && src2 < 4 &&
         rows == MATRIX_DIM && cols == MATRIX_DIM && depth <= MATRIX_K &&
         matrix.acc_valid[dest - 4] && matrix.ab_rows[src1] >= rows &&
         matrix.ab_rows[src2] >= cols && matrix.ab_cols[src1] >= depth &&
         matrix.ab_cols[src2] >= depth,
         "EVU reference supports one full INT8->INT32 MMA with fresh mzero/loads per FP32 launch");
  ++matrix.mma_count;
  // Compute independently of SHARE_REF's lazy adoption of DUT matrix results.
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      uint32_t value = matrix.acc[dest - 4][r][c];
      for (unsigned k = 0; k < depth; ++k) {
        int32_t product = (int32_t)matrix.ab[src1][r][k] * matrix.ab[src2][c][k];
        if (saturate) {
          int64_t sum = (int64_t)signed_word(value) + product;
          value = sum > INT32_MAX ? INT32_MAX : sum < INT32_MIN ? (uint32_t)INT32_MIN : (uint32_t)sum;
        } else value += (uint32_t)product;
      }
      matrix.acc[dest - 4][r][c] = value;
    }
  }
  const uint_fast8_t saved_rm = softfloat_roundingMode;
  const uint_fast8_t saved_flags = softfloat_exceptionFlags;
  const uint_fast8_t saved_tininess = softfloat_detectTininess;
  softfloat_roundingMode = softfloat_round_near_even;
  softfloat_detectTininess = softfloat_tininess_afterRounding;
  const float32_t zero = { .v = 0 };
  for (unsigned n = 0; n < rows; ++n) {
    for (unsigned m = 0; m < cols; ++m) {
      const unsigned i = n * MATRIX_DIM + m;
      softfloat_exceptionFlags = 0;
      const float32_t scale_a = { .v = matrix.scale_a[m] };
      const float32_t scale_b = { .v = matrix.scale_b[n] };
      float32_t value = i32_to_f32(signed_word(matrix.acc[dest - 4][n][m]));
      value = f32_mulAdd(value, scale_a, zero);
      matrix.output[i] = f32_mulAdd(value, scale_b, zero).v;
      matrix.flags[i] = softfloat_exceptionFlags;
    }
  }
  softfloat_roundingMode = saved_rm;
  softfloat_exceptionFlags = saved_flags;
  softfloat_detectTininess = saved_tininess;
  matrix.ready = true;
  if (!matrix.asynchronous) {
    Assert(matrix.base + MATRIX_BYTES >= matrix.base && in_pmem(matrix.base) &&
           in_pmem(matrix.base + MATRIX_BYTES - 1), "EVU matrix output is outside physical RAM");
    memcpy(guest_to_host(matrix.base), matrix.output, MATRIX_BYTES);
    memset(matrix.processed, 1, sizeof(matrix.processed));
    memset(matrix.observed, 1, sizeof(matrix.observed));
    matrix.processed_count = MATRIX_ELEMENTS;
    matrix.written = MATRIX_BYTES;
    matrix.last = true;
    for (unsigned i = 0; i < MATRIX_ELEMENTS; ++i) matrix.sticky_flags |= matrix.flags[i];
  }
}

static int matrix_reject(const char *reason, uint32_t row, uint32_t col,
                         const void *data, size_t bytes, uint64_t mask,
                         uint8_t flags, bool last) {
  fprintf(stderr, "EVU matrix mismatch reason=%s launch_pc=0x%lx active=%d ready=%d "
          "row=%u col=%u dim=%u bytes=%zu mask=0x%lx flags=0x%x last=%d "
          "seen_last=%d processed=%u/%u\n",
          reason, matrix.pc, matrix.active, matrix.ready, row, col,
          (unsigned)MATRIX_DIM, bytes, mask, flags, last, matrix.last,
          matrix.processed_count, (unsigned)MATRIX_ELEMENTS);
  fprintf(stderr, "  REF producer hooks: zero_mask=0x%x load_mask=0x%x mma_count=%u\n",
          matrix.zero_mask, matrix.load_mask, matrix.mma_count);
  if (!data || !bytes || bytes % 4 || bytes > 64) return 1;
  const unsigned lanes = bytes / 4;
  fprintf(stderr, "  DUT FP32 bits:");
  for (unsigned lane = 0; lane < lanes; ++lane) {
    uint32_t value;
    memcpy(&value, (const uint8_t *)data + lane * 4, sizeof(value));
    fprintf(stderr, " %08x", value);
  }
  fprintf(stderr, "\n");
  if (!matrix.ready || row >= MATRIX_DIM || col >= MATRIX_DIM || lanes > MATRIX_DIM - col) return 1;
  const unsigned start = row * MATRIX_DIM + col;
  uint8_t expected_flags = 0;
  fprintf(stderr, "  REF FP32 bits:");
  for (unsigned lane = 0; lane < lanes; ++lane) {
    fprintf(stderr, " %08x", matrix.output[start + lane]);
    expected_flags |= matrix.flags[start + lane];
  }
  fprintf(stderr, "\n  REF mask=0x%lx flags=0x%x last=%d\n",
          (UINT64_C(1) << lanes) - 1, expected_flags,
          matrix.processed_count + lanes == MATRIX_ELEMENTS);
  return 1;
}

int difftest_evu_matrix(uint32_t row, uint32_t col, const void *data, size_t bytes,
                        uint64_t mask, uint8_t flags, bool last) {
#define MATRIX_REJECT(reason) matrix_reject(reason, row, col, data, bytes, mask, flags, last)
  if (!matrix.active) return MATRIX_REJECT("inactive");
  if (!matrix.ready) return MATRIX_REJECT("not-ready");
  if (!data || !bytes || bytes % 4 || bytes > 64) return MATRIX_REJECT("data-size");
  if (row >= MATRIX_DIM || col >= MATRIX_DIM || bytes / 4 > MATRIX_DIM - col)
    return MATRIX_REJECT("coordinates");
  const unsigned lanes = bytes / 4;
  if ((lanes & (lanes - 1)) || col % lanes) return MATRIX_REJECT("alignment");
  const unsigned start = row * MATRIX_DIM + col;
  if (mask != (UINT64_C(1) << lanes) - 1) return MATRIX_REJECT("mask");
  if (matrix.last) return MATRIX_REJECT("after-last");
  uint8_t expected_flags = 0;
  for (unsigned lane = 0; lane < lanes; ++lane) {
    if (matrix.processed[start + lane]) return MATRIX_REJECT("duplicate");
    expected_flags |= matrix.flags[start + lane];
  }
  const bool complete = matrix.processed_count + lanes == MATRIX_ELEMENTS;
  if (memcmp(data, &matrix.output[start], bytes)) return MATRIX_REJECT("data");
  if (flags != expected_flags) return MATRIX_REJECT("flags");
  if (last != complete) return MATRIX_REJECT("last");
#undef MATRIX_REJECT
  memset(&matrix.processed[start], 1, lanes);
  matrix.processed_count += lanes;
  matrix.sticky_flags |= expected_flags;
  matrix.last = last;
  return 0;
}

bool evu_matrix_owns(uint64_t address) {
  return matrix.active && address <= UINT64_MAX - 63 &&
    matrix.base <= UINT64_MAX - MATRIX_BYTES && address < matrix.base + MATRIX_BYTES &&
    address + 63 >= matrix.base;
}

int evu_matrix_store(uint64_t address, const void *data, uint64_t mask) {
  if (!matrix.ready || mask != UINT64_MAX || (address & 63) ||
      !in_pmem(address) || !in_pmem(address + 63)) return 1;
  uint8_t expected[64];
  memcpy(expected, guest_to_host(address), sizeof(expected));
  const uint8_t *actual = data;
  const uint8_t *output = (const uint8_t *)matrix.output;
  bool intersects = false;
  for (unsigned byte = 0; byte < 64; ++byte) {
    uint64_t offset = address + byte - matrix.base;
    if (address + byte < matrix.base || offset >= MATRIX_BYTES || !matrix.processed[offset / 4]) continue;
    intersects = true;
    if (matrix.observed[offset] || actual[byte] == output[offset]) expected[byte] = output[offset];
  }
  if (!intersects || memcmp(data, expected, sizeof(expected))) {
    fprintf(stderr, "EVU matrix physical store/guard mismatch launch_pc=0x%lx addr=0x%lx\n", matrix.pc, address);
    return 1;
  }
  for (unsigned byte = 0; byte < 64; ++byte) {
    uint64_t offset = address + byte - matrix.base;
    if (address + byte >= matrix.base && offset < MATRIX_BYTES && matrix.processed[offset / 4] &&
        !matrix.observed[offset] && actual[byte] == output[offset]) {
      matrix.observed[offset] = 1;
      ++matrix.written;
    }
  }
  memcpy(guest_to_host(address), expected, sizeof(expected));
  return 0;
}

int evu_matrix_drain(void) {
  if (!matrix.active) return 0;
  if (!matrix.ready || !matrix.last || matrix.processed_count != MATRIX_ELEMENTS || matrix.written != MATRIX_BYTES) {
    fprintf(stderr, "EVU matrix drain incomplete launch_pc=0x%lx elements=%u/%u bytes=%u/%u\n",
            matrix.pc, matrix.processed_count, MATRIX_ELEMENTS, matrix.written, MATRIX_BYTES);
    return 1;
  }
  matrix.active = false;
  return 0;
}
#endif
