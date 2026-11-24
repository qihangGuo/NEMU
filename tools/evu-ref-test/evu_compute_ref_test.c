/* Native regression for the NEMU EVU vector execution model. */
#include <dlfcn.h>
#include <isa.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "evu/lowp.h"

enum { VLEN_BITS = CONFIG_CUSTOM_EVU_VLEN, VLEN_BYTES = VLEN_BITS / 8 };
_Static_assert(CONFIG_CUSTOM_EVU_VLEN == 4096, "Use riscv64-xs-evu-ref_defconfig");

#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      fprintf(stderr, "Failed at %s:%d: %s\n", __FILE__, __LINE__, #condition);                                        \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  } while (0)

static void (*copy_mem)(uint64_t, void *, size_t, bool);
static void (*copy_regs)(void *, bool);
static void (*copy_csrs)(void *, bool);
static void (*execute)(uint64_t);
static void (*set_async)(bool);
static int (*pending)(void);
static int (*get_reg)(uint32_t, void *, size_t);
static int (*get_csr)(void *, size_t);
static int (*inject_fault)(uint64_t, uint8_t, uint64_t, bool);
static void (*write_evu_csr)(uint32_t, uint64_t);
static int (*store_event)(uint64_t, const void *, uint64_t);
static int (*complete)(uint64_t, uint32_t, uint8_t, bool, uint8_t);
static CPU_state initial;
static FILE *softmax_trace;
static uint64_t code = UINT64_C(0x80008000);

static uint32_t insn(unsigned f7, unsigned f3, unsigned rd, unsigned rs1, unsigned rs2) {
  return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x0b;
}

static uint32_t setvli_insn(unsigned type, unsigned rd, unsigned rs1) {
  return insn(0x45, (type >> 5) & 7, rd, rs1, type & 31);
}

static uint64_t config_type_input;

static CPU_state run_raw(uint32_t instruction, uint64_t base) {
  code += 4;
  CPU_state state = initial;
  state.pc = code;
  state.mcause = 0;
  state.gpr[10]._64 = base;
  state.gpr[11]._64 = config_type_input;
  copy_mem(code, &instruction, sizeof(instruction), true);
  copy_regs(&state, true);
  execute(1);
  copy_regs(&state, false);
  return state;
}

static uint64_t run(uint32_t instruction, uint64_t base) {
  if (softmax_trace) write_evu_csr(0x8c1, 0);
  CPU_state state = run_raw(instruction, base);
  CHECK(state.mcause == 0 && state.pc == code + 4);
  if (softmax_trace) {
    uint32_t values[VLEN_BYTES / 4];
    const unsigned vd = (instruction >> 7) & 31;
    CHECK(get_reg(vd, values, sizeof(values)) == 0);
    uint64_t evu_state[13];
    CHECK(get_csr(evu_state, sizeof(evu_state)) == 0);
    fprintf(softmax_trace, "%02x %u %u %u %u %02x", instruction >> 25, (instruction >> 12) & 7,
            vd, (instruction >> 15) & 31, (instruction >> 20) & 31, (unsigned)evu_state[5]);
    for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
      fprintf(softmax_trace, " %08x", values[i]);
    fputc('\n', softmax_trace);
  }
  return code;
}

static void evu(uint32_t instruction, uint64_t base, uint8_t flags) {
  const uint64_t pc = run(instruction, base);
  CHECK(complete(pc, instruction, flags, false, 0) == 0);
}

static void load_reg(unsigned reg, uint64_t base, unsigned sew) { evu(insn(0x10, sew, reg, 10, 0), base, 0); }

static void read_reg(unsigned reg, void *data) { CHECK(get_reg(reg, data, VLEN_BYTES) == 0); }

static uint32_t float_bits(float value) {
  uint32_t result;
  memcpy(&result, &value, sizeof(result));
  return result;
}

static int8_t sat8(int value) {
  if (value > INT8_MAX)
    return INT8_MAX;
  if (value < INT8_MIN)
    return INT8_MIN;
  return (int8_t)value;
}

static void check_slides(void) {
  uint8_t source[2 * VLEN_BYTES], sentinel[3 * VLEN_BYTES];
  uint8_t expected[3 * VLEN_BYTES], actual[3 * VLEN_BYTES];
  uint64_t saved[4096];
  copy_csrs(saved, false);
  for (unsigned byte = 0; byte < sizeof(source); ++byte) source[byte] = (uint8_t)(byte * 29 + byte / 251 + 7);
  memset(sentinel, 0xa5, sizeof(sentinel));
  copy_mem(UINT64_C(0x80020000), source, sizeof(source), true);
  copy_mem(UINT64_C(0x80021000), sentinel, sizeof(sentinel), true);
  const unsigned lmuls[] = {0, 1, 7};
  unsigned cases = 0;
  for (unsigned sew = 0; sew < 3; ++sew) {
    const unsigned width = 1u << sew;
    for (unsigned mode = 0; mode < sizeof(lmuls) / sizeof(lmuls[0]); ++mode) {
      const unsigned lmul = lmuls[mode];
      const unsigned vlmax = (VLEN_BYTES / width) * (lmul == 1 ? 2 : 1) / (lmul == 7 ? 2 : 1);
      const uint64_t offsets[] = {0, 1, 64 / width - 1, 64 / width, 64 / width + 1,
        vlmax - 1, vlmax, vlmax + 1, UINT64_C(1) << 32, UINT64_MAX, 1, 1};
      for (unsigned down = 0; down < 2; ++down) {
        for (unsigned index = 0; index < sizeof(offsets) / sizeof(offsets[0]); ++index) {
          const unsigned vl = index == 10 ? 0 : index == 0 || index == 11 ? vlmax : vlmax - 3;
          const bool alias = down && index == 11;
          const unsigned dest = alias ? 4 : 8;
          const uint64_t offset = offsets[index];
          evu(setvli_insn(0, 0, 10), VLEN_BYTES, 0);
          load_reg(4, UINT64_C(0x80020000), 0);
          load_reg(5, UINT64_C(0x80020000) + VLEN_BYTES, 0);
          for (unsigned reg = 0; reg < 3; ++reg) load_reg(8 + reg, UINT64_C(0x80021000), 0);
          load_reg(6, UINT64_C(0x80021000), 0);
          memcpy(expected, sentinel, sizeof(expected));
          if (alias) memcpy(expected, source, sizeof(source));
          for (unsigned element = 0; element < vl; ++element) {
            if (!down && element < offset) continue;
            uint8_t *output = expected + element * width;
            if (down && (offset >= vlmax || element >= vlmax - offset)) memset(output, 0, width);
            else memcpy(output, source + (down ? element + offset : element - offset) * width, width);
          }
          evu(setvli_insn((sew << 3) | lmul, 0, 10), vl, 0);
          write_evu_csr(0x8c1, 21);
          write_evu_csr(0x8c2, 1);
          const uint64_t configuration = (UINT64_C(1) << 40) | ((uint64_t)((sew << 3) | lmul) << 32) | vl;
          config_type_input = offset;
          evu(insn(down ? 0x3f : 0x3e, sew, dest, index == 0 ? 0 : 11, 4), ~offset, 0);
          for (unsigned reg = 0; reg < 3; ++reg) read_reg(dest + reg, actual + reg * VLEN_BYTES);
          CHECK(memcmp(expected, actual, sizeof(actual)) == 0);
          uint64_t state[4096];
          copy_csrs(state, false);
          CHECK(state[0x8c0] == configuration && state[0x8c1] == 21 && state[0x8c2] == 1);
          ++cases;
        }
      }
    }
  }
  for (unsigned empty = 0; empty < 2; ++empty) {
    evu(setvli_insn(0x10, 0, 10), empty ? 0 : 13, 0);
    for (unsigned sew = 0; sew < 8; ++sew) {
      CPU_state rejected = run_raw(insn(0x19, sew, 0, 0, 0), 0);
      CHECK(rejected.mcause == 2 && pending() == 0);
    }
    CPU_state rejected = run_raw(insn(0x3e, 2, 4, 10, 4), 0);
    CHECK(rejected.mcause == 2 && pending() == 0);
    rejected = run_raw(insn(0x3f, 3, 8, 10, 4), 0);
    CHECK(rejected.mcause == 2 && pending() == 0);
    evu(setvli_insn(0x11, 0, 10), empty ? 0 : 13, 0);
    const unsigned groups[][2] = {{5, 4}, {8, 5}, {31, 4}, {8, 31}};
    for (unsigned group = 0; group < sizeof(groups) / sizeof(groups[0]); ++group) {
      rejected = run_raw(insn(0x3f, 2, groups[group][0], 10, groups[group][1]), 1);
      CHECK(rejected.mcause == 2 && pending() == 0);
    }
  }
  copy_csrs(saved, true);
  config_type_input = 0;
  printf("EVU slide: %u value/configuration cases and illegal encoding/group cases passed\n", cases);
}

static void check_rvv_math(const char *trace_path) {
  FILE *trace = trace_path ? fopen(trace_path, "w") : NULL;
  CHECK(!trace_path || trace);
  uint8_t source[4 * VLEN_BYTES], other[2 * VLEN_BYTES], sentinel[4 * VLEN_BYTES];
  uint8_t expected[4 * VLEN_BYTES], actual[4 * VLEN_BYTES];
  memset(sentinel, 0xa5, sizeof(sentinel));
  uint64_t saved[4096];
  copy_csrs(saved, false);
  const unsigned ops[] = {0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
                          0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e};
  unsigned cases = 0;
  for (unsigned operation = 0; operation < sizeof(ops) / sizeof(ops[0]); ++operation) {
    const unsigned op = ops[operation];
    const bool scalar = op == 0x62 || op == 0x63 || op == 0x65;
    const bool fp_unary = op == 0x60 || op == 0x61 || op == 0x64;
    const unsigned modes = scalar ? 2 : fp_unary ? 8 : op == 0x66 || op == 0x6b ? 6 : op >= 0x6c ? 5 : 4;
    for (unsigned mode = 0; mode < modes; ++mode) {
      const bool byte_case = mode >= 4 && op >= 0x6b;
      const unsigned sew = byte_case ? (op == 0x6c || op == 0x6d ? 1 : 0) :
        mode == 5 || mode == 6 || op == 0x67 || op == 0x68 || op == 0x6b || op == 0x6e ? 1 : 2;
      const unsigned input_width = op == 0x6c || op == 0x6d ? 1u << (sew - 1) :
        op == 0x68 || op == 0x6b || op == 0x6e ? 2u << sew : 1u << sew;
      const unsigned output_width = sew == 1 && op != 0x67 ? 2 : 4;
      const unsigned effective_output_width = byte_case && sew == 0 ? 1 : output_width;
      const unsigned source1 = op == 0x66 ? 12 : op == 0x6e ? (byte_case ? 8 : 16) :
        op == 0x6b && mode == 5 ? 1 : 0;
      const unsigned lmul = mode == 2 ? 7 : mode == 3 ? 1 : 0;
      const unsigned vl = mode == 1 ? 0 : mode == 3 ? 200 : 13;
      const unsigned active = scalar && vl ? 1 : vl;
      memset(source, 0, sizeof(source));
      memset(other, 0, sizeof(other));
      memcpy(expected, sentinel, sizeof(expected));
      unsigned flags = 0;
      bool saturation = false;
      for (unsigned element = 0; element < sizeof(source) / input_width; ++element) {
        uint32_t input = float_bits(4.0f), result = 0, rhs = UINT32_C(0x80000000);
        switch (op) {
        case 0x60: case 0x62: result = float_bits(2.0f); break;
        case 0x61: case 0x63: result = UINT32_C(0x3eff0000); break;
        case 0x64: case 0x65: result = UINT32_C(0x3e7f0000); break;
        case 0x66: input = UINT32_C(0xff800001); result = UINT32_C(0x7f800001); break;
        case 0x67: input = element & 1 ? 0xc000 : 0x3c00; result = float_bits(element & 1 ? -2.0f : 1.0f); break;
        case 0x68: input = float_bits(element & 1 ? -2.0f : 1.0f); result = element & 1 ? 0xc000 : 0x3c00; break;
        case 0x69: input = float_bits(element & 1 ? -3.5f : 2.5f); result = element & 1 ? (uint32_t)-4 : 2; break;
        case 0x6a: input = (uint32_t)((int)element - 100); result = float_bits((float)((int)element - 100)); break;
        case 0x6b: input = element & 1 ? (uint32_t)-100000 : 100000; result = element & 1 ? 0x8000 : 0x7fff; break;
        case 0x6c: input = 0xf123; result = 0xf123; break;
        case 0x6d: input = 0xf123; result = UINT32_C(0xfffff123); break;
        case 0x6e: input = UINT32_C(0x1234abcd) + element; result = input >> 16; break;
        }
        if (mode == 5 && !byte_case) {
          input = 0x4400;
          if (op == 0x60) result = 0x4000;
          if (op == 0x61) result = 0x37f8;
          if (op == 0x64) result = 0x33f8;
          if (op == 0x66) { input = 0xfc01; result = 0x7c01; rhs = 0x80008000; }
        }
        if ((mode == 4 || mode == 6) && fp_unary) {
          const uint32_t special32[] = {0, 0x80000000, 0x7f800000, 0xff800000, 0x7f800001, 0x7fc00000, 0x40800000, 0xc0800000};
          const uint32_t special16[] = {0, 0x8000, 0x7c00, 0xfc00, 0x7c01, 0x7e00, 0x4400, 0xc400};
          const uint32_t sqrt32[] = {0, 0x80000000, 0x7f800000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x40000000, 0x7fc00000};
          const uint32_t sqrt16[] = {0, 0x8000, 0x7c00, 0x7e00, 0x7e00, 0x7e00, 0x4000, 0x7e00};
          const uint32_t rsqrt32[] = {0x7f800000, 0xff800000, 0, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x3eff0000, 0x7fc00000};
          const uint32_t rsqrt16[] = {0x7c00, 0xfc00, 0, 0x7e00, 0x7e00, 0x7e00, 0x37f8, 0x7e00};
          const uint32_t rec32[] = {0x7f800000, 0xff800000, 0, 0x80000000, 0x7fc00000, 0x7fc00000, 0x3e7f0000, 0xbe7f0000};
          const uint32_t rec16[] = {0x7c00, 0xfc00, 0, 0x8000, 0x7e00, 0x7e00, 0x33f8, 0xb3f8};
          const unsigned offset = element % 8;
          input = sew == 1 ? special16[offset] : special32[offset];
          result = op == 0x60 ? (sew == 1 ? sqrt16[offset] : sqrt32[offset]) :
            op == 0x61 ? (sew == 1 ? rsqrt16[offset] : rsqrt32[offset]) :
            (sew == 1 ? rec16[offset] : rec32[offset]);
          flags = op == 0x60 ? 16 : 24;
        }
        if (mode == 7) {
          input = 1;
          if (op == 0x60) { input = 2; result = UINT32_C(0x1a800000); }
          if (op == 0x61) result = UINT32_C(0x64b40000);
          if (op == 0x64) { result = UINT32_C(0x7f800000); flags = 5; }
        }
        if (byte_case) {
          if (op == 0x6b) {
            input = element & 1 ? (uint16_t)-300 : 300;
            result = element & 1 ? 0x80 : 0x7f;
            if (mode == 5) {
              const uint16_t values[] = {255, (uint16_t)-257, (uint16_t)-255, 254};
              const uint8_t results[] = {127, 128, 129, 127};
              input = values[element % 4];
              result = results[element % 4];
            }
          }
          if (op == 0x6c) { input = 0x81; result = 0x81; }
          if (op == 0x6d) { input = 0x81; result = 0xff81; }
          if (op == 0x6e) { input = 0xabcd; result = 0xab; }
        }
        memcpy(source + input_width * element, &input, input_width);
        if (4 * element < sizeof(other)) memcpy(other + 4 * element, &rhs, 4);
        if (element < active) {
          memcpy(expected + effective_output_width * element, &result, effective_output_width);
          if (op == 0x69) flags = 1;
          if (op == 0x6b) saturation = true;
        }
      }
      const uint64_t enabled = UINT64_C(1) << 40;
      write_evu_csr(0x8c0, enabled | VLEN_BYTES);
      copy_mem(UINT64_C(0x80074000), source, sizeof(source), true);
      copy_mem(UINT64_C(0x80074800), other, sizeof(other), true);
      copy_mem(UINT64_C(0x80074c00), sentinel, sizeof(sentinel), true);
      for (unsigned reg = 0; reg < 4; ++reg) {
        load_reg(4 + reg, UINT64_C(0x80074000) + reg * VLEN_BYTES, 0);
        load_reg(8 + reg, UINT64_C(0x80074c00) + reg * VLEN_BYTES, 0);
      }
      load_reg(12, UINT64_C(0x80074800), 0);
      load_reg(13, UINT64_C(0x80074a00), 0);
      write_evu_csr(0x8c1, 0);
      write_evu_csr(0x8c2, 0);
      const unsigned vtype = (sew << 3) | lmul;
      write_evu_csr(0x8c0, enabled | ((uint64_t)vtype << 32) | vl);
      const uint32_t instruction = insn(op, sew, 8, source1, 4);
      evu(instruction, 0, flags);
      for (unsigned reg = 0; reg < 4; ++reg) read_reg(8 + reg, actual + reg * VLEN_BYTES);
      for (unsigned byte = 0; byte < sizeof(actual); ++byte) {
        if (actual[byte] != expected[byte]) {
          fprintf(stderr, "RVV math op=%02x mode=%u byte=%u actual=%02x expected=%02x\n",
            op, mode, byte, actual[byte], expected[byte]);
          CHECK(actual[byte] == expected[byte]);
        }
      }
      uint64_t state[13];
      CHECK(get_csr(state, sizeof(state)) == 0);
      CHECK(state[5] == flags && state[6] == saturation);
      if (trace) {
        fprintf(trace, "%08x %u %u %u %u", instruction, vtype, vl, flags, saturation);
        const uint8_t *buffers[] = {source, other, expected};
        const unsigned sizes[] = {sizeof(source), sizeof(other), sizeof(expected)};
        for (unsigned buffer = 0; buffer < 3; ++buffer)
          for (unsigned byte = 0; byte < sizes[buffer]; ++byte) fprintf(trace, " %02x", buffers[buffer][byte]);
        fputc('\n', trace);
      }
      ++cases;
    }
  }
  const unsigned invalid[][5] = {
    {0x60, 0, 0, 8, 4}, {0x62, 1, 0, 8, 4}, {0x67, 2, 0, 8, 4},
    {0x69, 1, 0, 8, 4}, {0x6b, 2, 0, 8, 4}, {0x6c, 0, 0, 8, 4},
    {0x67, 1, 0, 9, 4}, {0x67, 1, 0, 8, 8}, {0x68, 1, 0, 5, 4},
    {0x68, 1, 0, 8, 5}, {0x67, 1, 3, 16, 0}, {0x68, 1, 3, 16, 0},
    {0x6c, 2, 5, 8, 4}, {0x62, 2, 1, 8, 4}, {0x6c, 2, 0, 4, 4}
  };
  for (unsigned index = 0; index < sizeof(invalid) / sizeof(invalid[0]); ++index) {
    const unsigned *entry = invalid[index];
    write_evu_csr(0x8c0, (UINT64_C(1) << 40) | ((uint64_t)((entry[1] << 3) | entry[2]) << 32));
    read_reg(8, expected);
    const CPU_state state = run_raw(insn(entry[0], entry[1], entry[3], 0, entry[4]), 0);
    CHECK(state.mcause == 2 && pending() == 0);
    read_reg(8, actual);
    CHECK(memcmp(actual, expected, VLEN_BYTES) == 0);
  }
  write_evu_csr(0x8c0, saved[0x8c0]);
  CHECK(pending() == 0);
  if (trace) fclose(trace);
  printf("EVU RVV math: %u cases, 15 encodings, m1/m2/mf2, E0, tails, flags, saturation and illegal groups passed\n", cases);
}

static void check_blocks(const char *path) {
  FILE *trace = fopen(path, "w");
  CHECK(trace);
  uint32_t source[4 * VLEN_BYTES / 4] = {0}, other[2 * VLEN_BYTES / 4] = {0};
  uint8_t expected[4 * VLEN_BYTES], actual[VLEN_BYTES];
  const uint32_t values[] = {0xbf800000, 0xc0000000, 0xc0800000, 0x80000000};
  for (unsigned n = 0; n < 12; ++n) {
    const unsigned group = n % 4, slot = n % 8, index = 15 - n, vl = n < 4 ? 64 : 16;
    for (unsigned i = 0; i < 64; ++i) source[i] = values[i / 16];
    source[63] = 0x7f800001; /* Raises NV even when the product is subsequently cleared. */
    other[index] = 0x3f800000 | ((uint32_t)(n != 7) << 31);
    memset(expected, 0xa5, sizeof(expected));
    unsigned flags = 0;
    const unsigned control = 12 | (group << 5) | (index << 7) | (slot << 11);
    uint32_t inst;
    if (n < 4) {
      inst = insn(0x76, 2, 8, group, 4);
      flags = 16;
      for (unsigned i = 0; i < 4; ++i) {
        const uint32_t value = values[i] & 0x7fffffff;
        memcpy(expected + 16 * group + 4 * i, &value, 4);
      }
    } else {
      inst = insn(0x77, 2, 8, 10, 4);
      for (unsigned i = 0; i < 16; ++i) {
        uint32_t value = source[16 * group + i];
        if (value == 0x7f800001) { value = 0x7fc00000; flags |= 16; }
        EvuLowpResult v = evu_lowp_convert(n == 7 ? 0 : value, 0x75, 127);
        uint8_t *byte = &expected[8 * slot + i / 2];
        if (i & 1) *byte |= v.bits << 4; else *byte = v.bits;
        flags |= v.flags;
      }
    }
    write_evu_csr(0x8c0, (UINT64_C(1) << 40) | VLEN_BYTES);
    memset(actual, 0xa5, sizeof(actual));
    copy_mem(0x80074000, source, VLEN_BYTES, true);
    copy_mem(0x80074800, other, VLEN_BYTES, true);
    copy_mem(0x80074c00, actual, VLEN_BYTES, true);
    load_reg(4, 0x80074000, 0); load_reg(12, 0x80074800, 0); load_reg(8, 0x80074c00, 0);
    write_evu_csr(0x8c1, 0);
    write_evu_csr(0x8c0, (UINT64_C(1) << 40) | (UINT64_C(16) << 32) | vl);
    evu(inst, control, flags);
    read_reg(8, actual);
    CHECK(!memcmp(actual, expected, VLEN_BYTES));
    fprintf(trace, "%08x 16 %u %u 0 base=%x", inst, vl, flags, control);
    const void *buffers[] = {source, other, expected};
    const unsigned sizes[] = {sizeof(source), sizeof(other), sizeof(expected)};
    for (unsigned b = 0; b < 3; ++b)
      for (unsigned i = 0; i < sizes[b]; ++i) fprintf(trace, " %02x", ((const uint8_t *)buffers[b])[i]);
    fputc('\n', trace);
  }
  CHECK(fclose(trace) == 0);
}

static void check_low_precision(const char *trace_path) {
  FILE *trace = trace_path ? fopen(trace_path, "w") : NULL;
  CHECK(!trace_path || trace);
  const uint32_t inputs[] = {0, 0x80000000, 0x3e800000, 0x3f400000, 0x3fa00000,
    0x3fe00000, 0x40200000, 0x40600000, 0x40a00000, 0x40c00000, 0x43e00000,
    0x43e80000, 0x43f00000, 0x47700000, 0x47800000, 0x7f800000,
    0xff800000, 0x7fc00000, 0x7f800001, 1, 0x007fffff, 0x3b000000,
    0x3a800000, 0x42ff0000, 0xc3010000, 0x7f7fffff, 0x3f800000, 0xbf800000};
  const unsigned lengths[] = {0, 1, 2, 15, 16, 17, 63, 64, 65, 127, 128};
  const unsigned factors[] = {127, 128, 255};
  const uint32_t known[][5] = {
    {0x70, 0x40200000, 127, 2, 1}, {0x70, 0x42ff0000, 127, 127, 16},
    {0x71, 0x43e00000, 127, 0x7e, 0}, {0x71, 0x43e80000, 127, 0x7e, 1},
    {0x71, 0x43f00000, 127, 0x7e, 5}, {0x72, 0x47800000, 127, 0x7c, 5},
    {0x73, 1, 127, 0x3b000000, 0}, {0x73, 0x7e, 127, 0x43e00000, 0},
    {0x74, 0x3f800000, 128, 0x30, 0}, {0x75, 0x3e800000, 127, 0, 3},
    {0x75, 0x3f400000, 127, 2, 1}, {0x75, 0x40a00000, 127, 6, 1},
    {0x75, 0x7f800000, 127, 7, 5}, {0x75, 0x3f800000, 255, 7, 16},
    {0x27, 0xbf000000, 127, 0, 1}, {0x27, 0xbf000001, 127, 0, 16},
    {0x27, 0x437f8000, 127, 255, 16}, {0x28, 0x80, 127, 0xc3000000, 0},
    {0x29, 0xff, 127, 0x437f0000, 0}, {0x2a, 0xbfa00000, 127, 0xa, 1},
    {0x2b, 0x7d, 127, 0x7fc00000, 16}, {0x2b, 0xfc, 127, 0xff800000, 0},
    {0x2c, 0xf, 127, 0xc0c00000, 0}, {0x2c, 8, 127, 0x80000000, 0},
    {0x2d, 0x007fbfff, 127, 0x0080, 3}, {0x2d, 0x007fc000, 127, 0x0080, 1},
    {0x2d, 0x7f7f8000, 127, 0x7f80, 5}, {0x2d, 0xff800001, 127, 0x7fc0, 16},
    {0x2e, 0x8001, 127, 0x80010000, 0}, {0x2e, 0xff81, 127, 0x7fc00000, 16},
    {0x78, 0x40000000, 128, 0x3c, 0}, {0x78, 0x40000000, 255, 0x7e, 0},
    {0x79, 0x01, 0, 0x2000, 0}, {0x79, 0x7e, 254, 0x7f800000, 5},
    {0x7a, 0x01, 0, 0x40, 0}, {0x7a, 0x7d, 255, 0x7fc00000, 16},
    {0x7b, 0x09, 128, 0xbf800000, 0}, {0x7b, 0x00, 255, 0x7fc00000, 16}
  };
  for (unsigned index = 0; index < sizeof(known) / sizeof(known[0]); ++index) {
    EvuLowpResult result = evu_lowp_convert(known[index][1], known[index][0], known[index][2]);
    CHECK(result.bits == known[index][3] && result.flags == known[index][4]);
  }
  uint64_t saved[4096];
  copy_csrs(saved, false);
  uint8_t source[4 * VLEN_BYTES], other[2 * VLEN_BYTES], expected[4 * VLEN_BYTES], actual[4 * VLEN_BYTES];
  unsigned cases = 0;
  const struct { unsigned opcode, src_bits, dst_bits; bool scaled; } ops[] = {
    {0x70,32,8,0}, {0x71,32,8,0}, {0x72,32,8,0}, {0x73,8,32,0}, {0x74,32,8,1}, {0x75,32,4,1},
    {0x27,32,8,0}, {0x28,8,32,0}, {0x29,8,32,0}, {0x2a,32,4,0}, {0x2b,8,32,0}, {0x2c,4,32,0},
    {0x2d,32,16,0}, {0x2e,16,32,0}, {0x78,32,8,1}, {0x79,8,32,1}, {0x7a,8,32,1}, {0x7b,4,32,1}
  };
  for (unsigned op = 0; op < sizeof(ops) / sizeof(ops[0]); ++op) {
    const unsigned opcode = ops[op].opcode, src_bits = ops[op].src_bits, dst_bits = ops[op].dst_bits;
    for (unsigned mode = 0; mode < 3; ++mode) {
      memset(source, 0, sizeof(source));
      memset(other, 0xff, sizeof(other));
      other[0] = factors[mode];
      for (unsigned element = 0; element < 128; ++element) {
        const uint32_t value = inputs[(element + mode) % (sizeof(inputs) / sizeof(inputs[0]))];
        if (src_bits == 4) source[element / 2] |= ((element + mode) & 15) << (4 * (element & 1));
        else if (src_bits == 8) source[element] = (uint8_t)(element + mode * 64);
        else if (src_bits == 16) {
          const uint16_t half = value >> 16;
          memcpy(source + 2 * element, &half, 2);
        } else memcpy(source + 4 * element, &value, 4);
      }
      for (unsigned length = 0; length < sizeof(lengths) / sizeof(lengths[0]); ++length) {
        const unsigned vl = lengths[length];
        memset(expected, 0xa5, sizeof(expected));
        unsigned flags = 0;
        for (unsigned element = 0; element < vl; ++element) {
          uint32_t value = 0;
          if (src_bits == 4) value = (source[element / 2] >> (4 * (element & 1))) & 15;
          else memcpy(&value, source + element * (src_bits / 8), src_bits / 8);
          const EvuLowpResult result = evu_lowp_convert(value, opcode, other[0]);
          flags |= result.flags;
          if (dst_bits != 4) memcpy(expected + element * (dst_bits / 8), &result.bits, dst_bits / 8);
          else if (!(element & 1)) expected[element / 2] = result.bits & 15;
          else expected[element / 2] |= (result.bits & 15) << 4;
        }
        write_evu_csr(0x8c0, (UINT64_C(1) << 40) | VLEN_BYTES);
        memset(actual, 0xa5, sizeof(actual));
        copy_mem(0x80074000, source, sizeof(source), true);
        copy_mem(0x80074800, other, sizeof(other), true);
        copy_mem(0x80074c00, actual, sizeof(actual), true);
        for (unsigned reg = 0; reg < 4; ++reg) {
          load_reg(4 + reg, 0x80074000 + reg * VLEN_BYTES, 0);
          load_reg(8 + reg, 0x80074c00 + reg * VLEN_BYTES, 0);
        }
        load_reg(12, 0x80074800, 0);
        load_reg(13, 0x80074a00, 0);
        write_evu_csr(0x8c1, 0);
        write_evu_csr(0x8c2, 0);
        write_evu_csr(0x8c0, (UINT64_C(1) << 40) | (UINT64_C(16) << 32) | vl);
        const uint32_t instruction = insn(opcode, 2, 8, ops[op].scaled ? 12 : 0, 4);
        evu(instruction, 0, flags);
        for (unsigned reg = 0; reg < 4; ++reg) read_reg(8 + reg, actual + reg * VLEN_BYTES);
        CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
        uint64_t state[13];
        CHECK(get_csr(state, sizeof(state)) == 0 && state[5] == flags && state[6] == 0);
        if (trace) {
          fprintf(trace, "%08x 16 %u %u 0", instruction, vl, flags);
          const uint8_t *buffers[] = {source, other, expected};
          const unsigned sizes[] = {sizeof(source), sizeof(other), sizeof(expected)};
          for (unsigned buffer = 0; buffer < 3; ++buffer)
            for (unsigned byte = 0; byte < sizes[buffer]; ++byte) fprintf(trace, " %02x", buffers[buffer][byte]);
          fputc('\n', trace);
        }
        ++cases;
      }
    }
  }
  const unsigned invalid[][5] = {
    {0x70, 1, 0, 8, 0}, {0x71, 2, 1, 8, 0}, {0x72, 2, 7, 8, 0},
    {0x73, 2, 0, 4, 0}, {0x74, 2, 0, 12, 12}, {0x75, 2, 0, 4, 12},
    {0x70, 2, 0, 8, 12}, {0x27, 2, 0, 8, 12}, {0x2e, 2, 0, 4, 0},
    {0x2d, 1, 0, 8, 0}, {0x2c, 2, 1, 8, 0}, {0x79, 2, 0, 12, 12},
    {0x7b, 2, 0, 4, 12}, {0x2f, 2, 0, 8, 0}, {0x7c, 2, 0, 8, 0}
  };
  for (unsigned index = 0; index < sizeof(invalid) / sizeof(invalid[0]); ++index) {
    const unsigned *entry = invalid[index];
    write_evu_csr(0x8c0, (UINT64_C(1) << 40) | ((uint64_t)((entry[1] << 3) | entry[2]) << 32));
    const CPU_state state = run_raw(insn(entry[0], entry[1], entry[3], entry[4], 4), 0);
    CHECK(state.mcause == 2 && pending() == 0);
  }
  write_evu_csr(0x8c0, saved[0x8c0]);
  if (trace) fclose(trace);
  printf("EVU low precision: %u cases, 18 operations, tails, flags, scales and illegal encodings passed\n", cases);
}

static void check_scalar_fp_forms(void) {
  uint64_t saved[4096];
  copy_csrs(saved, false);
  const uint64_t config = (UINT64_C(1) << 40) | (UINT64_C(16) << 32);
  uint32_t input[VLEN_BYTES / 4], scale[VLEN_BYTES / 4], seed[VLEN_BYTES / 4], actual[VLEN_BYTES / 4];
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i) {
    input[i] = float_bits((float)(i + 1));
    scale[i] = i ? UINT32_C(0x7f800001) : float_bits(2.0f);
    seed[i] = float_bits(4.0f);
  }
  copy_mem(UINT64_C(0x80072000), input, sizeof(input), true);
  copy_mem(UINT64_C(0x80072400), scale, sizeof(scale), true);
  copy_mem(UINT64_C(0x80072800), seed, sizeof(seed), true);
  write_evu_csr(0x8c0, config | (VLEN_BYTES / 4));
  load_reg(8, UINT64_C(0x80072000), 2);
  load_reg(9, UINT64_C(0x80072400), 2);
  load_reg(10, UINT64_C(0x80072800), 2);
  load_reg(11, UINT64_C(0x80072800), 2);
  write_evu_csr(0x8c0, config | 64);
  // Inactive scalar-source lanes contain sNaNs; they must not contribute flags.
  evu(insn(0x5c, 2, 10, 9, 8), 0, 0);
  read_reg(10, actual);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(actual[i] == (i < 64 ? float_bits(2.0f * (float)(i + 1)) : seed[i]));
  evu(insn(0x5c, 2, 9, 9, 8), 0, 0); // Scalar source aliases the destination.
  read_reg(9, actual);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(actual[i] == (i < 64 ? float_bits(2.0f * (float)(i + 1)) : scale[i]));
  evu(insn(0x52, 2, 11, 10, 0), UINT64_C(0x80072000), 0);
  evu(insn(0x54, 2, 11, 8, 11), 0, 0);
  read_reg(11, actual);
  CHECK(actual[0] == float_bits(2.0f));
  for (unsigned i = 1; i < VLEN_BYTES / 4; ++i) CHECK(actual[i] == seed[i]);
  uint8_t guard[64], stored[64];
  memset(guard, 0xa5, sizeof(guard));
  memcpy(guard + 20, actual, 4);
  memset(stored, 0xa5, sizeof(stored));
  copy_mem(UINT64_C(0x80072c00), stored, sizeof(stored), true);
  const uint32_t store = insn(0x53, 2, 11, 10, 0);
  const uint64_t pc = run(store, UINT64_C(0x80072c14));
  CHECK(store_event(UINT64_C(0x80072c00), guard, UINT64_C(15) << 20) == 0);
  CHECK(complete(pc, store, 0, false, 0) == 0);
  copy_mem(UINT64_C(0x80072c00), stored, sizeof(stored), false);
  CHECK(memcmp(guard, stored, sizeof(guard)) == 0);
  uint64_t state[13];
  CHECK(get_csr(state, sizeof(state)) == 0);
  CHECK(state[0] == 64 && state[1] == 16);
  write_evu_csr(0x8c0, config); // VL=0: scalar arithmetic remains inactive.
  evu(insn(0x54, 2, 11, 8, 11), 0, 0);
  uint32_t after[VLEN_BYTES / 4];
  read_reg(11, after);
  CHECK(memcmp(actual, after, sizeof(after)) == 0);
  write_evu_csr(0x8c0, saved[0x8c0]);
  CHECK(pending() == 0);
  printf("EVU scalar FP32: source capture, aliases, tails, E0 load/store guards and VL preservation passed\n");
}

static void check_gather_imm(void) {
  uint64_t saved[4096];
  copy_csrs(saved, false);
  const uint64_t enabled = UINT64_C(1) << 40;
  uint8_t source[VLEN_BYTES], sentinel[VLEN_BYTES], actual[VLEN_BYTES], expected[VLEN_BYTES];
  memset(sentinel, 0xa5, sizeof(sentinel));
  const unsigned indices[] = {0, 7, 16, 31};
  const unsigned lengths[] = {0, 1, 13, 64};
  for (unsigned sew = 0; sew <= 2; ++sew) {
    const unsigned bytes = 1u << sew;
    const unsigned vlmax = VLEN_BYTES / bytes;
    const uint64_t config = enabled | ((uint64_t)(sew << 3) << 32);
    for (unsigned i = 0; i < sizeof(source); ++i) source[i] = (uint8_t)(i * 37 + (i >> 5) + 19);
    copy_mem(UINT64_C(0x80070000), source, sizeof(source), true);
    copy_mem(UINT64_C(0x80070400), sentinel, sizeof(sentinel), true);
    for (unsigned alias = 0; alias < 2; ++alias) {
      const unsigned dest = alias ? 8 : 16;
      for (unsigned n = 0; n < sizeof(indices) / sizeof(indices[0]); ++n) {
        for (unsigned v = 0; v < sizeof(lengths) / sizeof(lengths[0]); ++v) {
          const unsigned index = indices[n], vl = lengths[v];
          write_evu_csr(0x8c0, config | vlmax);
          load_reg(8, UINT64_C(0x80070000), sew);
          if (!alias) load_reg(dest, UINT64_C(0x80070400), sew);
          write_evu_csr(0x8c0, config | vl);
          memcpy(expected, alias ? source : sentinel, sizeof(expected));
          for (unsigned lane = 0; lane < vl; ++lane)
            memcpy(expected + lane * bytes, source + index * bytes, bytes);
          evu(insn(0x51, sew, dest, index, 8), 0, 0);
          read_reg(dest, actual);
          CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
          if (!alias) {
            read_reg(8, actual);
            CHECK(memcmp(actual, source, sizeof(actual)) == 0);
          }
        }
      }
    }
  }
  // Fractional LMUL can make an unsigned immediate exceed VLMAX.
  write_evu_csr(0x8c0, enabled | (UINT64_C(0x10) << 32) | (VLEN_BYTES / 4));
  load_reg(8, UINT64_C(0x80070000), 2);
  write_evu_csr(0x8c0, enabled | (UINT64_C(0x15) << 32) | 13);
  evu(insn(0x51, 2, 8, 31, 8), 0, 0);
  read_reg(8, actual);
  memset(expected, 0, 13 * 4);
  memcpy(expected + 13 * 4, source + 13 * 4, sizeof(expected) - 13 * 4);
  CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
  write_evu_csr(0x8c0, saved[0x8c0]);
  CHECK(pending() == 0);
  printf("EVU immediate gather: 3 widths, unsigned indices, VL=0/tails, aliases and out-of-range indices passed\n");
}

/* Mirror vector_add_evu_test.c without compiling a RISC-V workload. Keep the
 * entire instruction burst pending to exercise forwarding and store snapshots. */
static void check_vector_add_sequence(void) {
  enum { GUARD = 64, BUFFER_BYTES = VLEN_BYTES + 2 * GUARD };
  uint8_t input[2][BUFFER_BYTES], output[4][BUFFER_BYTES], actual[BUFFER_BYTES];
  for (unsigned sew = 0; sew < 3; ++sew) {
    const unsigned bytes = 1u << sew;
    const uint32_t mask = UINT32_MAX >> (32 - 8 * bytes);
    for (unsigned round = 0; round < 3; ++round) {
      for (unsigned v = 0; v < 6; ++v) {
        uint8_t *buffer = v < 2 ? input[v] : output[v - 2];
        memset(buffer, 0xa5, GUARD);
        memset(buffer + GUARD + VLEN_BYTES, 0x5a, GUARD);
        for (unsigned i = 0; i < VLEN_BYTES; ++i)
          buffer[GUARD + i] = v < 2 ? (uint8_t)((i * (v ? 73u : 37u) + round * 97u + v * 19u) ^ (i >> 2)) : 0xcc;
        copy_mem(UINT64_C(0x80030000) + v * 0x1000, buffer, BUFFER_BYTES, true);
      }
      for (unsigned i = 0; i < VLEN_BYTES / bytes; ++i) {
        uint32_t a = 0, b = 0;
        memcpy(&a, input[0] + GUARD + i * bytes, bytes);
        memcpy(&b, input[1] + GUARD + i * bytes, bytes);
        const uint32_t expected[] = {a, (a + b) & mask, (a + b) & mask, (a + b + b) & mask};
        for (unsigned v = 0; v < 4; ++v)
          memcpy(output[v] + GUARD + i * bytes, &expected[v], bytes);
      }
      const uint32_t instructions[] = {
          insn(0x10, sew, 1, 10, 0), insn(0x10, sew, 17, 10, 0), insn(0x11, sew, 1, 10, 0),
          insn(0x01, sew, 31, 1, 17), insn(0x11, sew, 31, 10, 0), insn(0x01, sew, 1, 1, 17),
          insn(0x11, sew, 1, 10, 0), insn(0x01, sew, 17, 31, 17), insn(0x11, sew, 17, 10, 0),
          insn(0x1a, 0, 0, 0, 0)};
      const unsigned buffers[] = {0, 1, 2, 0, 3, 0, 4, 0, 5, 0};
      uint64_t pcs[10];
      for (unsigned i = 0; i < 10; ++i)
        pcs[i] = run(instructions[i], UINT64_C(0x80030000) + buffers[i] * 0x1000 + GUARD);
      CHECK(pending() == 10);
      for (unsigned i = 0; i < 10; ++i) {
        if ((instructions[i] >> 25) == 0x11) {
          const unsigned v = buffers[i] - 2;
          for (unsigned offset = 0; offset < VLEN_BYTES; offset += 64)
            CHECK(store_event(UINT64_C(0x80030000) + buffers[i] * 0x1000 + GUARD + offset,
                              output[v] + GUARD + offset, UINT64_MAX) == 0);
        }
        CHECK(complete(pcs[i], instructions[i], 0, false, 0) == 0);
      }
      CHECK(pending() == 0);
      for (unsigned v = 0; v < 6; ++v) {
        copy_mem(UINT64_C(0x80030000) + v * 0x1000, actual, sizeof(actual), false);
        CHECK(memcmp(actual, v < 2 ? input[v] : output[v - 2], sizeof(actual)) == 0);
      }
      read_reg(31, actual);
      CHECK(memcmp(actual, output[1] + GUARD, VLEN_BYTES) == 0);
      read_reg(1, actual);
      CHECK(memcmp(actual, output[2] + GUARD, VLEN_BYTES) == 0);
      read_reg(17, actual);
      CHECK(memcmp(actual, output[3] + GUARD, VLEN_BYTES) == 0);
      uint64_t local_csrs[13];
      CHECK(get_csr(local_csrs, sizeof(local_csrs)) == 0);
      CHECK(local_csrs[0] == VLEN_BYTES / 4 && local_csrs[1] == 16 && local_csrs[2] == 0 &&
            local_csrs[3] == VLEN_BYTES && local_csrs[4] == 0);
      CPU_state state;
      copy_regs(&state, false);
      CHECK(memcmp(state.vr, initial.vr, sizeof(state.vr)) == 0);
    }
  }
  printf("EVU vector-add reference: 3 widths x 3 rounds, pending aliases, stores, guards and snapshots passed\n");
}

static void check_memory_window(void) {
  enum { JOBS = 320, PREFIX = 37 };
  uint64_t pcs[JOBS], saved[4096];
  copy_csrs(saved, false);
  CHECK(pending() == 0);
  const uint64_t config = (UINT64_C(1) << 40) | (UINT64_C(16) << 32);
  write_evu_csr(0x8c0, config | 1);
  const uint32_t add = insn(0x01, 2, 8, 8, 8);
  for (unsigned i = 0; i < 60; ++i) pcs[i] = run(add, 0);
  for (unsigned i = 0; i < PREFIX; ++i) CHECK(complete(pcs[i], add, 0, false, 0) == 0);
  for (unsigned i = 60; i < JOBS; ++i) pcs[i] = run(add, 0);
  CHECK(pending() == JOBS - PREFIX);
  for (unsigned i = PREFIX; i < JOBS; ++i) CHECK(complete(pcs[i], add, 0, false, 0) == 0);
  CHECK(pending() == 0);

  uint8_t line[64], actual[VLEN_BYTES], bad[64];
  memset(line, 0xa5, sizeof(line));
  const uint64_t address = UINT64_C(0x80075000);
  uint32_t a = 0x12345678, b = 0x23456789;
  copy_mem(address, line, sizeof(line), true);
  copy_mem(address + 64, &a, sizeof(a), true);
  copy_mem(address + 68, &b, sizeof(b), true);
  load_reg(8, address + 64, 2);
  load_reg(9, address + 68, 2);
  const uint32_t storeA = insn(0x53, 2, 8, 10, 0), storeB = insn(0x53, 2, 9, 10, 0);
  const uint32_t load = insn(0x52, 2, 10, 10, 0);
  const uint64_t pa = run(storeA, address), pb = run(storeB, address + 4);
  const uint64_t pl = run(load, address); // Reference forwards an unacknowledged store.
  memcpy(line, &a, 4);
  memcpy(line + 4, &b, 4);
  memcpy(bad, line, 64);
  bad[4] ^= 1;
  CHECK(store_event(address, bad, UINT64_C(255)) != 0);
  CHECK(store_event(address, line, UINT64_C(255)) == 0);
  CHECK(complete(pa, storeA, 0, false, 0) == 0);
  CHECK(complete(pb, storeB, 0, false, 0) == 0);
  CHECK(complete(pl, load, 0, false, 0) == 0);
  read_reg(10, actual);
  CHECK(memcmp(actual, &a, 4) == 0);

  // Equal-valued later writes must not be mistaken for an earlier physical Ack.
  const uint64_t p0 = run(storeA, address), p1 = run(storeA, address);
  CHECK(store_event(address, line, UINT64_C(15)) == 0);
  CHECK(complete(p0, storeA, 0, false, 0) == 0);
  CHECK(complete(p1, storeA, 0, false, 0) != 0);
  CHECK(store_event(address, line, UINT64_C(15)) == 0);
  CHECK(complete(p1, storeA, 0, false, 0) == 0);
  CHECK(pending() == 0);
  copy_csrs(saved, true);
  printf("EVU memory window: wrapped queue growth, merged ownership, guards, RAW forwarding and repeated writes passed\n");
}

static void check_maximum_group_memory(void) {
  enum { GROUP_BYTES = 8 * VLEN_BYTES, GUARD = 64 };
  uint8_t input[GROUP_BYTES], output[GROUP_BYTES + 2 * GUARD], actual[sizeof(output)];
  uint64_t old[13];
  CHECK(pending() == 0 && get_csr(old, sizeof(old)) == 0);
  // Start 32 bytes before a page boundary, exercise all eight registers,
  // then verify masked first/last cache lines and both memory guards.
  const uint64_t source = UINT64_C(0x80060fe0), target = UINT64_C(0x80064fe0);
  for (unsigned sew = 0; sew <= 2; ++sew) {
    for (unsigned i = 0; i < sizeof(input); ++i) input[i] = (uint8_t)(i * 73 + (i >> 7) + sew * 19);
    memset(output, 0xa5, sizeof(output));
    copy_mem(source, input, sizeof(input), true);
    copy_mem(target - GUARD, output, sizeof(output), true);
    write_evu_csr(0x8c0, (UINT64_C(1) << 40) | ((uint64_t)((sew << 3) | 3) << 32) | UINT32_MAX);
    uint64_t cfg[13];
    CHECK(get_csr(cfg, sizeof(cfg)) == 0 && cfg[0] == (unsigned)GROUP_BYTES >> sew);
    const uint32_t load = insn(0x10, sew, 8, 10, 0), store = insn(0x11, sew, 8, 10, 0);
    evu(load, source, 0);
    for (unsigned reg = 0; reg < 8; ++reg) {
      read_reg(8 + reg, actual);
      CHECK(memcmp(actual, input + reg * VLEN_BYTES, VLEN_BYTES) == 0);
    }
    const uint64_t pc = run(store, target);
    for (uint64_t line = target & ~UINT64_C(63); line < target + GROUP_BYTES; line += 64) {
      uint8_t data[64];
      copy_mem(line, data, sizeof(data), false);
      uint64_t mask = 0;
      for (unsigned b = 0; b < 64; ++b) {
        if (line + b >= target && line + b < target + GROUP_BYTES) {
          data[b] = input[line + b - target];
          mask |= UINT64_C(1) << b;
        }
      }
      // HBL2 sends a full-line RMW; the event retains logical byte ownership.
      CHECK(store_event(line, data, mask) == 0);
    }
    CHECK(complete(pc, store, 0, false, 0) == 0 && pending() == 0);
    memcpy(output + GUARD, input, sizeof(input));
    copy_mem(target - GUARD, actual, sizeof(actual), false);
    CHECK(memcmp(actual, output, sizeof(actual)) == 0);
    CHECK(run_raw(insn(0x10, sew, 9, 10, 0), source).mcause == 2 && pending() == 0);
    CHECK(run_raw(insn(0x39, 0, 16, 8, 8), 0).mcause == 2 && pending() == 0);
    CHECK(run_raw(insn(0x4c, sew, 0, 0, 8), 0).mcause == 2 && pending() == 0);
  }
  write_evu_csr(0x8c0, old[12]);
  printf("EVU m8 memory: e8/e16/e32, 4096 bytes across pages, eight registers, masked stores/guards and illegal groups passed\n");
}

static void check_int8_op(unsigned f7, unsigned rd, const uint8_t *lhs, const uint8_t *rhs) {
  uint8_t result[VLEN_BYTES];
  evu(insn(f7, 0, rd, 2, 1), 0, 0);
  read_reg(rd, result);
  for (unsigned i = 0; i < VLEN_BYTES; ++i) {
    const uint8_t a = lhs[i], b = rhs[i];
    const int8_t sa = (int8_t)a, sb = (int8_t)b;
    uint8_t expected = 0;
    switch (f7) {
    case 0x03:
      expected = a & b;
      break;
    case 0x04:
      expected = a | b;
      break;
    case 0x05:
      expected = a ^ b;
      break;
    case 0x06:
      expected = (uint8_t)(a * b);
      break;
    case 0x20:
      expected = (uint8_t)(a << (b & 7));
      break;
    case 0x21:
      expected = a >> (b & 7);
      break;
    case 0x22:
      expected = (uint8_t)(sa >> (b & 7));
      break;
    case 0x23:
      expected = sa < sb ? a : b;
      break;
    case 0x24:
      expected = sa > sb ? a : b;
      break;
    case 0x25:
      expected = a < b ? a : b;
      break;
    case 0x26:
      expected = a > b ? a : b;
      break;
    case 0x3c:
      expected = (uint8_t)sat8((int)sa + sb);
      break;
    case 0x3d:
      expected = (uint8_t)sat8((int)sa - sb);
      break;
    default:
      CHECK(0);
    }
    if (result[i] != expected) {
      fprintf(stderr, "f7=0x%x i=%u a=%u b=%u got=%u expected=%u\n", f7, i, a, b, result[i], expected);
      CHECK(0);
    }
  }
}

static void load_fp32(unsigned reg, const float *values) {
  const uint64_t address = UINT64_C(0x80050000);
  copy_mem(address, (void *)values, VLEN_BYTES, true);
  run(insn(0x10, 2, reg, 10, 0), address);
}

static void store_fp32(unsigned reg) {
  run(insn(0x11, 2, reg, 10, 0), UINT64_C(0x80060000));
}

static void fp_op(unsigned opcode, unsigned vd, unsigned vs1, unsigned vs2) {
  run(insn(opcode, 2, vd, vs1, vs2), 0);
}

static void check_unordered_reductions(void) {
  const unsigned lengths[] = {0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256};
  uint8_t source[VLEN_BYTES], seed[VLEN_BYTES], marker[VLEN_BYTES], actual[VLEN_BYTES];
  CHECK(pending() == 0);
  set_async(false);
  memset(seed, 0, sizeof(seed));
  memset(marker, 0xa5, sizeof(marker));
  for (unsigned sew = 1; sew <= 2; ++sew) {
    const unsigned bytes = 1u << sew, count = VLEN_BYTES / bytes;
    const uint32_t large = sew == 1 ? 0x6800 : 0x4b800000; // 2^11 or 2^24.
    const uint32_t negative = large | (UINT32_C(1) << (bytes * 8 - 1));
    const uint32_t one = sew == 1 ? 0x3c00 : 0x3f800000;
    const uint32_t snan = sew == 1 ? 0x7c01 : 0x7f800001;
    for (unsigned t = 0; t < sizeof(lengths) / sizeof(lengths[0]); ++t) {
      const unsigned vl = lengths[t];
      if (vl > count)
        continue;
      memset(source, 0, sizeof(source));
      if (vl) memcpy(source, &large, bytes);
      if (vl > 1) memcpy(source + bytes, &one, bytes);
      if (vl > 2) memcpy(source + 2 * bytes, &negative, bytes);
      if (vl < count) memcpy(source + vl * bytes, &snan, bytes);
      run(setvli_insn(sew << 3, 0, 10), count);
      const uint8_t *buffers[] = {source, seed, marker};
      for (unsigned reg = 1; reg <= 3; ++reg) {
        copy_mem(UINT64_C(0x80050000), (void *)buffers[reg - 1], VLEN_BYTES, true);
        run(insn(0x10, sew, reg, 10, 0), UINT64_C(0x80050000));
      }
      run(setvli_insn(sew << 3, 0, 10), vl);
      write_evu_csr(0x8c1, 0);
      run(insn(0x37, sew, 3, 2, 1), 0);
      read_reg(3, actual);
      uint32_t result = 0;
      memcpy(&result, actual, bytes);
      uint32_t expected = vl >= 3 ? one : vl ? large : sew == 1 ? 0xa5a5 : 0xa5a5a5a5;
      CHECK(result == expected);
      CHECK(memcmp(actual + bytes, marker + bytes, VLEN_BYTES - bytes) == 0);
      uint64_t local_csrs[4096];
      copy_csrs(local_csrs, false);
      CHECK((local_csrs[0x8c1] & 0x1f) == 0); // Cancellation is exact; the tail sNaN is inactive.
    }
    const unsigned lanes = CONFIG_CUSTOM_EVU_DLEN / (8u << sew);
    memset(source, 0, sizeof(source));
    memcpy(source, &large, bytes);
    memcpy(source + lanes * bytes, &large, bytes);
    memcpy(source + 2 * lanes * bytes, &negative, bytes);
    memcpy(source + 3 * lanes * bytes, &negative, bytes);
    memcpy(source + 4 * lanes * bytes, &one, bytes);
    run(setvli_insn(sew << 3, 0, 10), count);
    copy_mem(UINT64_C(0x80050000), source, sizeof(source), true);
    run(insn(0x10, sew, 1, 10, 0), UINT64_C(0x80050000));
    run(setvli_insn(sew << 3, 0, 10), 4 * lanes + 1);
    write_evu_csr(0x8c1, 0);
    run(insn(0x37, sew, 3, 2, 1), 0);
    read_reg(3, actual);
    for (unsigned b = 0; b < bytes; ++b) CHECK(actual[b] == 0);
    uint64_t local_csrs[4096];
    copy_csrs(local_csrs, false);
    CHECK((local_csrs[0x8c1] & 0x1f) == 1); // Slot 0 rounds large + 1 before the slot merges.
  }
  printf("EVU unordered reduction: FP16/FP32 cancellation, four slots, VL tails, flags and destination preservation passed\n");
  set_async(true);
}

static float softmax_input(unsigned test, unsigned block, unsigned element) {
  const float fraction = (float)((int)((element * 37u + block * 11u) % 65u) - 32) * 0.0625f;
  if (test == 0)
    return 0;
  if (test == 2)
    return (float)((3u - block) * 4u) + fraction;
  if (test == 3) {
    const int offsets[] = {0, 8, 3, 12};
    return (float)offsets[block] + fraction;
  }
  if (test == 6)
    return block == 2 && element == 101 ? 16.0f : -16.0f - (float)(element % 23u) * 0.25f;
  return (float)(block * 3u) + fraction + (test == 4 ? 64.0f : test == 5 ? -64.0f : 0.0f);
}

static void check_relative(float actual, long double expected, long double limit, long double *worst) {
  CHECK(isfinite(actual));
  const long double error = fabsl((long double)actual - expected) / fabsl(expected);
  if (error > *worst)
    *worst = error;
  if (error > limit)
    fprintf(stderr, "FP precision: actual=0x%08x expected=%.18Lg relative=%.9Lg limit=%.9Lg\n",
            float_bits(actual), expected, error, limit);
  CHECK(error <= limit);
}

/* Run the guest's seven online-softmax data sets through the reference decoder.
 * libm long-double exp is an independent accuracy oracle, not the EVU algorithm. */
static void check_online_softmax(const char *trace_path) {
  enum { ELEMENTS = VLEN_BYTES / 4, BLOCKS = 4 };
  float input[BLOCKS][ELEMENTS], values[ELEMENTS], result[ELEMENTS], shifted[BLOCKS][ELEMENTS];
  long double worst_exp = 0, worst_sum = 0, worst_output = 0, worst_total = 0;
  CHECK(pending() == 0);
  set_async(false);
  run(setvli_insn(0x10, 0, 10), ELEMENTS);
  if (trace_path) {
    softmax_trace = fopen(trace_path, "w");
    CHECK(softmax_trace);
    // Establish identical initial VRF contents for RTL replay, including the
    // preserved tails of scalar-result reduction instructions.
    memset(values, 0, sizeof(values));
    for (unsigned reg = 0; reg < 32; ++reg)
      load_fp32(reg, values);
  }
  for (unsigned test = 0; test < 7; ++test) {
    for (unsigned b = 0; b < BLOCKS; ++b)
      for (unsigned i = 0; i < ELEMENTS; ++i)
        input[b][i] = softmax_input(test, b, i);
    for (unsigned i = 0; i < ELEMENTS; ++i)
      values[i] = -0x1.fffffep127f;
    load_fp32(28, values);
    memset(values, 0, sizeof(values));
    load_fp32(29, values);
    for (unsigned i = 0; i < ELEMENTS; ++i)
      values[i] = 1.4426950408889634f;
    load_fp32(30, values);
    long double running_max = 0, running_sum = 0;
    for (unsigned b = 0; b < BLOCKS; ++b) {
      long double new_max = b ? running_max : input[b][0];
      for (unsigned i = 0; i < ELEMENTS; ++i)
        if (input[b][i] > new_max)
          new_max = input[b][i];
      long double new_sum = b ? running_sum * expl(running_max - new_max) : 0;
      load_fp32(1, input[b]);
      fp_op(0x44, 2, 28, 1);
      fp_op(0x47, 3, 29, 2);
      if (b) {
        fp_op(0x36, 3, 17, 3);
        fp_op(0x31, 4, 3, 17);
        fp_op(0x32, 4, 30, 4);
        fp_op(0x0f, 4, 0, 4);
        fp_op(0x32, 18, 4, 18);
      }
      fp_op(0x31, 5, 3, 1);
      fp_op(0x32, 5, 30, 5);
      fp_op(0x0f, 5, 0, 5);
      store_fp32(5);
      read_reg(5, result);
      for (unsigned i = 0; i < ELEMENTS; ++i) {
        const long double expected = expl((long double)input[b][i] - new_max);
        check_relative(result[i], expected, 2e-5L, &worst_exp);
        new_sum += expected;
      }
      fp_op(0x37, 6, 29, 5);
      read_reg(6, result);
      if (test == 1 && b == 1) {
        printf("EVU softmax log reproduction: sum=0x%08x, RTL=0x4200d92a\n", float_bits(result[0]));
        CHECK(float_bits(result[0]) == UINT32_C(0x4200d92a));
      }
      fp_op(0x47, 7, 29, 6);
      fp_op(b ? 0x30 : 0x36, 18, 7, b ? 18 : 7);
      fp_op(0x36, 17, 3, 3);
      store_fp32(17);
      store_fp32(18);
      read_reg(17, result);
      for (unsigned i = 0; i < ELEMENTS; ++i)
        CHECK((long double)result[i] == new_max);
      read_reg(18, result);
      for (unsigned i = 0; i < ELEMENTS; ++i)
        check_relative(result[i], new_sum, 3e-5L, &worst_sum);
      running_max = new_max;
      running_sum = new_sum;
    }
    long double probability_sum = 0;
    for (unsigned b = 0; b < BLOCKS; ++b) {
      load_fp32(1, input[b]);
      fp_op(0x31, 5, 17, 1);
      fp_op(0x32, 5, 30, 5);
      fp_op(0x0f, 31, 0, 5);
      fp_op(0x34, 31, 18, 31);
      store_fp32(31);
      read_reg(31, result);
      for (unsigned i = 0; i < ELEMENTS; ++i) {
        const long double expected = expl((long double)input[b][i] - running_max) / running_sum;
        check_relative(result[i], expected, 5e-5L, &worst_output);
        CHECK(result[i] > 0 && result[i] <= 1);
        probability_sum += result[i];
        if (test == 1)
          shifted[b][i] = result[i];
        if (test == 4 || test == 5) {
          long double ignored = 0;
          check_relative(result[i], shifted[b][i], 5e-5L, &ignored);
        }
      }
    }
    const long double total_error = fabsl(probability_sum - 1);
    if (total_error > worst_total)
      worst_total = total_error;
    CHECK(total_error <= 2e-4L);
  }
  printf("EVU online softmax: 7 cases x 512 elements passed; max relative exp=%.9Lg sum=%.9Lg output=%.9Lg; "
         "max |sum(p)-1|=%.9Lg\n", worst_exp, worst_sum, worst_output, worst_total);
  if (softmax_trace) {
    CHECK(fclose(softmax_trace) == 0);
    softmax_trace = NULL;
  }
  set_async(true);
}

/* Scalar checking code can set sticky FCSR flags before an EVU command.
 * A completion reports every exception raised by that command, even if the
 * same flag was already set. Exercise both delayed and synchronous publish. */
static void check_fp_flags_with_scalar_history(void) {
  const uint64_t saved_initial_fcsr = initial.fcsr;
  const uint64_t saved_status = initial.mstatus;
  initial.mstatus &= ~UINT64_C(0x6600);
  initial.vstart = 37;
  CHECK(pending() == 0);
  set_async(true);
  evu(setvli_insn(0x10, 0, 10), VLEN_BYTES / 4, 0);

  float source[VLEN_BYTES / 4], scale[VLEN_BYTES / 4];
  uint8_t baseline[VLEN_BYTES], actual[VLEN_BYTES];
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i) {
    source[i] = softmax_input(1, 0, i) - 2.0f;
    scale[i] = 1.4426950408889634f;
  }
  for (unsigned previous_nx = 0; previous_nx < 2; ++previous_nx) {
    initial.fcsr = 0;
    copy_mem(UINT64_C(0x80050000), source, sizeof(source), true);
    copy_mem(UINT64_C(0x80051000), scale, sizeof(scale), true);
    load_reg(5, UINT64_C(0x80050000), 2);
    load_reg(30, UINT64_C(0x80051000), 2);
    initial.fcsr = previous_nx;
    // Exactly the instruction in the new server log: FMUL v5, v30, v5.
    const uint32_t instruction = insn(0x32, 2, 5, 30, 5);
    CHECK(instruction == 0x645f228b);
    CPU_state state = run_raw(instruction, 0);
    CHECK(state.mcause == 0 && state.pc == code + 4 && state.fcsr == initial.fcsr);
    CHECK(pending() == 1);
    CHECK(complete(code, instruction, 1, false, 0) == 0);
    read_reg(5, actual);
    if (!previous_nx)
      memcpy(baseline, actual, sizeof(baseline));
    else
      CHECK(memcmp(baseline, actual, sizeof(baseline)) == 0);
    copy_regs(&state, false);
    CHECK(state.fcsr == initial.fcsr && pending() == 0);
  }
  printf("EVU softmax flags log reproduction: instr=0x645f228b, prior NX=0/1, completion NX=1, identical VRF\n");

  static const struct {
    unsigned opcode, flags;
    uint32_t lhs[2], rhs[2], result[2]; // FP16, FP32 encodings.
  } cases[] = {
    {0x32, 0x00, {0x3c00, 0x3f800000}, {0x4000, 0x40000000}, {0x4000, 0x40000000}},
    {0x32, 0x01, {0x3c01, 0x3f800001}, {0x3c01, 0x3f800001}, {0x3c02, 0x3f800002}},
    {0x32, 0x10, {0x7c00, 0x7f800000}, {0x0000, 0x00000000}, {0x7e00, 0x7fc00000}},
    {0x32, 0x05, {0x7bff, 0x7f7fffff}, {0x4000, 0x40000000}, {0x7c00, 0x7f800000}},
    {0x32, 0x03, {0x0001, 0x00000001}, {0x3800, 0x3f000000}, {0x0000, 0x00000000}},
    {0x34, 0x08, {0x3c00, 0x3f800000}, {0x0000, 0x00000000}, {0x7c00, 0x7f800000}},
    {0x34, 0x01, {0x3c00, 0x3f800000}, {0x4200, 0x40400000}, {0x3555, 0x3eaaaaab}},
  };
  unsigned checks = 0;
  for (unsigned sew = 1; sew <= 2; ++sew) {
    const unsigned bytes = 1u << sew, width = sew - 1, count = VLEN_BYTES / bytes;
    const uint32_t one = sew == 1 ? 0x3c00 : 0x3f800000;
    const uint32_t two = sew == 1 ? 0x4000 : 0x40000000;
    const uint32_t half = sew == 1 ? 0x3800 : 0x3f000000;
    initial.fcsr = 0;
    evu(setvli_insn(sew << 3, 0, 10), count, 0);
    for (unsigned test = 0; test < sizeof(cases) / sizeof(cases[0]); ++test) {
      uint8_t lhs[VLEN_BYTES], rhs[VLEN_BYTES];
      // Only the first element raises flags; later exact operations must not
      // erase them when rtl_vfpcall drains SoftFloat's exception accumulator.
      for (unsigned i = 0; i < count; ++i) {
        memcpy(lhs + i * bytes, i ? &one : &cases[test].lhs[width], bytes);
        memcpy(rhs + i * bytes, i ? &two : &cases[test].rhs[width], bytes);
      }
      copy_mem(UINT64_C(0x80050000), lhs, sizeof(lhs), true);
      copy_mem(UINT64_C(0x80051000), rhs, sizeof(rhs), true);
      load_reg(1, UINT64_C(0x80050000), sew);
      load_reg(2, UINT64_C(0x80051000), sew);
      for (unsigned asynchronous = 0; asynchronous < 2; ++asynchronous) {
        set_async(asynchronous);
        for (unsigned previous_flags = 0; previous_flags < 32; ++previous_flags) {
          initial.fcsr = previous_flags | ((previous_flags % 5) << 5);
          write_evu_csr(0x8c1, 0);
          const uint32_t instruction = insn(cases[test].opcode, sew, 3, 2, 1);
          CPU_state state = run_raw(instruction, 0);
          const uint64_t published = initial.fcsr;
          CHECK(state.mcause == 0 && state.pc == code + 4);
          CHECK(state.fcsr == initial.fcsr);
          CHECK(pending() == (int)asynchronous);
          if (asynchronous)
            CHECK(complete(code, instruction, cases[test].flags, false, 0) == 0);
          copy_regs(&state, false);
          CHECK(state.fcsr == published && pending() == 0);
          CHECK((state.mstatus & UINT64_C(0x6600)) == 0 && state.vstart == 37);
          uint64_t evu_state[13];
          CHECK(get_csr(evu_state, sizeof(evu_state)) == 0);
          CHECK(evu_state[5] == cases[test].flags);
          uint64_t csrs[4096];
          copy_csrs(csrs, false);
          CHECK((csrs[1] & 31) == (published & 31));
          read_reg(3, actual);
          for (unsigned i = 0; i < count; ++i) {
            uint32_t value = 0;
            memcpy(&value, actual + i * bytes, bytes);
            CHECK(value == (i ? (cases[test].opcode == 0x32 ? two : half) : cases[test].result[width]));
          }
          ++checks;
        }
      }
      set_async(true);
      initial.fcsr = 0;
    }
  }
  initial.fcsr = saved_initial_fcsr;
  initial.mstatus = saved_status;
  initial.vstart = 0;
  copy_regs(&initial, true);
  printf("EVU FP scalar history: %u FP16/FP32 cases, all 32 prior flags, sync/async publish and FRM preservation passed\n", checks);
}

/* Feed actual RTL flags/results/stores through the same asynchronous ABI as
 * EMU Difftest. This supplements, rather than replaces, full-core validation. */
static void check_softmax_dut_trace(const char *path, uint8_t scalar_flags) {
  const uint64_t saved_initial_fcsr = initial.fcsr;
  initial.fcsr = (initial.fcsr & ~UINT64_C(31)) | scalar_flags;
  FILE *trace = fopen(path, "r");
  CHECK(trace);
  CHECK(pending() == 0);
  set_async(true);
  evu(setvli_insn(0x10, 0, 10), VLEN_BYTES / 4, 0);
  unsigned opcode, sew, vd, vs1, vs2, flags, instructions = 0, stores = 0;
  uint32_t expected[VLEN_BYTES / 4], actual[VLEN_BYTES / 4];
  int fields;
  while ((fields = fscanf(trace, "%x %u %u %u %u %x", &opcode, &sew, &vd, &vs1, &vs2, &flags)) != EOF) {
    CHECK(fields == 6 && sew == 2 && vd < 32 && vs1 < 32 && vs2 < 32 && flags < 32);
    for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
      CHECK(fscanf(trace, "%x", &expected[i]) == 1);
    const uint64_t address = UINT64_C(0x80070000);
    if (opcode == 0x10)
      copy_mem(address, expected, sizeof(expected), true);
    const uint32_t instruction = insn(opcode, sew, vd, vs1, vs2);
    const uint64_t pc = run(instruction, address);
    if (opcode == 0x11) {
      for (unsigned offset = 0; offset < VLEN_BYTES; offset += 64)
        CHECK(store_event(address + offset, (const uint8_t *)expected + offset, UINT64_MAX) == 0);
      ++stores;
    }
    CHECK(complete(pc, instruction, flags, false, 0) == 0);
    CPU_state state;
    copy_regs(&state, false);
    CHECK((state.fcsr & 31) == scalar_flags);
    read_reg(vd, actual);
    for (unsigned i = 0; i < VLEN_BYTES / 4; ++i) {
      if (actual[i] != expected[i])
        fprintf(stderr, "RTL/NEMU instruction=%u opcode=0x%x vd=%u element=%u RTL=0x%08x NEMU=0x%08x\n",
                instructions, opcode, vd, i, expected[i], actual[i]);
      CHECK(actual[i] == expected[i]);
    }
    CHECK(pending() == 0);
    ++instructions;
  }
  CHECK(!ferror(trace) && fclose(trace) == 0);
  CHECK(instructions == 690 && stores == 112);
  initial.fcsr = saved_initial_fcsr;
  printf("EVU RTL/NEMU Difftest ABI: prior scalar flags=0x%02x, %u instructions, %u stores (%u physical beats), flags/VRF match; pending=0\n",
         scalar_flags, instructions, stores, stores * (VLEN_BYTES / 64));
}

static CPU_state csr_insn(unsigned addr, unsigned op, uint64_t value) {
  return run_raw((addr << 20) | (((value == 0 && op != 1) ? 0 : 10) << 15) | (op << 12) | (5 << 7) | 0x73, value);
}

static uint64_t csr_good(unsigned addr, unsigned op, uint64_t value) {
  CPU_state result = csr_insn(addr, op, value);
  if (result.mcause || result.pc != code + 4) fprintf(stderr, "CSR addr=%x op=%u value=%lx cause=%lu pc=%lx expected=%lx\n", addr, op, value, result.mcause, result.pc, code + 4);
  CHECK(result.mcause == 0 && result.pc == code + 4);
  CHECK(result.fcsr == initial.fcsr && result.vstart == initial.vstart);
  CHECK((result.mstatus & UINT64_C(0x6600)) == (initial.mstatus & UINT64_C(0x6600)));
  return result.gpr[5]._64;
}

static void check_config_v2(void) {
  CHECK(pending() == 0);
  CPU_state saved_initial = initial;
  uint64_t saved[4096];
  copy_csrs(saved, false);
  // Permit custom user/supervisor state before exercising those modes.
  csr_good(0x30c, 2, UINT64_C(1)); // mstateen0.C
  csr_good(0x10c, 2, UINT64_C(1)); // sstateen0.C
  csr_good(0xbc8, 1, 13); // Initial, permit U and S.
  for (unsigned type = 0; type < 256; ++type) {
    const unsigned sew = (type >> 3) & 7, lmul = type & 7;
    const bool invalid = sew > 2 || lmul == 4;
    const uint64_t base = VLEN_BYTES >> sew;
    const uint64_t maximum = invalid ? 0 : (lmul & 4) ? base >> (8 - lmul) : base << lmul;
    const uint64_t avls[] = {0, 1, 17, UINT32_MAX, UINT64_MAX, UINT64_C(1) << 32};
    for (unsigned a = 0; a < sizeof(avls) / sizeof(avls[0]); ++a) {
      for (unsigned reg_form = 0; reg_form < 2; ++reg_form) {
        config_type_input = type;
        const unsigned rd = reg_form ? 11 : 10; // Alias the VTYPE or AVL source.
        const uint32_t instruction = reg_form ? insn(0x46, 7, rd, 10, 11) : setvli_insn(type, rd, 10);
        CPU_state result = run_raw(instruction, avls[a]);
        uint64_t actual = avls[a] < maximum ? avls[a] : maximum;
        CHECK(result.mcause == 0 && result.gpr[rd]._64 == actual);
        // A later scalar write must survive asynchronous vector completion.
        result.gpr[rd]._64 = UINT64_C(0x123456789);
        copy_regs(&result, true);
        CHECK(complete(code, instruction, 0, false, 0) == 0);
        copy_regs(&result, false);
        CHECK(result.gpr[rd]._64 == UINT64_C(0x123456789));
        const uint64_t cfg = csr_good(0x8c0, 2, 0);
        CHECK((uint32_t)cfg == actual && (cfg >> 63) == invalid);
        CHECK(((cfg >> 32) & 255) == (invalid ? 0 : type));
      }
    }
  }
  config_type_input = UINT64_C(1) << 32;
  uint32_t instruction = insn(0x46, 7, 12, 10, 11);
  CHECK(run_raw(instruction, 128).gpr[12]._64 == 0);
  CHECK(complete(code, instruction, 0, false, 0) == 0);
  CHECK(csr_good(0x8c0, 2, 0) >> 63);
  CHECK(run_raw(insn(0x46, 0, 12, 10, 11), 128).mcause == 2);
  instruction = setvli_insn(0x10, 0, 10);
  CHECK(run_raw(instruction, 7).gpr[0]._64 == 0);
  CHECK(complete(code, instruction, 0, false, 0) == 0);
  CHECK((uint32_t)csr_good(0x8c0, 2, 0) == 7);
  instruction = setvli_insn(0x10, 12, 0);
  CHECK(run_raw(instruction, 128).gpr[12]._64 == 0);
  CHECK(complete(code, instruction, 0, false, 0) == 0);
  // The instruction remains available to enabled S/U; raw EVCFG writes do not.
#ifdef CONFIG_RV_PMP_CHECK
  // Allow instruction fetch from the test code in S/U, without granting data access.
  csr_good(0x3b0, 1, (code >> 18 << 16) | 0x7fff); // 256 KiB NAPOT
  csr_good(0x3a0, 1, 0x1c); // X | NAPOT
#endif
  for (unsigned mode = 0; mode <= 1; ++mode) {
    initial.mode = mode;
    CHECK(csr_insn(0x8c0, 1, 0).mcause == 2);
    instruction = setvli_insn(0x10, 12, 10);
    CPU_state result = run_raw(instruction, 9);
    CHECK(result.mcause == 0 && result.gpr[12]._64 == 9);
    CHECK(complete(code, instruction, 0, false, 0) == 0);
    CHECK((uint32_t)csr_good(0x8c0, 2, 0) == 9);
  }
  initial = saved_initial;
  config_type_input = 0;
  copy_regs(&initial, true);
  copy_csrs(saved, true);
  printf("EVU configuration ABI2: all VTYPE bytes, AVL clamp, aliases, no duplicate writeback and S/U restoration protection passed\n");
}

static void check_csr_isolation(void) {
  CHECK(pending() == 0);
  CPU_state saved_initial = initial;
  uint64_t saved_csrs[4096], snapshot[13], restored[13];
  copy_csrs(saved_csrs, false);
  initial.mstatus &= ~UINT64_C(0x6600); // Core FP and RVV are both disabled.
  initial.fcsr = 0x95;
  initial.vstart = 37;
  csr_good(0xbc8, 1, 0);
  CHECK(csr_insn(0x8c0, 2, 0).mcause == 2);
  CHECK(run_raw(insn(1, 0, 3, 2, 1), 0).mcause == 2);
  CHECK(csr_good(0xcd0, 2, 0) == VLEN_BYTES);
  CHECK((csr_good(0xcd1, 2, 0) & 255) == 2);
  CHECK(((csr_good(0xcd1, 2, 0) >> 8) & 255) == 0);
  CHECK(csr_good(0xcd1, 2, 0) & (UINT64_C(1) << 27));
  CHECK(((csr_good(0xcd1, 2, 0) >> 29) & 31) == 31);
  CHECK(csr_insn(0xcd0, 1, 0).mcause == 2);
  csr_good(0xbc8, 1, 1);
  const uint64_t enable = UINT64_C(1) << 40;
  csr_good(0x8c0, 1, enable | (UINT64_C(0x10) << 32) | UINT32_MAX);
  CHECK(csr_good(0x8c0, 2, 0) == (enable | (UINT64_C(0x10) << 32) | 128));
  csr_good(0x8c0, 1, enable | (UINT64_C(0x0f) << 32) | UINT32_MAX); // e16,mf2
  CHECK((uint32_t)csr_good(0x8c0, 2, 0) == 128);
  csr_good(0x8c0, 1, enable | (UINT64_C(0x10) << 32));
  CHECK((uint32_t)csr_good(0x8c0, 2, 0) == 0);
  csr_good(0x8c0, 1, enable | (UINT64_C(0x14) << 32) | 3); // reserved LMUL
  CHECK(csr_good(0x8c0, 2, 0) == (enable | (UINT64_C(1) << 63)));
  CHECK(run_raw(insn(1, 2, 3, 2, 1), 0).mcause == 2);
  csr_good(0x8c0, 1, enable | (UINT64_C(0x10) << 32) | 128);
  const uint32_t setvl = setvli_insn(0x10, 10, 10);
  CPU_state result = run_raw(setvl, UINT32_MAX);
  CHECK(result.mcause == 0 && result.gpr[10]._64 == 128); // rd=rs1, before external completion
  CHECK(complete(code, setvl, 0, false, 0) == 0);
  CHECK((uint32_t)csr_good(0x8c0, 2, 0) == 128);
  csr_good(0x8c1, 1, 5);
  CHECK(csr_good(0x8c1, 2, 8) == 5);
  CHECK(csr_good(0x8c1, 3, 1) == 13);
  CHECK(csr_good(0x8c1, 2, 0) == 12);
  csr_good(0x8c2, 1, UINT64_MAX);
  CHECK(csr_good(0x8c2, 2, 0) == 1);
  for (unsigned i = 0; i < 8; ++i) {
    CHECK(csr_insn(0x8d0 + i, 1, 1).mcause == 2);
    CHECK(csr_insn(0x8d0 + i, 2, 0).mcause == 2);
  }
  CHECK(get_csr(snapshot, sizeof(snapshot)) == 0);
  uint64_t checkpoint[4096];
  copy_csrs(checkpoint, false);
  csr_good(0x8c1, 1, 0);
  csr_good(0x8c2, 1, 0);
  copy_csrs(checkpoint, true);
  CHECK(get_csr(restored, sizeof(restored)) == 0);
  CHECK(memcmp(restored, snapshot, sizeof(restored)) == 0);
  csr_good(0xbc8, 1, 2); // Clean
  CHECK((csr_good(0x300, 2, 0) & (UINT64_C(3) << 15)) == (UINT64_C(2) << 15));
  csr_good(0x8c1, 1, 0);
  const uint64_t status = csr_good(0x300, 2, 0);
  CHECK(((status >> 15) & 3) == 3 && (status >> 63));
  CHECK((status & UINT64_C(0x6600)) == 0);
  csr_good(0xbc8, 1, 1);
  csr_good(0x8c0, 1, UINT64_MAX & ~(UINT64_C(1) << 40)); // normalized legacy state
  CHECK(csr_good(0x8c0, 2, 0) == ((UINT64_C(0x10) << 32) | 128));
  csr_good(0x8c0, 1, enable | (UINT64_C(0x10) << 32) | 128);
  const uint32_t failed_load = insn(0x10, 2, 7, 10, 0);
  uint8_t before[VLEN_BYTES], after[VLEN_BYTES];
  read_reg(7, before);
  uint64_t origin = run(failed_load, UINT64_C(0x80050000));
  CHECK(inject_fault(origin, 4, UINT64_C(0x80050020), true) == 0);
  CHECK(complete(origin, failed_load, 0, true, 4) == 0);
  CHECK(csr_good(0xcd2, 2, 0) == 6 && csr_good(0xcd3, 2, 0) == 4);
  CHECK(csr_good(0xcd4, 2, 0) == origin && csr_good(0xcd5, 2, 0) == UINT64_C(0x80050020));
  read_reg(7, after);
  CHECK(memcmp(before, after, sizeof(before)) == 0);
  uint64_t second = run(failed_load, UINT64_C(0x80050000));
  CHECK(inject_fault(second, 5, UINT64_C(0x80050040), false) == 0);
  CHECK(complete(second, failed_load, 0, true, 5) == 0);
  CHECK(csr_good(0xcd4, 2, 0) == origin && csr_good(0xcd3, 2, 0) == 4);
  csr_good(0xbc8, 1, 0x103);
  CHECK(csr_good(0xcd2, 2, 0) == 0 && csr_good(0xcd4, 2, 0) == 0);
  initial = saved_initial;
  copy_csrs(saved_csrs, true);
  copy_regs(&initial, true);
  printf("EVU CSR ABI2 isolation: permissions, returned VL, removed descriptors, context restore and core state preservation passed\n");
}

int main(int argc, char **argv) {
  CHECK(argc == 2 || argc == 3 || (argc == 4 &&
    (strcmp(argv[2], "--replay") == 0 || strcmp(argv[2], "--rvv-math") == 0 || strcmp(argv[2], "--lowp") == 0 || strcmp(argv[2], "--blocks") == 0)));
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    fprintf(stderr, "%s\n", dlerror());
    return 1;
  }
#define SYMBOL(target, name)                                                                                           \
  do {                                                                                                                 \
    *(void **)(&(target)) = dlsym(library, name);                                                                      \
    CHECK(target);                                                                                                     \
  } while (0)
  void (*init)(void);
  void (*ram_size)(size_t);
  void (*sync_interrupts)(void *);
  SYMBOL(init, "difftest_init");
  SYMBOL(ram_size, "difftest_set_ramsize");
  SYMBOL(sync_interrupts, "difftest_non_reg_interrupt_pending");
  SYMBOL(copy_mem, "difftest_memcpy");
  SYMBOL(copy_regs, "difftest_regcpy");
  SYMBOL(copy_csrs, "difftest_csrcpy");
  SYMBOL(execute, "difftest_exec");
  SYMBOL(set_async, "difftest_evu_set_async");
  SYMBOL(pending, "difftest_evu_pending");
  SYMBOL(get_reg, "difftest_evu_get_reg");
  SYMBOL(get_csr, "difftest_evu_get_csr");
  SYMBOL(write_evu_csr, "evu_csr_write");
  SYMBOL(inject_fault, "difftest_evu_inject_fault");
  SYMBOL(store_event, "difftest_evu_store");
  SYMBOL(complete, "difftest_evu_complete");

  ram_size(64 * 1024 * 1024);
  init();
  // EMU synchronizes pending interrupts before the first guest instruction.
  // Copied objects with stale CPU_state layouts used to crash here in AIA.
  struct NonRegInterruptPending no_interrupts = {0};
  for (unsigned i = 0; i < 20; ++i)
    sync_interrupts(&no_interrupts);
  printf("NEMU pre-commit interrupt synchronization passed\n");
  memset(&initial, 0, sizeof(initial));
  copy_regs(&initial, false);
  initial.mode = 3;
  initial.satp = 0;
  initial.mstatus = (UINT64_C(1) << 13) | (UINT64_C(1) << 9);
  initial.mtvec = UINT64_C(0x8000f000);
  initial.medeleg = initial.mideleg = initial.mie = 0;
  copy_regs(&initial, true);

  static uint64_t csrs[4096];
  memset(csrs, 0, sizeof(csrs));
  copy_csrs(csrs, false);
  csrs[0x744] |= UINT64_C(1) << 3;
  csrs[0xbc8] = 1;
  copy_csrs(csrs, true);
  set_async(true);
  if (argc == 4 && strcmp(argv[2], "--blocks") == 0) {
    check_blocks(argv[3]);
    dlclose(library);
    return 0;
  }
  if (argc == 3 && strcmp(argv[2], "--slide") == 0) {
    check_slides();
    dlclose(library);
    return 0;
  }
  if (argc == 4 && strcmp(argv[2], "--rvv-math") == 0) {
    check_rvv_math(argv[3]);
    dlclose(library);
    return 0;
  }
  if (argc == 4 && strcmp(argv[2], "--lowp") == 0) {
    check_low_precision(argv[3]);
    dlclose(library);
    return 0;
  }
  check_rvv_math(NULL);
  check_slides();
  check_vector_add_sequence();
  check_maximum_group_memory();
  check_gather_imm();
  check_scalar_fp_forms();
  check_memory_window();

  uint8_t lhs8[VLEN_BYTES], rhs8[VLEN_BYTES], result8[VLEN_BYTES];
  for (unsigned i = 0; i < VLEN_BYTES; ++i) {
    lhs8[i] = (uint8_t)(i * 3 + 1);
    rhs8[i] = (uint8_t)(i * 5 + 2);
  }
  copy_mem(UINT64_C(0x80010000), lhs8, sizeof(lhs8), true);
  copy_mem(UINT64_C(0x80010400), rhs8, sizeof(rhs8), true);
  load_reg(1, UINT64_C(0x80010000), 0);
  load_reg(2, UINT64_C(0x80010400), 0);
  const uint32_t add_instr = insn(0x01, 0, 3, 2, 1);
  const uint64_t add_pc = run(add_instr, 0);
  read_reg(3, result8);
  for (unsigned i = 0; i < VLEN_BYTES; ++i)
    CHECK(result8[i] == 0);
  CHECK(complete(add_pc, add_instr, 0, false, 0) == 0);
  read_reg(3, result8);
  for (unsigned i = 0; i < VLEN_BYTES; ++i)
    CHECK(result8[i] == (uint8_t)(lhs8[i] + rhs8[i]));
  check_int8_op(0x03, 18, lhs8, rhs8);
  check_int8_op(0x04, 19, lhs8, rhs8);
  check_int8_op(0x05, 20, lhs8, rhs8);
  check_int8_op(0x06, 21, lhs8, rhs8);
  check_int8_op(0x20, 22, lhs8, rhs8);
  check_int8_op(0x21, 23, lhs8, rhs8);
  check_int8_op(0x22, 24, lhs8, rhs8);
  check_int8_op(0x23, 25, lhs8, rhs8);
  check_int8_op(0x24, 26, lhs8, rhs8);
  check_int8_op(0x25, 27, lhs8, rhs8);
  check_int8_op(0x26, 28, lhs8, rhs8);
  check_int8_op(0x3c, 29, lhs8, rhs8);
  check_int8_op(0x3d, 30, lhs8, rhs8);
  copy_csrs(csrs, false);
  CHECK((csrs[0x8c2] & 1) == 1 && (csrs[0x009] & 1) == 0);

  uint8_t store_guard[VLEN_BYTES], store_actual[VLEN_BYTES];
  memset(store_guard, 0xa5, sizeof(store_guard));
  copy_mem(UINT64_C(0x80020000), store_guard, sizeof(store_guard), true);
  read_reg(1, lhs8);
  const uint32_t store_instr = insn(0x11, 0, 1, 10, 0);
  const uint64_t store_pc = run(store_instr, UINT64_C(0x80020000));
  for (unsigned line = 0; line < VLEN_BYTES / 64; ++line) {
    uint8_t data[64];
    memcpy(data, store_guard + line * 64, sizeof(data));
    memcpy(data, lhs8 + line * 64, sizeof(data));
    CHECK(store_event(UINT64_C(0x80020000) + line * 64, data, UINT64_MAX) == 0);
  }
  CHECK(complete(store_pc, store_instr, 0, false, 0) == 0);
  copy_mem(UINT64_C(0x80020000), store_actual, sizeof(store_actual), false);
  CHECK(memcmp(store_actual, lhs8, sizeof(store_actual)) == 0);

  uint8_t acc8[VLEN_BYTES], result16[VLEN_BYTES];
  for (unsigned i = 0; i < VLEN_BYTES; ++i)
    acc8[i] = (uint8_t)(i * 7 + 3);
  copy_mem(UINT64_C(0x8000f000), acc8, sizeof(acc8), true);
  load_reg(31, UINT64_C(0x8000f000), 0);
  evu(insn(0x07, 0, 31, 2, 1), 0, 0);
  read_reg(31, result8);
  for (unsigned i = 0; i < VLEN_BYTES; ++i)
    CHECK(result8[i] == (uint8_t)(lhs8[i] * rhs8[i] + acc8[i]));
  evu(insn(0x39, 0, 30, 2, 1), 0, 0);
  read_reg(30, result16);
  uint16_t *wide_result = (uint16_t *)result16;
  for (unsigned i = 0; i < VLEN_BYTES / 2; ++i) {
    const int16_t expected = (int16_t)((int8_t)lhs8[i] + (int8_t)rhs8[i]);
    CHECK(wide_result[i] == (uint16_t)expected);
  }

  uint16_t half_lhs[VLEN_BYTES / 2], half_rhs[VLEN_BYTES / 2], half_result[VLEN_BYTES / 2];
  for (unsigned i = 0; i < VLEN_BYTES / 2; ++i) {
    half_lhs[i] = 0x3c00; // 1.0
    half_rhs[i] = 0x3800; // 0.5
  }
  copy_mem(UINT64_C(0x8000e000), half_lhs, sizeof(half_lhs), true);
  copy_mem(UINT64_C(0x8000e400), half_rhs, sizeof(half_rhs), true);
  load_reg(18, UINT64_C(0x8000e000), 1);
  load_reg(19, UINT64_C(0x8000e400), 1);
  evu(insn(0x30, 1, 20, 19, 18), 0, 0);
  read_reg(20, half_result);
  for (unsigned i = 0; i < VLEN_BYTES / 2; ++i)
    CHECK(half_result[i] == 0x3e00); // 1.5

  uint32_t lhs32[VLEN_BYTES / 4], rhs32[VLEN_BYTES / 4], result32[VLEN_BYTES / 4];
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i) {
    lhs32[i] = i * 17u;
    rhs32[i] = i * 31u + 7u;
  }
  copy_mem(UINT64_C(0x80010800), lhs32, sizeof(lhs32), true);
  copy_mem(UINT64_C(0x80010c00), rhs32, sizeof(rhs32), true);
  load_reg(4, UINT64_C(0x80010800), 2);
  load_reg(5, UINT64_C(0x80010c00), 2);
  evu(insn(0x02, 2, 6, 5, 4), 0, 0);
  read_reg(6, result32);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(result32[i] == lhs32[i] - rhs32[i]);

  float fa[VLEN_BYTES / 4], fb[VLEN_BYTES / 4];
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i) {
    fa[i] = (float)(i & 7) + 0.25f;
    fb[i] = (float)(i & 3) + 0.5f;
  }
  copy_mem(UINT64_C(0x80011000), fa, sizeof(fa), true);
  copy_mem(UINT64_C(0x80011400), fb, sizeof(fb), true);
  load_reg(7, UINT64_C(0x80011000), 2);
  load_reg(8, UINT64_C(0x80011400), 2);
  evu(insn(0x30, 2, 9, 7, 8), 0, 0);
  read_reg(9, result32);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(result32[i] == float_bits(fa[i] + fb[i]));
  evu(insn(0x31, 2, 18, 8, 7), 0, 0);
  read_reg(18, result32);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(result32[i] == float_bits(fa[i] - fb[i]));
  evu(insn(0x32, 2, 19, 8, 7), 0, 0);
  read_reg(19, result32);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(result32[i] == float_bits(fa[i] * fb[i]));
  evu(insn(0x35, 2, 21, 8, 7), 0, 0);
  read_reg(21, result32);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(result32[i] == float_bits(fa[i] < fb[i] ? fa[i] : fb[i]));
  evu(insn(0x36, 2, 22, 8, 7), 0, 0);
  read_reg(22, result32);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(result32[i] == float_bits(fa[i] > fb[i] ? fa[i] : fb[i]));

  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    fb[i] = 1.0f;
  copy_mem(UINT64_C(0x80011400), fb, sizeof(fb), true);
  load_reg(8, UINT64_C(0x80011400), 2);
  load_reg(23, UINT64_C(0x80011000), 2);
  evu(insn(0x33, 2, 23, 8, 7), 0, 0);
  read_reg(23, result32);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(result32[i] == float_bits(fa[i] * 1.0f + fa[i]));

  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i) {
    fa[i] = 1.0f;
    fb[i] = 3.0f;
  }
  copy_mem(UINT64_C(0x80011000), fa, sizeof(fa), true);
  copy_mem(UINT64_C(0x80011400), fb, sizeof(fb), true);
  load_reg(27, UINT64_C(0x80011000), 2);
  load_reg(28, UINT64_C(0x80011400), 2);
  const uint32_t div_instr = insn(0x34, 2, 29, 28, 27);
  const uint64_t div_pc = run(div_instr, 0);
  copy_csrs(csrs, false);
  CHECK((csrs[0x8c1] & 1) == 0 && (csrs[1] & 1) == 0);
  CHECK(complete(div_pc, div_instr, 1, false, 0) == 0);
  read_reg(29, result32);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(result32[i] == float_bits(1.0f / 3.0f));
  copy_csrs(csrs, false);
  CHECK((csrs[0x8c1] & 1) == 1 && (csrs[1] & 1) == 0);

  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i) {
    fa[i] = 1.0f;
    fb[i] = 2.0f;
  }
  copy_mem(UINT64_C(0x80011000), fa, sizeof(fa), true);
  copy_mem(UINT64_C(0x80011400), fb, sizeof(fb), true);
  load_reg(24, UINT64_C(0x80011000), 2);
  load_reg(25, UINT64_C(0x80011400), 2);
  evu(insn(0x37, 2, 26, 24, 25), 0, 0);
  read_reg(26, result32);
  CHECK(result32[0] == float_bits(1.0f + 2.0f * (VLEN_BYTES / 4)));

  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    lhs32[i] = i + 3;
  lhs32[0] = 7;
  copy_mem(UINT64_C(0x80010800), lhs32, sizeof(lhs32), true);
  load_reg(10, UINT64_C(0x80010800), 2);
  evu(insn(0x0c, 2, 11, 10, 5), 0, 0);
  read_reg(11, result32);
  uint64_t reduction = 7;
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    reduction += rhs32[i];
  CHECK(result32[0] == (uint32_t)reduction);

  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    lhs32[i] = float_bits(i & 1 ? 2.0f : 1.0f);
  copy_mem(UINT64_C(0x80010800), lhs32, sizeof(lhs32), true);
  load_reg(12, UINT64_C(0x80010800), 2);
  evu(insn(0x0f, 2, 13, 0, 12), 0, 0);
  read_reg(13, result32);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(result32[i] == float_bits(i & 1 ? 4.0f : 2.0f));

  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    lhs32[i] = i * 3;
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    rhs32[i] = (i * 7) % (VLEN_BYTES / 4);
  copy_mem(UINT64_C(0x80010800), lhs32, sizeof(lhs32), true);
  copy_mem(UINT64_C(0x80010c00), rhs32, sizeof(rhs32), true);
  load_reg(1, UINT64_C(0x80010800), 2);
  load_reg(2, UINT64_C(0x80010c00), 2);
  evu(insn(0x47, 2, 3, 2, 1), 0, 0);
  read_reg(3, result32);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(result32[i] == lhs32[rhs32[i]]);

  rhs32[0] = 5;
  copy_mem(UINT64_C(0x80010c00), rhs32, sizeof(rhs32), true);
  load_reg(2, UINT64_C(0x80010c00), 2);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    result32[i] = 0xdead0000u + i;
  copy_mem(UINT64_C(0x80011400), result32, sizeof(result32), true);
  load_reg(3, UINT64_C(0x80011400), 2);
  evu(insn(0x3e, 2, 3, 10, 1), 5, 0);
  read_reg(3, result32);
  for (unsigned i = 0; i < VLEN_BYTES / 4; ++i)
    CHECK(result32[i] == (i < 5 ? 0xdead0000u + i : lhs32[i - 5]));

  uint8_t tail_marker[VLEN_BYTES];
  evu(setvli_insn(0, 0, 10), VLEN_BYTES, 0);
  for (unsigned i = 0; i < VLEN_BYTES; ++i)
    tail_marker[i] = (uint8_t)(0x80 + (i & 0x3f));
  copy_mem(UINT64_C(0x80012400), tail_marker, sizeof(tail_marker), true);
  load_reg(21, UINT64_C(0x80012400), 0);
  load_reg(0, UINT64_C(0x80012400), 0);
  evu(setvli_insn(0, 0, 10), 64, 0);
  uint64_t evu_csrs[13];
  CHECK(get_csr(evu_csrs, sizeof(evu_csrs)) == 0);
  CHECK(evu_csrs[0] == 64);
  CHECK(evu_csrs[1] == 0); // setvli_insn(0, ...) requests e8/m1, tail undisturbed.
  CHECK(evu_csrs[4] == 1);
  load_reg(21, UINT64_C(0x80010000), 0);
  read_reg(21, result8);
  for (unsigned i = 0; i < 64; ++i) {
    if (result8[i] != lhs8[i]) {
      fprintf(stderr, "load-vl i=%u got=%u expected=%u\n", i, result8[i], lhs8[i]);
      CHECK(0);
    }
  }
  for (unsigned i = 64; i < VLEN_BYTES; ++i)
    CHECK(result8[i] == tail_marker[i]);
  load_reg(0, UINT64_C(0x80010000), 0);
  read_reg(0, result8);
  for (unsigned i = 0; i < 64; ++i) {
    if (result8[i] != lhs8[i]) {
      fprintf(stderr, "load-v0 i=%u got=%u expected=%u\n", i, result8[i], lhs8[i]);
      CHECK(0);
    }
  }
  for (unsigned i = 64; i < VLEN_BYTES; ++i)
    CHECK(result8[i] == tail_marker[i]);

  // The retired-opcode checks below compare v0 against this committed image.
  read_reg(0, result8);

  CHECK(pending() == 0);
  set_async(false);

  CPU_state invalid = run_raw(insn(0x48, 2, 0, 0, 0), 0);
  CHECK(invalid.mcause == 2 && pending() == 0);
  invalid = run_raw(insn(0x39, 2, 0, 0, 0), 0);
  CHECK(invalid.mcause == 2 && pending() == 0);
  const unsigned retired[] = {0x09, 0x0b, 0x18, 0x19, 0x38, 0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x5e};
  for (unsigned vl_case = 0; vl_case < 2; ++vl_case) {
    write_evu_csr(0x8c0, (UINT64_C(1) << 40) | (UINT64_C(16) << 32) | (vl_case ? 9 : 0));
    for (unsigned op = 0; op < sizeof(retired) / sizeof(retired[0]); ++op)
      for (unsigned sew = 0; sew < 8; ++sew) {
        invalid = run_raw(insn(retired[op], sew, 0, 1, 0), 0);
        CHECK(invalid.mcause == 2 && pending() == 0);
        uint8_t mask_after_invalid[VLEN_BYTES];
        read_reg(0, mask_after_invalid);
        CHECK(memcmp(mask_after_invalid, result8, sizeof(result8)) == 0);
      }
  }

  check_unordered_reductions();
  check_fp_flags_with_scalar_history();
  check_online_softmax(argc == 3 ? argv[2] : NULL);
  if (argc == 4) {
    const uint8_t scalar_flags[] = {0x00, 0x01, 0x1f};
    for (unsigned i = 0; i < sizeof(scalar_flags) / sizeof(scalar_flags[0]); ++i)
      check_softmax_dut_trace(argv[3], scalar_flags[i]);
  }
  check_csr_isolation();
  check_config_v2();
  printf("EVU compute: integer/FP, reduction, Exp2, VL state and retired opcodes passed\n");
  dlclose(library);
  return 0;
}
