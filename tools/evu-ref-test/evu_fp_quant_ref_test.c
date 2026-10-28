/* Native instruction/API regression; does not build a guest or run RTL. */
#include <isa.h>
#include <dlfcn.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static void (*copy_mem)(uint64_t, void *, size_t, bool);
static void (*copy_regs)(void *, bool);
static void (*copy_csrs)(void *, bool);
static void (*execute)(uint64_t);
static void (*set_async)(bool);
static void (*write_csr)(uint32_t, uint64_t);
static int (*complete)(uint64_t, uint32_t, uint8_t, bool, uint8_t, bool);
static int (*cute_write)(uint32_t, uint32_t, uint32_t, const void *, size_t, uint64_t);
static int (*get_reg)(uint32_t, void *, size_t);
static int (*pending)(void);
static uint64_t pc = 0x80008000;
static uint32_t insn(unsigned opcode, unsigned f7, unsigned f3, unsigned rd, unsigned rs1, unsigned rs2) {
  return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opcode;
}
static void run(uint32_t instruction, uint64_t arg1, uint64_t arg2) {
  CPU_state state;
  copy_regs(&state, false);
  pc += 4; state.pc = pc; state.mcause = 0;
  state.gpr[10]._64 = arg1; state.gpr[11]._64 = arg2;
  copy_regs(&state, true); copy_mem(pc, &instruction, 4, true); execute(1);
  copy_regs(&state, false);
  CHECK(state.mcause == 0 && state.pc == pc + 4);
}
static uint32_t raw(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static float from_raw(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static uint8_t flags(void) {
  int f = fetestexcept(FE_ALL_EXCEPT);
  return ((f & FE_INVALID) ? 16 : 0) | ((f & FE_DIVBYZERO) ? 8 : 0) |
    ((f & FE_OVERFLOW) ? 4 : 0) | ((f & FE_UNDERFLOW) ? 2 : 0) | ((f & FE_INEXACT) ? 1 : 0);
}
static uint8_t quantize(float input, float scale, uint8_t *out_flags) {
  feclearexcept(FE_ALL_EXCEPT);
  volatile float a = input * scale;
  volatile float b = a + 0.5f;
  float c = fminf(b, 127.0f);
  c = fmaxf(c, 0.0f);
  uint8_t result = (uint8_t)c;
  *out_flags |= flags() | ((float)result != c ? 1 : 0);
  return result;
}
static void load(unsigned reg, unsigned vl, uint32_t *words, uint64_t address) {
  write_csr(0x8c0, (UINT64_C(1) << 40) | (UINT64_C(16) << 32) | vl);
  copy_mem(address, words, 512, true);
  uint32_t i = insn(0x0b, 0x10, 2, reg, 10, 0);
  run(i, address, 0);
  if (pending()) CHECK(complete(pc, i, 0, false, 0, false) == 0);
}
int main(int argc, char **argv) {
  CHECK(argc == 2);
  void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
#define SYM(target, name) do { *(void **)(&(target)) = dlsym(lib, name); CHECK(target); } while (0)
  void (*init)(void); void (*set_ram)(size_t);
  SYM(init, "difftest_init"); SYM(set_ram, "difftest_set_ramsize");
  SYM(copy_mem, "difftest_memcpy"); SYM(copy_regs, "difftest_regcpy"); SYM(copy_csrs, "difftest_csrcpy");
  SYM(execute, "difftest_exec"); SYM(set_async, "difftest_evu_set_async"); SYM(write_csr, "evu_csr_write");
  SYM(complete, "difftest_evu_complete_v2"); SYM(cute_write, "difftest_evu_cute_write");
  SYM(get_reg, "difftest_evu_get_reg"); SYM(pending, "difftest_evu_pending");
  set_ram(64 * 1024 * 1024); init();
  CPU_state state; copy_regs(&state, false);
  state.mode = 3; state.satp = 0; state.mtvec = 0x8000f000;
  state.mstatus = (UINT64_C(1) << 13) | (UINT64_C(1) << 9);
  copy_regs(&state, true);
  uint64_t csrs[4096]; copy_csrs(csrs, false); csrs[0xbc8] = 9;
  csrs[0x30c] |= 1; csrs[0x744] |= UINT64_C(1) << 3; copy_csrs(csrs, true);
  set_async(true); fesetround(FE_TONEAREST);
  uint32_t input[128], scale[128], snapshot[128], after[128];
  const uint32_t patterns[] = {0, 0x80000000, 0x3f000000, 0x3effffff, 0x3f000001,
    0x3fc00000, 0xc1200000, 0x42fe0000, 0x43000000, 0x7f800000, 0xff800000,
    0x7fc00001, 0x7f800001, 0x00800000, 1, 0x3eaaaaab};
  for (unsigned i = 0; i < 128; ++i) { input[i] = patterns[i % 16]; scale[i] = raw(i % 3 == 0 ? 0.5f : 1.0f); }
  load(3, 128, input, 0x80010000); load(4, 128, scale, 0x80010400);
  CHECK(get_reg(3, snapshot, 512) == 0);
  const unsigned lengths[] = {0, 1, 7, 8, 9, 31, 32, 33, 63, 64};
  unsigned writes = 0;
  for (unsigned t = 0; t < sizeof(lengths) / sizeof(lengths[0]); ++t) {
    unsigned vl = lengths[t], row = t % 2 ? 127 : 8, reg = t % 4;
    write_csr(0x8c0, (UINT64_C(1) << 40) | (UINT64_C(16) << 32) | vl);
    uint32_t q = insn(0x0b, 0x50, 2, 4, 10, 3);
    uint8_t expected[64], f = 0;
    for (unsigned i = 0; i < vl; ++i) expected[i] = quantize(from_raw(input[i]), from_raw(scale[i]), &f);
    run(q, (row << 2) | reg, 0);
    if (vl) CHECK(complete(pc, q, f, false, 0, false) != 0); // missing writes
    for (unsigned start = 0; start < vl; start += 8) {
      uint8_t data[32] = {0}; unsigned n = vl - start < 8 ? vl - start : 8;
      unsigned offset = start % 32, bank = row % 8, entry = (row / 8) * 2 + start / 32;
      uint64_t mask = ((UINT64_C(1) << n) - 1) << offset;
      memcpy(data + offset, expected + start, n);
      CHECK(cute_write(reg ^ 1, bank, entry, data, 32, mask) != 0);
      CHECK(cute_write(reg, bank ^ 1, entry, data, 32, mask) != 0);
      CHECK(cute_write(reg, bank, entry ^ 1, data, 32, mask) != 0);
      CHECK(cute_write(reg, bank, entry, data, 32, mask ^ (UINT64_C(1) << offset)) != 0);
      data[offset] ^= 1; CHECK(cute_write(reg, bank, entry, data, 32, mask) != 0); data[offset] ^= 1;
      CHECK(cute_write(reg, bank, entry, data, 32, mask) == 0);
      CHECK(cute_write(reg, bank, entry, data, 32, mask) != 0); // duplicate
      ++writes;
    }
    CHECK(complete(pc, q, f ^ 1, false, 0, false) != 0);
    CHECK(complete(pc, q, f, false, 0, true) != 0);
    CHECK(complete(pc, q, f, false, 0, false) == 0);
    CHECK(get_reg(3, after, 512) == 0 && !memcmp(after, snapshot, 512));
  }
  CHECK(pending() == 0);

  // Produce a complete P in AB register 2 before launch, then independently
  // evaluate PV. A launch must not clear the quantized AB bytes.
  set_async(false);
  for (unsigned i = 0; i < 128; ++i) scale[i] = raw(1.0f);
  load(4, 128, scale, 0x80010400);
  for (unsigned row = 0; row < 128; ++row) {
    for (unsigned k = 0; k < 128; ++k) input[k] = raw((float)((row + k) % 7));
    load(3, 64, input, 0x80010000);
    run(insn(0x0b, 0x50, 2, 4, 10, 3), (row << 2) | 2, 0);
  }
  load(1, 128, scale, 0x80010400); load(2, 128, scale, 0x80010400);
  run(insn(0x0b, 0x18, 4, 2, 10, 1), 0x80030000, 0);
  run(insn(0x2b, 0x21, 0, 0, 10, 0), 2, 0);
  run(insn(0x2b, 0x21, 0, 2, 10, 0), 2, 0);
  run(insn(0x2b, 0x21, 0, 4, 10, 0), 4, 0);
  run(insn(0x2b, 0x11, 0, 0, 10, 0), 128, 0);
  run(insn(0x2b, 0x19, 0, 0, 10, 0), 128, 0);
  run(insn(0x2b, 0x09, 0, 0, 10, 0), 64, 0);
  static int8_t v[128][64]; memset(v, 1, sizeof(v)); copy_mem(0x80020000, v, sizeof(v), true);
  run(insn(0x2b, 0, 3, 4, 0, 0), 0, 0);
  run(insn(0x2b, 0, 1, 0, 10, 11), 0x80020000, 64);
  run(insn(0x2b, 0, 2, 4, 2, 0), 0, 0);
  static float output[128][128]; copy_mem(0x80030000, output, sizeof(output), false);
  for (unsigned q = 0; q < 128; ++q) {
    int sum = 0; for (unsigned k = 0; k < 64; ++k) sum += (q + k) % 7;
    for (unsigned d = 0; d < 128; ++d) CHECK(output[q][d] == (float)sum);
  }
  run(insn(0x0b, 0x1a, 0, 0, 0, 0), 0, 0);
  CHECK(pending() == 0);
  printf("EVU FP quant: %u FP checked writes; tails, flags, special values, wrong/duplicate/missing writes, VRF preservation, P-before-launch and PV passed\n", writes);
  dlclose(lib); return 0;
}
