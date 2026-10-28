#ifndef NEMU_EVU_INTERNAL_H
#define NEMU_EVU_INTERNAL_H

#include <common.h>
#include <cpu/difftest/evu.h>

#define EVU_VLEN_BITS CONFIG_CUSTOM_EVU_VLEN
#define EVU_VLEN_BYTES (EVU_VLEN_BITS / 8)
#define EVU_DLEN_BYTES (CONFIG_CUSTOM_EVU_DLEN / 8)
#define EVU_MAX_ELEMS EVU_VLEN_BYTES
#define EVU_MAX_GROUP_BYTES (8 * EVU_VLEN_BYTES)
#define EVU_DESCRIPTORS 8
#define EVU_QUEUE 64

enum {
  EVU_INVALID = 0x00,
  EVU_ADD = 0x01,
  EVU_SUB = 0x02,
  EVU_AND = 0x03,
  EVU_OR = 0x04,
  EVU_XOR = 0x05,
  EVU_MUL = 0x06,
  EVU_MAC = 0x07,
  /* Retired VRF quantization opcodes 0x09 and 0x0b remain reserved. */
  EVU_REDUCE_SUM = 0x0c,
  EVU_REDUCE_MAX = 0x0d,
  EVU_REDUCE_MIN = 0x0e,
  EVU_EXP2 = 0x0f,
  EVU_LOAD = 0x10,
  EVU_STORE = 0x11,
  EVU_LAUNCH = 0x18,
  EVU_WAIT = 0x19,
  EVU_FENCE = 0x1a,
  EVU_SLL = 0x20,
  EVU_SRL = 0x21,
  EVU_SRA = 0x22,
  EVU_MIN = 0x23,
  EVU_MAX = 0x24,
  EVU_MINU = 0x25,
  EVU_MAXU = 0x26,
  EVU_FADD = 0x30,
  EVU_FSUB = 0x31,
  EVU_FMUL = 0x32,
  EVU_FMACC = 0x33,
  EVU_FDIV = 0x34,
  EVU_FMIN = 0x35,
  EVU_FMAX = 0x36,
  EVU_FREDUCE_SUM = 0x37,
  EVU_WADD = 0x39,
  EVU_WSUB = 0x3a,
  EVU_WMUL = 0x3b,
  EVU_SATADD = 0x3c,
  EVU_SATSUB = 0x3d,
  EVU_SLIDE_UP = 0x3e,
  EVU_SLIDE_DOWN = 0x3f,
  EVU_REDUCE_AND = 0x40,
  EVU_REDUCE_OR = 0x41,
  EVU_REDUCE_XOR = 0x42,
  EVU_FREDUCE_MIN = 0x43,
  EVU_FREDUCE_MAX = 0x44,
  EVU_SETVLI = 0x45,
  EVU_SETVL = 0x46,
  EVU_GATHER = 0x47,
  EVU_SCATTER = 0x48,
  EVU_FP_QUANT_TO_CUTE = 0x50,
  EVU_GATHER_IMM = 0x51,
  EVU_LOAD_E0 = 0x52,
  EVU_STORE_E0 = 0x53,
  EVU_FADD_E0 = 0x54,
  EVU_FSUB_E0 = 0x55,
  EVU_FMUL_E0 = 0x56,
  EVU_FDIV_E0 = 0x57,
  EVU_FMAX_E0 = 0x58,
  EVU_EXP2_E0 = 0x59,
  EVU_FADD_SCALAR = 0x5a,
  EVU_FSUB_SCALAR = 0x5b,
  EVU_FMUL_SCALAR = 0x5c,
  EVU_FMAX_SCALAR = 0x5d,
  EVU_FP_QUANT_SCALAR = 0x5e,
  /* Retired SETMASK opcode 0x4d remains reserved. */
  /* Retired fixed-point QuantToCute (0x4c) and SETCFG (0x4e) are reserved. */
  /* Retired VRF quantization opcode 0x4f remains reserved. */
};

static inline bool evu_element_zero(uint32_t opcode) {
  return opcode >= EVU_LOAD_E0 && opcode <= EVU_EXP2_E0;
}
static inline bool evu_scalar_source(uint32_t opcode) {
  return opcode >= EVU_FADD_SCALAR && opcode <= EVU_FMAX_SCALAR;
}
static inline bool evu_scalar_form(uint32_t opcode) {
  return evu_element_zero(opcode) || evu_scalar_source(opcode) || opcode == EVU_FP_QUANT_SCALAR;
}
static inline uint32_t evu_canonical_opcode(uint32_t opcode) {
  switch (opcode) {
  case EVU_LOAD_E0: return EVU_LOAD;
  case EVU_STORE_E0: return EVU_STORE;
  case EVU_FADD_E0: case EVU_FADD_SCALAR: return EVU_FADD;
  case EVU_FSUB_E0: case EVU_FSUB_SCALAR: return EVU_FSUB;
  case EVU_FMUL_E0: case EVU_FMUL_SCALAR: return EVU_FMUL;
  case EVU_FDIV_E0: return EVU_FDIV;
  case EVU_FMAX_E0: case EVU_FMAX_SCALAR: return EVU_FMAX;
  case EVU_EXP2_E0: return EVU_EXP2;
  case EVU_FP_QUANT_SCALAR: return EVU_FP_QUANT_TO_CUTE;
  default: return opcode;
  }
}

typedef struct {
  uint64_t pc;
  uint64_t base;
  uint32_t instr;

  uint8_t opcode;
  uint8_t sew;
  uint8_t vd;
  uint8_t vs1;
  uint8_t vs2;
  uint8_t aux;
  uint32_t vl;
  uint32_t vstart;

  bool writes_vrf;
  bool writes_memory;
  bool writes_cute;
  bool writes_scalar;
  bool uses_fp;
  bool updates_vector_state;
  uint8_t write_reg;
  uint8_t write_regs;
  uint8_t scalar_rd;
  uint64_t scalar_value;

  uint8_t output[EVU_MAX_GROUP_BYTES];
  uint32_t output_bytes;
  uint64_t physical[EVU_MAX_GROUP_BYTES];
  uint8_t observed[EVU_MAX_GROUP_BYTES];
  uint32_t written;

  uint8_t flags;
  bool vxsat;
  uint8_t cause;
  uint64_t fault_address;
  bool fault_address_valid;
} EvuJob;

typedef struct {
  uint32_t sew;
  uint32_t lmul;
  bool vta;
  bool vma;
  bool vill;
} EvuVType;

void evu_compute_reset(void);
void evu_compute_execute(Decode *s, EvuJob *job);
void evu_read_register(uint32_t reg, void *data);
void evu_publish_completion_flags(EvuJob *job);

EvuVType evu_current_vtype(void);
uint32_t evu_current_vl(uint32_t sew);
uint32_t evu_vlmax(uint32_t sew, uint32_t lmul);
uint8_t evu_lmul_regs(uint32_t lmul);

#endif
