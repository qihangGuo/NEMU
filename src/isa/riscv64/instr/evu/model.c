#include <common.h>

#ifdef CONFIG_CUSTOM_EVU
#include "../../local-include/csr.h"
#include "../../local-include/intr.h"
#include "evu-internal.h"
#include "matrix.h"
#include <cpu/cpu.h>
#include <cpu/difftest/evu.h>
#include <memory/host.h>
#include <memory/paddr.h>
#include <memory/vaddr.h>
#include <softfloat.h>
#include <stdint.h>
#include <string.h>

_Static_assert(CONFIG_CUSTOM_EVU_VLEN % 512 == 0, "EVU requires complete 64-byte packets");
_Static_assert(CONFIG_CUSTOM_EVU_DLEN % 64 == 0 && CONFIG_CUSTOM_EVU_VLEN % CONFIG_CUSTOM_EVU_DLEN == 0,
               "EVU DLEN must divide VLEN and contain complete beats");

static EvuJob jobs[EVU_QUEUE];
static unsigned head, count;
static bool asynchronous;
static uint32_t defined_registers;

static const uint16_t evu_csr_addresses[] = {
  0x8c0, 0x8c1, 0x8c2, 0x8d0, 0x8d1, 0x8d2, 0x8d3, 0x8d4, 0x8d5, 0x8d6, 0x8d7,
  0xbc8, 0xcd0, 0xcd1, 0xcd2, 0xcd3, 0xcd4, 0xcd5
};

bool evu_csr_address(uint32_t addr) {
  for (unsigned i = 0; i < sizeof(evu_csr_addresses) / sizeof(evu_csr_addresses[0]); ++i)
    if (addr == evu_csr_addresses[i]) return true;
  return false;
}

bool evu_permitted(void) {
  if (cpu.evu.state % 4 == 0) return false;
  IFDEF(CONFIG_RVH, if (cpu.v) return false);
#ifdef CONFIG_RV_SMSTATEEN
  if (cpu.mode != MODE_M && !mstateen0->c) return false;
  if (cpu.mode == MODE_U && !sstateen0->c) return false;
#endif
  return cpu.mode == MODE_M || (cpu.mode == MODE_S && (cpu.evu.state & 8)) ||
    (cpu.mode == MODE_U && (cpu.evu.state & 4));
}

void evu_csr_check(uint32_t addr) {
  IFDEF(CONFIG_RVH, if (cpu.v) longjmp_exception(EX_II));
  if (addr == 0xbc8) {
    if (cpu.mode != MODE_M) longjmp_exception(EX_II);
  } else if (addr != 0xcd0 && addr != 0xcd1 && !evu_permitted()) {
    longjmp_exception(EX_II);
  }
}

static void evu_summary(void) {
  mstatus->xs = cpu.evu.state & 3;
  mstatus->sd = mstatus->xs == 3 || mstatus->fs == 3 || mstatus->vs == 3 || mstatus->ms == 3;
  IFDEF(CONFIG_SHARE, csr_difftest_mark_dirty());
}

void evu_context_dirty(void) {
  cpu.evu.state |= 3;
  evu_summary();
}

static void configure(uint64_t avl, uint8_t type, bool enabled, bool force_vill) {
  const uint32_t sew = (type >> 3) & 7, lmul = type & 7;
  const uint32_t maximum = evu_vlmax(sew, lmul);
  const bool illegal = force_vill || sew > 2 || lmul == 4 || maximum == 0;
  cpu.evu.configured = enabled;
  cpu.evu.vtype = !enabled ? 0x10 : illegal ? UINT64_C(1) << 63 : type;
  cpu.evu.vl = !enabled ? EVU_VLEN_BITS / 32 : illegal ? 0 : avl < maximum ? avl : maximum;
  cpu.evu.vstart = 0;
  evu_context_dirty();
}

static uint64_t evu_csr_peek(uint32_t addr) {
  switch (addr) {
  case 0x8c0: return (cpu.evu.vtype & (UINT64_C(1) << 63)) | (cpu.evu.configured << 40) |
    ((cpu.evu.vtype & 255) << 32) | cpu.evu.vl;
  case 0x8c1: return cpu.evu.fflags;
  case 0x8c2: return cpu.evu.vxsat;
  case 0xbc8: return cpu.evu.state;
  case 0xcd0: return EVU_VLEN_BYTES;
  case 0xcd1: return UINT64_C(1) | (UINT64_C(8) << 8) | (UINT64_C(7) << 16) | (UINT64_C(1) << 24) |
    ((uint64_t)(CONFIG_CUSTOM_EVU_DLEN == 256) << 25) | (UINT64_C(1) << 26);
  case 0xcd2: return cpu.evu.status;
  case 0xcd3: return cpu.evu.cause;
  case 0xcd4: return cpu.evu.epc;
  case 0xcd5: return cpu.evu.tval;
  default: return cpu.evu.descriptors[addr - 0x8d0];
  }
}

static void evu_csr_drain(void) {
  Assert(count == 0, "EVU control access before older completion");
  Assert(!evu_matrix_drain(), "EVU control access before matrix writes drained");
  const uint8_t flags = evu_matrix_take_flags();
  cpu.evu.fflags |= flags;
  if (flags) evu_context_dirty();
}

uint64_t evu_csr_read(uint32_t addr) {
  if (addr != 0xcd0 && addr != 0xcd1) evu_csr_drain();
  return evu_csr_peek(addr);
}

void evu_csr_write(uint32_t addr, uint64_t value) {
  evu_csr_drain();
  switch (addr) {
  case 0x8c0: configure((uint32_t)value, value >> 32, (value >> 40) & 1, value >> 63); break;
  case 0x8c1: cpu.evu.fflags = value & 31; evu_context_dirty(); break;
  case 0x8c2: cpu.evu.vxsat = value & 1; evu_context_dirty(); break;
  case 0xbc8:
    cpu.evu.state = value & 15;
    if (value & 256) cpu.evu.status = cpu.evu.cause = cpu.evu.epc = cpu.evu.tval = 0;
    evu_summary();
    break;
  default:
    if (addr >= 0x8d0 && addr <= 0x8d7) {
      cpu.evu.descriptors[addr - 0x8d0] = ((value >> 48) & 3) <= 2 ? value & ((UINT64_C(1) << 51) - 1) : 0;
      evu_context_dirty();
    }
  }
}

void evu_csr_export(uint64_t *values) {
  for (unsigned i = 0; i < sizeof(evu_csr_addresses) / sizeof(evu_csr_addresses[0]); ++i)
    values[evu_csr_addresses[i]] = evu_csr_peek(evu_csr_addresses[i]);
}

void evu_csr_import(const uint64_t *values) {
  // Debugger/checkpoint transfer, not guest execution; do not drain or dirty.
  const uint64_t cfg = values[0x8c0];
  cpu.evu.configured = (cfg >> 40) & 1;
  cpu.evu.vtype = (cfg & (UINT64_C(1) << 63)) | ((cfg >> 32) & 255);
  cpu.evu.vl = (uint32_t)cfg;
  cpu.evu.vstart = 0;
  cpu.evu.fflags = values[0x8c1] & 31;
  cpu.evu.vxsat = values[0x8c2] & 1;
  cpu.evu.state = values[0xbc8] & 15;
  cpu.evu.status = values[0xcd2] & 6;
  cpu.evu.cause = values[0xcd3] & 15;
  cpu.evu.epc = values[0xcd4];
  cpu.evu.tval = values[0xcd5];
  for (unsigned i = 0; i < 8; ++i) cpu.evu.descriptors[i] = values[0x8d0 + i] & ((UINT64_C(1) << 51) - 1);
  evu_summary();
}

static unsigned write_group_regs(uint32_t lmul) { return evu_lmul_regs(lmul); }

static uint32_t write_group_mask(uint8_t regs) {
  if (regs >= 32)
    return UINT32_MAX;
  return (UINT32_C(1) << regs) - 1;
}

void evu_compute_reset(void);

void evu_reset(void) {
  memset(jobs, 0, sizeof(jobs));
  head = count = 0;
  asynchronous = false;
  defined_registers = 0;
  memset(&cpu.evu, 0, sizeof(cpu.evu));
  cpu.evu.vlenb = EVU_VLEN_BYTES;
  cpu.evu.vl = EVU_VLEN_BITS / 32;
  cpu.evu.vtype = 2 << 3;
  evu_compute_reset();
  evu_matrix_reset();
}

static void overlay_job_register(const EvuJob *job, uint32_t reg, uint8_t *data) {
  if (!job->writes_vrf || reg < job->write_reg || reg >= job->write_reg + job->write_regs)
    return;
  const unsigned group_index = reg - job->write_reg;
  memcpy(data, job->output + group_index * EVU_VLEN_BYTES, EVU_VLEN_BYTES);
}

void evu_read_register(uint32_t reg, void *data) {
  Assert(reg < 32, "EVU register index out of range");
  memcpy(data, cpu.evu.vr[reg], EVU_VLEN_BYTES);
  // Commands complete in order. Apply older pending architectural writes from
  // oldest to newest so chaining and in-place operations observe the same data
  // as the ordered RTL completion stream.
  for (unsigned i = 0; i < count; ++i) {
    overlay_job_register(&jobs[(head + i) % EVU_QUEUE], reg, data);
  }
}

static void write_architectural_register(uint8_t reg, uint8_t regs, const uint8_t *data) {
  Assert(reg + regs <= 32, "EVU architectural register group overflow");
  for (unsigned i = 0; i < regs; ++i) {
    memcpy(cpu.evu.vr[reg + i], data + i * EVU_VLEN_BYTES, EVU_VLEN_BYTES);
  }
  defined_registers |= write_group_mask(regs) << reg;
}

static void preflight(EvuJob *job) {
  const bool store = job->opcode == EVU_STORE;
  const bool write = store;
  const unsigned element_bytes = 1u << job->sew;
  const uint64_t byte_count = (uint64_t)job->vl * element_bytes;
  if (byte_count == 0)
    return;
  const uint64_t end = job->base + byte_count - 1;
  if ((job->base & (element_bytes - 1)) || end < job->base) {
    cpu.trapInfo.tval = job->base;
    longjmp_exception(end < job->base ? (store ? EX_SAF : EX_LAF) : (store ? EX_SAM : EX_LAM));
  }

  for (uint64_t line = job->base & ~UINT64_C(63);; line += 64) {
    const uint64_t probe = line < job->base ? job->base : line;
    paddr_t read_pa = 0;
    for (unsigned phase = 0; phase < (write ? 2u : 1u); ++phase) {
      const int type = phase ? MEM_TYPE_WRITE : MEM_TYPE_READ;
      int cause;
      PUSH_CONTEXT(&cause);
      if (!cause) {
        paddr_t pa = probe;
        const bool translated_mode = isa_mmu_check(probe, 1, type) != MMU_DIRECT;
        if (translated_mode) {
          const paddr_t translated = isa_mmu_translate(probe, 1, type);
          Assert((translated & PAGE_MASK) == MEM_RET_OK, "EVU translation failed without an exception");
          pa = (translated & ~PAGE_MASK) | (probe & PAGE_MASK);
        }
        pa &= ~UINT64_C(63);
        if (!check_paddr(pa, 64, type, store ? MEM_TYPE_WRITE : type, cpu.mode, probe) || !in_pmem(pa) ||
            !in_pmem(pa + 63) || (translated_mode && cpu.pbmt != 0)) {
          cpu.trapInfo.tval = probe;
          longjmp_exception(store || phase ? EX_SAF : EX_LAF);
        }
        if (phase && pa != read_pa) {
          cpu.trapInfo.tval = probe;
          longjmp_exception(EX_SAF);
        }
        if (!phase)
          read_pa = pa;
      }
      pop_context();
      if (cause) {
        if (store) {
          if (cause == EX_LPF)
            cause = EX_SPF;
          if (cause == EX_LAF)
            cause = EX_SAF;
          if (cause == EX_LGPF)
            cause = EX_SGPF;
        }
        longjmp_exception(cause);
      }
    }

    for (uint32_t i = 0; i < job->vl; ++i) {
      const uint64_t va = job->base + (uint64_t)i * element_bytes;
      if ((va & ~UINT64_C(63)) == line)
        job->physical[i] = read_pa | (va & 63);
    }
    if (line == (end & ~UINT64_C(63)))
      break;
  }
}

static void read_group_for_job(uint8_t *dst, uint8_t reg, unsigned regs) {
  if (reg + regs > 32 || (regs > 1 && (reg % regs) != 0))
    longjmp_exception(EX_II);
  memset(dst, 0, regs * EVU_VLEN_BYTES);
  for (unsigned i = 0; i < regs; ++i)
    evu_read_register(reg + i, dst + i * EVU_VLEN_BYTES);
}

static void execute_memory(EvuJob *job) {
  const EvuVType vt = evu_current_vtype();
  const unsigned regs = write_group_regs(vt.lmul);
  const uint8_t data_reg = (job->instr >> 7) & 31;
  const unsigned element_bytes = 1u << job->sew;
  memset(job->output, 0, sizeof(job->output));
  if (job->opcode == EVU_LOAD) {
    read_group_for_job(job->output, data_reg, regs);
    if (data_reg + regs > 32 || (regs > 1 && (data_reg % regs) != 0))
      longjmp_exception(EX_II);
    job->writes_vrf = true;
    job->write_reg = data_reg;
    job->write_regs = regs;
    job->output_bytes = regs * EVU_VLEN_BYTES;
  } else {
    read_group_for_job(job->output, data_reg, regs);
    job->writes_memory = true;
    job->output_bytes = regs * EVU_VLEN_BYTES;
  }

  preflight(job);
  if (job->vl == 0) {
    job->writes_vrf = false;
    job->writes_memory = false;
    return;
  }

  if (job->opcode == EVU_LOAD) {
    for (uint32_t i = job->vstart; i < job->vl; ++i) {
      memcpy(job->output + (size_t)i * element_bytes, guest_to_host(job->physical[i]), element_bytes);
    }
  }
}

static bool is_control(uint32_t opcode) {
  return opcode == EVU_LAUNCH || opcode == EVU_WAIT || opcode == EVU_FENCE || opcode == EVU_SETVLI ||
         opcode == EVU_SETVL;
}

static bool is_supported_opcode(uint32_t opcode) {
  switch (opcode) {
  case EVU_ADD:
  case EVU_SUB:
  case EVU_AND:
  case EVU_OR:
  case EVU_XOR:
  case EVU_MUL:
  case EVU_MAC:
  case EVU_REDUCE_SUM:
  case EVU_REDUCE_MAX:
  case EVU_REDUCE_MIN:
  case EVU_EXP2:
  case EVU_LOAD:
  case EVU_STORE:
  case EVU_SLL:
  case EVU_SRL:
  case EVU_SRA:
  case EVU_MIN:
  case EVU_MAX:
  case EVU_MINU:
  case EVU_MAXU:
  case EVU_FADD:
  case EVU_FSUB:
  case EVU_FMUL:
  case EVU_FMACC:
  case EVU_FDIV:
  case EVU_FMIN:
  case EVU_FMAX:
  case EVU_FREDUCE_SUM:
  case EVU_WADD:
  case EVU_WSUB:
  case EVU_WMUL:
  case EVU_SATADD:
  case EVU_SATSUB:
  case EVU_SLIDE_UP:
  case EVU_SLIDE_DOWN:
  case EVU_REDUCE_AND:
  case EVU_REDUCE_OR:
  case EVU_REDUCE_XOR:
  case EVU_FREDUCE_MIN:
  case EVU_FREDUCE_MAX:
  case EVU_SETVLI:
  case EVU_SETVL:
  case EVU_GATHER:
  case EVU_GATHER_IMM:
  case EVU_FP_QUANT_TO_CUTE:
  case EVU_LAUNCH:
  case EVU_WAIT:
  case EVU_FENCE:
    return true;
  default:
    return false;
  }
}

static bool valid_width(uint32_t opcode, uint32_t sew) {
  if (is_control(opcode))
    return true;
  if (sew > 2)
    return false;
  if (opcode == EVU_FP_QUANT_TO_CUTE) return sew == 2;
  if (opcode == EVU_WADD || opcode == EVU_WSUB || opcode == EVU_WMUL)
    return sew < 2;
  if (opcode == EVU_FADD || opcode == EVU_FSUB || opcode == EVU_FMUL || opcode == EVU_FMACC || opcode == EVU_FDIV ||
      opcode == EVU_FMIN || opcode == EVU_FMAX || opcode == EVU_FREDUCE_SUM || opcode == EVU_FREDUCE_MIN ||
      opcode == EVU_FREDUCE_MAX || opcode == EVU_EXP2)
    return sew >= 1;
  return true;
}

static uint8_t setv_config_aux(uint32_t instr) {
  return (uint8_t)((((instr >> 15) & 0x7) << 5) | ((instr >> 7) & 0x1f));
}

static void execute_setv(Decode *s, EvuJob *job) {
  const uint8_t aux = job->aux;
  const uint32_t mag = aux & 3;
  const uint32_t lmul = (aux & 4) ? (8 - mag) & 7 : mag;
  const uint32_t sew = job->opcode == EVU_SETVLI ? (aux >> 3) & 7 : job->sew;
  configure(job->base, (aux & 0xc0) | (sew << 3) | lmul, true, (aux & 4) && mag == 0);
  job->updates_vector_state = true;
  (void)s;
}

void evu_execute(Decode *s, uint32_t instr, uint64_t base) {
  save_globals(s);
  if (!evu_permitted()) longjmp_exception(EX_II);
  const uint32_t raw_opcode = instr >> 25;
  const uint32_t opcode = evu_canonical_opcode(raw_opcode);
  const uint32_t sew = (instr >> 12) & 0x7;
  if ((evu_scalar_form(raw_opcode) && sew != 2) || !is_supported_opcode(opcode) || !valid_width(opcode, sew) ||
      (opcode == EVU_LAUNCH && (sew != 4 || !evu_matrix_physical_mode()))) {
    longjmp_exception(EX_II);
  }
#ifndef CONFIG_RV_AME
  if (opcode == EVU_LAUNCH || opcode == EVU_FP_QUANT_TO_CUTE)
    longjmp_exception(EX_II);
#endif

  const EvuVType current = evu_current_vtype();
  if (evu_scalar_form(raw_opcode) && (current.lmul != 0 || (cpu.evu.configured && current.sew != 2)))
    longjmp_exception(EX_II);
  if (!is_control(opcode) && current.vill)
    longjmp_exception(EX_II);
  // The current CUTE quantizer streams one physical VRF register. Larger
  // groups would wrap its chunk counter and must fail before retirement.
  if (opcode == EVU_FP_QUANT_TO_CUTE && evu_lmul_regs(current.lmul) > 1)
    longjmp_exception(EX_II);
  if (opcode == EVU_FP_QUANT_TO_CUTE &&
      (base >> 9 || evu_current_vl(2) > 64 || CONFIG_CUSTOM_EVU_DLEN != 256))
    longjmp_exception(EX_II);
  Assert(count < EVU_QUEUE, "EVU reference command queue overflow");

  EvuJob *job = &jobs[(head + count) % EVU_QUEUE];
  memset(job, 0, sizeof(*job));
  job->pc = s->pc;
  job->instr = instr;
  job->base = base;
  job->opcode = opcode;
  job->sew = cpu.evu.configured && !is_control(opcode) ? current.sew : sew;
  job->vd = (instr >> 7) & 31;
  job->vs1 = (instr >> 15) & 31;
  job->vs2 = (instr >> 20) & 31;
  job->aux = (evu_element_zero(raw_opcode) ? 0x40 : 0) |
    (evu_scalar_source(raw_opcode) ? 0x20 : 0) | (raw_opcode == EVU_FP_QUANT_SCALAR ? 0x10 : 0);
  job->vstart = 0;
  if (!is_control(opcode) && !valid_width(opcode, job->sew)) longjmp_exception(EX_II);

  if (opcode == EVU_LAUNCH) {
    uint32_t scale_a[EVU_VLEN_BITS / 32], scale_b[EVU_VLEN_BITS / 32];
    evu_read_register(job->vs2, scale_a);
    evu_read_register(job->vd, scale_b);
    job->aux = (uint8_t)((((instr >> 15) & 0x7) << 5));
    evu_matrix_launch(job->pc, job->base, scale_a, scale_b, asynchronous);
  } else if (opcode == EVU_SETVLI || opcode == EVU_SETVL) {
    job->aux = setv_config_aux(instr);
    execute_setv(s, job);
    job->vl = (uint32_t)cpu.evu.vl;
  } else if (opcode != EVU_WAIT && opcode != EVU_FENCE) {
    job->vl = evu_current_vl(job->sew);
    if (evu_element_zero(raw_opcode))
      job->vl = job->vl != 0 && cpu.evu.vstart == 0 ? 1 : 0;
    if (opcode == EVU_LOAD || opcode == EVU_STORE)
      execute_memory(job);
    else
      evu_compute_execute(s, job);
  }

  if (opcode != EVU_STORE && opcode != EVU_WAIT && opcode != EVU_FENCE)
    evu_context_dirty();
  if (asynchronous) {
    ++count;
  } else {
    if (opcode == EVU_FENCE) {
      job->flags = evu_matrix_take_flags();
      Assert(!evu_matrix_drain(), "Incomplete synchronous EVU matrix stream");
    }
    if (job->writes_memory && !job->cause) {
      const unsigned element_bytes = 1u << job->sew;
      for (uint32_t i = job->vstart; i < job->vl; ++i) {
        memcpy(guest_to_host(job->physical[i]), job->output + (size_t)i * element_bytes, element_bytes);
      }
    }
    if (job->opcode == EVU_FP_QUANT_TO_CUTE && !job->cause) {
      for (unsigned i = 0; i < job->output_bytes; ++i) {
        const unsigned row = job->base >> 2;
        evu_matrix_cute_byte(job->base & 3, row % 8, (row / 8) * 2 + i / 32, i % 32, job->output[i]);
      }
    }
    if ((job->writes_vrf || job->writes_scalar) && !job->cause) {
      if (job->writes_vrf)
        write_architectural_register(job->write_reg, job->write_regs, job->output);
      if (job->writes_scalar && job->scalar_rd != 0)
        cpu.gpr[job->scalar_rd]._64 = job->scalar_value;
      cpu.evu.vstart = 0;
      }
    evu_publish_completion_flags(job);
  }
}

void difftest_evu_set_async(bool enable) {
  Assert(count == 0 && !evu_matrix_active(), "Cannot change EVU reference mode with pending work");
  asynchronous = enable;
}

int difftest_evu_pending(void) { return count + evu_matrix_active(); }

uint32_t difftest_evu_reg_valid(void) { return defined_registers; }

int difftest_evu_get_reg(uint32_t reg, void *data, size_t bytes) {
  if (reg >= 32 || bytes != EVU_VLEN_BYTES)
    return 1;
  memcpy(data, cpu.evu.vr[reg], bytes);
  return 0;
}

int difftest_evu_get_csr(void *data, size_t bytes) {
  const uint64_t values[] = {cpu.evu.vl, cpu.evu.vtype, 0, cpu.evu.vlenb, cpu.evu.configured,
    cpu.evu.fflags, cpu.evu.vxsat, cpu.evu.state, cpu.evu.status & ~UINT64_C(1),
    cpu.evu.cause, cpu.evu.epc, cpu.evu.tval,
    cpu.evu.descriptors[0], cpu.evu.descriptors[1], cpu.evu.descriptors[2], cpu.evu.descriptors[3],
    cpu.evu.descriptors[4], cpu.evu.descriptors[5], cpu.evu.descriptors[6], cpu.evu.descriptors[7], evu_csr_peek(0x8c0)};
  if (bytes != sizeof(values))
    return 1;
  memcpy(data, values, bytes);
  return 0;
}

int difftest_evu_store(uint64_t addr, const void *data, uint64_t mask) {
  if (evu_matrix_owns(addr))
    return evu_matrix_store(addr, data, mask);
  EvuJob *job = NULL;
  for (unsigned i = 0; i < count; ++i) {
    EvuJob *candidate = &jobs[(head + i) % EVU_QUEUE];
    if (candidate->writes_memory) {
      job = candidate;
      break;
    }
  }
  if (!job || job->cause || mask != UINT64_MAX || (addr & 63) || !in_pmem(addr) || !in_pmem(addr + 63)) {
    fprintf(stderr, "EVU store order/address/mask mismatch at 0x%lx mask=0x%lx\n", addr, mask);
    return 1;
  }

  const unsigned element_bytes = 1u << job->sew;
  uint8_t expected[64];
  memcpy(expected, guest_to_host(addr), sizeof(expected));
  bool intersects = false;
  for (unsigned byte = 0; byte < 64; ++byte) {
    for (uint32_t i = job->vstart; i < job->vl; ++i) {
      if (addr + byte < job->physical[i] || addr + byte >= job->physical[i] + element_bytes)
        continue;
      const unsigned index = i * element_bytes + addr + byte - job->physical[i];
      const uint8_t value = job->output[index];
      const uint8_t actual = ((const uint8_t *)data)[byte];
      intersects = true;
      if (job->observed[index] || actual == value)
        expected[byte] = value;
    }
  }
  if (!intersects || memcmp(data, expected, sizeof(expected))) {
    fprintf(stderr, "EVU physical store/guard mismatch at pc=0x%lx addr=0x%lx\n", job->pc, addr);
    return 1;
  }
  for (uint32_t i = job->vstart; i < job->vl; ++i) {
    for (unsigned byte = 0; byte < element_bytes; ++byte) {
      const uint64_t pa = job->physical[i] + byte;
      const unsigned index = i * element_bytes + byte;
      if (pa >= addr && pa < addr + 64 && !job->observed[index] &&
          ((const uint8_t *)data)[pa - addr] == job->output[index]) {
        job->observed[index] = 1;
        ++job->written;
      }
    }
  }
  memcpy(guest_to_host(addr), expected, sizeof(expected));
  return 0;
}

int difftest_evu_cute_write(uint32_t reg, uint32_t bank, uint32_t entry,
                             const void *data, size_t bytes, uint64_t mask) {
  EvuJob *job = NULL;
  for (unsigned i = 0; i < count; ++i) {
    EvuJob *candidate = &jobs[(head + i) % EVU_QUEUE];
    if (candidate->writes_cute && candidate->written < candidate->output_bytes) {
      job = candidate;
      break;
    }
  }
  if (!job || !data || bytes != 32) return 1;
  if (job->opcode != EVU_FP_QUANT_TO_CUTE) return 1;
  const unsigned row = job->base >> 2;
  const unsigned offset = job->written % 32;
  const unsigned lanes = 8;
  const unsigned remaining = job->output_bytes - job->written;
  const unsigned n = remaining < lanes ? remaining : lanes;
  const uint64_t expected_mask = ((UINT64_C(1) << n) - 1) << offset;
  if (reg != (job->base & 3) || bank != row % 8 ||
      entry != (row / 8) * 2 + job->written / 32 ||
      mask != expected_mask) return 1;
  const uint8_t *actual = data;
  for (unsigned i = 0; i < n; ++i)
    if (actual[offset + i] != job->output[job->written + i]) return 1;
  // Check the complete event before changing reference state.
  for (unsigned i = 0; i < n; ++i)
    evu_matrix_cute_byte(reg, bank, entry, offset + i, job->output[job->written + i]);
  job->written += n;
  return 0;
}

int difftest_evu_complete(uint64_t pc, uint32_t instr, uint8_t flags, bool exception, uint8_t cause) {
  if (!count)
    return 1;
  EvuJob *job = &jobs[head];
  if (job->opcode == EVU_FENCE)
    job->flags = evu_matrix_flags();
  const uint32_t expected_store_bytes = job->writes_memory ? (job->vl - job->vstart) * (1u << job->sew) : 0;
  const bool missing_stores = job->writes_memory && !job->cause && job->written != expected_store_bytes;
  if (job->pc != pc || job->instr != instr || job->flags != flags || (job->cause != 0) != exception ||
      job->cause != cause || missing_stores ||
      (job->opcode == EVU_FP_QUANT_TO_CUTE && job->written != job->output_bytes)) {
    fprintf(stderr, "EVU completion mismatch pc=0x%lx/0x%lx instr=0x%x/0x%x flags=0x%x/0x%x stores=%u/%u\n", pc,
            job->pc, instr, job->instr, flags, job->flags, job->written, expected_store_bytes);
    return 1;
  }
  if (job->opcode == EVU_FENCE) {
    if (evu_matrix_drain()) return 1;
    evu_matrix_take_flags();
  }
  if (!job->cause) {
    if (job->writes_vrf)
      write_architectural_register(job->write_reg, job->write_regs, job->output);
    if (job->writes_scalar && job->scalar_rd != 0)
      cpu.gpr[job->scalar_rd]._64 = job->scalar_value;
    if (job->writes_vrf || job->writes_scalar) {
      cpu.evu.vstart = 0;
      }
  }
  evu_publish_completion_flags(job);
  head = (head + 1) % EVU_QUEUE;
  --count;
  return 0;
}
int difftest_evu_complete_v2(uint64_t pc, uint32_t instr, uint8_t flags, bool exception,
                             uint8_t cause, bool vxsat) {
  if (!count || jobs[head].vxsat != vxsat) return 1;
  return difftest_evu_complete(pc, instr, flags, exception, cause);
}

int difftest_evu_drain(uint8_t flags, bool exception, uint8_t cause, uint64_t pc, uint64_t tval, bool valid) {
  (void)pc; (void)tval; (void)cause;
  // A physical bus failure must also be injected into the reference memory model.
  // Never silently accept an unexplained DUT failure or its numeric result.
  if (count || exception || valid || flags != evu_matrix_flags()) return 1;
  if (evu_matrix_drain()) return 1;
  evu_csr_drain();
  return 0;
}

int difftest_evu_inject_fault(uint64_t pc, uint8_t cause, uint64_t address, bool valid) {
  if (cause == 0 || cause > 15) return 1;
  for (unsigned i = 0; i < count; ++i) {
    EvuJob *job = &jobs[(head + i) % EVU_QUEUE];
    if (job->pc != pc || (job->opcode != EVU_LOAD && job->opcode != EVU_STORE)) continue;
    job->cause = cause;
    job->fault_address = address;
    job->fault_address_valid = valid;
    return 0;
  }
  return 1;
}

#endif
