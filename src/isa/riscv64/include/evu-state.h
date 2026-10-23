#ifndef RISCV64_EVU_STATE_H
#define RISCV64_EVU_STATE_H

typedef struct {
  /* Independent external-vector architectural VRF: 32 registers of VLEN bits. */
  uint64_t vr[32][CONFIG_CUSTOM_EVU_VLEN / 64];
  /* EVU-local vector control state exported through dedicated Difftest APIs. */
  uint64_t vl, vtype, vstart, vlenb, configured;
  uint64_t fflags, vxsat, state, status, cause, epc, tval;
  uint64_t descriptors[8];
} riscv64_EvuState;

#endif
