#ifndef NEMU_DIFFTEST_EVU_H
#define NEMU_DIFFTEST_EVU_H

#include <common.h>
#include <cpu/decode.h>

#ifdef CONFIG_CUSTOM_EVU
/*
 * The EVU has an architectural register bank independent from the regular
 * RVV bank. evu_execute models supported commands and queues their architectural
 * effects; difftest_evu_complete applies those effects in program order.
 */
bool evu_csr_address(uint32_t addr);
bool evu_permitted(void);
void evu_csr_check(uint32_t addr);
uint64_t evu_csr_read(uint32_t addr);
void evu_csr_write(uint32_t addr, uint64_t value);
void evu_csr_export(uint64_t *values);
void evu_csr_import(const uint64_t *values);
void evu_context_dirty(void);
/* Independent host memory-fault oracle; never populated from DUT completion data. */
int difftest_evu_inject_fault(uint64_t pc, uint8_t cause, uint64_t address, bool valid);
int difftest_evu_drain(uint8_t flags, bool exception, uint8_t cause, uint64_t pc, uint64_t tval, bool valid);
int difftest_evu_complete_v2(uint64_t pc, uint32_t instr, uint8_t flags, bool exception,
                             uint8_t cause, bool vxsat);
void evu_reset(void);
void evu_execute(Decode *s, uint32_t instr, uint64_t base);
void difftest_evu_set_async(bool enable);
int difftest_evu_store(uint64_t addr, const void *data, uint64_t mask);
int difftest_evu_matrix(uint32_t row, uint32_t col, const void *data, size_t bytes,
                        uint64_t mask, uint8_t flags, bool last);
int difftest_evu_complete(uint64_t pc, uint32_t instr, uint8_t flags, bool exception, uint8_t cause);
int difftest_evu_cute_write(uint32_t reg, uint32_t bank, uint32_t entry,
                             const void *data, size_t bytes, uint64_t mask);
int difftest_evu_pending(void);
/* Return only committed architectural state, never speculative job results. */
int difftest_evu_get_reg(uint32_t reg, void *data, size_t bytes);
uint32_t difftest_evu_reg_valid(void);
int difftest_evu_get_csr(void *data, size_t bytes);
void evu_matrix_zero(uint32_t reg);
void evu_matrix_load(uint32_t reg, uint64_t base, uint64_t stride,
                     unsigned rows, unsigned cols, unsigned size,
                     bool transpose, bool signed_int8);
void evu_matrix_mma(uint32_t dest, uint32_t src1, uint32_t src2,
                    unsigned rows, unsigned cols, unsigned depth,
                    bool signed_int8_int32, bool saturate);
#endif

#endif
