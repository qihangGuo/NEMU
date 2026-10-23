#ifndef NEMU_EVU_MATRIX_H
#define NEMU_EVU_MATRIX_H

#include <common.h>

void evu_matrix_reset(void);
bool evu_matrix_physical_mode(void);
bool evu_matrix_active(void);
void evu_matrix_launch(uint64_t pc, uint64_t base, const void *scale_a,
                       const void *scale_b, bool asynchronous);
uint8_t evu_matrix_flags(void);
uint8_t evu_matrix_take_flags(void);
bool evu_matrix_owns(uint64_t address);
int evu_matrix_store(uint64_t address, const void *data, uint64_t mask);
int evu_matrix_drain(void);

#endif
