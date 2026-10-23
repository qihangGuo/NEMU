NEMU_HOME ?= ../..
CC ?= cc
CFLAGS ?= -std=gnu11 -O2 -Wall -Wextra
CPPFLAGS += -D__ISA__=riscv64
CPPFLAGS += -I$(NEMU_HOME)/include
CPPFLAGS += -I$(NEMU_HOME)/src/isa/riscv64/include
CPPFLAGS += -I$(NEMU_HOME)/src/isa/riscv64/instr

all: evu_compute_ref_test

evu_compute_ref_test: evu_compute_ref_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ -ldl -lm

clean:
	rm -f evu_compute_ref_test

.PHONY: all clean
