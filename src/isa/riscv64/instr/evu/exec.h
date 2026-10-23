#include <cpu/difftest/evu.h>

def_EHelper(xv) {
  evu_execute(s, s->isa.instr.val, *dsrc1);
}
