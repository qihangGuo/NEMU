# EVU reference execution regression

This host test loads `build/riscv64-nemu-interpreter-so` and exercises the EVU
reference decoder without an emulator or RTL simulation. It covers SEW 8/16/32
integer and floating-point element-wise operations, reductions, Exp2,
generic vector LOAD/STORE and VL/tail state. It also checks LOAD into v0 and
rejection of the retired opcodes
`0x09/0x0b/0x18/0x19/0x38/0x4c/0x4d/0x4e/0x4f/0x50/0x5e` without modifying v0.
These encodings remain illegal; the dedicated quantization test is retired.

## Scalar-offset slides

On the server, `evu_compute_ref_test <reference.so> --slide` runs 216
directed cases for both slide operations, SEW=8/16/32, m1/m2/mf2, zero/full
VL and tails, data-path/register boundaries, large 64-bit offsets, x0,
an odd-numbered scalar GPR under m2, and in-place slidedown. Expected bytes
are computed independently of the EVU reference implementation. Additional
negative cases reject slideup overlap, bad widths/groups and retired WAIT
at both zero and nonzero VL. The default regression also runs these cases.
Offsets now come from the CPU GPR operand, not an EVU register element.

## RVV math extension

Funct7 0x60..0x6e adds sqrt/rsqrt7/rec7 (including FP32 E0), sign XOR,
FP16/FP32 conversion, FP32/INT32 conversion, signed immediate narrowing,
zero/sign extension and immediate logical narrowing. EVCAP bit 28 advertises
the complete set. FP arithmetic/conversion is RNE; integer clipping uses
fixed RNU and reports EVXSAT. The estimate helpers reuse NEMU's standard RVV
reference implementation without touching scalar FCSR.

On the build server, after rebuilding the matching reference:

```sh
make -C tools/evu-ref-test NEMU_HOME="$PWD"
./tools/evu-ref-test/evu_compute_ref_test \
  ./build/riscv64-nemu-interpreter-so --rvv-math /tmp/evu-rvv-math.trace
```

The focused run checks 73 positive cases and illegal width/register-group
cases before writing a trace for XSAI's `EvuRvvMathDifftestSpec`. Trace lines
contain the custom instruction, VTYPE, VL, expected flags/saturation, four
source registers, two sign-source registers and four expected destination
registers, as little-endian byte tokens. The ordinary full reference suite
also runs these cases. No generated trace or binary is checked in.

The first 54-case version passed locally before the user requested server-only
binary validation. The expanded cases require execution on the server.

Build NEMU with `configs/riscv64-xs-evu-ref_defconfig` first, then run:

```sh
make -C tools/evu-ref-test
./tools/evu-ref-test/evu_compute_ref_test \
  ./build/riscv64-nemu-interpreter-so
```

The test uses the existing `difftest_evu_*` ABI. It does not change the ABI or
require a guest binary.

The immediate-gather revision adds funct7=0x51 and EVCAP bit 26. The rs1
field is an unsigned five-bit element index; it does not name a scalar or
vector register. Index zero broadcasts the source element. The computation
test covers all three supported widths, indices 0/7/16/31, VL=0/1/13/64,
in-place destinations, preserved tails and out-of-range indices under
fractional LMUL. Rebuild the shared library before running these cases.

Before executing guest instructions, it also synchronizes pending interrupts
20 times through `difftest_non_reg_interrupt_pending`, matching EMU startup.
This covers the AIA initialization path that crashed when a copied build tree
mixed objects compiled with different EVU/CPU state layouts.

If an existing reference library crashes in `set_viprios_sort`, clean and rebuild
NEMU using the existing `.config` (`make clean`, then `make -j16`). Do not copy
`build/` between checkouts or machines. The build rules now rebuild objects whose
fixdep rules refer to another checkout, or whose dependency files are missing.
The host regression for these cases is `/usr/bin/python3 scripts/test-build-deps.py`
from the NEMU repository root.

## Scalar FCSR history and per-command exceptions

The reference collects each EVU command's exceptions independently of sticky
scalar FCSR flags. Re-raising NX/NV/OF/UF/DZ must still appear in the completion
when the same bits were already set by scalar code. The regression reproduces
the server softmax instruction `0x645f228b` with NX initially clear/set and checks
identical VRF bits and NX=1 in both completions. It also checks 896 FP16/FP32
cases: all 32 preexisting flag combinations, exact/inexact/invalid/overflow/
underflow/divide-by-zero operations, synchronous/asynchronous publication, and
FRM preservation. Only the first element raises exceptions; subsequent exact
elements must not erase the accumulated command flags. Async issue preserves
scalar FCSR; completion publishes the command flags only to independent EVU efflags.
Scalar FCSR/FRM and FS/VS/vstart must remain unchanged.

## Online softmax and RTL comparison

FReduceSum follows the Saturn EVU four-slot unordered tree, including
per-element-width rounding and flags. The regression checks FP16/FP32
cancellation, partial VL, inactive tail sNaNs, destination preservation and
slot reuse. The seven 512-element online-softmax cases also check every
exponent, broadcast running max/sum, final probability, shift invariance and
probability sum against independent long-double `expl`.

An optional second argument writes the FP32 softmax instruction trace:

```sh
./tools/evu-ref-test/evu_compute_ref_test \
  ./build/riscv64-nemu-interpreter-so /tmp/evu-softmax-reference.trace
```

The trace records opcode, SEW, vd, vs1, vs2, the per-instruction flags and all
128 destination words in element order. It starts with 32 zeroing LOADs so
preserved tails have a defined initial state. It contains 690 instructions,
including 112 vector stores. The RTL comparison is provided by
`XSAI_external_vector/software/evu/scripts/check_softmax_ref.sh` and
`EvuSoftmaxDifftestSpec`.

Replay a trace emitted by the RTL test using the actual asynchronous Difftest
completion/store API:

```sh
./tools/evu-ref-test/evu_compute_ref_test \
  ./build/riscv64-nemu-interpreter-so --replay /tmp/evu-softmax-rtl.trace
```

Replay runs three times, with prior scalar flags `0x00`, `0x01`, and `0x1f`.
Each run requires `flags/VRF match; pending=0`, including instructions that
re-raise existing flags and instructions that raise no flags. It reads the saved
RTL trace without generating RTL or building a guest. This tests the recorded
accelerator arithmetic and reference ABI; scalar core retirement and HBL2
require full EMU validation.


## Retired Matrix Coupling

Launch (0x18) and direct quantization (0x50/0x5e) are illegal instructions.
The compute reference regression checks all three with VL=0 and nonzero VL,
all encoded element widths, no queued completion and unchanged VRF contents.
Ordinary AME matrix instructions no longer feed a separate EVU matrix oracle.
CSR ABI 2 and the 13-word snapshot remain unchanged; EVCAP bit 25 is clear.
