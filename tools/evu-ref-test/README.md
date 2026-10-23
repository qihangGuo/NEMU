# EVU reference execution regression

This host test loads `build/riscv64-nemu-interpreter-so` and exercises the EVU
reference decoder without an emulator or RTL simulation. It covers SEW 8/16/32
integer and floating-point element-wise operations, reductions, Exp2,
fixed-point QuantToCute saturation and VRF preservation, generic vector
LOAD/STORE, and VL/tail state. CUTE A/B writes are not observed or applied by
this reference test; downstream QuantToCute results require RTL coverage.
It also checks LOAD into v0 and rejection of the retired opcodes
`0x09/0x0b/0x38/0x4d/0x4f` without modifying v0. These encodings remain illegal.

Build NEMU with `configs/riscv64-xs-evu-ref_defconfig` first, then run:

```sh
make -C tools/evu-ref-test
./tools/evu-ref-test/evu_compute_ref_test \
  ./build/riscv64-nemu-interpreter-so
```

The test uses the existing `difftest_evu_*` ABI. It does not change the ABI or
require a guest binary.

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
