// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-virtual-vector-lowering))" --split-input-file %s | iree-opt -iree-convert-to-llvm | mlir-translate --mlir-to-llvmir | llc -mtriple=aarch64-none-elf -mattr=+sve | FileCheck %s

// Checks that the generic vector FMA intrinsics emitted by the `inner_tiled`
// lowering are actually selected to AArch64 `fmla` instructions: `llvm.fma.v4f32`
// must become a NEON `fmla` on `v` registers, and `llvm.fma.nxv4f32` must become
// a predicated SVE `fmla` on `z` registers. Only the instruction mnemonic and
// register class are checked — register numbers and instruction scheduling are
// intentionally left unconstrained.

// -----

// NEON `fmla` natural orientation: fixed-width `llvm.fma.v4f32` -> `fmla v.4s`.

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @neon_fmla(
    %lhs: vector<1x1x1x1xf32>, %rhs: vector<1x1x4x1xf32>,
    %acc: vector<1x1x1x4xf32>) -> vector<1x1x1x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_ARM_NEON_FMLA_1x4x1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1x1x1xf32>, vector<1x1x4x1xf32> into vector<1x1x1x4xf32>
  return %0 : vector<1x1x1x4xf32>
}

// CHECK-LABEL: neon_fmla
// CHECK: fmla v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s

// -----

// SVE `fmla` natural orientation: scalable `llvm.fma.nxv4f32` -> predicated
// `fmla z.s` on `z` registers.

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @sve_fmla(
    %lhs: vector<1x1x1x1xf32>, %rhs: vector<1x1x[4]x1xf32>,
    %acc: vector<1x1x[4]x1xf32>) -> vector<1x1x[4]x1xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_ARM_SVE_FMLA_1x4VLx1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1x1x1xf32>, vector<1x1x[4]x1xf32> into vector<1x1x[4]x1xf32>
  return %0 : vector<1x1x[4]x1xf32>
}

// CHECK-LABEL: sve_fmla
// CHECK: fmla z{{[0-9]+}}.s, p{{[0-9]+}}/m, z{{[0-9]+}}.s, z{{[0-9]+}}.s
