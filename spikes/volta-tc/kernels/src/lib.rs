//! Spike: Volta (sm_70) tensor-core instructions written with cuda-oxide's `ptx_asm!`
//!
//! cuda-oxide has no Volta mma or wmma intrinsic, so a V100 port of speakrs's FP16 trunk
//! needs these instructions inline. Each kernel runs as one warp on FP16 operands with FP32
//! accumulation, and the host in `../host` compares the result with a CPU product:
//!
//! - [`volta_wmma_global`]: `wmma` m16n16k16 with both operands loaded from global memory
//! - [`volta_wmma_shared`]: the same with B staged in shared memory and loaded through a
//!   32-bit shared address, as a pipelined kernel would
//! - [`volta_m8n8k4`] (feature `m8n8k4`): one `mma.sync` m8n8k4 per quad pair, to check the
//!   fragment layouts a hand-tiled kernel would use
//!
//! Without the `aligned` feature, wmma is spelled as PTX ISA 6.0 defines it, the version
//! cuda-oxide emits for sm_70. With it, wmma carries `.aligned`, which ISA 6.3 and newer
//! require, so the module must be raised to ISA 6.5 by an llc wrapper

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, ptx_asm, thread, warp};

/// Eight f16x2 words of one wmma operand fragment
type Fragment = [u32; 8];

/// The A fragment of a row-major 16x16 FP16 matrix at a global address
#[inline(always)]
unsafe fn load_a_global(pointer: *const u32) -> Fragment {
    let (r0, r1, r2, r3, r4, r5, r6, r7): (u32, u32, u32, u32, u32, u32, u32, u32);
    // safety: a convergent warp-wide load of 512 bytes the caller owns
    unsafe {
        #[cfg(not(feature = "aligned"))]
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %8; wmma.load.a.sync.row.m16n16k16.global.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [g], 16; }",
            out("=r") r0, out("=r") r1, out("=r") r2, out("=r") r3,
            out("=r") r4, out("=r") r5, out("=r") r6, out("=r") r7,
            in("l") pointer as u64,
        );
        #[cfg(feature = "aligned")]
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %8; wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [g], 16; }",
            out("=r") r0, out("=r") r1, out("=r") r2, out("=r") r3,
            out("=r") r4, out("=r") r5, out("=r") r6, out("=r") r7,
            in("l") pointer as u64,
        );
    }
    [r0, r1, r2, r3, r4, r5, r6, r7]
}

/// The B fragment of a column-major 16x16 FP16 matrix at a global address
#[inline(always)]
unsafe fn load_b_global(pointer: *const u32) -> Fragment {
    let (r0, r1, r2, r3, r4, r5, r6, r7): (u32, u32, u32, u32, u32, u32, u32, u32);
    // safety: as for `load_a_global`
    unsafe {
        #[cfg(not(feature = "aligned"))]
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %8; wmma.load.b.sync.col.m16n16k16.global.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [g], 16; }",
            out("=r") r0, out("=r") r1, out("=r") r2, out("=r") r3,
            out("=r") r4, out("=r") r5, out("=r") r6, out("=r") r7,
            in("l") pointer as u64,
        );
        #[cfg(feature = "aligned")]
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %8; wmma.load.b.sync.aligned.col.m16n16k16.global.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [g], 16; }",
            out("=r") r0, out("=r") r1, out("=r") r2, out("=r") r3,
            out("=r") r4, out("=r") r5, out("=r") r6, out("=r") r7,
            in("l") pointer as u64,
        );
    }
    [r0, r1, r2, r3, r4, r5, r6, r7]
}

/// The B fragment of a column-major 16x16 FP16 matrix at a 32-bit shared address
#[inline(always)]
unsafe fn load_b_shared(address: u32) -> Fragment {
    let (r0, r1, r2, r3, r4, r5, r6, r7): (u32, u32, u32, u32, u32, u32, u32, u32);
    // safety: a convergent warp-wide load of a published, 32-byte aligned shared tile
    unsafe {
        #[cfg(not(feature = "aligned"))]
        ptx_asm!(
            "wmma.load.b.sync.col.m16n16k16.shared.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 16;",
            out("=r") r0, out("=r") r1, out("=r") r2, out("=r") r3,
            out("=r") r4, out("=r") r5, out("=r") r6, out("=r") r7,
            in("r") address,
            clobber("memory"),
        );
        #[cfg(feature = "aligned")]
        ptx_asm!(
            "wmma.load.b.sync.aligned.col.m16n16k16.shared.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 16;",
            out("=r") r0, out("=r") r1, out("=r") r2, out("=r") r3,
            out("=r") r4, out("=r") r5, out("=r") r6, out("=r") r7,
            in("r") address,
            clobber("memory"),
        );
    }
    [r0, r1, r2, r3, r4, r5, r6, r7]
}

/// `c += a * b` as one m16n16k16 product with FP32 accumulation
#[inline(always)]
fn wmma_mma(c: [f32; 8], a: Fragment, b: Fragment) -> [f32; 8] {
    let [mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6, mut c7] = c;
    // safety: a convergent warp-wide register operation; the caller runs all 32 lanes
    unsafe {
        #[cfg(not(feature = "aligned"))]
        ptx_asm!(
            "wmma.mma.sync.row.col.m16n16k16.f32.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19, %20, %21, %22, %23}, {%0, %1, %2, %3, %4, %5, %6, %7};",
            inout("+f") c0, inout("+f") c1, inout("+f") c2, inout("+f") c3,
            inout("+f") c4, inout("+f") c5, inout("+f") c6, inout("+f") c7,
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") a[4], in("r") a[5], in("r") a[6], in("r") a[7],
            in("r") b[0], in("r") b[1], in("r") b[2], in("r") b[3],
            in("r") b[4], in("r") b[5], in("r") b[6], in("r") b[7],
        );
        #[cfg(feature = "aligned")]
        ptx_asm!(
            "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19, %20, %21, %22, %23}, {%0, %1, %2, %3, %4, %5, %6, %7};",
            inout("+f") c0, inout("+f") c1, inout("+f") c2, inout("+f") c3,
            inout("+f") c4, inout("+f") c5, inout("+f") c6, inout("+f") c7,
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") a[4], in("r") a[5], in("r") a[6], in("r") a[7],
            in("r") b[0], in("r") b[1], in("r") b[2], in("r") b[3],
            in("r") b[4], in("r") b[5], in("r") b[6], in("r") b[7],
        );
    }
    [c0, c1, c2, c3, c4, c5, c6, c7]
}

/// Stores the accumulator as a row-major 16x16 FP32 matrix at a global address
#[inline(always)]
unsafe fn store_d_global(pointer: *mut f32, d: [f32; 8]) {
    // safety: a convergent warp-wide store of 1 KiB the caller owns
    unsafe {
        #[cfg(not(feature = "aligned"))]
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %0; wmma.store.d.sync.row.m16n16k16.global.f32 [g], {%1, %2, %3, %4, %5, %6, %7, %8}, 16; }",
            in("l") pointer as u64,
            in("f") d[0], in("f") d[1], in("f") d[2], in("f") d[3],
            in("f") d[4], in("f") d[5], in("f") d[6], in("f") d[7],
            clobber("memory"),
        );
        #[cfg(feature = "aligned")]
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %0; wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [g], {%1, %2, %3, %4, %5, %6, %7, %8}, 16; }",
            in("l") pointer as u64,
            in("f") d[0], in("f") d[1], in("f") d[2], in("f") d[3],
            in("f") d[4], in("f") d[5], in("f") d[6], in("f") d[7],
            clobber("memory"),
        );
    }
}

/// `d = a * b` for a row-major 16x16 FP16 `a` and a column-major 16x16 FP16 `b`, both as
/// f16x2 words (128 each), into a row-major 16x16 FP32 `d`
///
/// Launch one warp
#[kernel]
pub fn volta_wmma_global(a: &[u32], b: &[u32], mut d: DisjointSlice<f32>) {
    if a.len() < 128 || b.len() < 128 || d.len() < 256 {
        return;
    }
    // safety: one warp, all lanes active, with the lengths checked above
    unsafe {
        let product = wmma_mma([0.0; 8], load_a_global(a.as_ptr()), load_b_global(b.as_ptr()));
        store_d_global(d.as_mut_ptr(), product);
    }
}

/// As [`volta_wmma_global`], with `b` first copied to shared memory
///
/// Launch one warp with 512 bytes of dynamic shared memory
#[kernel]
pub fn volta_wmma_shared(a: &[u32], b: &[u32], mut d: DisjointSlice<f32>) {
    if a.len() < 128 || b.len() < 128 || d.len() < 256 {
        return;
    }
    let lane = warp::lane_id() as usize;
    let tile = DynamicSharedArray::<u32, 32>::get();
    for word in lane * 4..lane * 4 + 4 {
        // safety: 32 lanes write the 128 words of the 512-byte dynamic allocation once
        unsafe { tile.add(word).write(b[word]) };
    }
    thread::sync_threads();
    // safety: the tile is published by the barrier and 32-byte aligned
    unsafe {
        let address = cvta_generic_to_shared_u32(tile as *const u8);
        let product = wmma_mma([0.0; 8], load_a_global(a.as_ptr()), load_b_shared(address));
        store_d_global(d.as_mut_ptr(), product);
    }
}

/// Each quad pair computes `a * b` for a row-major 8x4 FP16 `a` and a column-major 4x8
/// FP16 `b`, both as f16x2 words (16 each), and writes its own row-major 8x8 FP32 copy of
/// the product to `d`, quad pair `q` at `d[64 * q..]`
///
/// Quad pair `q` is lanes `4q..4q + 4` and `16 + 4q..20 + 4q`. Launch one warp
#[cfg(feature = "m8n8k4")]
#[kernel]
pub fn volta_m8n8k4(a: &[u32], b: &[u32], mut d: DisjointSlice<f32>) {
    if a.len() < 16 || b.len() < 16 || d.len() < 256 {
        return;
    }
    let lane = warp::lane_id() as usize;
    let upper = if lane >= 16 { 4 } else { 0 };
    // the row of `a` and the column of `b` this lane holds, four k values each
    let line = lane % 4 + upper;
    let (a0, a1) = (a[2 * line], a[2 * line + 1]);
    let (b0, b1) = (b[2 * line], b[2 * line + 1]);
    let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6, mut c7) =
        (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    // safety: a convergent warp-wide register operation with all 32 lanes active
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};",
            inout("+f") c0, inout("+f") c1, inout("+f") c2, inout("+f") c3,
            inout("+f") c4, inout("+f") c5, inout("+f") c6, inout("+f") c7,
            in("r") a0, in("r") a1, in("r") b0, in("r") b1,
        );
    }
    // the f32 accumulator layout of the PTX ISA for m8n8k4, to be confirmed by the host
    let quad_pair = lane % 16 / 4;
    let out = d.as_mut_ptr();
    for (i, value) in [c0, c1, c2, c3, c4, c5, c6, c7].into_iter().enumerate() {
        let row = (lane & 1) + (i & 2) + upper;
        let col = (i & 4) + (lane & 2) + (i & 1);
        // safety: every (quad pair, row, col) is written by exactly one lane, inside `d`
        unsafe { out.add(64 * quad_pair + 8 * row + col).write(value) };
    }
}
