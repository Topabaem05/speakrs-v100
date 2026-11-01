//! Runs the Volta tensor-core spike kernels from a PTX or cubin file and compares each
//! result with a CPU product of the same FP16 operands
//!
//! The operands are multiples of 1/8 in [-1, 1], so they are exact in FP16 and every sum
//! of their products is exact in FP32: any difference is a fragment layout or encoding
//! error, never rounding
//!
//! Usage: `volta-tc-host <kernels.ptx | kernels.cubin>`; exits nonzero on any mismatch

use std::error::Error;
use std::sync::Arc;

use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use half::f16;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Packs FP16 values two per word, the first in the low half
fn packed(values: &[f32]) -> Vec<u32> {
    values
        .chunks(2)
        .map(|pair| {
            let lo = u32::from(f16::from_f32(pair[0]).to_bits());
            let hi = u32::from(f16::from_f32(pair[1]).to_bits());
            lo | hi << 16
        })
        .collect()
}

/// An exact FP16 operand in [-1, 1] from a small integer pattern
fn eighths(pattern: usize, modulus: usize) -> f32 {
    (pattern % modulus) as f32 / 8.0 - (modulus / 2) as f32 / 8.0
}

/// Runs `kernel` as one warp on `a` and `b` and returns its 256-float output
fn run(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    kernel: &str,
    a: &[u32],
    b: &[u32],
    shared_mem_bytes: u32,
) -> Result<Vec<f32>> {
    let function = module.load_function(kernel)?;
    let a_dev = stream.clone_htod(a)?;
    let b_dev = stream.clone_htod(b)?;
    let mut d_dev = stream.alloc_zeros::<f32>(256)?;
    let (a_len, b_len, d_len) = (a.len() as u64, b.len() as u64, 256_u64);
    let mut launch = stream.launch_builder(&function);
    launch
        .arg(&a_dev)
        .arg(&a_len)
        .arg(&b_dev)
        .arg(&b_len)
        .arg(&mut d_dev)
        .arg(&d_len);
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes,
    };
    // SAFETY: each kernel takes three slices as (pointer, length) pairs, reads at most
    // the words of `a` and `b` given here and writes at most the 256 floats of `d`
    unsafe { launch.launch(config) }?;
    Ok(stream.clone_dtoh(&d_dev)?)
}

/// Prints a kernel's outcome; returns whether it failed
fn report(kernel: &str, result: Result<Vec<f32>>, expected: &[f32], cols: usize) -> bool {
    let actual = match result {
        Ok(actual) => actual,
        Err(error) => {
            println!("{kernel}: ERROR {error}");
            return true;
        }
    };
    let mismatches: Vec<_> = actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (got, want))| got != want)
        .collect();
    if mismatches.is_empty() {
        println!("{kernel}: PASS, {} values exact", expected.len());
        return false;
    }
    println!(
        "{kernel}: FAIL, {} of {} values differ",
        mismatches.len(),
        expected.len()
    );
    for (index, (got, want)) in mismatches.iter().take(8) {
        let (copy, cell) = (index / (cols * cols), index % (cols * cols));
        println!(
            "  copy {copy} row {} col {}: got {got}, want {want}",
            cell / cols,
            cell % cols
        );
    }
    true
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: volta-tc-host <kernels.ptx | kernels.cubin>")?;
    let context = cudarc::driver::CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = if path.ends_with(".cubin") {
        context.load_module(Ptx::from_binary(std::fs::read(&path)?))?
    } else {
        context.load_module(Ptx::from_src(std::fs::read_to_string(&path)?))?
    };
    let mut failed = false;

    // wmma m16n16k16: row-major A, column-major B (element (k, c) at c * 16 + k)
    let a: Vec<f32> = (0..256)
        .map(|i| eighths(3 * (i / 16) + 5 * (i % 16), 17))
        .collect();
    let b: Vec<f32> = (0..256)
        .map(|i| eighths(7 * (i % 16) + 11 * (i / 16), 13))
        .collect();
    let expected: Vec<f32> = (0..256)
        .map(|i| {
            let (row, col) = (i / 16, i % 16);
            (0..16)
                .map(|k| f64::from(a[row * 16 + k]) * f64::from(b[col * 16 + k]))
                .sum::<f64>() as f32
        })
        .collect();
    for (kernel, shared) in [("volta_wmma_global", 0), ("volta_wmma_shared", 512)] {
        let result = run(&stream, &module, kernel, &packed(&a), &packed(&b), shared);
        failed |= report(kernel, result, &expected, 16);
    }

    // m8n8k4: row-major 8x4 A, column-major 4x8 B, one 8x8 copy per quad pair
    let a: Vec<f32> = (0..32)
        .map(|i| eighths(5 * (i / 4) + 3 * (i % 4), 11))
        .collect();
    let b: Vec<f32> = (0..32)
        .map(|i| eighths(7 * (i / 4) + 2 * (i % 4), 9))
        .collect();
    let product: Vec<f32> = (0..64)
        .map(|i| {
            let (row, col) = (i / 8, i % 8);
            (0..4)
                .map(|k| f64::from(a[row * 4 + k]) * f64::from(b[col * 4 + k]))
                .sum::<f64>() as f32
        })
        .collect();
    let expected: Vec<f32> = product.iter().cycle().take(256).copied().collect();
    match module.load_function("volta_m8n8k4") {
        Ok(_) => {
            let result = run(
                &stream,
                &module,
                "volta_m8n8k4",
                &packed(&a),
                &packed(&b),
                0,
            );
            failed |= report("volta_m8n8k4", result, &expected, 8);
        }
        Err(_) => println!("volta_m8n8k4: not in this module (built without `m8n8k4`)"),
    }

    if failed {
        return Err("some kernels did not match the CPU product".into());
    }
    Ok(())
}
