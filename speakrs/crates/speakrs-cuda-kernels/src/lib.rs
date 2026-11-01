//! cuda-oxide device kernels for the speakrs native CUDA backend
//!
//! Each area is a module behind its own feature. `cargo xtask cuda-kernels build`
//! compiles the crate once per area feature and writes
//! `src/inference/cuda/ptx/<area>.ptx` in the speakrs tree, so PTX entry names only
//! have to be unique within an area. Prefix them with the area name anyway, because
//! the host looks kernels up by that name
//!
//! An area may also ship variants for newer GPUs. The baseline variant targets
//! `sm_75` with no tier feature, and a higher variant turns on `tier-sm80`,
//! `tier-sm90` or `tier-sm120`. Gate tier-specific code with
//! `#[cfg(feature = "tier-sm80")]`. Every variant of an area must export the same
//! kernels with the same parameters, because the host picks one variant at run time
//! and looks kernels up by name
//!
//! The `sm70` variant for Volta sits below the baseline and turns on `tier-sm70`, which
//! removes code that needs sm_75. An entry it cannot implement stays as a stub that
//! traps, and the host never plans it below the sm75 tier

#[cfg(all(feature = "tier-sm70", feature = "tier-sm80"))]
compile_error!("`tier-sm70` removes sm_75 code that the higher tiers build on; enable one of them");

#[cfg(feature = "embedding")]
pub mod embedding;
#[cfg(feature = "fbank")]
pub mod fbank;
#[cfg(feature = "fbankdft")]
pub mod fbankdft;
#[cfg(feature = "lstm")]
pub mod lstm;
#[cfg(feature = "lstmproj")]
pub mod lstmproj;
#[cfg(feature = "probe")]
pub mod probe;
#[cfg(feature = "resnet")]
pub mod resnet;
#[cfg(feature = "segdense")]
pub mod segdense;
#[cfg(feature = "segmentation")]
pub mod segmentation;
#[cfg(feature = "sincnet")]
pub mod sincnet;
#[cfg(feature = "wideconv")]
pub mod wideconv;
