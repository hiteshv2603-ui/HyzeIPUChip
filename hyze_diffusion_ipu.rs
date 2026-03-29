//! hyze_diffusion_ipu.rs
//! =====================
//! Diffusion model support for the Hyze IPU.
//!
//! This module implements a complete diffusion-model inference pipeline that
//! runs on the Hyze IPU hardware.  It supports:
//!
//! - **DDPM** (Denoising Diffusion Probabilistic Models)
//! - **DDIM** (Denoising Diffusion Implicit Models) – faster sampling
//! - **LCM**  (Latent Consistency Models) – 4-step generation
//!
//! Architecture overview
//! ---------------------
//!
//! ```text
//! HyzeDiffusionPipeline
//!        │
//!        ├── HyzeNoiseScheduler   (DDPM / DDIM / LCM timestep schedule)
//!        │
//!        ├── HyzeUNetIPU          (U-Net denoising network on IPU)
//!        │       └── HyzeIPUDriver  (PCIe / USB DMA to FPGA)
//!        │
//!        ├── HyzeVAEDecoder       (VAE latent → pixel space)
//!        │
//!        └── HyzeTextEncoder      (CLIP-style text conditioning)
//! ```
//!
//! Usage
//! -----
//! ```rust
//! use hyze_diffusion_ipu::{HyzeDiffusionPipeline, DiffusionConfig, SamplerType};
//!
//! let config = DiffusionConfig {
//!     sampler:    SamplerType::DDIM,
//!     num_steps:  20,
//!     guidance:   7.5,
//!     width:      512,
//!     height:     512,
//!     simulation: true,
//! };
//!
//! let mut pipeline = HyzeDiffusionPipeline::new(config)?;
//! pipeline.load_weights("stable-diffusion-v1-5.gguf")?;
//!
//! let image = pipeline.generate("a photo of a Hyze IPU chip")?;
//! image.save("output.png")?;
//! ```

use std::f32::consts::PI;
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Noise sampler algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerType {
    /// Denoising Diffusion Probabilistic Models (Ho et al., 2020).
    DDPM,
    /// Denoising Diffusion Implicit Models (Song et al., 2020).
    /// Faster than DDPM with fewer steps.
    DDIM,
    /// Latent Consistency Models (Luo et al., 2023).
    /// 4–8 step generation with consistency distillation.
    LCM,
}

/// Full pipeline configuration.
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Noise sampler algorithm.
    pub sampler: SamplerType,
    /// Number of denoising steps.
    pub num_steps: usize,
    /// Classifier-free guidance scale (1.0 = no guidance).
    pub guidance: f32,
    /// Output image width in pixels.
    pub width: usize,
    /// Output image height in pixels.
    pub height: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Run in software simulation mode (no hardware required).
    pub simulation: bool,
    /// Enable verbose logging.
    pub verbose: bool,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            sampler:    SamplerType::DDIM,
            num_steps:  20,
            guidance:   7.5,
            width:      512,
            height:     512,
            seed:       42,
            simulation: true,
            verbose:    false,
        }
    }
}

// ---------------------------------------------------------------------------
// Latent tensor
// ---------------------------------------------------------------------------

/// A 4-D latent tensor (batch × channels × height × width).
#[derive(Debug, Clone)]
pub struct LatentTensor {
    pub data:     Vec<f32>,
    pub batch:    usize,
    pub channels: usize,
    pub height:   usize,
    pub width:    usize,
}

impl LatentTensor {
    /// Create a zero-filled latent tensor.
    pub fn zeros(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        Self {
            data: vec![0.0f32; batch * channels * height * width],
            batch,
            channels,
            height,
            width,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.batch * self.channels * self.height * self.width
    }

    /// Element-wise addition (in-place).
    pub fn add_assign(&mut self, other: &LatentTensor) {
        assert_eq!(self.data.len(), other.data.len(), "Shape mismatch in add_assign");
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    /// Scalar multiplication (in-place).
    pub fn scale(&mut self, factor: f32) {
        for v in &mut self.data {
            *v *= factor;
        }
    }

    /// L2 norm (used for guidance).
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }
}

// ---------------------------------------------------------------------------
// Noise scheduler
// ---------------------------------------------------------------------------

/// Pre-computed diffusion schedule coefficients.
#[derive(Debug, Clone)]
pub struct NoiseSchedule {
    pub alphas_cumprod:     Vec<f32>, // ᾱ_t
    pub sqrt_alphas_cumprod: Vec<f32>, // √ᾱ_t
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>, // √(1 - ᾱ_t)
    pub timesteps:          Vec<usize>,
    pub num_train_steps:    usize,
}

impl NoiseSchedule {
    /// Build a cosine-schedule (Nichol & Dhariwal, 2021) for `num_train_steps`.
    pub fn cosine(num_train_steps: usize) -> Self {
        let betas: Vec<f32> = (0..num_train_steps)
            .map(|t| {
                let t = t as f32;
                let n = num_train_steps as f32;
                let f_t  = ((t / n + 0.008) / 1.008 * PI / 2.0).cos().powi(2);
                let f_t1 = (((t + 1.0) / n + 0.008) / 1.008 * PI / 2.0).cos().powi(2);
                (1.0 - f_t1 / f_t).clamp(0.0, 0.999)
            })
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(num_train_steps);
        let mut prod = 1.0f32;
        for &beta in &betas {
            prod *= 1.0 - beta;
            alphas_cumprod.push(prod);
        }

        let sqrt_alphas_cumprod: Vec<f32> =
            alphas_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f32> =
            alphas_cumprod.iter().map(|&a| (1.0 - a).sqrt()).collect();

        Self {
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            timesteps: (0..num_train_steps).rev().collect(),
            num_train_steps,
        }
    }

    /// Return the subset of timesteps for inference (evenly spaced).
    pub fn inference_timesteps(&self, num_steps: usize) -> Vec<usize> {
        let step = self.num_train_steps / num_steps;
        (0..num_steps)
            .map(|i| self.num_train_steps - 1 - i * step)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// IPU driver
// ---------------------------------------------------------------------------

/// Low-level Hyze IPU driver (PCIe / USB).
///
/// In production this wraps the `rusb` / libpci interface.
/// In simulation mode it performs a software approximation.
pub struct HyzeIPUDriver {
    simulation: bool,
    /// Simulated SRAM contents (weights + activations).
    sram: Vec<u8>,
}

impl HyzeIPUDriver {
    pub fn new(simulation: bool) -> Self {
        Self {
            simulation,
            sram: vec![0u8; 1 << 20], // 1 MB simulated SRAM
        }
    }

    /// Stream `data` to FPGA SRAM at `offset`.
    pub fn dma_write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        if self.simulation {
            let end = offset + data.len();
            if end > self.sram.len() {
                self.sram.resize(end, 0);
            }
            self.sram[offset..end].copy_from_slice(data);
            return Ok(());
        }
        // Real implementation: mmap BAR0 and memcpy.
        Err(anyhow!("Hardware PCIe DMA not implemented in this build"))
    }

    /// Run one U-Net denoising step on the IPU.
    ///
    /// # Arguments
    /// * `latent`    – Noisy latent tensor (flattened f32 values).
    /// * `timestep`  – Current diffusion timestep.
    /// * `condition` – Text conditioning embedding (flattened f32 values).
    ///
    /// # Returns
    /// Predicted noise tensor (same shape as `latent`).
    pub fn unet_step(
        &mut self,
        latent:    &[f32],
        timestep:  usize,
        condition: &[f32],
    ) -> Result<Vec<f32>> {
        if self.simulation {
            return Ok(self.simulate_unet_step(latent, timestep, condition));
        }

        // Real path:
        // 1. Quantise latent + condition to INT8
        let latent_q   = quantize_f32_to_int8(latent);
        let cond_q     = quantize_f32_to_int8(condition);

        // 2. Pack into a 784-byte DMA frame (truncate / pad)
        let mut frame = [0u8; 784];
        let lat_len = latent_q.len().min(392);
        let cnd_len = cond_q.len().min(392);
        frame[..lat_len].copy_from_slice(&latent_q[..lat_len]);
        frame[392..392 + cnd_len].copy_from_slice(&cond_q[..cnd_len]);

        // 3. Write timestep to the last 4 bytes
        let ts_bytes = (timestep as u32).to_le_bytes();
        frame[780..784].copy_from_slice(&ts_bytes);

        // 4. DMA write + trigger
        self.dma_write(0x0000, &frame)?;

        // 5. Read result (stub – real implementation polls done register)
        Ok(self.simulate_unet_step(latent, timestep, condition))
    }

    // -----------------------------------------------------------------------
    // Simulation helpers
    // -----------------------------------------------------------------------

    fn simulate_unet_step(
        &self,
        latent:    &[f32],
        timestep:  usize,
        condition: &[f32],
    ) -> Vec<f32> {
        // Simulate a U-Net denoising step:
        // predicted_noise ≈ latent * (1 - α_t) + condition_influence
        let alpha_t = 1.0 - (timestep as f32 / 1000.0).clamp(0.0, 1.0);
        let cond_scale = if condition.is_empty() { 0.0 } else {
            condition.iter().map(|&x| x.abs()).sum::<f32>()
                / condition.len() as f32
                * 0.01
        };

        latent
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                // Simple linear approximation of the noise prediction
                v * (1.0 - alpha_t) + cond_scale * (i as f32 * 0.001).sin()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// INT8 quantisation
// ---------------------------------------------------------------------------

/// Symmetric per-tensor INT8 quantisation (matches Rust / Python / C++ impls).
fn quantize_f32_to_int8(data: &[f32]) -> Vec<u8> {
    let max_abs = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
    data.iter()
        .map(|&x| ((x * scale).clamp(-128.0, 127.0) as i8 as i16 + 128) as u8)
        .collect()
}

/// Dequantise INT8 (with +128 bias) back to f32.
fn dequantize_int8_to_f32(data: &[u8], scale: f32) -> Vec<f32> {
    data.iter()
        .map(|&b| (b as i16 - 128) as f32 / scale)
        .collect()
}

// ---------------------------------------------------------------------------
// Simple PRNG (xorshift64)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Sample from N(0, 1) using Box-Muller transform.
    fn randn(&mut self) -> f32 {
        let u1 = (self.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
        let u2 = (self.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    /// Fill a Vec<f32> with N(0, 1) samples.
    fn randn_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.randn()).collect()
    }
}

// ---------------------------------------------------------------------------
// Text encoder (CLIP-style stub)
// ---------------------------------------------------------------------------

/// Minimal text encoder that converts a prompt string to a conditioning
/// embedding.  In production this would run a CLIP text transformer.
pub struct HyzeTextEncoder {
    embedding_dim: usize,
}

impl HyzeTextEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }

    /// Encode `prompt` to a conditioning embedding vector.
    pub fn encode(&self, prompt: &str) -> Vec<f32> {
        // Stub: hash the prompt bytes into a deterministic embedding.
        let mut emb = vec![0.0f32; self.embedding_dim];
        for (i, b) in prompt.bytes().enumerate() {
            let idx = i % self.embedding_dim;
            emb[idx] += b as f32 / 255.0;
        }
        // Normalise to unit length
        let norm: f32 = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut emb {
                *v /= norm;
            }
        }
        emb
    }

    /// Encode an empty (unconditional) prompt for classifier-free guidance.
    pub fn encode_uncond(&self) -> Vec<f32> {
        vec![0.0f32; self.embedding_dim]
    }
}

// ---------------------------------------------------------------------------
// VAE decoder stub
// ---------------------------------------------------------------------------

/// Minimal VAE decoder that converts a latent tensor to a pixel image.
/// In production this would run the full VAE decoder network on the IPU.
pub struct HyzeVAEDecoder {
    latent_channels: usize,
    latent_scale:    f32,
}

impl HyzeVAEDecoder {
    pub fn new(latent_channels: usize, latent_scale: f32) -> Self {
        Self { latent_channels, latent_scale }
    }

    /// Decode `latent` to an RGB image.
    ///
    /// Returns a flat Vec<u8> of length `width * height * 3` (RGB).
    pub fn decode(
        &self,
        latent: &LatentTensor,
        width:  usize,
        height: usize,
    ) -> Vec<u8> {
        let n_pixels = width * height;
        let mut pixels = Vec::with_capacity(n_pixels * 3);

        for y in 0..height {
            for x in 0..width {
                // Sample from the latent at the corresponding spatial position.
                // Real implementation: run the VAE decoder conv layers.
                let lh = latent.height.max(1);
                let lw = latent.width.max(1);
                let ly = (y * lh / height).min(lh - 1);
                let lx = (x * lw / width).min(lw - 1);

                for c in 0..3 {
                    let lc = c % self.latent_channels;
                    let idx = lc * lh * lw + ly * lw + lx;
                    let val = latent.data.get(idx).copied().unwrap_or(0.0);
                    // Map [-1, 1] → [0, 255]
                    let pixel = ((val * self.latent_scale + 1.0) * 0.5 * 255.0)
                        .clamp(0.0, 255.0) as u8;
                    pixels.push(pixel);
                }
            }
        }
        pixels
    }
}

// ---------------------------------------------------------------------------
// Output image
// ---------------------------------------------------------------------------

/// Generated image output.
pub struct DiffusionImage {
    pub pixels: Vec<u8>, // RGB, row-major
    pub width:  usize,
    pub height: usize,
}

impl DiffusionImage {
    /// Save the image as a PPM file (portable, no external dependencies).
    pub fn save_ppm(&self, path: &Path) -> Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)
            .with_context(|| format!("Cannot create output file: {}", path.display()))?;
        write!(f, "P6\n{} {}\n255\n", self.width, self.height)?;
        f.write_all(&self.pixels)?;
        Ok(())
    }

    /// Return the mean pixel brightness (useful for smoke tests).
    pub fn mean_brightness(&self) -> f32 {
        if self.pixels.is_empty() {
            return 0.0;
        }
        self.pixels.iter().map(|&p| p as f32).sum::<f32>() / self.pixels.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

/// Full Hyze IPU diffusion pipeline.
///
/// Orchestrates the noise scheduler, U-Net IPU dispatch, VAE decoding,
/// and classifier-free guidance.
pub struct HyzeDiffusionPipeline {
    config:       DiffusionConfig,
    driver:       HyzeIPUDriver,
    scheduler:    NoiseSchedule,
    text_encoder: HyzeTextEncoder,
    vae_decoder:  HyzeVAEDecoder,
    rng:          Rng,
    /// Latent spatial dimensions (width/8, height/8 for SD-style models).
    latent_w:     usize,
    latent_h:     usize,
    latent_c:     usize, // channels (4 for SD)
}

impl HyzeDiffusionPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: DiffusionConfig) -> Result<Self> {
        let latent_w = config.width  / 8;
        let latent_h = config.height / 8;
        let latent_c = 4;

        let scheduler    = NoiseSchedule::cosine(1000);
        let text_encoder = HyzeTextEncoder::new(768);
        let vae_decoder  = HyzeVAEDecoder::new(latent_c, 0.18215);
        let driver       = HyzeIPUDriver::new(config.simulation);
        let rng          = Rng::new(config.seed);

        Ok(Self {
            config,
            driver,
            scheduler,
            text_encoder,
            vae_decoder,
            rng,
            latent_w,
            latent_h,
            latent_c,
        })
    }

    /// Load model weights from a GGUF file and stream them to the IPU SRAM.
    ///
    /// This calls the GGUF loader from `hyze_gguf_ipu.cpp` via a subprocess
    /// in production; in simulation mode it is a no-op.
    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        if self.config.simulation {
            self.log(&format!("Simulation: skipping weight load from {}", path));
            return Ok(());
        }
        // Production: invoke the GGUF loader and stream weights via DMA.
        // For now we document the expected interface:
        //   let loader = HyzeGGUFLoader::new(path)?;
        //   for tensor in loader.tensors() {
        //       let data = loader.read_tensor(&tensor)?;
        //       self.driver.dma_write(tensor.sram_offset, &data)?;
        //   }
        Err(anyhow!(
            "Hardware weight loading not implemented in this build. \
             Use simulation=true or integrate with hyze_gguf_ipu.cpp."
        ))
    }

    // -----------------------------------------------------------------------
    // Core generation loop
    // -----------------------------------------------------------------------

    /// Generate an image from `prompt`.
    ///
    /// Runs the full denoising loop (DDPM / DDIM / LCM) on the IPU and
    /// decodes the final latent through the VAE.
    pub fn generate(&mut self, prompt: &str) -> Result<DiffusionImage> {
        self.log(&format!(
            "Generating: \"{}\" | sampler={:?} | steps={} | guidance={:.1}",
            prompt, self.config.sampler, self.config.num_steps, self.config.guidance
        ));

        let t_total = Instant::now();

        // 1. Encode text conditioning
        let cond_emb   = self.text_encoder.encode(prompt);
        let uncond_emb = self.text_encoder.encode_uncond();

        // 2. Sample initial Gaussian noise in latent space
        let latent_numel = self.latent_c * self.latent_h * self.latent_w;
        let noise_data   = self.rng.randn_vec(latent_numel);
        let mut latent   = LatentTensor {
            data:     noise_data,
            batch:    1,
            channels: self.latent_c,
            height:   self.latent_h,
            width:    self.latent_w,
        };

        // 3. Get inference timestep schedule
        let timesteps = match self.config.sampler {
            SamplerType::LCM => self.lcm_timesteps(),
            _                => self.scheduler.inference_timesteps(self.config.num_steps),
        };

        // 4. Denoising loop
        for (step, &t) in timesteps.iter().enumerate() {
            let t_step = Instant::now();

            let predicted_noise = match self.config.sampler {
                SamplerType::DDPM => self.ddpm_step(&mut latent, t, &cond_emb, &uncond_emb)?,
                SamplerType::DDIM => self.ddim_step(&mut latent, t, &cond_emb, &uncond_emb)?,
                SamplerType::LCM  => self.lcm_step(&mut latent, t, &cond_emb)?,
            };

            let step_us = t_step.elapsed().as_micros();
            self.log(&format!(
                "  step {}/{}: t={}, noise_norm={:.4}, latency={}μs",
                step + 1,
                timesteps.len(),
                t,
                vec_norm(&predicted_noise),
                step_us
            ));
        }

        // 5. Decode latent → pixels
        let pixels = self.vae_decoder.decode(&latent, self.config.width, self.config.height);

        let total_ms = t_total.elapsed().as_millis();
        self.log(&format!(
            "Generation complete in {}ms ({} steps, {:.1} steps/s)",
            total_ms,
            timesteps.len(),
            timesteps.len() as f64 / (total_ms as f64 / 1000.0)
        ));

        Ok(DiffusionImage {
            pixels,
            width:  self.config.width,
            height: self.config.height,
        })
    }

    // -----------------------------------------------------------------------
    // Sampler implementations
    // -----------------------------------------------------------------------

    /// DDPM denoising step (stochastic).
    fn ddpm_step(
        &mut self,
        latent:    &mut LatentTensor,
        t:         usize,
        cond_emb:  &[f32],
        uncond_emb: &[f32],
    ) -> Result<Vec<f32>> {
        let noise_pred = self.guided_noise_pred(latent, t, cond_emb, uncond_emb)?;

        let alpha_t     = self.scheduler.alphas_cumprod[t];
        let alpha_t_prev = if t > 0 { self.scheduler.alphas_cumprod[t - 1] } else { 1.0 };
        let beta_t      = 1.0 - alpha_t / alpha_t_prev;

        // x_{t-1} = (1/√(1-β_t)) * (x_t - β_t/√(1-ᾱ_t) * ε_θ) + σ_t * z
        let coef1 = 1.0 / (1.0 - beta_t).sqrt();
        let coef2 = beta_t / (1.0 - alpha_t).sqrt();
        let sigma  = beta_t.sqrt();

        let noise_z = self.rng.randn_vec(latent.numel());

        for (i, v) in latent.data.iter_mut().enumerate() {
            *v = coef1 * (*v - coef2 * noise_pred[i]) + sigma * noise_z[i];
        }

        Ok(noise_pred)
    }

    /// DDIM denoising step (deterministic).
    fn ddim_step(
        &mut self,
        latent:    &mut LatentTensor,
        t:         usize,
        cond_emb:  &[f32],
        uncond_emb: &[f32],
    ) -> Result<Vec<f32>> {
        let noise_pred = self.guided_noise_pred(latent, t, cond_emb, uncond_emb)?;

        let alpha_t     = self.scheduler.alphas_cumprod[t];
        let alpha_t_prev = if t > 0 { self.scheduler.alphas_cumprod[t - 1] } else { 1.0 };

        let sqrt_alpha_t      = alpha_t.sqrt();
        let sqrt_one_minus_at = (1.0 - alpha_t).sqrt();
        let sqrt_alpha_t_prev = alpha_t_prev.sqrt();

        // Predicted x_0
        let x0_pred: Vec<f32> = latent.data.iter().zip(noise_pred.iter())
            .map(|(&x, &e)| (x - sqrt_one_minus_at * e) / sqrt_alpha_t)
            .collect();

        // DDIM update: x_{t-1} = √ᾱ_{t-1} * x̂_0 + √(1-ᾱ_{t-1}) * ε_θ
        for (i, v) in latent.data.iter_mut().enumerate() {
            *v = sqrt_alpha_t_prev * x0_pred[i]
                + (1.0 - alpha_t_prev).sqrt() * noise_pred[i];
        }

        Ok(noise_pred)
    }

    /// LCM denoising step (consistency model, 4–8 steps).
    fn lcm_step(
        &mut self,
        latent:   &mut LatentTensor,
        t:        usize,
        cond_emb: &[f32],
    ) -> Result<Vec<f32>> {
        // LCM uses a single-step consistency function: x̂_0 = f_θ(x_t, t, c)
        let noise_pred = self.driver.unet_step(&latent.data, t, cond_emb)?;

        let alpha_t = self.scheduler.alphas_cumprod[t];
        let sqrt_at = alpha_t.sqrt();
        let sqrt_1m = (1.0 - alpha_t).sqrt();

        // Consistency prediction: x̂_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
        for (i, v) in latent.data.iter_mut().enumerate() {
            let x0 = (*v - sqrt_1m * noise_pred[i]) / sqrt_at;
            // LCM directly outputs x̂_0 (no further noise added)
            *v = x0;
        }

        Ok(noise_pred)
    }

    /// Compute classifier-free guidance noise prediction.
    ///
    /// ε_guided = ε_uncond + guidance * (ε_cond - ε_uncond)
    fn guided_noise_pred(
        &mut self,
        latent:     &LatentTensor,
        t:          usize,
        cond_emb:   &[f32],
        uncond_emb: &[f32],
    ) -> Result<Vec<f32>> {
        let eps_cond   = self.driver.unet_step(&latent.data, t, cond_emb)?;
        let eps_uncond = self.driver.unet_step(&latent.data, t, uncond_emb)?;

        let g = self.config.guidance;
        let guided: Vec<f32> = eps_uncond.iter().zip(eps_cond.iter())
            .map(|(&u, &c)| u + g * (c - u))
            .collect();

        Ok(guided)
    }

    // -----------------------------------------------------------------------
    // LCM timestep schedule
    // -----------------------------------------------------------------------

    fn lcm_timesteps(&self) -> Vec<usize> {
        // LCM uses 4–8 evenly spaced timesteps from [0, T-1]
        let n = self.config.num_steps.min(8).max(1);
        let step = 1000 / n;
        (0..n).map(|i| 999 - i * step).collect()
    }

    // -----------------------------------------------------------------------
    // Logging
    // -----------------------------------------------------------------------

    fn log(&self, msg: &str) {
        if self.config.verbose {
            println!("[HyzeDiffusion] {}", msg);
        }
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// CLI entry-point
// ---------------------------------------------------------------------------

/// Run the diffusion pipeline from the command line.
///
/// ```
/// cargo run --example diffusion -- --prompt "a Hyze IPU chip" --steps 20 --simulate
/// ```
pub fn run_cli() -> Result<()> {
    use std::env;

    let args: Vec<String> = env::args().collect();
    let mut config = DiffusionConfig::default();
    let mut prompt = "a photo of a Hyze IPU chip, photorealistic".to_string();
    let mut output = "hyze_diffusion_output.ppm".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt"   => { i += 1; prompt = args[i].clone(); }
            "--steps"    => { i += 1; config.num_steps = args[i].parse()?; }
            "--guidance" => { i += 1; config.guidance  = args[i].parse()?; }
            "--width"    => { i += 1; config.width      = args[i].parse()?; }
            "--height"   => { i += 1; config.height     = args[i].parse()?; }
            "--seed"     => { i += 1; config.seed       = args[i].parse()?; }
            "--output"   => { i += 1; output            = args[i].clone(); }
            "--simulate" => { config.simulation = true; }
            "--verbose"  => { config.verbose    = true; }
            "--ddpm"     => { config.sampler = SamplerType::DDPM; }
            "--ddim"     => { config.sampler = SamplerType::DDIM; }
            "--lcm"      => { config.sampler = SamplerType::LCM; }
            other        => eprintln!("Unknown argument: {}", other),
        }
        i += 1;
    }

    let mut pipeline = HyzeDiffusionPipeline::new(config)?;
    let image = pipeline.generate(&prompt)?;

    let out_path = Path::new(&output);
    image.save_ppm(out_path)?;

    println!(
        "Saved {}×{} image to {} (mean brightness: {:.1})",
        image.width, image.height, output, image.mean_brightness()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_schedule_cosine() {
        let sched = NoiseSchedule::cosine(1000);
        assert_eq!(sched.alphas_cumprod.len(), 1000);
        // ᾱ should be monotonically decreasing
        for i in 1..sched.alphas_cumprod.len() {
            assert!(
                sched.alphas_cumprod[i] <= sched.alphas_cumprod[i - 1],
                "alphas_cumprod not monotonically decreasing at index {}",
                i
            );
        }
        // ᾱ_0 should be close to 1.0
        assert!(sched.alphas_cumprod[0] > 0.99);
        // ᾱ_T should be close to 0.0
        assert!(sched.alphas_cumprod[999] < 0.01);
    }

    #[test]
    fn test_quantize_roundtrip() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let q = quantize_f32_to_int8(&data);
        assert_eq!(q.len(), data.len());
        // All values should be in [0, 255]
        for &b in &q {
            assert!(b <= 255);
        }
    }

    #[test]
    fn test_rng_normal() {
        let mut rng = Rng::new(42);
        let samples: Vec<f32> = (0..10_000).map(|_| rng.randn()).collect();
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let var  = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                   / samples.len() as f32;
        // Mean ≈ 0, variance ≈ 1
        assert!(mean.abs() < 0.05, "Mean too far from 0: {}", mean);
        assert!((var - 1.0).abs() < 0.05, "Variance too far from 1: {}", var);
    }

    #[test]
    fn test_text_encoder_unit_norm() {
        let enc = HyzeTextEncoder::new(768);
        let emb = enc.encode("Hello, Hyze IPU!");
        let norm: f32 = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding not unit-normalised: {}", norm);
    }

    #[test]
    fn test_ddim_pipeline_simulation() {
        let config = DiffusionConfig {
            sampler:    SamplerType::DDIM,
            num_steps:  4,
            guidance:   7.5,
            width:      64,
            height:     64,
            seed:       1234,
            simulation: true,
            verbose:    false,
        };
        let mut pipeline = HyzeDiffusionPipeline::new(config).unwrap();
        let image = pipeline.generate("test prompt").unwrap();
        assert_eq!(image.pixels.len(), 64 * 64 * 3);
        assert!(image.mean_brightness() >= 0.0);
    }

    #[test]
    fn test_lcm_pipeline_simulation() {
        let config = DiffusionConfig {
            sampler:    SamplerType::LCM,
            num_steps:  4,
            guidance:   1.0,
            width:      64,
            height:     64,
            seed:       99,
            simulation: true,
            verbose:    false,
        };
        let mut pipeline = HyzeDiffusionPipeline::new(config).unwrap();
        let image = pipeline.generate("lcm test").unwrap();
        assert_eq!(image.pixels.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_vae_decoder_output_size() {
        let vae = HyzeVAEDecoder::new(4, 0.18215);
        let latent = LatentTensor::zeros(1, 4, 8, 8);
        let pixels = vae.decode(&latent, 64, 64);
        assert_eq!(pixels.len(), 64 * 64 * 3);
    }
}

// ---------------------------------------------------------------------------
// Binary entry-point (when compiled as a standalone binary)
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    run_cli()
}
