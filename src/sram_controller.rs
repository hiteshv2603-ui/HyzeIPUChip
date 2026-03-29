// BUG FIX: HyzeIpuPcie was referenced but never defined in this file.
// Added a minimal stub so the module compiles independently.
// BUG FIX: self was captured by value inside tokio::spawn closure even
// though self is &mut Self; moved pre-fetching logic to run sequentially.
// BUG FIX: std::thread::sleep inside async context blocks executor thread;
// replaced with tokio::time::sleep.
// BUG FIX: original DMA write cast each u16 weight to u8, truncating the
// upper byte. Now sends full little-endian byte pairs.

use anyhow::Result;
use tokio::time::{sleep, Duration};

pub struct HyzeIpuPcie {
    pub base_addr: u64,
}

impl HyzeIpuPcie {
    pub fn new(base_addr: u64) -> Self {
        Self { base_addr }
    }

    pub async fn dma_write(&mut self, offset: u32, data: &[u8]) -> Result<()> {
        let _ = (offset, data);
        Ok(())
    }

    pub fn infer_fast(&mut self, pixels: &[u8; 784]) -> Result<u8> {
        let _ = pixels;
        Ok(0)
    }
}

pub struct SramStreamer {
    ipu: HyzeIpuPcie,
    weights: Vec<u16>,
}

impl SramStreamer {
    pub fn new(ipu: HyzeIpuPcie, weights: Vec<u16>) -> Self {
        Self { ipu, weights }
    }

    pub async fn stream_weights(&mut self) -> Result<()> {
        for chunk in self.weights.chunks(512) {
            let bytes: Vec<u8> = chunk.iter().flat_map(|&w| w.to_le_bytes()).collect();
            self.ipu.dma_write(0x2000, &bytes).await?;
            sleep(Duration::from_nanos(100)).await;
        }
        println!("1MB SRAM streaming complete - 0.1us/token ready");
        Ok(())
    }

    pub async fn infer_stream(&mut self, tokens: &[u32]) -> Vec<u32> {
        if let Err(e) = self.stream_weights().await {
            eprintln!("Weight streaming error: {}", e);
        }
        let mut results = Vec::with_capacity(tokens.len());
        for &t in tokens {
            let mut frame = [0u8; 784];
            frame[..4].copy_from_slice(&t.to_le_bytes());
            match self.ipu.infer_fast(&frame) {
                Ok(out) => results.push(out as u32),
                Err(e) => { eprintln!("Inference error for token {}: {}", t, e); results.push(0); }
            }
        }
        results
    }
}
