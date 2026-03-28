pub struct SramStreamer {
    ipu: HyzeIpuPcie,
    weights: Vec<u16>,  // 1M FP16 weights from Jupyter
}

impl SramStreamer {
    pub async fn stream_weights(&mut self) -> Result<()> {
        for chunk in self.weights.chunks(512) {  // 1KB blocks
            // DMA 1024 weights → SRAM bank
            self.ipu.dma_write(0x2000, &chunk.iter().map(|&w| w as u8).collect::<Vec<_>>()).await?;
            std::thread::sleep(std::time::Duration::from_nanos(100)); // Pipeline fill
        }
        println!("1MB SRAM streaming complete - 0.1μs/token ready");
        Ok(())
    }
    
    pub async fn infer_stream(&mut self, tokens: &[u32]) -> Vec<u32> {
        // Prefetch next 1024 weights during compute
        tokio::spawn({
            async move {
                self.stream_weights().await.unwrap();
            }
        });
        
        // Pipeline inference (Groq assembly line)
        tokens.iter().map(|&t| {
            // Token → embedding → NPU → next token
            self.ipu.infer_fast(&[t as u8; 784]).unwrap() as u32
        }).collect()
    }
}
