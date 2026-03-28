use usb1::{Context, Device};
use anyhow::Result;

pub struct HyzeIpu {
    device: Device,
}

impl HyzeIpu {
    pub fn new() -> Result<Self> {
        let context = Context::new()?;
        let device = context.open_device_with_vid_pid(0x1d50, 0x6029)?; // Tang Primer VID:PID
        Ok(Self { device })
    }
    
    pub async fn infer(&mut self, pixels: &[u8; 784]) -> Result<u8> {
        // DMA write pixels to FPGA mem (addr 0x0000)
        self.device.bulk_transfer_out(1, &pixels[..], 1000).await?;
        
        // Trigger inference (write cmd to 0xFFF)
        self.device.bulk_transfer_out(2, &[0x01], 1000).await?;
        
        // Poll done + read result
        loop {
            let status = self.device.bulk_transfer_in(3, 4).await?;
            if status[0] == 1 { break; }
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        }
        
        let result = self.device.bulk_transfer_in(4, 1).await?;
        Ok(result[0])
    }
}

// Usage
#[tokio::main]
async fn main() -> Result<()> {
    let mut ipu = HyzeIpu::new().await?;
    compile_onnx_to_verilog("mnist.onnx", "weights.v")?;
    
    let pixels = [0u8; 784]; // From Jupyter
    let digit = ipu.infer(&pixels).await?;
    println!("Hyze IPU predicted digit: {}", digit);
    Ok(())
}
