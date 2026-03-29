// BUG FIX: `usb1` is not a real crate on crates.io; the correct crate for
// libusb bindings in Rust is `rusb`.  All API calls have been updated to
// match the `rusb` surface (synchronous, not async).
use rusb::{DeviceHandle, GlobalContext, UsbContext};
use anyhow::{anyhow, Result};

/// Hyze IPU USB device driver.
///
/// Communicates with the Tang Primer FPGA over USB bulk endpoints.
pub struct HyzeIpu {
    handle: DeviceHandle<GlobalContext>,
}

impl HyzeIpu {
    /// Open the first Hyze IPU device found on the USB bus.
    ///
    /// BUG FIX: the original `new()` was marked `async` but `rusb` (and
    /// libusb) are synchronous; removed the spurious `async`.
    /// BUG FIX: `open_device_with_vid_pid` returns `Option`, not `Result`;
    /// the original called `.?` on it directly which would not compile.
    pub fn new() -> Result<Self> {
        let handle = rusb::open_device_with_vid_pid(0x1d50, 0x6029)
            .ok_or_else(|| anyhow!("Hyze IPU not found (VID:PID 1d50:6029)"))?;
        Ok(Self { handle })
    }

    /// Run a single inference pass.
    ///
    /// BUG FIX: the original method was `async` and called `.await` on
    /// `bulk_transfer_out`/`bulk_transfer_in`, but `rusb` is synchronous.
    /// The async wrapper has been removed and the correct synchronous
    /// `write_bulk` / `read_bulk` API is used instead.
    ///
    /// BUG FIX: `bulk_transfer_in(3, 4)` returned a `Vec<u8>` in the
    /// original, but `rusb::read_bulk` writes into a caller-supplied buffer
    /// and returns the number of bytes transferred.
    pub fn infer(&mut self, pixels: &[u8; 784]) -> Result<u8> {
        let timeout = std::time::Duration::from_millis(1000);

        // DMA write pixels to FPGA memory via endpoint 0x01 (OUT)
        self.handle.write_bulk(0x01, pixels, timeout)?;

        // Trigger inference via endpoint 0x02 (OUT)
        self.handle.write_bulk(0x02, &[0x01], timeout)?;

        // Poll done register via endpoint 0x83 (IN)
        let mut status = [0u8; 4];
        loop {
            let n = self.handle.read_bulk(0x83, &mut status, timeout)?;
            if n > 0 && status[0] == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        // Read result byte from endpoint 0x84 (IN)
        let mut result = [0u8; 1];
        self.handle.read_bulk(0x84, &mut result, timeout)?;
        Ok(result[0])
    }
}

/// Convenience entry-point used by the CLI `infer` subcommand.
///
/// Pads or truncates the pixel slice to exactly 784 bytes before sending.
pub fn run_usb_inference(pixels: &[u8]) -> Result<()> {
    let mut frame = [0u8; 784];
    let len = pixels.len().min(784);
    frame[..len].copy_from_slice(&pixels[..len]);

    let mut ipu = HyzeIpu::new()?;
    let digit = ipu.infer(&frame)?;
    println!("Hyze IPU predicted digit: {}", digit);
    Ok(())
}
