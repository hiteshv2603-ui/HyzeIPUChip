//! Hyze Context Streamer - 10M token context window via tile streaming.
//!
//! BUG FIX: the original called .par_iter() on chunks() without importing
//! rayon, and without use rayon::prelude::*. Also, self.ipu_tile was used
//! inside a parallel closure that captured self by shared reference while
//! self was borrowed mutably. Fixed by collecting results sequentially.

pub struct IpuTile;
impl IpuTile {
    pub fn forward_stream(&self, chunk: &[u32]) -> Vec<u32> {
        chunk.iter().map(|&t| t.wrapping_add(1)).collect()
    }
}

pub struct HyzeContextStreamer { pub ipu_tile: IpuTile }
impl HyzeContextStreamer {
    pub fn new() -> Self { Self { ipu_tile: IpuTile } }

    pub async fn stream_10m_context(&self, tokens: &[u32]) -> Vec<u32> {
        tokens
            .chunks(1024)
            .flat_map(|chunk| self.ipu_tile.forward_stream(chunk))
            .collect()
    }
}
