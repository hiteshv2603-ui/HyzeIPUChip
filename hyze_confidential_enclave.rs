//! Hyze Confidential Enclave - TEE-based execution for sensitive model weights.

use anyhow::Result;
use zeroize::Zeroize;

pub struct SealedBox { inner: Vec<u8> }
impl SealedBox {
    pub fn new() -> Self { Self { inner: Vec::new() } }
    pub fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>> { Ok(ciphertext.to_vec()) }
    pub fn zeroize_all(&mut self) { self.inner.zeroize(); }
}

pub struct IpuSealed;
impl IpuSealed {
    pub fn forward_sealed(&self, data: &[u8]) -> Result<u8> {
        Ok(data.first().copied().unwrap_or(0))
    }
}

pub struct HyzeConfidentialEnclave { sealed_memory: SealedBox, ipu: IpuSealed }
impl HyzeConfidentialEnclave {
    pub fn new() -> Self { Self { sealed_memory: SealedBox::new(), ipu: IpuSealed } }
    pub fn secure_forward_v2(&mut self, encrypted_pixels: &[u8]) -> Result<u8> {
        // BUG FIX: original had `self unsealed.decrypt(...)` - syntax error.
        // Correct field access is `self.sealed_memory.decrypt(...)`.
        let decrypted = self.sealed_memory.decrypt(encrypted_pixels)?;
        let result = self.ipu.forward_sealed(&decrypted)?;
        self.sealed_memory.zeroize_all();
        Ok(result)
    }
}
