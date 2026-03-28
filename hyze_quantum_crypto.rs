//! Hyze Quantum-Safe Crypto v1.0
//! Post-quantum encryption for AI weights + data
//! Kyber + Dilithium (NIST PQC standards)

use pqcrypto_kyber::*;
use pqcrypto_dilithium::*;
use anyhow::Result;
use std::fs;
use zeroize::Zeroize;

pub struct HyzeQuantumCrypto {
    kyber_kem: Kyber512,
    dilithium_sig: Dilithium3,
}

impl HyzeQuantumCrypto {
    pub fn new() -> Self {
        Self {
            kyber_kem: Kyber512::new(),
            dilithium_sig: Dilithium3::new(),
        }
    }

    // Encrypt AI weights (quantum-resistant)
    pub fn encrypt_weights(&self, weights: &[u8], pk: &[u8]) -> Result<Vec<u8>> {
        let (ciphertext, shared_secret) = self.kyber_kem.encap(pk)?;
        let mut encrypted = encrypt_aes_gcm(&weights, &shared_secret)?;
        encrypted.extend_from_slice(&ciphertext);
        Ok(encrypted)
    }

    // Sign model updates (unforgeable)
    pub fn sign_model_update(&self, update: &[u8], sk: &[u8]) -> Result<Vec<u8>> {
        let signature = self.dilithium_sig.sign(sk, update)?;
        let mut signed = update.to_vec();
        signed.extend_from_slice(&signature);
        Ok(signed)
    }

    // Verify + decrypt inference
    pub fn secure_inference(&self, encrypted_input: &[u8], sk: &[u8]) -> Result<Vec<u8>> {
        let (ciphertext, encrypted_data) = split_encrypted(encrypted_input)?;
        let (shared_secret, _) = self.kyber_kem.decap(sk, &ciphertext)?;
        
        let decrypted = decrypt_aes_gcm(&encrypted_data, &shared_secret)?;
        decrypted.zeroize_on_drop();
        Ok(decrypted)
    }
}

fn encrypt_aes_gcm(data: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
    use aes_gcm::Aes256Gcm;
    use aes_gcm::{Key, Nonce};
    
    let key = Key::<Aes256Gcm>::from_slice(key);
    let cipher = Aes256Gcm::new(key);
    let nonce = Nonce::from_slice(b"hyze_quantum12");
    
    let ciphertext = cipher.encrypt(nonce, data)?;
    Ok(ciphertext)
}
