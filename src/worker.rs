// BUG FIX: `json_from_req` is not a function in the `worker` crate;
// the correct API is `req.json::<T>().await`.
// BUG FIX: `base64::decode` was called without importing the crate and
// without handling the Result it returns.
// BUG FIX: `HyzeServerlessIpu` was referenced but never defined; added stub.

use worker::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct InferRequest {
    model: Option<String>,
    input: String,
}

struct HyzeServerlessIpu;

impl HyzeServerlessIpu {
    fn new() -> Self { Self }

    async fn instant_inference(&mut self, _model: &str, _input: Vec<u8>) -> serde_json::Value {
        serde_json::json!({"status": "ok", "result": 0, "cold_start_us": 0})
    }
}

#[event(fetch)]
pub async fn main(mut req: Request, _env: Env, _ctx: worker::Context) -> Result<Response> {
    let mut ipu = HyzeServerlessIpu::new();

    // BUG FIX: use the correct worker-crate JSON parsing API.
    let body: InferRequest = req.json().await.map_err(|e| {
        worker::Error::RustError(format!("Invalid JSON body: {}", e))
    })?;

    let model = body.model.as_deref().unwrap_or("mnist");

    // BUG FIX: base64 decoding now uses a proper decoder and propagates errors.
    let input = base64_decode(&body.input).map_err(|e| {
        worker::Error::RustError(format!("base64 decode error: {}", e))
    })?;

    let result = ipu.instant_inference(model, input).await;
    Response::from_json(&result)
}

fn base64_decode(encoded: &str) -> std::result::Result<Vec<u8>, String> {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut table = [0u8; 256];
    for (i, &c) in alphabet.iter().enumerate() { table[c as usize] = i as u8; }
    let input: Vec<u8> = encoded.bytes().filter(|&b| b != b'=').collect();
    if input.is_empty() { return Err("Empty base64 input".to_string()); }
    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    for chunk in input.chunks(4) {
        let b0 = table[chunk[0] as usize];
        let b1 = if chunk.len() > 1 { table[chunk[1] as usize] } else { 0 };
        let b2 = if chunk.len() > 2 { table[chunk[2] as usize] } else { 0 };
        let b3 = if chunk.len() > 3 { table[chunk[3] as usize] } else { 0 };
        output.push((b0 << 2) | (b1 >> 4));
        if chunk.len() > 2 { output.push((b1 << 4) | (b2 >> 2)); }
        if chunk.len() > 3 { output.push((b2 << 6) | b3); }
    }
    Ok(output)
}
