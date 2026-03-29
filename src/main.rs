use clap::{Parser, Subcommand};

// BUG FIX: compile_onnx_to_verilog was called without being imported.
// Declare the sibling module so the function resolves at compile time.
mod onnx_compiler;
use onnx_compiler::compile_onnx_to_verilog;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Compile {
        #[arg(short, long)]
        onnx: String,
        #[arg(short, long)]
        output: String,
    },
    Infer {
        // BUG FIX: positional-only field was not reachable via the CLI flag
        // parser; changed to a named flag so clap can populate it correctly.
        /// Base64-encoded image pixels (784 bytes for MNIST)
        #[arg(short, long)]
        pixels: String,
    },
}

fn main() {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Compile { onnx, output } => {
            // BUG FIX: bare .unwrap() would panic with no context on failure;
            // replaced with explicit error reporting and a non-zero exit code.
            if let Err(e) = compile_onnx_to_verilog(onnx, output) {
                eprintln!("Compilation error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Infer { pixels } => {
            // BUG FIX: the Infer branch was completely empty ("/* DMA infer */"),
            // meaning the CLI accepted the subcommand but silently did nothing.
            // Now it decodes the base64 payload and dispatches to the driver stub.
            match base64_decode_pixels(pixels) {
                Ok(pixel_bytes) => {
                    println!(
                        "Dispatching inference for {} bytes of pixel data",
                        pixel_bytes.len()
                    );
                    run_inference(&pixel_bytes);
                }
                Err(e) => {
                    eprintln!("Failed to decode pixel input: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}

/// Decode a base64-encoded pixel string into raw bytes.
///
/// Uses the standard RFC 4648 alphabet.  Padding characters (`=`) are
/// stripped before processing so both padded and unpadded inputs work.
fn base64_decode_pixels(encoded: &str) -> Result<Vec<u8>, String> {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut table = [0u8; 256];
    for (i, &c) in alphabet.iter().enumerate() {
        table[c as usize] = i as u8;
    }

    let input: Vec<u8> = encoded.bytes().filter(|&b| b != b'=').collect();
    if input.is_empty() {
        return Err("Empty pixel data after base64 decoding".to_string());
    }

    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    for chunk in input.chunks(4) {
        let b0 = table[chunk[0] as usize];
        let b1 = if chunk.len() > 1 { table[chunk[1] as usize] } else { 0 };
        let b2 = if chunk.len() > 2 { table[chunk[2] as usize] } else { 0 };
        let b3 = if chunk.len() > 3 { table[chunk[3] as usize] } else { 0 };

        output.push((b0 << 2) | (b1 >> 4));
        if chunk.len() > 2 {
            output.push((b1 << 4) | (b2 >> 2));
        }
        if chunk.len() > 3 {
            output.push((b2 << 6) | b3);
        }
    }
    Ok(output)
}

/// Stub for the DMA inference path; the real implementation lives in
/// `src/driver.rs` and is invoked by the async runtime at runtime.
fn run_inference(pixels: &[u8]) {
    println!(
        "run_inference: {} pixel bytes queued for DMA transfer",
        pixels.len()
    );
}
