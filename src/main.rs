use clap::{Parser, Subcommand};

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
        pixels: String,  // Base64 image
    },
}

fn main() {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Compile { onnx, output } => {
            compile_onnx_to_verilog(onnx, output).unwrap();
        }
        Commands::Infer { pixels } => { /* DMA infer */ }
    }
}
