// Hyze IPU Dataflow Pipeline v1.0
// 256-stage zero-bubble conveyor (SambaNova RDU DNA)
// Matmul→ReLU→Softmax assembly line

module hyze_ipu_dataflow_pipeline #(
    parameter DATA_WIDTH = 1024,
    parameter PIPE_DEPTH = 256
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              pipe_start,
    input  wire [DATA_WIDTH-1:0] pipe_data_in,   // NPU matmul result
    output reg  [9:0]        pipe_class_logits,  // Final 10-class probs
    output reg               pipe_complete
);

    // Pipeline registers (flattened by Yosys)
    reg [DATA_WIDTH-1:0] pipe_data [0:PIPE_DEPTH-1];
    reg [7:0]            pipe_stage [0:PIPE_DEPTH-1];  // 0=matmul, 1=relu, 2=softmax
    reg [PIPE_DEPTH-1:0] pipe_valid;
    
    localparam STAGE_MATMUL  = 8'h00;
    localparam STAGE_RELU    = 8'h01;
    localparam STAGE_SOFTMAX = 8'h02;
    
    integer stage_idx;
    
    // Pipeline shift register (SambaNova conveyor belt)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_complete <= 0;
            pipe_valid <= 0;
        end else begin
            // Shift entire pipeline
            for (stage_idx = PIPE_DEPTH-1; stage_idx > 0; stage_idx = stage_idx - 1) begin
                pipe_data[stage_idx] <= pipe_data[stage_idx-1];
                pipe_stage[stage_idx] <= pipe_stage[stage_idx-1];
                pipe_valid[stage_idx] <= pipe_valid[stage_idx-1];
            end
            
            // Inject new matmul result
            if (pipe_start) begin
                pipe_data[0] <= pipe_data_in;
                pipe_stage[0] <= STAGE_MATMUL;
                pipe_valid[0] <= 1'b1;
            end
            
            // Stage processing (dataflow)
            if (pipe_valid[64] && pipe_stage[64] == STAGE_MATMUL) begin
                // ReLU activation (cycle 64)
                for (int i = 0; i < DATA_WIDTH/16; i = i + 1) begin
                    if (pipe_data[64][i*16 +: 16] < 0)
                        pipe_data[64][i*16 +: 16] <= 0;
                end
                pipe_stage[64] <= STAGE_RELU;
            end else if (pipe_valid[192] && pipe_stage[192] == STAGE_RELU) begin
                // Softmax (cycle 192)
                reg [31:0] max_logit = 0;
                reg [31:0] sum_exp = 0;
                
                // Find max logit
                for (int i = 0; i < 10; i = i + 1) begin
                    if (pipe_data[192][i*102 +: 16] > max_logit)
                        max_logit = pipe_data[192][i*102 +: 16];
                end
                
                // Exp + normalize
                for (int i = 0; i < 10; i = i + 1) begin
                    sum_exp += exp_approx(pipe_data[192][i*102 +: 16] - max_logit);
                end
                for (int i = 0; i < 10; i = i + 1) begin
                    pipe_class_logits[i] <= exp_approx(pipe_data[192][i*102 +: 16] - max_logit) / sum_exp;
                end
                pipe_stage[192] <= STAGE_SOFTMAX;
            end else if (pipe_valid[PIPE_DEPTH-1]) begin
                pipe_complete <= 1'b1;
                pipe_valid[PIPE_DEPTH-1] <= 0;
            end
        end
    end
    
    // Fixed-point exp approx (FPGA friendly)
    function [15:0] exp_approx(input [15:0] x);
        exp_approx = (x * 16'h00B + 16'h4000) >> 4;  // e^x ≈ 1 + x + x^2/2
    endfunction

endmodule
