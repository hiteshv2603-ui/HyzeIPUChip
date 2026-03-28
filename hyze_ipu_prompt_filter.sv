module hyze_ipu_prompt_filter (
    input [7:0] prompt_byte,
    output reg safe_byte
);
    // 256-entry danger LUT (prompt injection triggers)
    reg [255:0] danger_lut = {
        8'h21, "!", 8'h3f, "?", 8'h25, "%",  // Special chars
        "ignore", "pretend", "forget", "jailbreak", "DAN"
    };
    
    always_comb begin
        safe_byte = !danger_lut[prompt_byte];
    end
endmodule
