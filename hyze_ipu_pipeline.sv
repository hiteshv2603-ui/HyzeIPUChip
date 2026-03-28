pipeline_reg #(.WIDTH(1024)) pipe[0:255];  // 256 stage conveyor
always_ff @(posedge clk) begin
    pipe[0] <= matmul_out;
    for (int i = 1; i < 256; i++) pipe[i] <= pipe[i-1];
end
