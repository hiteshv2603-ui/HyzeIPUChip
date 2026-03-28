module tb_hyze_ipu_dma;
    logic clk = 0, rst_n = 0;
    logic [31:0] host_data; logic [11:0] host_addr; logic host_en;
    wire [3:0] class_id; wire done, busy;
    
    hyze_ipu_dma_if dma_if();
    assign dma_if.clk = clk; assign dma_if.rst_n = rst_n;
    assign dma_if.host_wr_data = host_data; assign dma_if.host_wr_addr = host_addr;
    assign dma_if.host_wr_en = host_en;
    
    hyze_ipu_dma_controller dut (.dma_if(dma_if));
    
    always #1 clk = ~clk;
    
    initial begin
        rst_n = 1; #10;
        // Load 784 pixels + start cmd
        repeat(785) begin @(posedge clk); host_en = 1; /* data/addr */ end
        host_en = 0;
        wait(done) $display("IPU predicted: %d", class_id);
    end
endmodule
