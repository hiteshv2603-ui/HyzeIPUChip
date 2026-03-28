// Hyze IPU PCIe Driver v1.0
// Gen4 x16 DMA → FPGA NPU (32 GB/s)
// Xilinx PCIe IP + your Verilog DMA interface

#include <pci/pci.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cstring>

class HyzeIpuPcie {
private:
    struct pci_dev* dev;
    void* bar0;      // PCIe BAR0 (DMA mem)
    size_t bar_size;
    
public:
    HyzeIpuPcie() : dev(nullptr), bar0(nullptr) {}
    
    ~HyzeIpuPcie() {
        if (bar0) munmap(bar0, bar_size);
        if (dev) pci_cleanup_dev(dev);
    }
    
    bool init(uint16_t vendor=0x10ee, uint16_t device=0x7021) {  // Xilinx VU9P VID:PID
        pci_init(nullptr);
        dev = pci_get_dev(nullptr, 0, 0, 0, vendor, device);
        if (!dev) {
            std::cerr << "FPGA PCIe not found\\n";
            return false;
        }
        
        pci_access_dev(dev);
        bar_size = pci_bar_size(dev->access, dev, 0);
        int fd = open("/dev/mem", O_RDWR | O_SYNC);
        bar0 = mmap(nullptr, bar_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, pci_resource_address(dev, 0));
        close(fd);
        
        // Reset IPU
        *(volatile uint32_t*)(bar0 + 0x1000) = 0xDEADBEEF;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::cout << "Hyze IPU PCIe ready (" << bar_size/1024 << "KB BAR0)\\n";
        return true;
    }
    
    uint8_t infer(const uint8_t* pixels_784) {
        auto t0 = std::chrono::high_resolution_clock::now();
        
        // DMA write: 784 pixels → FPGA mem (addr 0x0000)
        memcpy(bar0, pixels_784, 784);
        
        // Trigger inference (write cmd to control reg)
        *(volatile uint32_t*)(bar0 + 0xFFF0) = 0x00000001;  // START bit
        
        // Spin on DONE (2μs typical)
        volatile uint32_t* status = (volatile uint32_t*)(bar0 + 0xFFF4);
        while (!(*status & 0x00000001)) { /* 500MHz busy wait */ }
        
        // Read result (addr 0xFFF8)
        uint8_t class_id = *(volatile uint8_t*)(bar0 + 0xFFF8) & 0x0F;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
        std::cout << "IPU infer: " << (int)class_id << " (" << us << "μs)\\n";
        
        return class_id;
    }
    
    void benchmark(int iters=1000) {
        std::vector<uint8_t> pixels(784, 128);  // Fake image
        uint64_t total_us = 0;
        
        for (int i = 0; i < iters; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            infer(pixels.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            total_us += std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
        }
        std::cout << iters << " infers: " << (total_us*1e-3/iters) << "ms avg, "
                  << (iters*784*8/total_us*1e-6) << " TOPS\\n";
    }
};

int main(int argc, char** argv) {
    HyzeIpuPcie ipu;
    if (!ipu.init()) return 1;
    
    if (argc > 1 && std::string(argv[1]) == "bench") {
        ipu.benchmark(10000);
    } else {
        std::vector<uint8_t> test_img(784, 85);  // Test pattern
        uint8_t digit = ipu.infer(test_img.data());
        std::cout << "Predicted digit: " << (int)digit << "\\n";
    }
    
    return 0;
}
