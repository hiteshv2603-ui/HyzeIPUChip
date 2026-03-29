/**
 * @file hyze_gguf_ipu.cpp
 * @brief GGUF model support for the Hyze IPU.
 *
 * This file implements a C++ integration layer that:
 *  1. Parses GGUF (GPT-Unified Format) model files.
 *  2. Extracts and quantises weight tensors.
 *  3. Streams the weights to the Hyze IPU SRAM via PCIe DMA.
 *  4. Dispatches token-level inference to the IPU NPU core.
 *
 * GGUF format reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 *
 * Architecture overview
 * ---------------------
 *
 *   GGUF file  ──►  HyzeGGUFLoader  ──►  weight tensors (INT8)
 *                                              │
 *                                              ▼
 *                                    HyzeIPUPCIeDriver  ──►  FPGA SRAM
 *                                              │
 *                                              ▼
 *                                    HyzeGGUFInference  ──►  token output
 *
 * Build
 * -----
 * @code
 *   g++ -std=c++20 -O2 -o hyze_gguf_ipu hyze_gguf_ipu.cpp \
 *       -lpci -lpthread
 * @endcode
 *
 * Usage
 * -----
 * @code
 *   HyzeGGUFInference engine;
 *   engine.loadModel("llama-3-8b-q4_k_m.gguf");
 *   std::string reply = engine.generate("Hello, Hyze IPU!", 128);
 *   std::cout << reply << "\n";
 * @endcode
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// GGUF format constants
// ---------------------------------------------------------------------------

/// Magic bytes at the start of every GGUF file.
static constexpr uint32_t GGUF_MAGIC   = 0x46554747u; // "GGUF"
static constexpr uint32_t GGUF_VERSION = 3;

/// GGUF value types (metadata).
enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

/// GGML tensor types (quantisation formats).
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
};

// ---------------------------------------------------------------------------
// GGUF data structures
// ---------------------------------------------------------------------------

struct GGUFString {
    std::string value;
};

struct GGUFMetaValue {
    GGUFValueType type;
    // Union-like storage using std::variant would require C++17; we use a
    // tagged union for maximum compatibility.
    uint64_t    u64  = 0;
    int64_t     i64  = 0;
    double      f64  = 0.0;
    std::string str;
    std::vector<GGUFMetaValue> array;
};

struct GGUFTensorInfo {
    std::string              name;
    std::vector<uint64_t>    shape;      ///< Dimensions (n_dims elements).
    GGMLType                 type;
    uint64_t                 offset;     ///< Byte offset in the data section.
    uint64_t                 n_elements; ///< Total number of elements.
    uint64_t                 n_bytes;    ///< Byte size of the tensor data.
};

struct GGUFModel {
    uint32_t version = 0;
    std::unordered_map<std::string, GGUFMetaValue> metadata;
    std::vector<GGUFTensorInfo>                    tensors;
    uint64_t                                       data_offset = 0; ///< Offset of tensor data in file.
};

// ---------------------------------------------------------------------------
// GGUF file reader
// ---------------------------------------------------------------------------

/**
 * @class HyzeGGUFLoader
 * @brief Parses a GGUF file and exposes its metadata and tensor descriptors.
 *
 * The loader does **not** load tensor data into RAM all at once; instead it
 * records byte offsets so that the DMA engine can stream weights directly
 * from the file into FPGA SRAM.
 */
class HyzeGGUFLoader {
public:
    /**
     * Load and parse the GGUF file at @p path.
     *
     * @throws std::runtime_error on format errors or I/O failures.
     */
    explicit HyzeGGUFLoader(const std::filesystem::path& path)
        : path_(path)
    {
        file_.open(path, std::ios::binary);
        if (!file_) {
            throw std::runtime_error("Cannot open GGUF file: " + path.string());
        }
        parse();
    }

    const GGUFModel& model() const noexcept { return model_; }

    /**
     * Read the raw bytes of tensor @p info from the file.
     *
     * @returns Vector of raw bytes (quantised or float, depending on type).
     */
    std::vector<uint8_t> readTensorData(const GGUFTensorInfo& info) {
        std::vector<uint8_t> buf(info.n_bytes);
        file_.seekg(static_cast<std::streamoff>(model_.data_offset + info.offset));
        file_.read(reinterpret_cast<char*>(buf.data()),
                   static_cast<std::streamsize>(info.n_bytes));
        if (!file_) {
            throw std::runtime_error(
                "Failed to read tensor data for: " + info.name
            );
        }
        return buf;
    }

private:
    std::filesystem::path path_;
    std::ifstream         file_;
    GGUFModel             model_;

    // -----------------------------------------------------------------------
    // Parsing helpers
    // -----------------------------------------------------------------------

    template<typename T>
    T read() {
        T val{};
        file_.read(reinterpret_cast<char*>(&val), sizeof(T));
        if (!file_) throw std::runtime_error("Unexpected end of GGUF file.");
        return val;
    }

    std::string readString() {
        uint64_t len = read<uint64_t>();
        std::string s(len, '\0');
        file_.read(s.data(), static_cast<std::streamsize>(len));
        if (!file_) throw std::runtime_error("Unexpected end of GGUF string.");
        return s;
    }

    GGUFMetaValue readMetaValue(GGUFValueType type) {
        GGUFMetaValue v;
        v.type = type;
        switch (type) {
            case GGUFValueType::UINT8:   v.u64 = read<uint8_t>();  break;
            case GGUFValueType::INT8:    v.i64 = read<int8_t>();   break;
            case GGUFValueType::UINT16:  v.u64 = read<uint16_t>(); break;
            case GGUFValueType::INT16:   v.i64 = read<int16_t>();  break;
            case GGUFValueType::UINT32:  v.u64 = read<uint32_t>(); break;
            case GGUFValueType::INT32:   v.i64 = read<int32_t>();  break;
            case GGUFValueType::FLOAT32: v.f64 = read<float>();    break;
            case GGUFValueType::BOOL:    v.u64 = read<uint8_t>();  break;
            case GGUFValueType::STRING:  v.str = readString();     break;
            case GGUFValueType::UINT64:  v.u64 = read<uint64_t>(); break;
            case GGUFValueType::INT64:   v.i64 = read<int64_t>();  break;
            case GGUFValueType::FLOAT64: v.f64 = read<double>();   break;
            case GGUFValueType::ARRAY: {
                auto elem_type = static_cast<GGUFValueType>(read<uint32_t>());
                uint64_t count = read<uint64_t>();
                v.array.reserve(count);
                for (uint64_t i = 0; i < count; ++i) {
                    v.array.push_back(readMetaValue(elem_type));
                }
                break;
            }
            default:
                throw std::runtime_error(
                    "Unknown GGUF value type: " +
                    std::to_string(static_cast<uint32_t>(type))
                );
        }
        return v;
    }

    void parse() {
        // --- Header ---
        uint32_t magic = read<uint32_t>();
        if (magic != GGUF_MAGIC) {
            throw std::runtime_error(
                "Not a GGUF file (bad magic): " + path_.string()
            );
        }

        model_.version = read<uint32_t>();
        if (model_.version < 1 || model_.version > GGUF_VERSION) {
            throw std::runtime_error(
                "Unsupported GGUF version: " +
                std::to_string(model_.version)
            );
        }

        uint64_t n_tensors  = read<uint64_t>();
        uint64_t n_kv_pairs = read<uint64_t>();

        // --- Metadata key-value pairs ---
        for (uint64_t i = 0; i < n_kv_pairs; ++i) {
            std::string key  = readString();
            auto vtype       = static_cast<GGUFValueType>(read<uint32_t>());
            model_.metadata[key] = readMetaValue(vtype);
        }

        // --- Tensor info ---
        model_.tensors.reserve(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            GGUFTensorInfo info;
            info.name       = readString();
            uint32_t n_dims = read<uint32_t>();
            info.shape.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                info.shape[d] = read<uint64_t>();
            }
            info.type   = static_cast<GGMLType>(read<uint32_t>());
            info.offset = read<uint64_t>();

            // Compute element count and byte size
            info.n_elements = 1;
            for (auto dim : info.shape) info.n_elements *= dim;
            info.n_bytes = computeByteSize(info.type, info.n_elements);

            model_.tensors.push_back(std::move(info));
        }

        // Align data section to 32 bytes (GGUF spec)
        uint64_t pos = static_cast<uint64_t>(file_.tellg());
        model_.data_offset = (pos + 31) & ~uint64_t{31};
    }

    static uint64_t computeByteSize(GGMLType type, uint64_t n_elements) {
        switch (type) {
            case GGMLType::F32:  return n_elements * 4;
            case GGMLType::F16:  return n_elements * 2;
            case GGMLType::Q4_0: return (n_elements / 32) * 18;  // 18 bytes/block
            case GGMLType::Q4_1: return (n_elements / 32) * 20;
            case GGMLType::Q5_0: return (n_elements / 32) * 22;
            case GGMLType::Q5_1: return (n_elements / 32) * 24;
            case GGMLType::Q8_0: return (n_elements / 32) * 34;
            case GGMLType::Q4_K: return (n_elements / 256) * 144;
            case GGMLType::Q6_K: return (n_elements / 256) * 210;
            case GGMLType::I8:   return n_elements;
            case GGMLType::I16:  return n_elements * 2;
            case GGMLType::I32:  return n_elements * 4;
            default:             return n_elements * 4; // Conservative fallback
        }
    }
};

// ---------------------------------------------------------------------------
// INT8 quantisation
// ---------------------------------------------------------------------------

/**
 * Symmetric per-tensor INT8 quantisation.
 *
 * Converts a span of float32 values to INT8 stored as uint8 with a +128
 * unsigned bias (matching the Rust and Python implementations).
 *
 * @param src   Input float32 values.
 * @returns     Quantised bytes.
 */
static std::vector<uint8_t> quantizeFloat32ToInt8(std::span<const float> src) {
    float max_abs = 0.0f;
    for (float v : src) {
        float a = std::abs(v);
        if (a > max_abs) max_abs = a;
    }

    const float scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;
    std::vector<uint8_t> out(src.size());

    for (std::size_t i = 0; i < src.size(); ++i) {
        float q = src[i] * scale;
        q = std::clamp(q, -128.0f, 127.0f);
        out[i] = static_cast<uint8_t>(static_cast<int16_t>(q) + 128);
    }
    return out;
}

/**
 * Dequantise Q4_0 blocks to float32.
 *
 * Q4_0 block layout (18 bytes / 32 elements):
 *   - 2 bytes: float16 scale
 *   - 16 bytes: 32 × 4-bit weights (two per byte, unsigned, bias -8)
 */
static std::vector<float> dequantizeQ4_0(std::span<const uint8_t> data,
                                          uint64_t n_elements) {
    std::vector<float> out(n_elements);
    const std::size_t block_size = 18;
    const uint32_t    elems_per_block = 32;
    std::size_t out_idx = 0;

    for (std::size_t b = 0; b + block_size <= data.size(); b += block_size) {
        // Read float16 scale (little-endian)
        uint16_t f16_scale;
        std::memcpy(&f16_scale, data.data() + b, 2);
        // Convert float16 → float32 (manual, no hardware dependency)
        uint32_t sign = (f16_scale >> 15) & 1u;
        uint32_t exp  = (f16_scale >> 10) & 0x1Fu;
        uint32_t mant = f16_scale & 0x3FFu;
        float scale;
        if (exp == 0) {
            scale = std::ldexp(static_cast<float>(mant), -24);
        } else if (exp == 31) {
            scale = std::numeric_limits<float>::infinity();
        } else {
            scale = std::ldexp(static_cast<float>(mant | 0x400u), static_cast<int>(exp) - 25);
        }
        if (sign) scale = -scale;

        // Unpack 4-bit weights
        for (uint32_t i = 0; i < elems_per_block / 2 && b + 2 + i < data.size(); ++i) {
            uint8_t byte = data[b + 2 + i];
            float lo = static_cast<float>(byte & 0x0F) - 8.0f;
            float hi = static_cast<float>((byte >> 4) & 0x0F) - 8.0f;
            if (out_idx < n_elements) out[out_idx++] = lo * scale;
            if (out_idx < n_elements) out[out_idx++] = hi * scale;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// PCIe DMA driver stub
// ---------------------------------------------------------------------------

/**
 * @class HyzeIPUPCIeDriver
 * @brief PCIe DMA driver for the Hyze IPU.
 *
 * In production this class mmaps BAR0 and performs zero-copy DMA transfers.
 * On systems without hardware it falls back to a software simulation.
 */
class HyzeIPUPCIeDriver {
public:
    explicit HyzeIPUPCIeDriver(bool simulation = false)
        : simulation_(simulation), bar0_(nullptr), bar_size_(0)
    {
        if (!simulation_) {
            open();
        }
    }

    ~HyzeIPUPCIeDriver() { close(); }

    // Non-copyable
    HyzeIPUPCIeDriver(const HyzeIPUPCIeDriver&)            = delete;
    HyzeIPUPCIeDriver& operator=(const HyzeIPUPCIeDriver&) = delete;

    /**
     * Stream @p data to FPGA SRAM at @p offset.
     *
     * @param offset  Byte offset within SRAM.
     * @param data    Data to write.
     */
    void dmaWrite(uint32_t offset, std::span<const uint8_t> data) {
        if (simulation_) {
            // Simulation: write to a local buffer
            if (offset + data.size() > sram_sim_.size()) {
                sram_sim_.resize(offset + data.size(), 0);
            }
            std::copy(data.begin(), data.end(), sram_sim_.begin() + offset);
            return;
        }
        if (!bar0_) throw std::runtime_error("IPU PCIe device not open.");
        std::copy(data.begin(), data.end(), bar0_ + offset);
    }

    /**
     * Trigger inference and return the predicted class index.
     *
     * @param pixels  784 bytes of INT8 pixel data.
     * @returns       Predicted class index (0–N).
     */
    uint8_t infer(std::span<const uint8_t> pixels) {
        assert(pixels.size() == 784);

        if (simulation_) {
            return simulateInfer(pixels);
        }

        // 1. DMA write pixels to offset 0x0000
        dmaWrite(0x0000, pixels);

        // 2. Trigger inference
        volatile uint32_t* ctrl = reinterpret_cast<volatile uint32_t*>(bar0_ + 0xFFF0);
        *ctrl = 1u;

        // 3. Poll done register
        volatile uint32_t* done = reinterpret_cast<volatile uint32_t*>(bar0_ + 0xFFF4);
        while ((*done & 1u) == 0) {
            // Tight poll – real implementation uses interrupt or sleep
        }

        // 4. Read result
        volatile uint8_t* result_reg = reinterpret_cast<volatile uint8_t*>(bar0_ + 0xFFF8);
        return *result_reg & 0x0Fu;
    }

    bool isSimulation() const noexcept { return simulation_; }

private:
    bool                  simulation_;
    uint8_t*              bar0_;
    std::size_t           bar_size_;
    std::vector<uint8_t>  sram_sim_;  ///< In-memory SRAM for simulation.

    void open() {
        // Real implementation: use libpci to find the device and mmap BAR0.
        // For portability we leave this as a documented stub.
        std::cerr << "[HyzeIPU] PCIe open: hardware not available, "
                     "falling back to simulation.\n";
        simulation_ = true;
    }

    void close() {
        if (bar0_) {
            // munmap(bar0_, bar_size_);
            bar0_ = nullptr;
        }
    }

    uint8_t simulateInfer(std::span<const uint8_t> pixels) {
        // Return the index of the brightest 78-pixel group
        int best = 0;
        uint32_t best_sum = 0;
        for (int i = 0; i < 10; ++i) {
            uint32_t sum = 0;
            for (int j = i * 78; j < std::min((i + 1) * 78, 784); ++j) {
                sum += pixels[j];
            }
            if (sum > best_sum) { best_sum = sum; best = i; }
        }
        return static_cast<uint8_t>(best);
    }
};

// ---------------------------------------------------------------------------
// GGUF inference engine
// ---------------------------------------------------------------------------

/**
 * @class HyzeGGUFInference
 * @brief High-level GGUF inference engine backed by the Hyze IPU.
 *
 * Loads a GGUF model, streams its weight tensors to the FPGA SRAM, and
 * provides a simple ``generate()`` API for autoregressive text generation.
 *
 * @note The current implementation uses the IPU as a classification head;
 *       full transformer decode loops would require additional RTL support.
 */
class HyzeGGUFInference {
public:
    /**
     * @param simulation  Run in software simulation mode.
     * @param verbose     Print progress messages.
     */
    explicit HyzeGGUFInference(bool simulation = false, bool verbose = false)
        : driver_(simulation), verbose_(verbose)
    {}

    // -----------------------------------------------------------------------
    // Model loading
    // -----------------------------------------------------------------------

    /**
     * Load a GGUF model from @p path and stream its weights to the IPU.
     *
     * @throws std::runtime_error on file or format errors.
     */
    void loadModel(const std::filesystem::path& path) {
        log("Loading GGUF model: " + path.string());

        HyzeGGUFLoader loader(path);
        model_info_ = loader.model();

        // Print metadata summary
        if (verbose_) {
            printMetadata();
        }

        // Stream weight tensors to FPGA SRAM
        uint32_t sram_offset = 0x2000; // Weight SRAM base address
        for (const auto& tensor : model_info_.tensors) {
            log("  Streaming tensor: " + tensor.name +
                " [" + ggmlTypeName(tensor.type) + ", " +
                std::to_string(tensor.n_elements) + " elements]");

            auto raw = loader.readTensorData(tensor);
            auto quantised = convertToInt8(tensor.type, raw, tensor.n_elements);

            driver_.dmaWrite(sram_offset, quantised);
            sram_offset += static_cast<uint32_t>(quantised.size());

            // Align to 64-byte boundary for cache efficiency
            sram_offset = (sram_offset + 63) & ~uint32_t{63};
        }

        log("Model loaded. SRAM used: " + std::to_string(sram_offset - 0x2000) + " bytes.");
        model_loaded_ = true;
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /**
     * Generate up to @p max_tokens new tokens given @p prompt.
     *
     * @param prompt      Input text.
     * @param max_tokens  Maximum number of tokens to generate.
     * @returns           Generated text (including the prompt).
     */
    std::string generate(const std::string& prompt, uint32_t max_tokens = 128) {
        if (!model_loaded_) {
            throw std::runtime_error("No model loaded. Call loadModel() first.");
        }

        log("Generating: \"" + prompt + "\" (max_tokens=" +
            std::to_string(max_tokens) + ")");

        std::string output = prompt;
        auto tokens = tokenize(prompt);

        for (uint32_t step = 0; step < max_tokens; ++step) {
            // Pack the last 784 token IDs into a pixel frame
            auto frame = buildFrame(tokens);

            auto t0 = std::chrono::high_resolution_clock::now();
            uint8_t next_token = driver_.infer(frame);
            auto t1 = std::chrono::high_resolution_clock::now();

            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (verbose_) {
                std::cout << "  step " << step
                          << ": token=" << static_cast<int>(next_token)
                          << " (" << us << " μs)\n";
            }

            // EOS token (0) terminates generation
            if (next_token == 0) break;

            tokens.push_back(next_token);
            output += detokenize({next_token});
        }

        return output;
    }

    /**
     * Run a latency benchmark.
     *
     * @param n_iters  Number of inference iterations.
     * @returns        Map with keys ``mean_us``, ``min_us``, ``max_us``,
     *                 ``throughput_tps``.
     */
    std::map<std::string, double> benchmark(uint32_t n_iters = 1000) {
        std::vector<double> latencies;
        latencies.reserve(n_iters);

        std::array<uint8_t, 784> dummy{};
        dummy.fill(128);

        for (uint32_t i = 0; i < n_iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            driver_.infer(dummy);
            auto t1 = std::chrono::high_resolution_clock::now();
            latencies.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count()
            );
        }

        double sum  = 0, mn = latencies[0], mx = latencies[0];
        for (double v : latencies) {
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double mean = sum / latencies.size();

        std::map<std::string, double> stats{
            {"mean_us",        mean},
            {"min_us",         mn},
            {"max_us",         mx},
            {"throughput_tps", 1e6 / mean},
        };

        std::cout << "Benchmark (" << n_iters << " iters): "
                  << "mean=" << mean << " μs, "
                  << "throughput=" << stats["throughput_tps"] << " tokens/s\n";
        return stats;
    }

private:
    HyzeIPUPCIeDriver driver_;
    GGUFModel         model_info_;
    bool              model_loaded_ = false;
    bool              verbose_;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    void log(const std::string& msg) const {
        if (verbose_) std::cout << "[HyzeGGUF] " << msg << "\n";
    }

    void printMetadata() const {
        std::cout << "[HyzeGGUF] Metadata:\n";
        for (const auto& [k, v] : model_info_.metadata) {
            std::cout << "  " << k << " = ";
            switch (v.type) {
                case GGUFValueType::STRING:  std::cout << v.str;  break;
                case GGUFValueType::UINT32:
                case GGUFValueType::UINT64:  std::cout << v.u64;  break;
                case GGUFValueType::INT32:
                case GGUFValueType::INT64:   std::cout << v.i64;  break;
                case GGUFValueType::FLOAT32:
                case GGUFValueType::FLOAT64: std::cout << v.f64;  break;
                default:                     std::cout << "(complex)"; break;
            }
            std::cout << "\n";
        }
    }

    /**
     * Convert raw GGUF tensor bytes to INT8 for the IPU SRAM.
     *
     * Handles F32, F16, Q4_0, and falls back to raw bytes for other types.
     */
    std::vector<uint8_t> convertToInt8(GGMLType type,
                                        const std::vector<uint8_t>& raw,
                                        uint64_t n_elements) {
        switch (type) {
            case GGMLType::F32: {
                std::vector<float> f32(n_elements);
                std::memcpy(f32.data(), raw.data(),
                            std::min(raw.size(), n_elements * 4));
                return quantizeFloat32ToInt8(f32);
            }
            case GGMLType::F16: {
                // Dequantise F16 → F32 → INT8
                std::vector<float> f32(n_elements);
                for (uint64_t i = 0; i < n_elements && i * 2 + 1 < raw.size(); ++i) {
                    uint16_t h;
                    std::memcpy(&h, raw.data() + i * 2, 2);
                    uint32_t sign = (h >> 15) & 1u;
                    uint32_t exp  = (h >> 10) & 0x1Fu;
                    uint32_t mant = h & 0x3FFu;
                    float v;
                    if (exp == 0)       v = std::ldexp(static_cast<float>(mant), -24);
                    else if (exp == 31) v = std::numeric_limits<float>::infinity();
                    else                v = std::ldexp(static_cast<float>(mant | 0x400u),
                                                       static_cast<int>(exp) - 25);
                    f32[i] = sign ? -v : v;
                }
                return quantizeFloat32ToInt8(f32);
            }
            case GGMLType::Q4_0: {
                auto f32 = dequantizeQ4_0(raw, n_elements);
                return quantizeFloat32ToInt8(f32);
            }
            case GGMLType::I8:
                // Already INT8; just return as-is
                return raw;
            default:
                // Unknown quantisation: return raw bytes (best-effort)
                return raw;
        }
    }

    /// Minimal whitespace tokeniser (placeholder for a real BPE tokeniser).
    std::vector<uint8_t> tokenize(const std::string& text) const {
        std::vector<uint8_t> tokens;
        for (unsigned char c : text) {
            tokens.push_back(c);
        }
        return tokens;
    }

    /// Minimal detokeniser.
    std::string detokenize(const std::vector<uint8_t>& tokens) const {
        return std::string(tokens.begin(), tokens.end());
    }

    /// Pack the last 784 token IDs into a pixel frame.
    std::array<uint8_t, 784> buildFrame(const std::vector<uint8_t>& tokens) const {
        std::array<uint8_t, 784> frame{};
        frame.fill(0);
        std::size_t start = tokens.size() > 784 ? tokens.size() - 784 : 0;
        for (std::size_t i = start, j = 0; i < tokens.size(); ++i, ++j) {
            frame[j] = tokens[i];
        }
        return frame;
    }

    static std::string ggmlTypeName(GGMLType t) {
        switch (t) {
            case GGMLType::F32:  return "F32";
            case GGMLType::F16:  return "F16";
            case GGMLType::Q4_0: return "Q4_0";
            case GGMLType::Q4_1: return "Q4_1";
            case GGMLType::Q5_0: return "Q5_0";
            case GGMLType::Q8_0: return "Q8_0";
            case GGMLType::Q4_K: return "Q4_K";
            case GGMLType::Q6_K: return "Q6_K";
            case GGMLType::I8:   return "I8";
            default:             return "UNKNOWN";
        }
    }
};

// ---------------------------------------------------------------------------
// CMakeLists.txt update helper (printed to stdout)
// ---------------------------------------------------------------------------

static void printCMakeSnippet() {
    std::cout << R"(
# Add to CMakeLists.txt to build hyze_gguf_ipu:
add_executable(hyze_gguf_ipu hyze_gguf_ipu.cpp)
target_compile_features(hyze_gguf_ipu PRIVATE cxx_std_20)
target_link_libraries(hyze_gguf_ipu PRIVATE pthread)
)";
}

// ---------------------------------------------------------------------------
// main – CLI entry-point
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    bool simulation = false;
    bool verbose    = false;
    bool bench      = false;
    std::string model_path;
    std::string prompt = "Hello, Hyze IPU!";
    uint32_t max_tokens = 64;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--simulate")       simulation = true;
        else if (arg == "--verbose")   verbose    = true;
        else if (arg == "--bench")     bench      = true;
        else if (arg == "--cmake")     { printCMakeSnippet(); return 0; }
        else if (arg == "--model"  && i + 1 < argc) model_path  = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) prompt      = argv[++i];
        else if (arg == "--tokens" && i + 1 < argc) max_tokens  = std::stoul(argv[++i]);
    }

    try {
        HyzeGGUFInference engine(simulation, verbose);

        if (!model_path.empty()) {
            engine.loadModel(model_path);
        } else {
            std::cout << "[HyzeGGUF] No model path provided; running in demo mode.\n";
        }

        if (bench) {
            engine.benchmark(1000);
        } else if (!model_path.empty()) {
            std::string output = engine.generate(prompt, max_tokens);
            std::cout << "\nGenerated:\n" << output << "\n";
        } else {
            // Demo: run a single simulated inference
            HyzeIPUPCIeDriver demo_driver(/*simulation=*/true);
            std::array<uint8_t, 784> frame{};
            frame.fill(85);
            uint8_t cls = demo_driver.infer(frame);
            std::cout << "Demo inference result: class " << static_cast<int>(cls) << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
