@RestController
@RequestMapping("/secure")
public class HyzeSecureDpController {
    @PreAuthorize("hasRole('HIPAA')")
    @PostMapping("/dp_inference_v2")
    public SecureResult dpInferV2(@Valid @RequestBody EncryptedBatchV2 batch) {
        return confidentialIpu.dpInference(batch);
    }
}
