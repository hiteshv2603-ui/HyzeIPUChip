@echo off
REM Hyze MegaChip Security CLI - Batch + Curl
REM Deploys confidential inference, DP, prompt guard via curl
REM Windows CMD native - no PowerShell!

REM === CONFIG ===
set API_URL=http://localhost:8080/secure
set HIPAA_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
set PCI_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

REM === 1. Prompt Injection Scan ===
echo Scanning for prompt injection...
curl -X POST "%API_URL%/prompt_guard_v3" ^
  -H "Authorization: Bearer %HIPAA_TOKEN%" ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\": \"%1\"}" ^
  --silent | findstr "BLOCKED" >nul && (
    echo [!] PROMPT BLOCKED: Jailbreak detected!
    exit /b 1
  ) || echo [+] Prompt safe

REM === 2. Confidential DP Inference ===
echo Running hardware DP inference...
curl -X POST "%API_URL%/dp_inference_v3" ^
  -H "Authorization: Bearer %PCI_TOKEN%" ^
  -H "Content-Type: application/json" ^
  -d "{\"pixels\": [85,128,42,0,255], \"epsilon\": 1.0}" ^
  --compressed ^
  -w "Latency: %%{time_total}s\n" ^
  | findstr "class_id" > result.json

REM === 3. Supply Chain Verification ===
echo Verifying SBOM + RoT...
curl -X GET "%API_URL%/supply_chain_check_v2" ^
  -H "Authorization: Bearer %HIPAA_TOKEN%" ^
  --silent | findstr "trusted" || (
    echo [!] SUPPLY CHAIN COMPROMISE!
    exit /b 2
  )

REM === 4. ZK-Proof Validation ===
echo Validating zero-knowledge proof...
curl -X POST "%API_URL%/zk_verify_v2" ^
  -H "Authorization: Bearer %PCI_TOKEN%" ^
  -d @zk_proof.json ^
  | findstr "verified:true" || (
    echo [!] ZK-PROOF FAILED!
    exit /b 3
  )

REM === 5. Multi-Party Computation ===
echo Running MPC across 3 IPUs...
curl -X POST "%API_URL%/mpc_inference_v3" ^
  -H "Authorization: Bearer %HIPAA_TOKEN%" ^
  -d "{\"encrypted_shares\": [%1,%2,%3]}" ^
  | findstr "reconstructed" > mpc_result.json

echo.
echo [+] Hyze MegaChip Security Pipeline COMPLETE
echo Results: result.json, mpc_result.json
type result.json
pause
