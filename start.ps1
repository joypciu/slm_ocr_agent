# Usage:  .\start.ps1                   -> full profile: Qwen3-0.6B text reasoner + SmolVLM-256M vision specialist + OCR (~691 MB of weights)
#         .\start.ps1 -Profile docs     -> text reasoner + OCR only (~411 MB, no picture understanding)
#         .\start.ps1 -Profile vision   -> one SmolVLM model for everything (~313 MB, weakest on documents)
param([string]$Profile = "full")
Set-Location $PSScriptRoot
$exe = ".\runtime\llama\llama-server.exe"
$qwen = @("-m","models\Qwen3-0.6B-Q4_K_M.gguf","--chat-template-kwargs",'{\"enable_thinking\":false}',"-np","2","-t","6","-c","4096","--port","8081")
$smol = @("-m","models\vision\SmolVLM-256M-Instruct-Q8_0.gguf","--mmproj","models\vision\mmproj-SmolVLM-256M-Instruct-Q8_0.gguf","-np","2","-t","6","-c","8192")
if ($Profile -eq "vision") {
  $env:OMNI_VISION = "1"
  Start-Process -WindowStyle Hidden -FilePath $exe -ArgumentList ($smol + @("--port","8081"))
} elseif ($Profile -eq "docs") {
  $env:OMNI_VISION = "0"
  Start-Process -WindowStyle Hidden -FilePath $exe -ArgumentList $qwen
} else {
  $env:OMNI_VISION = "0"
  $env:OMNI_VISION_URL = "http://127.0.0.1:8082"
  Start-Process -WindowStyle Hidden -FilePath $exe -ArgumentList $qwen
  Start-Process -WindowStyle Hidden -FilePath $exe -ArgumentList ($smol + @("--port","8082"))
}
Start-Sleep 8
.\.venv\Scripts\python.exe -m uvicorn server:app --port 8090
