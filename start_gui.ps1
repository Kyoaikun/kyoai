$ErrorActionPreference = 'Stop'
$env:LLAMA_MODEL = 'qwen2.5-14b-q5km'
$env:LLAMA_BASE_URL = 'http://127.0.0.1:8080/v1'
$env:LLAMA_AUTO_SAVE_PATTERN = '1'
try {
    $port = 8080
    $listening = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $listening) {
        Start-Process powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File F:\kimi\agent\start_server.ps1
        Start-Sleep -Seconds 2
    }
} catch {
    Start-Process powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File F:\kimi\agent\start_server.ps1
    Start-Sleep -Seconds 2
}
python F:\kimi\agent\gui_client.py
