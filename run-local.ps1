#Requires -Version 5.1
<#
.SYNOPSIS
    Launches the extra-code GGUF on llama-server (OpenAI-compatible API).

.DESCRIPTION
    Picks the Vulkan build when it is present and the device is visible, falls
    back to the CPU build otherwise. Quantization defaults to Q4_K_M; pass -Quant q8_0
    for the safer quality trade, or -Quant f16 for the unquantized file.

.EXAMPLE
    .\run-local.ps1
    .\run-local.ps1 -Port 8080 -Ctx 2048
    .\run-local.ps1 -Quant q8_0 -Cpu
#>
param(
    [ValidateSet("q4_k_m", "q8_0", "f16")]
    [string]$Quant = "q4_k_m",
    [int]$Port = 8080,
    [int]$Ctx = 1024,
    [int]$Parallel = 2,
    [switch]$Cpu,
    [switch]$NoFlashAttention
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$model = Join-Path $PSScriptRoot "export\gguf\extra-code-$Quant.gguf"
if (-not (Test-Path -LiteralPath $model)) {
    throw "model not found: $model -- build it first, see RUN.md section 3"
}

$vulkanDir = Join-Path $PSScriptRoot "tools\llama.cpp-vulkan"
$cpuDir    = Join-Path $PSScriptRoot "tools\llama.cpp"

$useVulkan = (-not $Cpu) -and (Test-Path -LiteralPath (Join-Path $vulkanDir "llama-server.exe"))
$serverDir = if ($useVulkan) { $vulkanDir } else { $cpuDir }
$server    = Join-Path $serverDir "llama-server.exe"
if (-not (Test-Path -LiteralPath $server)) { throw "llama-server.exe not found in $serverDir" }

$serverArgs = @(
    "-m", $model,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "-c", "$Ctx",
    "--parallel", "$Parallel",
    "--cont-batching",
    "--no-warmup"
)
if ($useVulkan) { $serverArgs += @("-ngl", "99") }
if (-not $NoFlashAttention) { $serverArgs += @("-fa", "on") }

Write-Host "model   : $model"
Write-Host "backend : $(if ($useVulkan) { 'vulkan' } else { 'cpu' })"
Write-Host "api     : http://127.0.0.1:$Port/v1"
Write-Host "ui      : http://127.0.0.1:$Port"
Write-Host ""

& $server @serverArgs
