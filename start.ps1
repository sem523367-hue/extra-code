#Requires -Version 5.1
<#
.SYNOPSIS
    Starts the whole extra-code stack with one command:
    the LLM inside llama-server (engine, :8080) and the extra-code
    FastAPI server on top (:8000, auth + queue + web search).

.DESCRIPTION
    Both processes are detached: closing this window does NOT stop them.
    Shut everything down with stop.ps1 / stop.bat.
    Re-running is safe: parts that are already healthy are reused.

.EXAMPLE
    .\start.ps1
    .\start.ps1 -ApiPort 8001 -Ctx 2048
#>
param(
    [int]$ModelPort = 8080,
    [int]$ApiPort = 8000,
    # 0 = pick from the model's KV cost. The 4B has 8 KV heads against
    # 2 on the 3B, so its cache is 4x heavier per token: 12K context
    # costs the same VRAM as 20K on the 3B.
    [int]$Ctx = 0,
    [string]$ModelFile = ""
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# --- resolve binaries ----------------------------------------------------
$py = "C:\Users\Sem52\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path -LiteralPath $py)) { $py = (Get-Command python).Source }

# Prefer the biggest brain that exists on disk: 3B Instruct general model
# beats the 1.5B coder at reasoning, still fits the 4GB RX 580 fully.
# -ModelFile overrides for benchmarking (see docs/WHAT-WE-ADDED.md).
if ($ModelFile) {
    $model = Join-Path $PSScriptRoot "models\$ModelFile"
    if (-not (Test-Path -LiteralPath $model)) { throw "model not found: $model" }
} else {
    # Newest model on disk wins. The 4B-2507 instruct is the 2025
    # refresh: same 36 layers, far better instruction following and code
    # than the 2024 3B, and 2.50 GB of weights still fit the 4 GB card.
    # The 3B stays as the rollback: if the 4B ever fails to load, start
    # falls back instead of leaving the stack dead.
    $model4b = Join-Path $PSScriptRoot "models\extra-code-4b-2507-q4_k_m.gguf"
    $model3b = Join-Path $PSScriptRoot "models\extra-code-3b-q4_k_m.gguf"
    if     (Test-Path -LiteralPath $model4b) { $model = $model4b }
    elseif (Test-Path -LiteralPath $model3b) { $model = $model3b }
    else { throw "no model found under models\" }
}

# Context sizing follows the model's KV footprint, measured on this card
# (RX 580 4 GB, ~3.84 GB free after the desktop):
#   3B (2 KV heads): 1.96 GB weights + 19.5 KB/token -> 20480
#   4B (8 KV heads): 2.50 GB weights + 78   KB/token -> 10240
# Measured ceilings for the 4B Q4_K_M, q8_0 KV, -ub 256:
#   10240 -> 3.48 GB, loads;  11264 -> OOM;  12288 -> OOM.
# q8_0 KV both, so the totals land at ~2.4 GB and ~3.5 GB of the 4 GB card.
if ($Ctx -le 0) {
    if ((Split-Path $model -Leaf) -match "extra-code-4b") { $Ctx = 10240 }
    elseif ((Split-Path $model -Leaf) -match "extra-code-8b") { $Ctx = 8192 }
    else { $Ctx = 20480 }
}

$vulkanExe = Join-Path $PSScriptRoot "tools\llama.cpp-vulkan\llama-server.exe"
$cpuExe    = Join-Path $PSScriptRoot "tools\llama.cpp\llama-server.exe"
if     (Test-Path -LiteralPath $vulkanExe) { $llama = $vulkanExe; $gpu = $true }
elseif (Test-Path -LiteralPath $cpuExe)    { $llama = $cpuExe;    $gpu = $false }
else { throw "llama-server.exe not found under tools\" }

# Identity lives in the chat template, not only in the API layer: the
# llama.cpp web UI on :8080 talks to the engine directly and never passes
# through the FastAPI guard. Without this template the engine introduces
# itself under the base model's real name and vendor.
$templateFile = Join-Path $PSScriptRoot "tools\extra-code-chat-template.jinja"
if (-not (Test-Path -LiteralPath $templateFile)) {
    throw "chat template not found: $templateFile"
}

function Test-Health([int]$port) {
    try {
        (Invoke-WebRequest "http://127.0.0.1:$port/health" -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200
    } catch { $false }
}

function Wait-Health([int]$port, [int]$timeoutS) {
    $deadline = (Get-Date).AddSeconds($timeoutS)
    while ((Get-Date) -lt $deadline) {
        if (Test-Health $port) { return $true }
        Start-Sleep -Seconds 2
        Write-Host -NoNewline "."
    }
    return $false
}

# --- engine: llama-server (:8080) ------------------------------------------
if (Test-Health $ModelPort) {
    Write-Host "engine  : http://127.0.0.1:$ModelPort already up - reusing"
} else {
    $srvArgs = @(
        "-m", "`"$model`"",
        "--host", "127.0.0.1",
        "--port", "$ModelPort",
        "-c", "$Ctx",
        "--parallel", "1",
        "--cont-batching",
        "--no-warmup",
        "--chat-template-file", "`"$templateFile`"",
        "--alias", "`"Extra Code v1`""
    )
    if ($gpu) {
        # q8_0 KV halves the cache: on the 3B that is 2.0 GB weights + 0.6 GB
        # KV, on the 4B 2.5 GB + 0.8 GB, both inside the 4 GB RX 580 where
        # fp16 KV OOMs mid-generation.
        $srvArgs += @(
            "-ngl", "99",
            "-fa", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            # Prefix reuse: every turn repeats the same system prompt, and
            # cache-reuse shifts the matching KV chunk instead of
            # re-prefilling it. Off by default upstream; on this box it
            # removes ~3.5K tokens of prefill from every turn after the
            # first.
            "--cache-reuse", "256",
            # Host-side prompt cache (MiB). 8 GB of 32 GB RAM: keeps the
            # whole conversation prefix hot across requests.
            "--cache-ram", "8192",
            # Physical batch 256 instead of 512. The compute buffer scales
            # with ubatch and on a 4 GB card that buffer is what decides
            # between 10K and 12K of context: 12288 + ub 512 OOMs, 10240 +
            # ub 256 loads at 3.48 GB.
            "-ub", "256",
            "-b", "512"
        )
    }

    $cmdLine = "`"$llama`" " + ($srvArgs -join " ")
    # llama-server's own output used to go to a console that is gone by the
    # time it crashes, leaving a dead engine and no reason. Route both
    # streams to files next to the API logs.
    $engineRunner = Join-Path $PSScriptRoot "run-engine.cmd"
    $engineContent = @(
        '@echo off',
        'rem Generated by start.ps1 - do not edit by hand.',
        "cd /d `"$PSScriptRoot`"",
        "$cmdLine >> engine-out.log 2>> engine-err.log"
    ) -join "`r`n"
    Set-Content -LiteralPath $engineRunner -Value $engineContent -Encoding ASCII
    $proc = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "cmd.exe /c `"$engineRunner`""
    }
    if ($proc.ReturnValue -ne 0) { throw "failed to start llama-server (code $($proc.ReturnValue))" }

    Write-Host -NoNewline "engine  : starting llama-server (pid $($proc.ProcessId))"
    if (-not (Wait-Health $ModelPort 180)) {
        throw "llama-server did not become healthy on :$ModelPort within 180s"
    }
    Write-Host " ok - http://127.0.0.1:$ModelPort"
}

# --- api: extra-code FastAPI (:8000) ---------------------------------------
if (Test-Health $ApiPort) {
    Write-Host "api     : http://127.0.0.1:$ApiPort already up - reusing"
} else {
    # The runner is regenerated on every start: ports and paths always match
    # this invocation. Env vars here override .env (load_dotenv never
    # overwrites an existing process variable).
    $runner = Join-Path $PSScriptRoot "run-server.cmd"
    $runnerContent = @(
        '@echo off',
        'rem Generated by start.ps1 - do not edit by hand.',
        "cd /d `"$PSScriptRoot`"",
        'set PYTHONIOENCODING=utf-8',
        'set BACKEND=openai',
        "set OPENAI_BASE_URL=http://127.0.0.1:$ModelPort/v1",
        'set MODEL_NAME=extra-code',
        "set N_CTX=$Ctx",
        'set API_KEY=local-dev-key',
        'set GENERATION_TIMEOUT_S=180',
        "`"$py`" -m uvicorn server.main:app --host 127.0.0.1 --port $ApiPort >> server-out.log 2>> server-err.log"
    ) -join "`r`n"
    Set-Content -LiteralPath $runner -Value $runnerContent -Encoding ASCII

    $apiCmd = 'cmd.exe /c "' + $runner + '"'
    $proc = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $apiCmd }
    if ($proc.ReturnValue -ne 0) { throw "failed to start the api server (code $($proc.ReturnValue))" }

    Write-Host -NoNewline "api     : starting extra-code server (pid $($proc.ProcessId))"
    if (-not (Wait-Health $ApiPort 90)) {
        throw "api did not become healthy on :$ApiPort within 90s - check server-err.log"
    }
    Write-Host " ok - http://127.0.0.1:$ApiPort"
}

Write-Host ""
Write-Host "extra-code is up and ready:"
Write-Host "  ask a question : .\ask.bat `"your question`"     (web on, sources included)"
Write-Host "  openai api     : http://127.0.0.1:$ApiPort/v1    (key: local-dev-key)"
Write-Host "  swagger ui     : http://127.0.0.1:$ApiPort/docs"
Write-Host "  engine direct  : http://127.0.0.1:$ModelPort     (no web/kb, raw model)"
Write-Host "  stop everything: .\stop.bat"
