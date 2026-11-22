#Requires -Version 5.1
<#
.SYNOPSIS
    Stops everything start.ps1 launched: the extra-code API (:8000) and
    llama-server (:8080).

.DESCRIPTION
    Kills the process owning each listening port, then sweeps any surviving
    llama-server by name. Safe to run when nothing is up.
#>
param(
    [int]$ModelPort = 8080,
    [int]$ApiPort = 8000
)

foreach ($p in @($ApiPort, $ModelPort)) {
    $owners = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($owner in $owners) {
        try {
            $proc = Get-Process -Id $owner -ErrorAction Stop
            Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
            Write-Host "stopped :$p  ($($proc.ProcessName), pid $owner)"
        } catch { }
    }
}
Get-Process llama-server -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    Write-Host "stopped llama-server (pid $($_.Id))"
}

Start-Sleep -Seconds 1
$left = @()
foreach ($p in @($ApiPort, $ModelPort)) {
    $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if ($conn) { $left += ":$p" }
}
if ($left) { Write-Host "still listening: $($left -join ', ')" }
else       { Write-Host "all stopped." }
