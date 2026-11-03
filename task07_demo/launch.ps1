# Arguments are forwarded directly to Python without building a shell command.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$parent = Split-Path -Parent $repo
$candidates = @($env:TASK07_PYTHON, (Join-Path $repo '.venv\Scripts\python.exe'), (Join-Path $parent 'nnUNet\.venv\Scripts\python.exe'))
$pythonExe = $null
foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
        $ErrorActionPreference = 'Continue'
        & $candidate (Join-Path $PSScriptRoot 'check_env.py') 2>$null
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE -eq 0) { $pythonExe = $candidate; break }
    }
}
if (-not $pythonExe) { throw 'No CUDA environment found. Read TASK07_DEMO.md or set TASK07_PYTHON to a GPU Python executable.' }
$entry = if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'run_nnunet.py')) { 'run_nnunet.py' } else { 'run_unet.py' }
Write-Host "Python: $pythonExe"
Write-Host 'Default: 90 minutes TRAINING, then full-volume prediction/evaluation. Do not run both models at once.'
& $pythonExe (Join-Path $PSScriptRoot $entry) @args
if ($LASTEXITCODE -ne 0) { throw "Task07 exited with code $LASTEXITCODE. Keep console.log and report_evidence.zip." }
