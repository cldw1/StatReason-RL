# Preserves the existing CUDA PyTorch wheel; run in your own terminal.
[CmdletBinding()]
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Existing .venv was not found. No environment was created or replaced.'
}
& $pythonPath -c "import torch; assert torch.__version__ == '2.8.0+cu128', torch.__version__"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch wheel does not match constraints-torch.txt.' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$setupDirectory = Join-Path $projectRoot "outputs\setup\install_$stamp"
New-Item -ItemType Directory -Path $setupDirectory -Force | Out-Null
& $pythonPath -m pip freeze | Set-Content -LiteralPath (Join-Path $setupDirectory 'pip_freeze.before.txt') -Encoding utf8
$pipArguments = @('-m', 'pip', 'install', '--disable-pip-version-check', '--timeout', '30', '--retries', '1', '-r', (Join-Path $projectRoot 'requirements.txt'), '--report', (Join-Path $setupDirectory 'pip_install_report.json'))
if ($DryRun) { $pipArguments += '--dry-run' }
& $pythonPath @pipArguments 2>&1 | Tee-Object -FilePath (Join-Path $setupDirectory 'install.log')
if ($LASTEXITCODE -ne 0) { throw 'Dependency resolution/install failed. See install.log; no compatibility success is claimed.' }
if ($DryRun) { return }
& $pythonPath -m pip check 2>&1 | Tee-Object -FilePath (Join-Path $setupDirectory 'pip_check.txt')
if ($LASTEXITCODE -ne 0) { throw 'pip check failed.' }
& $pythonPath (Join-Path $PSScriptRoot '01_check_env.py') --require-training-stack
if ($LASTEXITCODE -ne 0) { throw 'GPU or training API audit failed. Environment is not ready.' }
& $pythonPath -m pip freeze | Set-Content -LiteralPath (Join-Path $setupDirectory 'pip_freeze.after.txt') -Encoding utf8
Write-Output "Installation and import checks passed. Records: $setupDirectory"

