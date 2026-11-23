# BRLO - environment setup for Windows (PyTorch 2.15, CUDA 13.x or CPU)
# Creates .venv inside this folder, installs PyTorch with the right compute
# platform, installs the package, verifies the GPU, runs the tests and a
# 1-minute POMO smoke training. Everything is logged to install_log.txt.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$log = Join-Path $root "install_log.txt"
Set-Content -Path $log -Value "BRLO setup log - $(Get-Date -Format s)" -Encoding UTF8

function Say([string]$s) { Write-Host $s; Add-Content -Path $log -Value $s -Encoding UTF8 }
function Run([string]$title, [scriptblock]$sb) {
    Say ""
    Say "=== $title ==="
    & $sb 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; Add-Content -Path $log -Value $line -Encoding UTF8 }
    $code = $LASTEXITCODE
    Say "[exit code $code]"
    return $code
}

# ---------------------------------------------------------------- 1. Python
function Get-PyVersion($exe, $pre) {
    try {
        $v = & $exe @pre -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { return "$v".Trim() }
    } catch {}
    return $null
}
$cands = @(
    @{exe = "py"; pre = @("-3.13")}, @{exe = "py"; pre = @("-3.12")}, @{exe = "py"; pre = @("-3.11")},
    @{exe = "py"; pre = @("-3.14")}, @{exe = "python"; pre = @()}, @{exe = "python3"; pre = @()}
)
$py = $null
foreach ($c in $cands) {
    $v = Get-PyVersion $c.exe $c.pre
    if ($v) {
        Say "Found Python $v via '$($c.exe) $($c.pre -join ' ')'"
        if ([version]$v -ge [version]"3.11" -and [version]$v -le [version]"3.14") { $py = $c; break }
    }
}
if (-not $py) {
    Say "ERROR: no Python 3.11-3.14 found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH') and run this again."
    exit 1
}

# ---------------------------------------------------------------- 2. venv
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Run "Create virtual environment .venv" { & $py.exe @($py.pre) -m venv .venv } | Out-Null
}
$vpy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Say "ERROR: could not create .venv"; exit 1 }
Run "Upgrade pip" { & $vpy -m pip install --upgrade pip } | Out-Null

# ---------------------------------------------------------------- 3. GPU
$index = "https://download.pytorch.org/whl/cpu"; $flavor = "CPU"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $out = (& nvidia-smi 2>&1) -join "`n"
    Say ""; Say "=== nvidia-smi ==="; Say $out
    if ($out -match "CUDA (?:UMD )?Version:\s*([0-9]+\.[0-9]+)") {   # driver >= 6xx prints "CUDA UMD Version"
        $cv = [version]$Matches[1]
        if ($cv -ge [version]"13.4") { $index = "https://download.pytorch.org/whl/cu134"; $flavor = "CUDA 13.4" }
        elseif ($cv -ge [version]"13.2") { $index = "https://download.pytorch.org/whl/cu132"; $flavor = "CUDA 13.2" }
        else { Say "WARNING: the NVIDIA driver supports CUDA $cv only; PyTorch 2.15 needs a driver for CUDA >= 13.2. Installing the CPU build; update the driver and re-run to use the GPU." }
    }
} else {
    Say "No NVIDIA GPU driver found (nvidia-smi missing): installing the CPU build."
}
Say "Selected PyTorch build: $flavor ($index)"

# ---------------------------------------------------------------- 4. install
$have = (& $vpy -c "import torch;print(torch.__version__)" 2>$null)
if ($LASTEXITCODE -eq 0 -and $have) {
    $have = "$have".Trim()
    $wantCpu = ($flavor -eq "CPU")
    if (($wantCpu -and $have -notlike "*+cpu*") -or ((-not $wantCpu) -and $have -like "*+cpu*")) {
        Run "Remove existing torch $have (wrong compute platform)" { & $vpy -m pip uninstall -y torch } | Out-Null
    }
}
$rc = Run "Install PyTorch ($flavor) - large download, please wait" { & $vpy -m pip install torch --index-url $index }
if ($rc -ne 0) { Say "ERROR: PyTorch installation failed (see above)."; exit 1 }
Run "Install brlo (editable) + dev + app extras" { & $vpy -m pip install -e ".[dev,app]" } | Out-Null

# ---------------------------------------------------------------- 5. verify
Run "Verify PyTorch" { & $vpy -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" } | Out-Null
Run "Run test suite (includes PyTorch tests)" { & $vpy -m pytest -q } | Out-Null
Run "POMO smoke training (n=20, 2 short epochs)" { & $vpy -m brlo.firstmile.train --n 20 --epochs 2 --steps 20 --batch 32 --layers 3 --val-size 32 --out runs_smoke } | Out-Null
Say ""
Say "SETUP FINISHED - $(Get-Date -Format s)"
