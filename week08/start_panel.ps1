param(
    [string]$CheckpointRoot = "",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [int]$Port = 7860,
    [string]$Python = "python",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$panelDirectory = $PSScriptRoot
$projectDirectory = Split-Path -Parent $panelDirectory
$dependencyDirectory = Join-Path $panelDirectory ".panel_dependencies"
$runtimeDirectory = Join-Path $panelDirectory ".panel_venv"
$panelPython = Join-Path $runtimeDirectory "Scripts\python.exe"
$projectPackages = Join-Path $projectDirectory ".venv\Lib\site-packages"
$previousPythonPath = $env:PYTHONPATH
$previousEncoding = $env:PYTHONIOENCODING

try {
    if (-not (Test-Path -LiteralPath $panelPython)) {
        Write-Host "Creating an isolated panel runtime..."
        & $Python -m venv $runtimeDirectory
        if ($LASTEXITCODE -ne 0) { throw "Could not create the panel runtime." }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $dependencyDirectory "gradio"))) {
        Write-Host "Installing optional panel dependencies in the project-local isolated folder..."
        & $panelPython -m pip install --target $dependencyDirectory -r (Join-Path $panelDirectory "ui_requirements.txt")
        if ($LASTEXITCODE -ne 0) { throw "Optional dependency installation failed." }
    }
    $packagePaths = @($dependencyDirectory)
    if (Test-Path -LiteralPath $projectPackages) { $packagePaths += $projectPackages }
    if ($previousPythonPath) { $packagePaths += $previousPythonPath }
    $env:PYTHONPATH = $packagePaths -join [IO.Path]::PathSeparator
    $env:PYTHONIOENCODING = "utf-8"
    if (-not $CheckpointRoot) {
        $localCheckpointRoot = Join-Path $panelDirectory "outputs\checkpoints"
        $englishCheckpoint = Join-Path $localCheckpointRoot "en_a100_layernorm\best.pt"
        $turkishCheckpoint = Join-Path $localCheckpointRoot "tr_a100_layernorm\best.pt"
        if ((Test-Path -LiteralPath $englishCheckpoint) -or (Test-Path -LiteralPath $turkishCheckpoint)) {
            $CheckpointRoot = $localCheckpointRoot
            Write-Host "Connecting the owner's local model checkpoints..."
        }
    }
    $panelArguments = @("-u", (Join-Path $panelDirectory "panel.py"), "--device", $Device, "--port", "$Port")
    if (-not $NoBrowser) { $panelArguments += "--open" }
    if ($CheckpointRoot) { $panelArguments += @("--checkpoint-root", $CheckpointRoot) }
    & $panelPython @panelArguments
    if ($LASTEXITCODE -ne 0) { throw "Panel stopped with an error. Verify that PyTorch is installed in the selected Python environment." }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:PYTHONIOENCODING = $previousEncoding
}
