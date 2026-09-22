param([switch]$SkipModels)
$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pythonLauncher) { throw 'Install 64-bit Python 3.12 with the Python launcher, then retry.' }
    & $pythonLauncher.Source -3.12 -m venv (Join-Path $repoRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 virtual environment creation failed.' }
}
& $venvPython -m pip install -r (Join-Path $repoRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not $SkipModels) {
    & $venvPython (Join-Path $repoRoot 'prepare_models.py')
    if ($LASTEXITCODE -ne 0) { throw 'OCR model preparation failed. Check network access and retry.' }
}
$pythonWindowed = Join-Path $repoRoot '.venv\Scripts\pythonw.exe'
$entryPoint = Join-Path $repoRoot 'launch.pyw'
$shortcutShell = New-Object -ComObject WScript.Shell
$shortcut = $shortcutShell.CreateShortcut((Join-Path $repoRoot 'Start-ApexHighlights.lnk'))
$shortcut.TargetPath = $pythonWindowed
$shortcut.Arguments = '"' + $entryPoint + '"'
$shortcut.WorkingDirectory = $repoRoot
$shortcut.Description = 'Apex Highlights'
$shortcut.Save()
Write-Host 'Setup complete. Open Start-ApexHighlights.lnk (no console window).'
Write-Host 'Damage OCR also requires Tesseract with the English language data installed separately.'
