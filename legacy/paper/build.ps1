param([switch]$Clean)
$ErrorActionPreference = 'Stop'
$paperRoot = $PSScriptRoot
$miktexBin = 'C:\Program Files\MiKTeX\miktex\bin\x64'
if (-not (Test-Path -LiteralPath "$miktexBin\pdflatex.exe")) {
    throw 'The configured MiKTeX installation was not found.'
}
Push-Location -LiteralPath $paperRoot
try {
    if ($Clean) {
        foreach ($extension in @('aux', 'bbl', 'blg', 'log', 'out', 'toc')) {
            $temporaryFile = Join-Path $paperRoot "main.$extension"
            if (Test-Path -LiteralPath $temporaryFile) {
                Remove-Item -LiteralPath $temporaryFile
            }
        }
    }
    & "$miktexBin\pdflatex.exe" -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    if ($LASTEXITCODE -ne 0) { throw 'First MiKTeX pass failed.' }
    & "$miktexBin\bibtex.exe" main
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
    foreach ($pass in 1..2) {
        & "$miktexBin\pdflatex.exe" -interaction=nonstopmode -halt-on-error -file-line-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "MiKTeX pass $pass failed." }
    }
} finally {
    Pop-Location
}
