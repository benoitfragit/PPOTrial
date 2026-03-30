[CmdletBinding()]
param(
    [switch]$NoOpen
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Doxyfile = Join-Path $RepoRoot "docs\\doxygen\\Doxyfile"
$OutputRoot = Join-Path $RepoRoot "docs\\generated"
$HtmlIndex = Join-Path $OutputRoot "html\\index.html"
$WarningLog = Join-Path $OutputRoot "doxygen-warnings.log"

function Find-Executable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [string[]]$Candidates = @()
    )

    $commandInfo = Get-Command $Command -ErrorAction SilentlyContinue
    if ($commandInfo -and $commandInfo.Source) {
        return $commandInfo.Source
    }

    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }

        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return $null
}

$doxygenCandidates = @(
    (Join-Path $env:ProgramFiles "doxygen\\bin\\doxygen.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "doxygen\\bin\\doxygen.exe")
)

$graphvizCandidates = @(
    (Join-Path $env:ProgramFiles "Graphviz\\bin\\dot.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Graphviz\\bin\\dot.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Graphviz2.38\\bin\\dot.exe")
)

$doxygenExe = Find-Executable -Command "doxygen" -Candidates $doxygenCandidates
if (-not $doxygenExe) {
    Write-Host ""
    Write-Host "Doxygen introuvable." -ForegroundColor Red
    Write-Host "Installe Doxygen puis relance generatedoc.bat."
    Write-Host "Graphviz est recommande si tu veux les graphes d'appel et de dependances."
    exit 1
}

$dotExe = Find-Executable -Command "dot" -Candidates $graphvizCandidates
if ($dotExe) {
    $env:Path = "$(Split-Path $dotExe -Parent);$env:Path"
    Write-Host "Graphviz detecte : $dotExe"
}
else {
    Write-Warning "Graphviz (dot.exe) introuvable. La doc HTML sera generee sans graphes."
}

if (-not (Test-Path $Doxyfile)) {
    throw "Fichier Doxyfile introuvable : $Doxyfile"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

Push-Location $RepoRoot
try {
    Write-Host "Generation de la documentation avec $doxygenExe"
    & $doxygenExe $Doxyfile

    if ($LASTEXITCODE -ne 0) {
        throw "Doxygen a retourne le code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

if (-not (Test-Path $HtmlIndex)) {
    throw "Generation terminee, mais index.html est introuvable : $HtmlIndex"
}

Write-Host ""
Write-Host "Documentation generee :" -ForegroundColor Green
Write-Host "  $HtmlIndex"

if ((Test-Path $WarningLog) -and (Get-Item $WarningLog).Length -gt 0) {
    Write-Host ""
    Write-Host "Warnings Doxygen :" -ForegroundColor Yellow
    Write-Host "  $WarningLog"
}

if (-not $NoOpen) {
    Start-Process $HtmlIndex | Out-Null
}
