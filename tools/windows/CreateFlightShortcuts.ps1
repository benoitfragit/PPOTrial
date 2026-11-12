param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [switch]$Verify
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$shell = New-Object -ComObject WScript.Shell
foreach ($configuration in @('Debug', 'Release')) {
    $launcher = Join-Path $root "Lancer_FlightControl3D_$configuration.bat"
    $linkPath = Join-Path $root "FlightControl3D_$configuration.lnk"
    if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
        throw "Missing FlightControl3D launcher: $launcher"
    }
    if ($Verify -and -not (Test-Path -LiteralPath $linkPath -PathType Leaf)) {
        throw "Missing generated shortcut: $linkPath"
    }
    $link = $shell.CreateShortcut($linkPath)
    if (-not $Verify) {
        $link.TargetPath = $launcher
        $link.WorkingDirectory = $root
        $link.Arguments = ''
        $link.Description = "FlightControl3D JSBSim - $configuration"
        $link.Save()
        $link = $shell.CreateShortcut($linkPath)
    }
    if ($link.TargetPath -ine $launcher -or $link.WorkingDirectory -ine $root -or $link.Arguments -ne '') {
        throw "Incorrect FlightControl3D shortcut target: $linkPath"
    }
}
Write-Output 'Verified FlightControl3D root shortcuts'
