param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [switch]$Verify
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$shell = New-Object -ComObject WScript.Shell
# Remove only the obsolete root shortcuts previously created for FlightControl3D.
foreach ($obsoleteName in @('FlightControl3D_Debug.lnk', 'FlightControl3D_Release.lnk', 'FlightControl3D_T6_Release.lnk')) {
    $obsoletePath = Join-Path $root $obsoleteName
    if (Test-Path -LiteralPath $obsoletePath -PathType Leaf) {
        if ($Verify) { throw "Obsolete FlightControl3D shortcut: $obsoletePath" }
        Remove-Item -LiteralPath $obsoletePath
    }
}
$curriculumLauncher = Join-Path $root 'Lancer_FlightControl3D_Curriculum_Release.bat'
$curriculumLinkPath = Join-Path $root 'FlightControl3D_Curriculum_Release.lnk'
if (-not (Test-Path -LiteralPath $curriculumLauncher -PathType Leaf)) {
    throw "Missing FlightControl3D curriculum launcher: $curriculumLauncher"
}
if ($Verify -and -not (Test-Path -LiteralPath $curriculumLinkPath -PathType Leaf)) {
    throw "Missing generated shortcut: $curriculumLinkPath"
}
$curriculumLink = $shell.CreateShortcut($curriculumLinkPath)
if (-not $Verify) {
    $curriculumLink.TargetPath = $curriculumLauncher
    $curriculumLink.WorkingDirectory = $root
    $curriculumLink.Arguments = ''
    $curriculumLink.Description = 'FlightControl3D T-6 Joint training - Release'
    $curriculumLink.Save()
    $curriculumLink = $shell.CreateShortcut($curriculumLinkPath)
}
if ($curriculumLink.TargetPath -ine $curriculumLauncher -or $curriculumLink.WorkingDirectory -ine $root -or $curriculumLink.Arguments -ne '') {
    throw "Incorrect FlightControl3D curriculum shortcut target: $curriculumLinkPath"
}
Write-Output 'Verified FlightControl3D root shortcuts'
