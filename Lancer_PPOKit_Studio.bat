@echo off
rem PPOKit studio: open, create, train and run PPOKit projects.
rem Without argument: the start screen (open a folder, new project, recent projects).
rem A project folder as argument opens it directly.
setlocal
cd /d "%~dp0"
set "STUDIO_EXECUTABLE=%~dp0build\release\PPOKitStudio\Release\ppokit-studio.exe"
if not exist "%STUDIO_EXECUTABLE%" (
    echo [STUDIO] Construction incrementale Release de ppokit-studio...
    cmake --build build/release --config Release --target ppokit-studio
    if errorlevel 1 (
        echo [STUDIO] Echec de la compilation. Configurez build/release puis relancez.
        pause
        exit /b 1
    )
)
start "" "%STUDIO_EXECUTABLE%" %*
exit /b 0
