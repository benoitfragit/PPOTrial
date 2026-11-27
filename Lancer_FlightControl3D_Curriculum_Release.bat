@echo off
rem T-6 landing (FlightControl3D) under PPOKit: the studio opens the T6 project.
rem Train, Evaluate and Inference run ppokit-run; View opens ppokit-viewer on the flight.
setlocal
cd /d "%~dp0"
set "STUDIO_EXECUTABLE=%~dp0build\release\PPOKitStudio\Release\ppokit-studio.exe"
set "T6_PROJECT=%~dp0Examples\FlightControl3D\projects\T6"
echo [T6] Construction incrementale Release du studio, de ppokit-run et des bibliotheques FlightControl3D...
cmake --build build/release --config Release --target ppokit-studio ppokit-run ppokit-viewer flight_environment ppokit_graph_network flight_visualiser
if errorlevel 1 (
    echo [T6] Echec de la compilation. Configurez build/release puis relancez.
    pause
    exit /b 1
)
start "" "%STUDIO_EXECUTABLE%" "%T6_PROJECT%"
exit /b 0
