@echo off
setlocal enabledelayedexpansion
REM Zero-Shot LLM system for MedMax

echo Starting MedMax Zero-Shot System...
echo.
echo Model: gpt-4o-mini
echo.
echo Choose mode:
echo [1] Interactive zero-shot session
echo [2] Single zero-shot query
echo.
set /p choice="Enter choice (1-2): "

if "%choice%"=="1" goto interactive_zero_shot
if "%choice%"=="2" goto single_query_zero_shot
goto invalid_choice

:interactive_zero_shot
echo Starting interactive zero-shot session...
cmd /c "micromamba activate medproj && python -m src.rag.main interactive --mode zero_shot --model gpt-4o-mini"
goto end

:single_query_zero_shot
set /p question="Enter your medical question: "
echo Running single zero-shot query...
cmd /c "micromamba activate medproj && python -m src.rag.main query ^"!question!^" --mode zero_shot --model gpt-4o-mini"
goto end

:invalid_choice
echo Invalid choice. Starting interactive zero-shot session...
cmd /c "micromamba activate medproj && python -m src.rag.main interactive --mode zero_shot --model gpt-4o-mini"

:end
if %errorlevel% neq 0 (
    echo Failed to run Zero-Shot system!
    pause
    exit /b 1
)
echo Zero-Shot session completed!
pause
