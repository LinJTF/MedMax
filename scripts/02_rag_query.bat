@echo off
setlocal enabledelayedexpansion
REM RAG system for MedMax

echo Starting MedMax RAG System...
echo.
echo Collection: medmax_pubmed
echo.
echo Choose mode:
echo [1] Interactive session (recommended)
echo [2] Single query example
echo [3] Test with sample questions
echo [4] Advanced query with custom settings
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto interactive
if "%choice%"=="2" goto single_query
if "%choice%"=="3" goto test_questions
if "%choice%"=="4" goto advanced_query
goto invalid_choice

:interactive
echo Starting interactive RAG session...
cmd /c "micromamba activate medproj && python -m src.rag.main interactive"
goto end

:single_query
set /p question="Enter your medical question: "
echo Running single query...
cmd /c "micromamba activate medproj && python -m src.rag.main query \"!question!\" --verbose"
goto end

:test_questions
echo Testing with sample medical questions...
echo.
echo Question 1: What is diabetes?
cmd /c "micromamba activate medproj && python -m src.rag.main query \"What is diabetes?\" --verbose"
echo.
echo Question 2: Treatment options for hypertension?
cmd /c "micromamba activate medproj && python -m src.rag.main query \"What are the treatment options for hypertension?\" --verbose"
goto end

:advanced_query
set /p question="Enter your medical question: "
set /p top_k="Number of sources to retrieve (default 5): "
if "!top_k!"=="" set top_k=5
set /p model="OpenAI model (default gpt-4o-mini): "
if "!model!"=="" set model=gpt-4o-mini

echo Running advanced query...
cmd /c "micromamba activate medproj && python -m src.rag.main query \"!question!\" --top-k !top_k! --model !model! --engine-type enhanced --verbose"
goto end

:invalid_choice
echo Invalid choice. Starting interactive session...
cmd /c "micromamba activate medproj && python -m src.rag.main interactive"

:end
if %errorlevel% neq 0 (
    echo Failed to run RAG system!
    pause
    exit /b 1
)
echo RAG session completed!
pause
