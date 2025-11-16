@echo off
setlocal EnableDelayedExpansion

echo ================================================
echo TEST EVALUATION (Quick Test Mode)
echo ================================================
echo.

REM Dataset selection
echo Choose dataset:
echo [1] MedREQAL
echo [2] PubMedQA
echo.
set /p dataset_choice="Enter choice (1-2): "

if "!dataset_choice!"=="1" (
    set "DATASET_TYPE=medreqal"
    set "DATA_PATH=data\MedREQAL.csv"
    set "COLLECTION=medmax_pubmed"
) else if "!dataset_choice!"=="2" (
    set "DATASET_TYPE=pubmedqa"
    set "DATA_PATH=data\pqa_labeled_train.parquet"
    set "COLLECTION=medmax_pubmed_full"
) else (
    echo Invalid choice!
    pause
    exit /b 1
)

echo Selected: !DATASET_TYPE!
echo.

REM Provider selection
echo Choose model provider:
echo [1] OpenAI (Cloud, Paid)
echo [2] Ollama (Local, Free)
echo [3] HuggingFace (Local GPU, Free)
echo.
set /p provider_choice="Enter choice (1-3): "

if "!provider_choice!"=="1" goto openai
if "!provider_choice!"=="2" goto ollama
if "!provider_choice!"=="3" goto huggingface
echo Invalid choice!
pause
exit /b 1

:openai
echo.
echo OpenAI models:
echo [1] GPT-4o-mini
echo [2] GPT-4o
echo [3] GPT-3.5-turbo
set /p model_choice="Enter choice (1-3): "

if "!model_choice!"=="1" set "MODEL=gpt-4o-mini"
if "!model_choice!"=="2" set "MODEL=gpt-4o"
if "!model_choice!"=="3" set "MODEL=gpt-3.5-turbo"
set "FLAGS="
goto mode_select

:ollama
echo.
echo Ollama models:
echo [1] Mistral 7B
echo [2] Qwen 2.5 7B
set /p model_choice="Enter choice (1-2): "

if "!model_choice!"=="1" set "MODEL=mistral:7b"
if "!model_choice!"=="2" set "MODEL=qwen2.5:7b"
set "FLAGS=--use_ollama"
goto mode_select

:huggingface
echo.
echo HuggingFace models (GPU required):
echo [1] Mistral 7B Instruct
echo [2] Qwen 2.5 7B Instruct
echo [3] Llama 3 8B Instruct
set /p model_choice="Enter choice (1-3): "

if "!model_choice!"=="1" set "MODEL=mistral-7b-instruct"
if "!model_choice!"=="2" set "MODEL=qwen2.5-7b-instruct"
if "!model_choice!"=="3" set "MODEL=llama3-8b-instruct"
set "FLAGS=--use_huggingface"
goto mode_select

:mode_select
if not defined MODEL (
    echo Invalid model choice!
    pause
    exit /b 1
)

echo.
echo Choose mode:
echo [1] RAG (with retrieval)
echo [2] Zero-Shot (no retrieval)
set /p mode_choice="Enter choice (1-2): "

if "!mode_choice!"=="1" set "MODE=rag"
if "!mode_choice!"=="2" set "MODE=zero_shot"

if not defined MODE (
    echo Invalid mode choice!
    pause
    exit /b 1
)

REM Summary
echo.
echo ================================================
echo CONFIGURATION SUMMARY
echo ================================================
echo Dataset: !DATASET_TYPE!
echo Model: !MODEL!
echo Mode: !MODE!
echo Flags: !FLAGS!
echo ================================================
echo.

REM Check dataset exists
if not exist "!DATA_PATH!" (
    echo ERROR: Dataset not found at !DATA_PATH!
    pause
    exit /b 1
)

REM Create output directory
if not exist "evaluation_results" mkdir evaluation_results

REM Build and execute command
echo Running evaluation...
echo.

set "CMD=python -m src.evaluation.main"
set "CMD=!CMD! --data_path "!DATA_PATH!""
set "CMD=!CMD! --dataset_type "!DATASET_TYPE!""
set "CMD=!CMD! --output_dir "evaluation_results""
set "CMD=!CMD! --collection_name "!COLLECTION!""
set "CMD=!CMD! --engine_type "standard""
set "CMD=!CMD! --delay 0.1"
set "CMD=!CMD! --limit 100"
set "CMD=!CMD! --llm_model "!MODEL!""
if defined FLAGS set "CMD=!CMD! !FLAGS!"
set "CMD=!CMD! --mode "!MODE!""

echo [DEBUG] Executing: !CMD!
echo.

!CMD!

set EXIT_CODE=!ERRORLEVEL!

echo.
echo ================================================
if !EXIT_CODE! equ 0 (
    echo  TEST COMPLETED SUCCESSFULLY
) else (
    echo  TEST FAILED (Exit code: !EXIT_CODE!)
)
echo ================================================
echo.

pause
exit /b !EXIT_CODE!
