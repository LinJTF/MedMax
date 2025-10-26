@echo off
setlocal EnableDelayedExpansion

echo ================================================
echo TEST EVALUATION (Quick Test Mode)
echo ================================================
echo.
echo Choose dataset for testing:
echo [1] MedREQAL (5 questions)
echo [2] PubMedQA (5 questions)
echo.
set /p dataset_choice="Enter choice (1-2): "
set dataset_choice=!dataset_choice: =!

REM Set dataset variables
if "!dataset_choice!"=="1" (
    call :set_medreqal_dataset
) else if "!dataset_choice!"=="2" (
    call :set_pubmedqa_dataset
) else (
    echo Invalid choice!
    pause
    exit /b 1
)

echo.
echo Choose model:
echo [1] GPT-4o-mini (OpenAI - Paid API)
echo [2] Mistral 7B (Ollama - Local, Free)
echo [3] Qwen 2.5 7B (Ollama - Local, Free)
echo.
set /p model_choice="Enter choice (1-3): "
set model_choice=!model_choice: =!

REM Set model variables
if "!model_choice!"=="1" (
    call :set_openai_model
) else if "!model_choice!"=="2" (
    call :set_mistral_model
) else if "!model_choice!"=="3" (
    call :set_qwen_model
) else (
    echo Invalid choice!
    pause
    exit /b 1
)

echo.
echo Choose mode:
echo [1] RAG (with context retrieval)
echo [2] Zero-Shot (no context)
echo.
set /p mode_choice="Enter choice (1-2): "
set mode_choice=!mode_choice: =!

REM Set mode variables
if "!mode_choice!"=="1" (
    set MODE=rag
    echo Selected: RAG mode
) else if "!mode_choice!"=="2" (
    set MODE=zero_shot
    echo Selected: Zero-Shot mode
) else (
    echo Invalid choice!
    pause
    exit /b 1
)

echo.
echo Running quick test with 5 questions...
echo Dataset: !DATASET_TYPE!
echo Model: !LLM_MODEL!
echo Mode: !MODE!
echo Ollama flag: !USE_OLLAMA!
echo.

REM Check if dataset file exists
if not exist "!DATA_PATH!" (
    echo ERROR: Dataset not found at !DATA_PATH!
    pause
    exit /b 1
)

REM Create test results directory
if not exist "evaluation_results" mkdir evaluation_results

echo Testing !MODE! mode...
echo.
echo [DEBUG] Command will be:
echo python -m src.evaluation.main --data_path "!DATA_PATH!" --dataset_type "!DATASET_TYPE!" --output_dir "evaluation_results" --collection_name "!COLLECTION!" --engine_type "standard" --delay 0.1 --limit 5 --llm_model "!LLM_MODEL!" !USE_OLLAMA! --mode "!MODE!"
echo.

python -m src.evaluation.main ^
    --data_path "!DATA_PATH!" ^
    --dataset_type "!DATASET_TYPE!" ^
    --output_dir "evaluation_results" ^
    --collection_name "!COLLECTION!" ^
    --engine_type "standard" ^
    --delay 0.1 ^
    --limit 5 ^
    --llm_model "!LLM_MODEL!" ^
    !USE_OLLAMA! ^
    --mode "!MODE!"

if !ERRORLEVEL! neq 0 (
    echo ERROR: Test evaluation failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo TEST COMPLETED SUCCESSFULLY!
echo ================================================
echo Results saved in evaluation_results directory
echo Check the output for any issues
echo.
pause
goto :eof

REM Subroutines for setting variables
:set_medreqal_dataset
set DATASET_TYPE=medreqal
set DATA_PATH=data\MedREQAL.csv
set COLLECTION=medmax_pubmed
echo Selected: MedREQAL dataset test
goto :eof

:set_pubmedqa_dataset
set DATASET_TYPE=pubmedqa
set DATA_PATH=data\pqa_labeled_train.parquet
set COLLECTION=medmax_pubmed_full
echo Selected: PubMedQA dataset test
goto :eof

:set_openai_model
set LLM_MODEL=gpt-4o-mini
set USE_OLLAMA=
echo Selected: GPT-4o-mini (OpenAI)
goto :eof

:set_mistral_model
set LLM_MODEL=mistral:7b
set USE_OLLAMA=--use_ollama
echo Selected: Mistral 7B (Ollama)
goto :eof

:set_qwen_model
set LLM_MODEL=qwen2.5:7b
set USE_OLLAMA=--use_ollama
echo Selected: Qwen 2.5 7B (Ollama)
goto :eof