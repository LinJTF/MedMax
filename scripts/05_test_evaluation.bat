@echo off
setlocal EnableDelayedExpansion
REM Quick test evaluation script for development/testing
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

if "!dataset_choice!"=="1" (
    set DATASET_TYPE=medreqal
    set DATA_PATH=data\MedREQAL.csv
    set COLLECTION=medmax_pubmed
    echo Selected: MedREQAL dataset test
    goto dataset_selected
)
if "!dataset_choice!"=="2" (
    set DATASET_TYPE=pubmedqa
    set DATA_PATH=data\pqa_labeled_train.parquet
    set COLLECTION=medmax_pubmed_full
    echo Selected: PubMedQA dataset test
    goto dataset_selected
)

echo Invalid choice!
pause
exit /b 1

:dataset_selected

echo.
echo Choose model:
echo [1] GPT-4o-mini (OpenAI - Paid API)
echo [2] Mistral 7B (Ollama - Local, Free)
echo [3] Qwen 2.5 7B (Ollama - Local, Free)
echo.
set /p model_choice="Enter choice (1-3): "
set model_choice=!model_choice: =!
echo [DEBUG] Model choice captured: [!model_choice!]

if "!model_choice!"=="1" (
    set LLM_MODEL=gpt-4o-mini
    set USE_OLLAMA=
    echo Selected: GPT-4o-mini (OpenAI)
    goto model_selected
)
if "!model_choice!"=="2" (
    set LLM_MODEL=mistral:7b
    set USE_OLLAMA=--use_ollama
    echo Selected: Mistral 7B (Ollama)
    goto model_selected
)
if "!model_choice!"=="3" (
    echo [DEBUG] Setting LLM_MODEL to qwen2.5:7b
    set LLM_MODEL=qwen2.5:7b
    echo [DEBUG] LLM_MODEL is now: !LLM_MODEL!
    set USE_OLLAMA=--use_ollama
    echo [DEBUG] USE_OLLAMA is now: !USE_OLLAMA!
    echo Selected: Qwen 2.5 7B (Ollama)
    goto model_selected
)

echo Invalid choice! Choice was: [!model_choice!]
pause
exit /b 1

:model_selected

echo.
echo Choose mode:
echo [1] RAG (with context retrieval)
echo [2] Zero-Shot (no context)
echo.
set /p mode_choice="Enter choice (1-2): "
set mode_choice=!mode_choice: =!

if "!mode_choice!"=="1" (
    set MODE=rag
    echo Selected: RAG mode
    goto mode_selected
)
if "!mode_choice!"=="2" (
    set MODE=zero_shot
    echo Selected: Zero-Shot mode
    goto mode_selected
)

echo Invalid choice!
pause
exit /b 1

:mode_selected

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
