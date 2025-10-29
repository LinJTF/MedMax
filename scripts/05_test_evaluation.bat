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
    set "DATASET_TYPE=medreqal"
    set "DATA_PATH=data\MedREQAL.csv"
    set "COLLECTION=medmax_pubmed"
    echo Selected: MedREQAL dataset test
)
if "!dataset_choice!"=="2" (
    set "DATASET_TYPE=pubmedqa"
    set "DATA_PATH=data\pqa_labeled_train.parquet"
    set "COLLECTION=medmax_pubmed_full"
    echo Selected: PubMedQA dataset test
)

if not defined DATASET_TYPE (
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
echo [DEBUG] Model choice captured: [!model_choice!]

if "!model_choice!"=="1" (
    set "LLM_MODEL=gpt-4o-mini"
    set "USE_OLLAMA_FLAG="
    echo Selected: GPT-4o-mini (OpenAI)
)
if "!model_choice!"=="2" (
    set "LLM_MODEL=mistral:7b"
    set "USE_OLLAMA_FLAG=YES"
    echo Selected: Mistral 7B (Ollama)
)
if "!model_choice!"=="3" (
    set "LLM_MODEL=qwen2.5:7b"
    set "USE_OLLAMA_FLAG=YES"
    echo Selected: Qwen 2.5 7B (Ollama)
)

if not defined LLM_MODEL (
    echo Invalid choice! Choice was: [!model_choice!]
    pause
    exit /b 1
)

echo [DEBUG] After model selection
echo [DEBUG] LLM_MODEL = [!LLM_MODEL!]
echo [DEBUG] USE_OLLAMA_FLAG = [!USE_OLLAMA_FLAG!]

echo.
echo Choose mode:
echo [1] RAG (with context retrieval)
echo [2] Zero-Shot (no context)
echo.
set /p mode_choice="Enter choice (1-2): "
set mode_choice=!mode_choice: =!

if "!mode_choice!"=="1" (
    set "MODE=rag"
    echo Selected: RAG mode
)
if "!mode_choice!"=="2" (
    set "MODE=zero_shot"
    echo Selected: Zero-Shot mode
)

if not defined MODE (
    echo Invalid choice!
    pause
    exit /b 1
)

echo.
echo Running quick test with 5 questions...
echo Dataset: !DATASET_TYPE!
echo Model: !LLM_MODEL!
echo Mode: !MODE!
echo Ollama flag: !USE_OLLAMA_FLAG!
echo.

echo [DEBUG] Before command construction:
echo [DEBUG] LLM_MODEL = [!LLM_MODEL!]
echo [DEBUG] USE_OLLAMA_FLAG = [!USE_OLLAMA_FLAG!]
echo [DEBUG] MODE = [!MODE!]
echo [DEBUG] DATA_PATH = [!DATA_PATH!]
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
echo [DEBUG] Executing command...
echo.

if "!USE_OLLAMA_FLAG!"=="YES" (
    echo [DEBUG] Running with Ollama flag
    python -m src.evaluation.main --data_path "!DATA_PATH!" --dataset_type "!DATASET_TYPE!" --output_dir "evaluation_results" --collection_name "!COLLECTION!" --engine_type "standard" --delay 0.1 --limit 5 --llm_model "!LLM_MODEL!" --use_ollama --mode "!MODE!"
) else (
    echo [DEBUG] Running without Ollama flag
    python -m src.evaluation.main --data_path "!DATA_PATH!" --dataset_type "!DATASET_TYPE!" --output_dir "evaluation_results" --collection_name "!COLLECTION!" --engine_type "standard" --delay 0.1 --limit 5 --llm_model "!LLM_MODEL!" --mode "!MODE!"
)

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
