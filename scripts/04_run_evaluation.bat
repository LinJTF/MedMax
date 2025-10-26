@echo off
setlocal EnableDelayedExpansion
REM Batch script to run comparative evaluation (RAG vs Zero-Shot) for MedREQAL or PubMedQA
echo ================================================
echo COMPARATIVE EVALUATION (RAG vs Zero-Shot)
echo ================================================
echo.
echo Choose dataset:
echo [1] MedREQAL (requires MedREQAL.csv)
echo [2] PubMedQA (uses pqa_labeled_train.parquet)
echo.
set /p dataset_choice="Enter choice (1-2): "
set dataset_choice=!dataset_choice: =!

REM Set dataset variables using subroutines
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

REM Set model variables using subroutines
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
echo WARNING: This will run both RAG and Zero-Shot evaluations!
echo Dataset: !DATASET_TYPE!
echo File: !DATA_PATH!
echo Collection: !COLLECTION!
echo Model: !LLM_MODEL!
echo Limit: !LIMIT! questions
echo.
echo Press Ctrl+C to cancel or any key to continue...
pause

REM Check if dataset file exists
if not exist "!DATA_PATH!" (
    echo ERROR: Dataset not found at !DATA_PATH!
    echo Please ensure your dataset file is in the data directory
    pause
    exit /b 1
)

REM Create evaluation results directory
if not exist "evaluation_results" mkdir evaluation_results

echo Running evaluation with both RAG and Zero-Shot approaches...
echo Processing !LIMIT! questions...
echo Estimated time: ~30 minutes for !LIMIT! questions
echo.

echo ================================================
echo STEP 1: Running RAG evaluation...
echo ================================================
python -m src.evaluation.main ^
    --data_path "!DATA_PATH!" ^
    --dataset_type "!DATASET_TYPE!" ^
    --output_dir "evaluation_results" ^
    --collection_name "!COLLECTION!" ^
    --engine_type "standard" ^
    --delay 0.5 ^
    --baseline_accuracy 0.65 ^
    --limit !LIMIT! ^
    --llm_model "!LLM_MODEL!" ^
    !USE_OLLAMA! ^
    --mode "rag"

if !ERRORLEVEL! neq 0 (
    echo ERROR: RAG evaluation failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo STEP 2: Running Zero-Shot evaluation...
echo ================================================
python -m src.evaluation.main ^
    --data_path "!DATA_PATH!" ^
    --dataset_type "!DATASET_TYPE!" ^
    --output_dir "evaluation_results" ^
    --collection_name "!COLLECTION!" ^
    --engine_type "standard" ^
    --delay 0.5 ^
    --baseline_accuracy 0.65 ^
    --mode "zero_shot" ^
    --limit !LIMIT! ^
    --llm_model "!LLM_MODEL!" ^
    !USE_OLLAMA!

if !ERRORLEVEL! neq 0 (
    echo ERROR: Zero-shot evaluation failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo BOTH EVALUATIONS COMPLETED SUCCESSFULLY!
echo ================================================
echo Results saved in evaluation_results directory:
echo - RAG results: !DATASET_TYPE!_rag_results_TIMESTAMP.csv
echo - Zero-shot results: !DATASET_TYPE!_zero_shot_results_TIMESTAMP.csv
echo Check the accuracy and F1 scores for comparison
echo.
pause
goto :eof

REM Subroutines for setting variables
:set_medreqal_dataset
set DATASET_TYPE=medreqal
set DATA_PATH=data\MedREQAL.csv
set COLLECTION=medmax_pubmed
set LIMIT=50
echo Selected: MedREQAL dataset
goto :eof

:set_pubmedqa_dataset
set DATASET_TYPE=pubmedqa
set DATA_PATH=data\pqa_labeled_train.parquet
set COLLECTION=medmax_pubmed_full
set LIMIT=1000
echo Selected: PubMedQA dataset
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
