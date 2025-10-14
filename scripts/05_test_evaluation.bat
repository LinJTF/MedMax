@echo off
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

if "%dataset_choice%"=="1" (
    set DATASET_TYPE=medreqal
    set DATA_PATH=data\MedREQAL.csv
    set COLLECTION=medmax_pubmed
    echo Selected: MedREQAL dataset test
) else if "%dataset_choice%"=="2" (
    set DATASET_TYPE=pubmedqa
    set DATA_PATH=data\pqa_labeled_train.parquet
    set COLLECTION=medmax_pubmed_full
    echo Selected: PubMedQA dataset test
) else (
    echo Invalid choice!
    pause
    exit /b 1
)

echo.
echo Running quick test with 5 questions...
echo This will test RAG functionality without full evaluation
echo.

REM Check if dataset file exists
if not exist "%DATA_PATH%" (
    echo ERROR: Dataset not found at %DATA_PATH%
    pause
    exit /b 1
)

REM Create test results directory
if not exist "evaluation_results" mkdir test_results

echo Testing RAG system...
python -m src.evaluation.main ^
    --data_path "%DATA_PATH%" ^
    --dataset_type "%DATASET_TYPE%" ^
    --output_dir "evaluation_results" ^
    --collection_name "%COLLECTION%" ^
    --engine_type "standard" ^
    --delay 0.1 ^
    --limit 5 ^
    --mode "rag"

if %ERRORLEVEL% neq 0 (
    echo ERROR: Test evaluation failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo TEST COMPLETED SUCCESSFULLY!
echo ================================================
echo Results saved in test_results directory
echo Check the output for any issues
echo.
pause
