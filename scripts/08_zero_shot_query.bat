@echo off
REM Zero-shot evaluation script for both datasets
echo ================================================
echo ZERO-SHOT EVALUATION
echo ================================================
echo.
echo Choose dataset:
echo [1] MedREQAL dataset
echo [2] PubMedQA dataset
echo.
set /p dataset_choice="Enter choice (1-2): "

if "%dataset_choice%"=="1" (
    set DATASET_TYPE=medreqal
    set DATA_PATH=data\MedREQAL.csv
    set LIMIT=50
    echo Selected: MedREQAL zero-shot evaluation
) else if "%dataset_choice%"=="2" (
    set DATASET_TYPE=pubmedqa
    set DATA_PATH=data\pqa_labeled_train.parquet
    set LIMIT=20
    echo Selected: PubMedQA zero-shot evaluation
) else (
    echo Invalid choice!
    pause
    exit /b 1
)

echo.
echo Zero-shot evaluation (no retrieval)
echo Dataset: %DATASET_TYPE%
echo Limit: %LIMIT% questions
echo Model: gpt-4o-mini
echo.
echo Press any key to continue...
pause

REM Check if dataset file exists
if not exist "%DATA_PATH%" (
    echo ERROR: Dataset not found at %DATA_PATH%
    pause
    exit /b 1
)

REM Create results directory
if not exist "zero_shot_results" mkdir zero_shot_results

echo Running zero-shot evaluation...
python -m src.evaluation.main ^
    --data_path "%DATA_PATH%" ^
    --dataset_type "%DATASET_TYPE%" ^
    --output_dir "zero_shot_results" ^
    --collection_name "medmax_pubmed_full" ^
    --engine_type "standard" ^
    --delay 0.5 ^
    --mode "zero_shot" ^
    --limit %LIMIT% ^
    --llm_model "gpt-4o-mini"

if %ERRORLEVEL% neq 0 (
    echo ERROR: Zero-shot evaluation failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo ZERO-SHOT EVALUATION COMPLETED!
echo ================================================
echo Results saved in zero_shot_results directory
echo.
pause
