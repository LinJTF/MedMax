@echo off
REM Batch script to test MedREQAL evaluation with limited questions
echo ================================================
echo MedREQAL Evaluation TEST (10 questions)
echo ================================================

REM Check if MedREQAL CSV file exists
if not exist "data\MedREQAL.csv" (
    echo ERROR: MedREQAL dataset not found at data\MedREQAL.csv
    echo Please place your MedREQAL CSV file in the data directory
    pause
    exit /b 1
)

REM Create evaluation results directory
if not exist "evaluation_results" mkdir evaluation_results

echo Testing with Standard Engine (10 questions)...
python -m src.evaluation.main ^
    --csv_path "data\MedREQAL.csv" ^
    --output_dir "evaluation_results" ^
    --collection_name "medmax_pubmed" ^
    --engine_type "standard" ^
    --limit 10 ^
    --delay 0.5 ^
    --baseline_accuracy 0.92

echo.
echo Test evaluation completed!
echo Results saved in evaluation_results directory
echo.
pause
