@echo off
REM Batch script to test MedREQAL evaluation with limited questions
echo Starting MedREQAL Evaluation Test (10 questions)...

REM Check if MedREQAL CSV file exists
if not exist "data\medreqal_dataset.csv" (
    echo ERROR: MedREQAL dataset not found at data\medreqal_dataset.csv
    echo Please place your MedREQAL CSV file in the data directory
    pause
    exit /b 1
)

REM Create evaluation results directory
if not exist "evaluation_results" mkdir evaluation_results

REM Run evaluation test with only 10 questions
echo Running test evaluation on 10 questions...
C:/Users/Lin/anaconda3/Scripts/conda.exe run -p C:\Users\Lin\micromamba\envs\medproj python -m src.evaluation.main ^
    --csv_path "data\medreqal_dataset.csv" ^
    --output_dir "evaluation_results" ^
    --collection_name "medmax_pubmed" ^
    --limit 10 ^
    --delay 0.5 ^
    --baseline_accuracy 0.65

echo.
echo Test evaluation completed!
echo Results saved in evaluation_results directory
pause
