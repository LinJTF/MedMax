@echo off
REM Batch script to run MedREQAL evaluation
echo Starting MedREQAL Evaluation...

REM Check if MedREQAL CSV file exists
if not exist "data\medreqal_dataset.csv" (
    echo ERROR: MedREQAL dataset not found at data\medreqal_dataset.csv
    echo Please place your MedREQAL CSV file in the data directory
    pause
    exit /b 1
)

REM Create evaluation results directory
if not exist "evaluation_results" mkdir evaluation_results

REM Run evaluation with default settings
echo Running evaluation on MedREQAL dataset...
C:/Users/Lin/anaconda3/Scripts/conda.exe run -p C:\Users\Lin\micromamba\envs\medproj python -m src.evaluation.main ^
    --csv_path "data\medreqal_dataset.csv" ^
    --output_dir "evaluation_results" ^
    --collection_name "medmax_pubmed" ^
    --delay 1.0

echo.
echo Evaluation completed!
echo Results saved in evaluation_results directory
pause
