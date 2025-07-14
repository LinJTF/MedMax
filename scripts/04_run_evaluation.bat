@echo off
REM Batch script to run complete MedREQAL evaluation
echo ================================================
echo 🚀 MedREQAL COMPLETE EVALUATION (ALL 2786 questions)
echo ================================================
echo ⚠️  WARNING: This will take several hours!
echo Press Ctrl+C to cancel or any key to continue...
pause

REM Check if MedREQAL CSV file exists
if not exist "data\MedREQAL.csv" (
    echo ❌ ERROR: MedREQAL dataset not found at data\MedREQAL.csv
    echo Please place your MedREQAL CSV file in the data directory
    pause
    exit /b 1
)

REM Create evaluation results directory
if not exist "evaluation_results" mkdir evaluation_results

echo 🎯 Running complete evaluation with Standard Engine...
echo 📊 Processing all 2786 questions...
echo ⏱️  Estimated time: ~2-3 hours
echo.

python -m src.evaluation.main ^
    --csv_path "data\MedREQAL.csv" ^
    --output_dir "evaluation_results" ^
    --collection_name "medmax_pubmed" ^
    --engine_type "standard" ^
    --delay 1.0 ^
    --baseline_accuracy 0.65

echo.
echo ✅ Complete evaluation finished!
echo 📊 Results saved in evaluation_results directory
echo 🎯 Check the final accuracy and F1 scores above
echo.
pause
