@echo off
REM Batch script to compare all three query engines
echo ================================================
echo 🔬 ENGINE COMPARISON (Simple vs Standard vs Enhanced)
echo ================================================
echo 📊 Testing 20 questions with each engine type
echo.

REM Check if MedREQAL CSV file exists
if not exist "data\MedREQAL.csv" (
    echo ❌ ERROR: MedREQAL dataset not found at data\MedREQAL.csv
    echo Please place your MedREQAL CSV file in the data directory
    pause
    exit /b 1
)

REM Create evaluation results directory
if not exist "evaluation_results" mkdir evaluation_results

echo 🟡 Testing SIMPLE Engine...
python -m src.evaluation.main ^
    --csv_path "data\MedREQAL.csv" ^
    --output_dir "evaluation_results" ^
    --collection_name "medmax_pubmed" ^
    --engine_type "simple" ^
    --limit 20 ^
    --delay 0.5

echo.
echo 🟢 Testing STANDARD Engine...
python -m src.evaluation.main ^
    --csv_path "data\MedREQAL.csv" ^
    --output_dir "evaluation_results" ^
    --collection_name "medmax_pubmed" ^
    --engine_type "standard" ^
    --limit 20 ^
    --delay 0.5

echo.
echo 🔵 Testing ENHANCED Engine...
python -m src.evaluation.main ^
    --csv_path "data\MedREQAL.csv" ^
    --output_dir "evaluation_results" ^
    --collection_name "medmax_pubmed" ^
    --engine_type "enhanced" ^
    --limit 20 ^
    --delay 0.5

echo.
echo ✅ Engine comparison completed!
echo 📊 Check evaluation_results directory for detailed comparison
echo 🎯 Compare the accuracy scores to see which engine performs best
echo.
pause
