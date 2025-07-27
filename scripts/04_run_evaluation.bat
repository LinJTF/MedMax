@echo off
REM Batch script to run comparative MedREQAL evaluation (RAG vs Zero-Shot)
echo ================================================
echo MedREQAL COMPARATIVE EVALUATION (RAG vs Zero-Shot)
echo ================================================
echo WARNING: This will run both RAG and Zero-Shot evaluations!
echo Press Ctrl+C to cancel or any key to continue...
pause

REM Check if MedREQAL CSV file exists
if not exist "data\MedREQAL.csv" (
    echo ERROR: MedREQAL dataset not found at data\MedREQAL.csv
    echo Please place your MedREQAL CSV file in the data directory
    pause
    exit /b 1
)

REM Create evaluation results directory
if not exist "evaluation_results" mkdir evaluation_results

echo Running evaluation with both RAG and Zero-Shot approaches...
echo Processing 3 questions with each approach...
echo Estimated time: ~2-3 minutes for testing
echo.

echo ================================================
echo STEP 1: Running RAG evaluation...
echo ================================================
@REM C:\Users\Lin\micromamba\condabin\micromamba.bat activate medproj && python -m src.evaluation.main ^
python -m src.evaluation.main ^
    --csv_path "data\MedREQAL.csv" ^
    --output_dir "evaluation_results" ^
    --collection_name "medmax_pubmed" ^
    --engine_type "standard" ^
    --delay 0.5 ^
    --baseline_accuracy 0.65 ^
    --mode "rag"

if %ERRORLEVEL% neq 0 (
    echo ERROR: RAG evaluation failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo STEP 2: Running Zero-Shot evaluation...
echo ================================================
@REM C:\Users\Lin\micromamba\condabin\micromamba.bat activate medproj && python -m src.evaluation.main ^
python -m src.evaluation.main ^
    --csv_path "data\MedREQAL.csv" ^
    --output_dir "evaluation_results" ^
    --collection_name "medmax_pubmed" ^
    --engine_type "standard" ^
    --delay 0.5 ^
    --baseline_accuracy 0.65 ^
    --mode "zero_shot" ^
    --llm_model "gpt-4o-mini"

if %ERRORLEVEL% neq 0 (
    echo ERROR: Zero-shot evaluation failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo BOTH EVALUATIONS COMPLETED SUCCESSFULLY!
echo ================================================
echo Results saved in evaluation_results directory:
echo - RAG results: rag_results_TIMESTAMP.csv and rag_metrics_TIMESTAMP.json
echo - Zero-shot results: zero_shot_results_TIMESTAMP.csv and zero_shot_metrics_TIMESTAMP.json
echo Check the accuracy and F1 scores for comparison
echo.
pause
