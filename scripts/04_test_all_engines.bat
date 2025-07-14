@echo off
REM Test all RAG engine types with the same question
echo Testing all RAG engine types...

set QUESTION="Are group 2 innate lymphoid cells increased in chronic rhinosinusitis"

echo.
echo ========================================
echo Testing SIMPLE Engine
echo ========================================
C:/Users/Lin/anaconda3/Scripts/conda.exe run -p C:\Users\Lin\micromamba\envs\medproj python -m src.rag.main query %QUESTION% --engine-type simple --verbose

echo.
echo ========================================
echo Testing STANDARD Engine  
echo ========================================
C:/Users/Lin/anaconda3/Scripts/conda.exe run -p C:\Users\Lin\micromamba\envs\medproj python -m src.rag.main query %QUESTION% --engine-type standard --verbose

echo.
echo ========================================
echo Testing ENHANCED Engine
echo ========================================
C:/Users/Lin/anaconda3/Scripts/conda.exe run -p C:\Users\Lin\micromamba\envs\medproj python -m src.rag.main query %QUESTION% --engine-type enhanced --verbose

echo.
echo ========================================
echo Testing completed!
echo ========================================
pause
