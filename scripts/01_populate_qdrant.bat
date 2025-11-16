@echo off
setlocal enabledelayedexpansion
REM Populate Qdrant with PubMed data - WITH SMART COLLECTION NAMING
set DATA_PATH=data/PubMed-compact/pubmedqa.jsonl

REM Activate Python environment
call cmd /c "micromamba activate medproj"

echo ============================================================
echo    MedMax Vector Store Population
echo    WITH OPEN SOURCE EMBEDDINGS SUPPORT!
echo ============================================================
echo.

REM ========================================
REM STEP 1: Choose Embedding Provider
REM ========================================
echo STEP 1: Choose Embedding Provider
echo ============================================================
echo.
echo [1] OpenAI Embeddings (Paid, High Quality)
echo     - Model: text-embedding-3-small
echo     - Cost: ~$0.02 per 1M tokens
echo     - Dimension: 1536
echo.
echo [2] HuggingFace Embeddings (FREE, Local) *** RECOMMENDED ***
echo     - Model: all-MiniLM-L6-v2
echo     - Cost: $0 (FREE!)
echo     - Dimension: 384
echo     - Processes locally (faster if you have GPU)
echo.
set /p embedding_choice="Enter embedding provider choice (1/2): "

if "%embedding_choice%"=="2" goto huggingface_embeddings
if "%embedding_choice%"=="1" goto openai_embeddings

REM Default to OpenAI if invalid choice
echo Invalid choice, defaulting to OpenAI embeddings...
goto openai_embeddings

:huggingface_embeddings
echo.
echo Selected: HuggingFace Embeddings (FREE!)
echo.
set USE_HF=--use-huggingface-embeddings
set EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
set EMBEDDING_SUFFIX=hf_minilm

REM Check device for HuggingFace
echo Choose device for embedding generation:
echo [A] Auto-detect (recommended - uses GPU if available)
echo [B] Force GPU (CUDA)
echo [C] Force CPU
echo.
set /p device_choice="Enter device choice (A/B/C): "

if /i "%device_choice%"=="B" (
    set DEVICE_FLAG=--embedding-device cuda
    echo Using: GPU (CUDA)
) else if /i "%device_choice%"=="C" (
    set DEVICE_FLAG=--embedding-device cpu
    echo Using: CPU
) else (
    set DEVICE_FLAG=--embedding-device auto
    echo Using: Auto-detect (GPU if available, else CPU)
)

goto choose_data_source

:openai_embeddings
echo.
echo Selected: OpenAI Embeddings (Paid)
echo.
set USE_HF=
set EMBEDDING_MODEL=
set DEVICE_FLAG=
set EMBEDDING_SUFFIX=openai_small
goto choose_data_source

:choose_data_source
echo.
echo ============================================================
echo STEP 2: Choose Data Source
echo ============================================================
echo.
echo [A] Original JSONL data (data/PubMed-compact/pubmedqa.jsonl)
echo [B] New PubMedQA parquet files (data/*.parquet) - RECOMMENDED
echo.
set /p data_choice="Enter data source choice (A/B): "

if /i "%data_choice%"=="B" goto parquet_mode
if /i "%data_choice%"=="b" goto parquet_mode

REM ========================================
REM JSONL MODE
REM ========================================
echo.
echo Using JSONL data source: %DATA_PATH%
echo.
echo ============================================================
echo STEP 3: Choose Operation Mode
echo ============================================================
echo.
echo [1] Test mode (10 records only)
echo [2] Small dataset (100 records)
echo [3] Medium dataset (1000 records)
echo [4] Full dataset
echo [5] Custom limit
echo.
set /p mode_choice="Enter mode choice (1-5): "

if "%mode_choice%"=="1" (
    set COLLECTION_NAME=medmax_test_%EMBEDDING_SUFFIX%
    set LIMIT=10
    set MODE_DESC=test
) else if "%mode_choice%"=="2" (
    set COLLECTION_NAME=medmax_small_%EMBEDDING_SUFFIX%
    set LIMIT=100
    set MODE_DESC=small
) else if "%mode_choice%"=="3" (
    set COLLECTION_NAME=medmax_medium_%EMBEDDING_SUFFIX%
    set LIMIT=1000
    set MODE_DESC=medium
) else if "%mode_choice%"=="4" (
    set COLLECTION_NAME=medmax_pubmed_%EMBEDDING_SUFFIX%
    set LIMIT=
    set MODE_DESC=full
) else if "%mode_choice%"=="5" (
    set /p custom_limit="Enter number of records: "
    set COLLECTION_NAME=medmax_custom_%EMBEDDING_SUFFIX%
    set LIMIT=!custom_limit!
    set MODE_DESC=custom
) else (
    echo Invalid choice. Exiting...
    goto end
)

echo.
echo ============================================================
echo CONFIGURATION SUMMARY
echo ============================================================
echo Collection name: %COLLECTION_NAME%
echo Data source: JSONL
echo Mode: %MODE_DESC%
if defined LIMIT (
    echo Limit: %LIMIT% records
) else (
    echo Limit: None (full dataset)
)
echo Embedding: %EMBEDDING_SUFFIX%
echo ============================================================
echo.
echo This will create a NEW collection: %COLLECTION_NAME%
echo Your existing collections will NOT be affected!
echo.
set /p confirm="Proceed? (y/N): "
if /i not "!confirm!"=="y" goto end

if defined LIMIT (
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --data-path %DATA_PATH% --limit %LIMIT% %USE_HF% %DEVICE_FLAG%"
) else (
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --data-path %DATA_PATH% %USE_HF% %DEVICE_FLAG%"
)
goto end

REM ========================================
REM PARQUET MODE
REM ========================================
:parquet_mode
echo.
echo Using PubMedQA parquet files (unlabeled + labeled + artificial)
echo This is the RECOMMENDED data source for full dataset!
echo.
echo ============================================================
echo STEP 3: Choose Operation Mode
echo ============================================================
echo.
echo [1] Test mode (10 records per dataset)
echo [2] Small dataset (100 records per dataset)
echo [3] Medium dataset (1000 records per dataset)
echo [4] Full datasets (ALL data - ~200k records)
echo [5] Custom limit per dataset
echo.
set /p mode_choice="Enter mode choice (1-5): "

if "%mode_choice%"=="1" (
    set COLLECTION_NAME=medmax_test_%EMBEDDING_SUFFIX%
    set LIMIT=10
    set MODE_DESC=test
) else if "%mode_choice%"=="2" (
    set COLLECTION_NAME=medmax_small_%EMBEDDING_SUFFIX%
    set LIMIT=100
    set MODE_DESC=small
) else if "%mode_choice%"=="3" (
    set COLLECTION_NAME=medmax_medium_%EMBEDDING_SUFFIX%
    set LIMIT=1000
    set MODE_DESC=medium
) else if "%mode_choice%"=="4" (
    set COLLECTION_NAME=medmax_pubmed_full_%EMBEDDING_SUFFIX%
    set LIMIT=
    set MODE_DESC=full
) else if "%mode_choice%"=="5" (
    set /p custom_limit="Enter number of records per dataset: "
    set COLLECTION_NAME=medmax_custom_%EMBEDDING_SUFFIX%
    set LIMIT=!custom_limit!
    set MODE_DESC=custom
) else (
    echo Invalid choice. Exiting...
    goto end
)

echo.
echo ============================================================
echo CONFIGURATION SUMMARY
echo ============================================================
echo Collection name: %COLLECTION_NAME%
echo Data source: Parquet files
echo Mode: %MODE_DESC%
if defined LIMIT (
    echo Limit: %LIMIT% records per dataset
) else (
    echo Limit: None (full datasets - ~200k total records)
)
echo Embedding: %EMBEDDING_SUFFIX%
echo ============================================================
echo.
echo This will create a NEW collection: %COLLECTION_NAME%
echo Your existing collections will NOT be affected!
echo.

REM Show existing collections for reference
echo Your existing collections:
echo   - medmax_pubmed (211269 vectors, 1536 dim)
echo   - medmax_pubmed_full (273518 vectors, 1536 dim)
echo.

set /p confirm="Proceed? (y/N): "
if /i not "!confirm!"=="y" goto end

if defined LIMIT (
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --use-parquet --limit %LIMIT% %USE_HF% %DEVICE_FLAG%"
) else (
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --use-parquet %USE_HF% %DEVICE_FLAG%"
)
goto end

:end
if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo  FAILED TO POPULATE VECTOR STORE!
    echo ============================================================
    if not "%USE_HF%"=="" (
        echo.
        echo Troubleshooting for HuggingFace Embeddings:
        echo   1. Install: pip install sentence-transformers torch
        echo   2. Check GPU: python -c "import torch; print(torch.cuda.is_available())"
        echo   3. Try CPU mode instead of GPU
        echo.
    )
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  VECTOR STORE POPULATED SUCCESSFULLY!
echo ============================================================
echo.
echo Collection created: %COLLECTION_NAME%
