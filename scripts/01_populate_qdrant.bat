@echo off
setlocal enabledelayedexpansion
REM Populate Qdrant with PubMed data
set COLLECTION_NAME=medmax_pubmed
set DATA_PATH=data/PubMed-compact/pubmedqa.jsonl

REM Activate Python environment and set python command
call cmd /c "micromamba activate medproj"
set PYTHON_CMD=cmd /c "micromamba activate medproj && python"

echo Starting MedMax Vector Store Population...
echo.
echo Collection: %COLLECTION_NAME%
echo Data source: %DATA_PATH%
echo.
echo Choose data source:
echo [A] Original JSONL data (data/PubMed-compact/pubmedqa.jsonl)
echo [B] New PubMedQA parquet files (data/*.parquet)
echo.
set /p data_choice="Enter data source choice (A/B): "

if /i "%data_choice%"=="B" goto parquet_mode
if /i "%data_choice%"=="b" goto parquet_mode

REM Original JSONL mode
echo.
echo Using JSONL data source: %DATA_PATH%
echo.
echo Choose mode:
echo [1] Test mode (10 records only)
echo [2] Full dataset
echo [3] Custom limit
echo [4] Force reindex (recreate collection)
echo [5] Custom collection name
echo [6] Delete collection and populate with custom limit
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto test_mode
if "%choice%"=="2" goto full_dataset
if "%choice%"=="3" goto custom_limit
if "%choice%"=="4" goto force_reindex
if "%choice%"=="5" goto custom_collection
if "%choice%"=="6" goto delete_and_custom
goto invalid_choice

:parquet_mode
echo.
echo Using PubMedQA parquet files (unlabeled + labeled + artificial)
echo.
echo Choose mode:
echo [1] Test mode (10 records per dataset)
echo [2] Full datasets
echo [3] Custom limit per dataset
echo [4] Force reindex (recreate collection)
echo [5] Custom collection name
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto parquet_test_mode
if "%choice%"=="2" goto parquet_full_dataset
if "%choice%"=="3" goto parquet_custom_limit
if "%choice%"=="4" goto parquet_force_reindex
if "%choice%"=="5" goto parquet_custom_collection
goto invalid_choice

:parquet_test_mode
echo Running in TEST MODE with 10 records per dataset...
cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --use-parquet --limit 10"
goto end

:parquet_full_dataset
echo Running with FULL DATASETS...
cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --use-parquet"
goto end

:parquet_custom_limit
set /p custom_limit="Enter number of records per dataset: "
echo Running with !custom_limit! records per dataset...
cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --use-parquet --limit !custom_limit!"
goto end

:parquet_force_reindex
echo Running with FORCE REINDEX (will recreate collection)...
cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --use-parquet --force-reindex"
goto end

:parquet_custom_collection
set /p custom_collection="Enter collection name: "
echo Choose sub-option:
echo [a] Test mode (10 records per dataset)
echo [b] Full datasets
echo [c] Force reindex
set /p sub_choice="Enter sub-choice (a-c): "

if "!sub_choice!"=="a" (
    echo Running with custom collection '!custom_collection!' - 10 records per dataset...
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name !custom_collection! --use-parquet --limit 10"
) else if "!sub_choice!"=="b" (
    echo Running with custom collection '!custom_collection!' - full datasets...
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name !custom_collection! --use-parquet"
) else if "!sub_choice!"=="c" (
    echo Running with custom collection '!custom_collection!' - force reindex...
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name !custom_collection! --use-parquet --force-reindex"
) else (
    echo Invalid sub-choice. Using default test mode...
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name !custom_collection! --use-parquet --limit 10"
)
goto end

:test_mode
echo Running in TEST MODE with 10 records...
cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --data-path %DATA_PATH% --limit 10"
goto end

:full_dataset
echo Running with FULL DATASET...
cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --data-path %DATA_PATH%"
goto end

:custom_limit
set /p custom_limit="Enter number of records: "
echo Running with !custom_limit! records...
cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --data-path %DATA_PATH% --limit !custom_limit!"
goto end

:force_reindex
echo Running with FORCE REINDEX (will recreate collection)...
cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --data-path %DATA_PATH% --force-reindex"
goto end

:custom_collection
set /p custom_collection="Enter collection name: "
echo Choose sub-option:
echo [a] Test mode (10 records)
echo [b] Full dataset
echo [c] Force reindex
set /p sub_choice="Enter sub-choice (a-c): "

if "!sub_choice!"=="a" (
    echo Running with custom collection '!custom_collection!' - 10 records...
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name !custom_collection! --data-path %DATA_PATH% --limit 10"
) else if "!sub_choice!"=="b" (
    echo Running with custom collection '!custom_collection!' - full dataset...
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name !custom_collection! --data-path %DATA_PATH%"
) else if "!sub_choice!"=="c" (
    echo Running with custom collection '!custom_collection!' - force reindex...
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name !custom_collection! --data-path %DATA_PATH% --force-reindex"
) else (
    echo Invalid sub-choice. Using default test mode...
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name !custom_collection! --data-path %DATA_PATH% --limit 10"
)
goto end

:delete_and_custom
set /p custom_limit="Enter number of records to populate: "
echo Deleting collection 'medmax_pubmed' and populating with !custom_limit! records...
echo WARNING: This will delete all existing data in the collection!
set /p confirm="Are you sure? (y/N): "
if /i "!confirm!"=="y" (
    echo Proceeding with deletion and repopulation...
    cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --data-path %DATA_PATH% --limit !custom_limit! --force-reindex"
) else (
    echo Operation cancelled.
)
goto end

:invalid_choice
echo Invalid choice. Running in test mode...
cmd /c "micromamba activate medproj && python -m src.vector_store.main populate --collection-name %COLLECTION_NAME% --data-path %DATA_PATH% --limit 10"

:end
if %errorlevel% neq 0 (
    echo Failed to populate vector store!
    pause
    exit /b 1
)
echo Vector store populated successfully!
pause
