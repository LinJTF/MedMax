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
echo Choose mode:
echo [1] Test mode (10 PubMed records)
echo [2] PubMed full dataset
echo [3] PubMed custom limit
echo [4] PubMed force reindex
echo [5] PubMed custom collection name
echo [6] PubMed delete + custom limit
echo [7] Collect Wikipedia health titles (writes data/wiki_health_titles.txt)
echo [8] Populate Wikipedia-only collection from titles file
echo [9] Populate combined PubMed + Wikipedia into new collection
echo.
set /p choice="Enter choice (1-9): "

if "%choice%"=="1" goto test_mode
if "%choice%"=="2" goto full_dataset
if "%choice%"=="3" goto custom_limit
if "%choice%"=="4" goto force_reindex
if "%choice%"=="5" goto custom_collection
if "%choice%"=="6" goto delete_and_custom
if "%choice%"=="7" goto collect_wiki
if "%choice%"=="8" goto populate_wiki
if "%choice%"=="9" goto populate_combined
goto invalid_choice

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

:collect_wiki
echo Collecting Wikipedia health-related titles (default limits)...
cmd /c "micromamba activate medproj && python -m src.vector_store.wiki_collect --output data/wiki_health_titles.txt"
echo Done collecting. File: data/wiki_health_titles.txt
goto end

:populate_wiki
set /p WIKI_COLLECTION="Enter target Wikipedia collection name (default: medmax_wiki_health): "
if "%WIKI_COLLECTION%"=="" set WIKI_COLLECTION=medmax_wiki_health
if not exist data\wiki_health_titles.txt (
    echo Missing data/wiki_health_titles.txt. Run option 7 first or create it.
    goto end
)
echo Populating Wikipedia-only collection %WIKI_COLLECTION% ...
cmd /c "micromamba activate medproj && python -m src.vector_store.main populate_wiki --collection-name %WIKI_COLLECTION% --wikipedia-titles-file data/wiki_health_titles.txt"
goto end

:populate_combined
set /p COMBINED_COLLECTION="Enter combined collection name (default: medmax_pubmed_wiki): "
if "%COMBINED_COLLECTION%"=="" set COMBINED_COLLECTION=medmax_pubmed_wiki
if not exist data\wiki_health_titles.txt (
    echo Missing data/wiki_health_titles.txt. Run option 7 first or create it.
    goto end
)
echo Building combined collection %COMBINED_COLLECTION% from PubMed + Wikipedia...
cmd /c "micromamba activate medproj && python -m src.vector_store.main populate_combined --wikipedia-titles-file data/wiki_health_titles.txt --combined-collection-name %COMBINED_COLLECTION%"
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
