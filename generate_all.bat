@echo off
REM ============================================================================
REM All-in-One Pipeline Executor
REM Runs complete signal optimization pipeline and generates visualizations
REM ============================================================================

echo.
echo ================================================================================
echo   SIGNAL OPTIMIZATION PIPELINE - Complete Execution
echo ================================================================================
echo.
echo This script will run:
echo   1. ML-based optimization
echo   2. Webster-based optimization  
echo   3. Performance comparison
echo   4. SUMO microsimulation
echo   5. Visualization generation
echo.
echo Expected runtime: 1-2 minutes
echo.
pause

echo.
echo [1/5] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo.
echo [2/5] Checking dependencies...
python -c "import plotly, pandas, numpy, sklearn" 2>nul
if errorlevel 1 (
    echo WARNING: Some dependencies missing. Installing...
    pip install -r requirements.txt
)

echo.
echo [3/5] Checking kaleido (for PNG export)...
python -c "import kaleido" 2>nul
if errorlevel 1 (
    echo Installing kaleido for image export...
    pip install kaleido==0.2.1
)

echo.
echo [4/5] Creating output directories...
if not exist "outputs" mkdir outputs
if not exist "outputs\visualizations" mkdir outputs\visualizations

echo.
echo [5/5] Running complete pipeline...
echo ================================================================================
python generate_visualizations.py

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ERROR: Pipeline encountered an issue!
    echo Check the error messages above.
    echo ================================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Pipeline completed successfully.
echo ================================================================================
echo.
echo Generated files location: outputs\visualizations\
echo.
echo To view results:
echo   - Check outputs\visualizations\ for PNG images and CSV
echo   - Check outputs\ for JSON files
echo.
echo Next steps:
echo   1. Copy images to report\images\
echo   2. Capture Streamlit screenshots (see VISUALIZATION_GUIDE.md)
echo   3. Compile LaTeX report (cd report, then pdflatex main.tex)
echo.
echo Press any key to open output folder...
pause >nul
explorer outputs\visualizations

