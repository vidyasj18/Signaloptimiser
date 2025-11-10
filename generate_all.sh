#!/bin/bash
# ============================================================================
# All-in-One Pipeline Executor (Linux/Mac)
# Runs complete signal optimization pipeline and generates visualizations
# ============================================================================

echo ""
echo "================================================================================"
echo "  SIGNAL OPTIMIZATION PIPELINE - Complete Execution"
echo "================================================================================"
echo ""
echo "This script will run:"
echo "  1. ML-based optimization"
echo "  2. Webster-based optimization"
echo "  3. Performance comparison"
echo "  4. SUMO microsimulation"
echo "  5. Visualization generation"
echo ""
echo "Expected runtime: 1-2 minutes"
echo ""
read -p "Press Enter to continue..."

echo ""
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found! Please install Python 3.8 or higher."
    exit 1
fi
python3 --version

echo ""
echo "[2/5] Checking dependencies..."
if ! python3 -c "import plotly, pandas, numpy, sklearn" 2>/dev/null; then
    echo "WARNING: Some dependencies missing. Installing..."
    pip3 install -r requirements.txt
fi

echo ""
echo "[3/5] Checking kaleido (for PNG export)..."
if ! python3 -c "import kaleido" 2>/dev/null; then
    echo "Installing kaleido for image export..."
    pip3 install kaleido==0.2.1
fi

echo ""
echo "[4/5] Creating output directories..."
mkdir -p outputs/visualizations

echo ""
echo "[5/5] Running complete pipeline..."
echo "================================================================================"
python3 generate_visualizations.py

if [ $? -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: Pipeline encountered an issue!"
    echo "Check the error messages above."
    echo "================================================================================"
    exit 1
fi

echo ""
echo "================================================================================"
echo "SUCCESS! Pipeline completed successfully."
echo "================================================================================"
echo ""
echo "Generated files location: outputs/visualizations/"
echo ""
echo "To view results:"
echo "  - Check outputs/visualizations/ for PNG images and CSV"
echo "  - Check outputs/ for JSON files"
echo ""
echo "Next steps:"
echo "  1. Copy images to report/images/"
echo "  2. Capture Streamlit screenshots (see VISUALIZATION_GUIDE.md)"
echo "  3. Compile LaTeX report (cd report && pdflatex main.tex)"
echo ""
echo "Output directory: outputs/visualizations/"

