# Signal Optimiser (PSU → Timings)

This repository estimates signal timings from approach-level traffic demand (Passenger Car Units, PSU). The main script is `final.py`, which reads approach totals and outputs a 2‑phase plan (NS vs EW) with per‑approach Green/Amber/Red and a phase‑timeline chart.

## What `final.py` does
- Reads approach PSUs (N,S,E,W) from `outputs/intersection_summary.json` produced by the Streamlit app.
- Computes the signal cycle length using a Webster-based relationship between degree of saturation (Y) and cycle.
- Splits the effective green between NS and EW proportionally to demand; then splits within each approach proportionally.
- Renders an exclusive phase timeline (when NS is green/amber, EW is red, and vice versa) with optional all‑red between phases.

## Inputs
`outputs/intersection_summary.json` (example):
```json
{
  "NB": {"vehicle_counts": {"car": 120}, "total_pcu": 194.0, "duration": 600.0},
  "SB": {"vehicle_counts": {"car": 100}, "total_pcu": 161.5, "duration": 600.0},
  "EB": {"vehicle_counts": {"car": 150}, "total_pcu": 244.0, "duration": 600.0},
  "WB": {"vehicle_counts": {"car": 130}, "total_pcu": 205.5, "duration": 600.0}
}
```
`final.py` maps NB/SB/EB/WB → N/S/E/W via `total_pcu`. Missing approaches are treated as absent (works for T‑junctions).

## How to run
```bash
# in a virtualenv
pip install -r requirements.txt
python final.py
```
- If `outputs/intersection_summary.json` exists, it will be used automatically and echoed in the console.
- The script prints cycle, per‑approach G/A/R, and shows a Plotly phase timeline.

## Key parameters and assumptions
- DEFAULT_LANES: default lanes per approach for capacity (used in data synthesis and Webster ML; can be adjusted).
- SAT_PER_LANE: 1800 PCU/hr/lane (typical design value; tune per context).
- DEFAULT_LOST_TIME: 12 s (startup + change intervals per cycle).
- YELLOW (Amber): 3 s (fixed per approach in current model).
- ALL_RED: 2 s between phases (safety clearance).

## Method (high level)
1) Degree of saturation and cycle (Webster):
   - Compute total demand `Q = N + S + E + W` (PSU/hr).
   - Compute capacity `C_cap = (#present_approaches * lanes * sat_per_lane)`.
   - Degree of saturation `Y = Q / C_cap` (capped at 0.95).
   - Cycle `C = (1.5*L + 5) / (1 - Y)`, clamped to [60, 180] s.
2) Green splits:
   - Effective green `G_eff = C - L`.
   - Phase split: `G_NS = G_eff * (NS/(NS+EW))`, `G_EW = G_eff - G_NS`.
   - Approach split (proportional within the phase group).
3) Timings:
   - Each approach: Green, Amber (fixed), Red = C − (G + Amber).
   - Phase timeline: NS (green→amber) → All‑red → EW (green→amber) → All‑red.

## Notes on Indian practice (IRC)
- Passenger Car Units (PCU) factors and saturation flows are drawn from IRC guidance; common references include:
  - IRC:106 — Guidelines for Capacity of Urban Roads in Plain Areas.
  - IRC:SP:41 — Guidelines for the Design of At‑Grade Intersections in Rural & Urban Areas.
  - IRC:SP:90 — Manual on Road Traffic Signal Control.
- Webster’s method is standard for fixed‑time optimisation; Indian practice often adapts:
  - Saturation flow per lane (typical 1800 PCU/hr/lane, adjust for local conditions).
  - Lost time (startup + change intervals), amber and all‑red durations per IRC signal design tables.

Please confirm the exact edition/sections required by your course; the above are the commonly cited manuals for Indian signal design. Replace constants to match the jurisdiction and site calibration.

# Vehicle Counting & PCU Calculator

A machine learning-based application that detects vehicles in video footage and calculates Passenger Car Units (PCU) based on IRC 106-1990 standards using YOLOv8 and Streamlit.

## Features

- 🚗 **Vehicle Detection**: Uses YOLOv8 to detect cars, trucks, buses, motorcycles, and bicycles
- 📊 **PCU Calculation**: Applies IRC 106-1990 PCU factors automatically
- 📈 **Real-time Analysis**: Frame-by-frame processing with progress tracking
- 📊 **Visualization**: Interactive charts and graphs using Plotly
- 💾 **Data Export**: Download results as CSV files
- 🎥 **Multiple Formats**: Supports MP4, AVI, MOV, MKV video formats

## PCU Factors (IRC 106-1990)

| Vehicle Type | PCU Factor |
|--------------|------------|
| Car          | 1.0        |
| Truck        | 3.0        |
| Bus          | 3.0        |
| Motorcycle   | 0.5        |
| Bicycle      | 0.5        |

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd miniproject
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

## Usage

### 1. Start the Application
```bash
# Make sure your virtual environment is activated
streamlit run main.py
```

### 2. Using the Application
1. Open your web browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)
2. Upload a video file using the file uploader
3. Click "Process Video" to start analysis
4. View results including:
   - Vehicle counts by type
   - Total PCU calculation
   - Interactive charts
   - Frame-by-frame analysis
5. Download results as CSV files

## Project Structure

```
miniproject/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .venv/              # Virtual environment (created during setup)
```

## Technical Details

- **Model**: YOLOv8n (nano) - lightweight and fast
- **Detection Confidence**: >50% threshold for reliable detections
- **Processing**: Frame-by-frame analysis with progress tracking
- **UI Framework**: Streamlit for web-based interface
- **Visualization**: Plotly for interactive charts

## Supported Video Formats

- MP4
- AVI
- MOV
- MKV

## Output Files

The application generates two types of CSV files:

1. **Summary CSV**: Contains vehicle counts, PCU factors, and total PCU
2. **Frame Data CSV**: Contains frame-by-frame vehicle detection data

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure you have a stable internet connection for the first run (YOLOv8 model will be downloaded automatically)

2. **Memory Issues**: For large videos, consider:
   - Using shorter video clips
   - Processing in smaller segments
   - Ensuring sufficient RAM (8GB+ recommended)

3. **CUDA Issues**: The application works with CPU by default. For GPU acceleration, ensure CUDA is properly installed.

### Performance Tips

- Use shorter video clips for faster processing
- Ensure good lighting in videos for better detection accuracy
- Close other applications to free up system resources

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 