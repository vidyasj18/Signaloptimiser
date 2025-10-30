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