# 📊 Visualization Generation Guide

This guide explains how to generate all visualizations and screenshots needed for your report.

## 🚀 Quick Start

### 1. Install Dependencies

First, install the required package for image export:

```bash
pip install kaleido
```

Or update all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (All-in-One) ⭐

**NEW: Single command to run everything!**

```bash
python generate_visualizations.py
```

This script now runs the complete pipeline:
1. ✅ ML-based optimization (final.py)
2. ✅ Webster-based optimization (webster.py)
3. ✅ Performance comparison (compare_signal_plans.py)
4. ✅ SUMO simulation (sumo_simulation.py)
5. ✅ Generates all PNG images and CSV datasets

All outputs will be saved to `outputs/visualizations/`

### Alternative: Run Scripts Individually

If you prefer to run each step separately:

```bash
# Step 1: Generate ML signal plan
python final.py

# Step 2: Generate Webster signal plan  
python webster.py

# Step 3: Compare both plans
python compare_signal_plans.py

# Step 4: Run SUMO simulation
python sumo_simulation.py

# Step 5: Generate visualizations only
# (Skip if you already ran generate_visualizations.py above)
```

---

## 📁 Generated Files

### Phase Diagrams
- `phase_diagram_ml.png` - ML-based signal timing phase diagram
- `phase_diagram_webster.png` - Webster-based signal timing phase diagram

### Performance Comparison Charts
- `delay_comparison.png` - Delay comparison by approach
- `queue_length_comparison.png` - Queue length comparison
- `throughput_comparison.png` - Throughput comparison
- `average_metrics_comparison.png` - Overall average metrics

### Dataset Files
- `synthetic_training_dataset.csv` - 1000 synthetic training samples with all features

### Dataset Distribution Plots
- `pcu_distribution.png` - PCU value distribution across approaches
- `time_of_day_distribution.png` - Traffic distribution by hour
- `cycle_length_distribution.png` - Optimized cycle length distribution
- `weather_impact.png` - Average PCU by weather condition
- `day_of_week_impact.png` - Average PCU by day of week

### SUMO Simulation Results
- `sumo_average_delay.png` - Average delay comparison
- `sumo_average_waiting_time.png` - Average waiting time comparison
- `sumo_average_travel_time.png` - Average travel time comparison
- `sumo_average_time_loss.png` - Average time loss comparison
- `sumo_total_throughput.png` - Total throughput comparison
- `sumo_time_metrics_comparison.png` - All time-based metrics
- `sumo_throughput_comparison.png` - Throughput-specific chart

---

## 🖼️ Streamlit UI Screenshots

To capture Streamlit interface screenshots:

### 1. Start the Streamlit App

```bash
streamlit run main.py
```

### 2. Navigate Through Features

Capture screenshots of:

#### Main Detection Page
- Home page with file uploaders
- Video upload interface for all 4 approaches (NB, SB, EB, WB)
- ROI mask upload section
- Detection parameters settings

#### Detection Results
- Annotated video frames with YOLO bounding boxes (one per approach)
- Vehicle counts bar chart
- Time series detection plot
- PCU summary table
- JSON output display

#### Signal Timing Simulator
- "Start Simulation" page interface
- Phase diagram displays (both ML and Webster)
- Cycle preview charts
- Signal timing parameters table

#### Full Simulation
- "Run Full Simulation" page
- Cycle length history chart
- Green splits history chart
- Comparison results

### 3. Use Windows Snipping Tool

Press `Win + Shift + S` to capture screenshots

---

## 📸 Additional Screenshots Needed

### Code/Data Files
Capture screenshots of these files in VS Code:

1. **JSON Files:**
   - `outputs/intersection_summary.json`
   - `outputs/ml_signal_plan.json`
   - `outputs/webster_signal_plan.json`
   - `outputs/signalplan_comparision.json`
   - `outputs/sumo_comparison.json`

2. **CSV Files:**
   - `outputs/NB_summary.csv`
   - `outputs/visualizations/synthetic_training_dataset.csv`

3. **SUMO XML Files:**
   - `intersection.nod.xml`
   - `intersection.edg.xml`
   - `sumo_network.net.xml`
   - `ml_traffic_lights.add.xml`
   - `webster_traffic_lights.add.xml`
   - `routes.rou.xml`

### SUMO GUI Screenshots

1. **Open SUMO GUI:**
   ```bash
   sumo-gui -c ml_config.sumocfg
   ```

2. **Capture:**
   - Network visualization
   - Traffic light states
   - Vehicles on network
   - Simulation running

3. **Repeat for Webster:**
   ```bash
   sumo-gui -c webster_config.sumocfg
   ```

### Terminal Outputs

Capture terminal output from:
- `python final.py` (showing ML training and optimization)
- `python webster.py` (showing Webster calculations)
- `python compare_signal_plans.py` (showing comparison table)
- `python sumo_simulation.py` (showing simulation metrics)
- `python generate_visualizations.py` (showing file generation)

---

## 🎯 Where to Use These Images

### Chapter 3: Methodology
- Project workflow diagram (create manually)
- YOLO detection samples

### Chapter 4: IRC Standards
- IRC document covers (download from references)
- PCU table screenshot

### Chapter 5: ML Optimization
- `synthetic_training_dataset.csv` screenshot
- All dataset distribution plots
- Model training visualizations

### Chapter 6: YOLO Optimization
- All Streamlit UI screenshots
- YOLO detection frames
- Vehicle classification charts
- PCU calculation results
- Phase diagrams (ML and Webster)
- JSON output files

### Chapter 7: SUMO Validation
- All SUMO network files
- SUMO GUI screenshots
- All SUMO comparison charts
- Performance metrics tables

### Chapter 8: Conclusion
- Key comparison charts
- Overall performance summary

---

## 📝 Image Checklist

Use this checklist to track your progress:

**Automated (✅ Generated by script):**
- [ ] Phase diagrams (2)
- [ ] Performance comparison charts (4)
- [ ] Synthetic dataset CSV (1)
- [ ] Dataset distribution plots (5)
- [ ] SUMO comparison charts (7)

**Manual Screenshots (📸 Capture yourself):**
- [ ] Streamlit home page
- [ ] Streamlit video upload
- [ ] Streamlit detection results (4 approaches)
- [ ] Streamlit phase diagrams in UI
- [ ] JSON file screenshots (5)
- [ ] CSV file screenshots (2)
- [ ] SUMO XML file screenshots (6)
- [ ] SUMO GUI screenshots (4)
- [ ] Terminal outputs (5)

**External/Created:**
- [ ] Intersection location photos (2)
- [ ] IRC standard covers (2)
- [ ] Workflow diagram (1)
- [ ] System architecture diagram (1)

**Total: ~55 images**

---

## 💡 Tips

1. **High Quality:** Use scale=2 in plotly exports for high-resolution images
2. **Consistency:** Use the same color scheme (ML=#3498db blue, Webster=#e74c3c red)
3. **File Names:** Use descriptive names with prefixes (`ch5_`, `sumo_`, etc.)
4. **Size:** Aim for 900-1200px width for good readability
5. **Format:** PNG is preferred for screenshots and charts

---

## 🔧 Troubleshooting

### Kaleido Installation Issues

If `pip install kaleido` fails:

```bash
# Try with specific version
pip install kaleido==0.2.1

# Or use conda
conda install -c conda-forge python-kaleido
```

### Plotly Image Export Errors

If you get "Image export using the kaleido package requires the kaleido package":

1. Uninstall and reinstall:
   ```bash
   pip uninstall kaleido
   pip install kaleido==0.2.1 --no-cache-dir
   ```

2. Restart your Python kernel/terminal

### SUMO Not Found

If SUMO commands don't work:

1. Verify installation: `sumo --version`
2. Add to PATH (see SUMO_SETUP.md)
3. Use full path: `"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"`

---

## ✅ Verification

After running `generate_visualizations.py`, verify all files exist:

```bash
dir outputs\visualizations
```

You should see:
- 2 phase diagram PNGs
- 4 comparison chart PNGs  
- 5 distribution plot PNGs
- 7 SUMO comparison PNGs
- 1 CSV dataset file

**Total: 18 automated files + ~37 manual screenshots = ~55 images for report**

---

Need help? Check the main README.md or run scripts with `-h` flag for help.

