# 🚀 QUICK START: Generate All Report Visualizations

## ⚡ ONE COMMAND TO RULE THEM ALL

Just run this single command to execute the complete pipeline:

```bash
python generate_visualizations.py
```

## 📦 What This Script Does

### Automatically Runs:
1. **ML-based Signal Optimization** (`final.py`)
   - Trains ML models on synthetic data
   - Generates optimized signal timing plan
   - Saves to `outputs/ml_signal_plan.json`

2. **Webster-based Optimization** (`webster.py`)
   - Calculates traditional Webster signal timing
   - Saves to `outputs/webster_signal_plan.json`

3. **Performance Comparison** (`compare_signal_plans.py`)
   - Compares ML vs Webster on delay, queue, throughput, LOS
   - Saves to `outputs/signalplan_comparision.json`

4. **SUMO Microsimulation** (`sumo_simulation.py`)
   - Validates both plans in traffic simulator
   - Saves to `outputs/sumo_comparison.json`

5. **Generates 18 Visualizations + 1 Dataset:**
   - Phase diagrams (2)
   - Performance comparison charts (4)
   - Synthetic training dataset CSV (1)
   - Dataset distribution plots (5)
   - SUMO comparison charts (7)

## 📊 Output Location

All files saved to: `outputs/visualizations/`

### Generated Files:
```
outputs/visualizations/
├── phase_diagram_ml.png                    # ML timing diagram
├── phase_diagram_webster.png               # Webster timing diagram
├── delay_comparison.png                    # Delay: ML vs Webster
├── queue_length_comparison.png             # Queue length comparison
├── throughput_comparison.png               # Throughput comparison
├── average_metrics_comparison.png          # Overall performance
├── synthetic_training_dataset.csv          # Training data (1000 samples) ⭐
├── pcu_distribution.png                    # PCU histogram
├── time_of_day_distribution.png            # Traffic by hour
├── cycle_length_distribution.png           # Cycle length histogram
├── weather_impact.png                      # Weather effects
├── day_of_week_impact.png                  # Weekday patterns
├── sumo_average_delay.png                  # SUMO: Delay
├── sumo_average_waiting_time.png           # SUMO: Waiting time
├── sumo_average_travel_time.png            # SUMO: Travel time
├── sumo_average_time_loss.png              # SUMO: Time loss
├── sumo_time_metrics_comparison.png        # SUMO: All metrics
├── sumo_throughput_comparison.png          # SUMO: Throughput
└── (18 total files)
```

## ⚙️ Configuration

### Hardcoded PCU Values (Fallback)
If `outputs/intersection_summary.json` doesn't exist, the script uses:

```python
PCU_FALLBACK = {"N": 2542.0, "S": 2760.0, "E": 0.0, "W": 1500.0}  # 3-way T-junction
```

**To change this:** Edit line 19 in `generate_visualizations.py`

For 4-way intersection, uncomment line 20:
```python
# PCU_FALLBACK = {"N": 2880.0, "S": 2760.0, "E": 1560.0, "W": 3480.0}
```

## 🔧 Prerequisites

Install dependencies first:
```bash
pip install -r requirements.txt
```

Ensure SUMO is installed and in PATH:
```bash
sumo --version
```

## ⏱️ Expected Runtime

- **ML Optimization:** ~10-15 seconds
- **Webster Optimization:** ~2-3 seconds
- **Comparison:** ~1 second
- **SUMO Simulation:** ~30-60 seconds
- **Visualization Generation:** ~20-30 seconds

**Total: ~1-2 minutes** ⏰

## 📋 Next Steps

After running the script:

1. **Copy images to report folder:**
   ```bash
   copy outputs\visualizations\*.png report\images\
   copy outputs\visualizations\*.csv report\images\
   ```

2. **Capture manual screenshots:**
   - Streamlit UI (26 images) - See `VISUALIZATION_GUIDE.md`
   - SUMO GUI (16 images) - See `VISUALIZATION_GUIDE.md`
   - JSON/CSV files (10 images)
   - Terminal outputs (5 images)

3. **Compile LaTeX report:**
   ```bash
   cd report
   pdflatex main.tex
   ```

## 🐛 Troubleshooting

### "kaleido not found"
```bash
pip install kaleido==0.2.1
```

### "SUMO not found"
- Install SUMO from https://sumo.dlr.de/
- Add to PATH: `C:\Program Files (x86)\Eclipse\Sumo\bin`

### "intersection_summary.json not found"
- Script will use hardcoded PCU values
- Or run Streamlit app first to generate JSON:
  ```bash
  streamlit run main.py
  ```

### Script fails at any step
- Check terminal output for specific error
- Steps are independent - other visualizations will still be generated
- Run individual scripts manually if needed

## 📖 Documentation

- **Complete guide:** `VISUALIZATION_GUIDE.md`
- **Image checklist:** `IMAGE_CHECKLIST.md` (78 total images)
- **SUMO setup:** `SUMO_SETUP.md`
- **SUMO GUI guide:** `SUMO_GUI_GUIDE.md`

## 💡 Pro Tips

1. **Check output folder size:** Should be ~15-20 MB after completion
2. **Verify file count:** 18 files in `outputs/visualizations/`
3. **View generated images:** Open any PNG to verify quality
4. **Check CSV:** Open in Excel to verify synthetic data
5. **Review JSON files:** Check `outputs/*.json` for all metrics

## ✅ Success Checklist

After running `python generate_visualizations.py`, verify:

- [ ] Script completed without errors
- [ ] `outputs/ml_signal_plan.json` exists
- [ ] `outputs/webster_signal_plan.json` exists
- [ ] `outputs/signalplan_comparision.json` exists
- [ ] `outputs/sumo_comparison.json` exists
- [ ] `outputs/visualizations/` contains 18 files
- [ ] All PNG images are visible and clear
- [ ] CSV contains 1000 rows

**If all checked: You're ready for manual screenshots!** 🎉

---

**Questions?** See `VISUALIZATION_GUIDE.md` or `IMAGE_CHECKLIST.md`

