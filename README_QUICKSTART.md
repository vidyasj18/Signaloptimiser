# 🚀 Quick Start Guide

**Everything is ready! Here's how to generate all visualizations.**

---

## ⚡ TL;DR (Too Long; Didn't Read)

```bash
# One command to generate everything:
python generate_visualizations.py
```

**That's it!** This runs the complete pipeline and generates 18 PNG images + 1 CSV dataset.

---

## 📊 What Gets Generated

### Output Location: `outputs/visualizations/`

**18 Images + 1 Dataset:**
- 2 Phase diagrams (ML & Webster)
- 4 Performance comparison charts
- 6 Dataset distribution plots
- 1 Synthetic training dataset (CSV)
- 7 SUMO simulation comparison charts

---

## 🔍 Verify Setup First

```bash
python check_setup.py
```

You should see:
```
✅ CRITICAL COMPONENTS: ALL OK
🎉 Your setup is ready!
```

---

## 🎯 Full Workflow

### Step 1: Generate Visualizations (1-2 minutes)
```bash
python generate_visualizations.py
```

This automatically runs:
1. ML optimization (`final.py`)
2. Webster optimization (`webster.py`)
3. Performance comparison (`compare_signal_plans.py`)
4. SUMO simulation (`sumo_simulation.py`)
5. Visualization generation

### Step 2: Copy to Report
```bash
copy outputs\visualizations\*.png report\images\
copy outputs\visualizations\*.csv report\images\
```

### Step 3: Compile LaTeX Report
```bash
cd report
pdflatex main.tex
```

---

## 📱 Alternative: Manual Screenshots

For images not auto-generated, see `IMAGE_CHECKLIST.md`:

### Streamlit UI (26 images)
```bash
streamlit run main.py
```
Screenshot the UI pages.

### SUMO GUI (16 images)
```bash
sumo-gui -c ml_config.sumocfg
```
Screenshot the network and simulation.

---

## 🐛 Troubleshooting

### "No module named 'kaleido'"
```bash
pip install kaleido==0.2.1
```

### "SUMO not found"
```bash
# Add to PATH or reinstall from:
https://sumo.dlr.de/docs/Downloads.php
```

### "Files not generated"
```bash
# Clean and retry:
del *.xml *.sumocfg
python generate_visualizations.py
```

---

## 📚 Documentation

- **Setup verification:** `python check_setup.py`
- **Complete guide:** `VISUALIZATION_GUIDE.md`
- **Image checklist:** `IMAGE_CHECKLIST.md` (78 total images)
- **Setup report:** `SETUP_VERIFIED.md`
- **Main README:** `README.md`

---

## ✅ Quick Checklist

- [ ] Run `python check_setup.py` ✅
- [ ] Run `python generate_visualizations.py` ✅
- [ ] Check `outputs/visualizations/` (18 files) ✅
- [ ] Copy to `report/images/` ✅
- [ ] Capture manual screenshots (see checklist) ⏳
- [ ] Compile LaTeX report ⏳

---

**Your environment is ready. Just run the commands!** 🎉

