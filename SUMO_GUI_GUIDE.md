# SUMO GUI Usage Guide

This guide explains how to use the generated SUMO files in SUMO GUI to visualize and compare the two signal plans.

## Generated Files

You have 6 files:
1. **sumo_network.net.xml** - The intersection network (shared)
2. **routes.rou.xml** - Vehicle routes (shared)
3. **ml_traffic_lights.add.xml** - ML-based signal timings
4. **webster_traffic_lights.add.xml** - Webster-based signal timings
5. **ml_config.sumocfg** - ML simulation configuration
6. **webster_config.sumocfg** - Webster simulation configuration

## Method 1: Load Configuration File (Easiest)

### For ML-based Plan:
1. Open SUMO GUI:
   ```bash
   sumo-gui
   ```

2. Load the ML configuration:
   - Click **File → Open Simulation** (or press `Ctrl+O`)
   - Navigate to your project folder
   - Select **`ml_config.sumocfg`**
   - Click **Open**

3. The simulation will load with:
   - Network: `sumo_network.net.xml`
   - Routes: `routes.rou.xml`
   - Traffic lights: `ml_traffic_lights.add.xml`

4. Click **Play** (▶) to start simulation

### For Webster-based Plan:
1. In SUMO GUI, click **File → Open Simulation**
2. Select **`webster_config.sumocfg`**
3. Click **Play** to start

## Method 2: Load Files Manually

If you want more control:

1. Open SUMO GUI:
   ```bash
   sumo-gui
   ```

2. Load network:
   - Click **File → Open Network**
   - Select **`sumo_network.net.xml`**

3. Load routes:
   - Click **File → Load Routes**
   - Select **`routes.rou.xml`**

4. Load traffic lights:
   - For ML: Click **File → Load Additionals** → Select **`ml_traffic_lights.add.xml`**
   - For Webster: Click **File → Load Additionals** → Select **`webster_traffic_lights.add.xml`**

5. Start simulation:
   - Click **Simulation → Start** (or press `Ctrl+Space`)

## Comparing Both Plans Side-by-Side

### Option A: Two SUMO GUI Windows
1. Open first SUMO GUI window:
   ```bash
   sumo-gui -c ml_config.sumocfg
   ```

2. Open second SUMO GUI window:
   ```bash
   sumo-gui -c webster_config.sumocfg
   ```

3. Arrange windows side-by-side
4. Start both simulations simultaneously

### Option B: Run One at a Time
1. Run ML plan, observe, note metrics
2. Close and open Webster plan
3. Compare results

## Useful SUMO GUI Features

### View Traffic Lights
- Right-click on the intersection center
- Select **"Open Traffic Light Dialog"**
- See current phase, timing, and phase sequence

### View Statistics
- Click **View → Show Vehicle Parameter**
- Select metrics to display (delay, speed, etc.)
- Hover over vehicles to see values

### View Delays
- Click **View → Show Vehicle Parameter → Waiting Time**
- Vehicles will show color-coded delays
- Red = high delay, Green = low delay

### View Queues
- Click **View → Show Vehicle Parameter → Speed**
- Stopped vehicles (speed = 0) indicate queues

### Pause and Step
- **Pause** (⏸): Pause simulation
- **Step** (⏭): Advance one simulation step
- **Delay**: Adjust simulation speed (slower = easier to observe)

### Take Screenshots
- Click **File → Make Snapshot**
- Save images for comparison

## Viewing Metrics in GUI

### Real-time Metrics
1. Click **View → Open SUMO Configuration**
2. Enable output options:
   - Tripinfo output
   - Summary output
   - Queue output

### View Trip Information
- Right-click on a vehicle
- Select **"Show Parameter"**
- See: travel time, waiting time, delay, route

### View Junction Statistics
- Right-click on intersection
- Select **"Show Junction Statistics"**
- See: vehicle count, average speed, etc.

## Tips for Comparison

### 1. Use Same View Settings
- Set same zoom level for both simulations
- Use same camera angle
- Enable same visualization options

### 2. Record Metrics
- Note average delays you observe
- Count vehicles in queues
- Time how long vehicles wait

### 3. Use Time Controls
- Set same start time: **Simulation → Start Time**
- Use same end time: **Simulation → End Time**
- Run for same duration (e.g., 3600 seconds)

### 4. Compare Visually
- **Queue Length**: Count stopped vehicles
- **Delay**: Observe vehicle waiting times
- **Throughput**: Count vehicles passing through

## Command Line Alternative

If you prefer command line (faster, no GUI overhead):

```bash
# Run ML simulation
sumo -c ml_config.sumocfg --summary-output ml_summary.xml --tripinfo-output ml_tripinfo.xml

# Run Webster simulation
sumo -c webster_config.sumocfg --summary-output webster_summary.xml --tripinfo-output webster_tripinfo.xml
```

Then use the Python script to extract and compare metrics:
```bash
python sumo_simulation.py
```

## Troubleshooting

### Network Not Loading
- Check that `sumo_network.net.xml` exists
- Verify file path is correct
- Try regenerating network

### No Vehicles Appearing
- Check `routes.rou.xml` has flow definitions
- Verify PCU values are > 0
- Check simulation time range

### Traffic Lights Not Working
- Verify traffic light file is loaded
- Check traffic light ID matches junction ID
- Ensure phase durations are valid (> 0)

### Simulation Too Fast/Slow
- Adjust delay in GUI (bottom toolbar)
- Use pause/step for detailed observation
- Slow down to 0.1x speed for careful analysis

## Quick Start Checklist

- [ ] SUMO GUI installed and in PATH
- [ ] All 6 files generated in project folder
- [ ] Open SUMO GUI
- [ ] Load `ml_config.sumocfg` or `webster_config.sumocfg`
- [ ] Click Play
- [ ] Observe simulation
- [ ] Switch to other config file
- [ ] Compare results

## Next Steps

After observing in GUI:
1. Run command-line simulations for metrics
2. Use `sumo_simulation.py` to extract detailed statistics
3. Compare results in `outputs/sumo_comparison.json`
4. Make decision based on real simulation data

