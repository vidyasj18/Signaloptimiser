# SUMO Simulation Setup Guide

This guide explains how to set up and run SUMO simulations to compare ML-based and Webster-based signal plans.

## Prerequisites

1. **Install SUMO**
   - Download from: https://sumo.dlr.de/docs/Downloads.php
   - Windows: Use the installer or download the zip file
   - Linux: `sudo apt-get install sumo` or build from source
   - Mac: `brew install sumo`

2. **Verify Installation**
   ```bash
   sumo --version
   sumo-gui --version
   ```

3. **Python Requirements**
   - The script uses standard libraries (json, xml, subprocess)
   - Optional: `sumolib` and `traci` (usually included with SUMO)
   - Add SUMO to PYTHONPATH if needed:
     ```bash
     # Windows
     set PYTHONPATH=%PYTHONPATH%;C:\path\to\sumo\tools
     
     # Linux/Mac
     export PYTHONPATH=$PYTHONPATH:/path/to/sumo/tools
     ```

## Quick Start

1. **Generate Signal Plans**
   ```bash
   python final.py      # Generates outputs/ml_signal_plan.json
   python webster.py    # Generates outputs/webster_signal_plan.json
   ```

2. **Run SUMO Simulation**
   ```bash
   python sumo_simulation.py
   ```

3. **View Results**
   - Console output shows comparison
   - Detailed results in `outputs/sumo_comparison.json`
   - Tripinfo files: `tripinfo_ML.xml`, `tripinfo_Webster.xml`

## What the Script Does

1. **Creates SUMO Network** (`sumo_network.net.xml`)
   - 4-way intersection
   - 4 approaches: North (NB), South (SB), East (EB), West (WB)
   - 2 lanes per approach (incoming + outgoing)

2. **Converts Signal Timings** to SUMO traffic light format
   - ML plan: `ml_traffic_lights.add.xml`
   - Webster plan: `webster_traffic_lights.add.xml`
   - Uses 2-phase operation: NS (North-South) vs EW (East-West)

3. **Generates Vehicle Routes** (`routes.rou.xml`)
   - Based on PCU values from `intersection_summary.json`
   - Converts PCU/hour to vehicles/second
   - Vehicles spawn throughout simulation period

4. **Runs Simulations**
   - ML plan: `ml_config.sumocfg`
   - Webster plan: `webster_config.sumocfg`
   - Simulation time: 3600 seconds (1 hour)
   - Warmup: 300 seconds

5. **Extracts Metrics**
   - Average delay
   - Average waiting time
   - Average travel time
   - Average time loss
   - Vehicle counts

## Manual Network Creation (Advanced)

For a more realistic network, create proper SUMO network files:

1. **Create .nod file** (nodes/junctions):
   ```
   <?xml version="1.0" encoding="UTF-8"?>
   <nodes>
       <node id="north" x="0.0" y="250.0" type="priority"/>
       <node id="south" x="500.0" y="250.0" type="priority"/>
       <node id="east" x="250.0" y="0.0" type="priority"/>
       <node id="west" x="250.0" y="500.0" type="priority"/>
       <node id="center" x="250.0" y="250.0" type="traffic_light"/>
   </nodes>
   ```

2. **Create .edg file** (edges/lanes):
   ```
   <?xml version="1.0" encoding="UTF-8"?>
   <edges>
       <edge id="NB_in" from="north" to="center" priority="13" numLanes="2"/>
       <edge id="NB_out" from="center" to="south" priority="13" numLanes="2"/>
       <edge id="SB_in" from="south" to="center" priority="13" numLanes="2"/>
       <edge id="SB_out" from="center" to="north" priority="13" numLanes="2"/>
       <edge id="EB_in" from="east" to="center" priority="13" numLanes="2"/>
       <edge id="EB_out" from="center" to="west" priority="13" numLanes="2"/>
       <edge id="WB_in" from="west" to="center" priority="13" numLanes="2"/>
       <edge id="WB_out" from="center" to="east" priority="13" numLanes="2"/>
   </edges>
   ```

3. **Convert to .net.xml**:
   ```bash
   netconvert --node-files=intersection.nod.xml --edge-files=intersection.edg.xml --output-file=sumo_network.net.xml
   ```

## Running with GUI

To visualize the simulation:

```bash
# ML plan
sumo-gui -c ml_config.sumocfg

# Webster plan
sumo-gui -c webster_config.sumocfg
```

## Troubleshooting

### SUMO not found
- Ensure SUMO is installed and in PATH
- Windows: Add SUMO bin directory to system PATH
- Linux/Mac: Verify with `which sumo`

### Network errors
- The auto-generated network is simplified
- For production, use netconvert with proper .nod/.edg files
- Or use SUMO GUI to create network interactively

### No vehicles in simulation
- Check `routes.rou.xml` for flow definitions
- Verify PCU values in `intersection_summary.json`
- Ensure routes match network edge IDs

### Traffic lights not working
- Check `*_traffic_lights.add.xml` files
- Verify phase durations match cycle length
- Ensure traffic light ID matches junction ID in network

## Output Files

- `sumo_network.net.xml` - Network definition
- `ml_traffic_lights.add.xml` - ML signal timings
- `webster_traffic_lights.add.xml` - Webster signal timings
- `routes.rou.xml` - Vehicle routes
- `ml_config.sumocfg` - ML simulation config
- `webster_config.sumocfg` - Webster simulation config
- `tripinfo_ML.xml` - ML simulation results
- `tripinfo_Webster.xml` - Webster simulation results
- `outputs/sumo_comparison.json` - Comparison results

## Interpreting Results

The comparison shows:
- **Average Delay**: Time spent waiting at intersection
- **Average Waiting Time**: Time stopped/waiting
- **Average Travel Time**: Total time from entry to exit
- **Average Time Loss**: Difference from free-flow travel time
- **Vehicle Count**: Number of vehicles that completed trip

Lower values are better for delay, waiting time, travel time, and time loss.

## Next Steps

1. **Calibrate Network**: Adjust speeds, lane lengths, turning movements
2. **Add More Vehicle Types**: Trucks, buses, motorcycles with different properties
3. **Extend Simulation Time**: Run longer simulations for better statistics
4. **Add Random Seed**: Use different seeds for multiple runs
5. **Export Detailed Metrics**: Use SUMO's output tools for queue lengths, emissions, etc.

