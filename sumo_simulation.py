"""
SUMO Simulation Script (Webster-only)

Workflow:
1. Read intersection PCU demand data
2. Generate Webster signal plan JSON
3. Build a SUMO network and routing files
4. Run a single SUMO simulation for the Webster plan
5. Extract and report performance metrics
"""

import json
import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Tuple
import math
from pathlib import Path
import webster

# Try to import SUMO libraries
try:
    import sumolib
    import traci
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False
    print("Warning: SUMO libraries not found. Install SUMO and ensure sumolib/traci are in PYTHONPATH")


# Constants
NETWORK_SIZE = 500.0  # meters - size of intersection area
LANE_LENGTH = 200.0   # meters - approach length
LANE_WIDTH = 3.5      # meters
SIMULATION_TIME = 3600  # seconds - 1 hour simulation
WARMUP_TIME = 300     # seconds - warmup period before metrics collection


def prettify_xml(elem):
    """Return a pretty-printed XML string"""
    rough_string = ET.tostring(elem, 'unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_sumo_network_files(signal_plans: List[Dict] = None):
    """
    Create .nod and .edg files for netconvert (proper SUMO network generation).
    Returns the list of present approaches.
    """
    # Determine which approaches are present
    present_approaches = set()
    if signal_plans:
        for plan in signal_plans:
            approaches = plan.get('approaches', {})
            for appr_id in approaches.keys():
                if approaches[appr_id].get('green', 0) > 0:
                    present_approaches.add(appr_id)
    
    if not present_approaches:
        present_approaches = {'NB', 'SB', 'EB', 'WB'}
    
    print(f"  Detected approaches: {', '.join(sorted(present_approaches))}")
    is_3way = len(present_approaches) == 3
    if is_3way:
        print(f"  Type: 3-way T-junction")
    else:
        print(f"  Type: 4-way intersection")
    
    center_x = NETWORK_SIZE / 2
    center_y = NETWORK_SIZE / 2
    
    # Create .nod file
    nod_root = ET.Element('nodes')
    nod_root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    nod_root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/nodes_file.xsd')
    
    # Map approaches to positions
    approach_positions = {
        'NB': {'x': center_x, 'y': 0},
        'SB': {'x': center_x, 'y': NETWORK_SIZE},
        'EB': {'x': NETWORK_SIZE, 'y': center_y},
        'WB': {'x': 0, 'y': center_y},
    }
    
    # Create nodes - map approaches to node names
    approach_to_node_map = {
        'NB': 'north',
        'SB': 'south',
        'EB': 'east',
        'WB': 'west',
    }
    
    needed_nodes = {'center'}  # Always need center
    for appr_id in present_approaches:
        if appr_id in approach_to_node_map:
            # Add the approach's node
            needed_nodes.add(approach_to_node_map[appr_id])
            
            # Also add opposite node for outgoing traffic
            if appr_id == 'NB':
                needed_nodes.add('south')
            elif appr_id == 'SB':
                needed_nodes.add('north')
            elif appr_id == 'EB':
                needed_nodes.add('west')
            elif appr_id == 'WB':
                needed_nodes.add('east')
    
    # Create all needed nodes
    all_node_positions = {
        'north': {'x': center_x, 'y': 0},
        'south': {'x': center_x, 'y': NETWORK_SIZE},
        'east': {'x': NETWORK_SIZE, 'y': center_y},
        'west': {'x': 0, 'y': center_y},
        'center': {'x': center_x, 'y': center_y},
    }
    
    for node_id in needed_nodes:
        if node_id in all_node_positions:
            node = ET.SubElement(nod_root, 'node')
            node.set('id', node_id)
            pos = all_node_positions[node_id]
            node.set('x', str(pos['x']))
            node.set('y', str(pos['y']))
            if node_id == 'center':
                node.set('type', 'traffic_light')
            else:
                node.set('type', 'priority')
    
    # Write .nod file
    nod_tree = ET.ElementTree(nod_root)
    with open('intersection.nod.xml', 'wb') as f:
        nod_tree.write(f, encoding='utf-8', xml_declaration=True)
    
    # Create .edg file
    edg_root = ET.Element('edges')
    edg_root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    edg_root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/edges_file.xsd')
    
    approach_to_nodes = {
        'NB': {'from': 'north', 'to': 'center', 'out': 'south'},
        'SB': {'from': 'south', 'to': 'center', 'out': 'north'},
        'EB': {'from': 'east', 'to': 'center', 'out': 'west'},
        'WB': {'from': 'west', 'to': 'center', 'out': 'east'},
    }
    
    for appr_id in present_approaches:
        if appr_id in approach_to_nodes:
            nodes = approach_to_nodes[appr_id]
            
            # Incoming edge
            edge_in = ET.SubElement(edg_root, 'edge')
            edge_in.set('id', f'{appr_id}_in')
            edge_in.set('from', nodes['from'])
            edge_in.set('to', nodes['to'])
            edge_in.set('priority', '13')
            edge_in.set('numLanes', '1')
            edge_in.set('speed', '13.89')
            
            # Outgoing edge
            edge_out = ET.SubElement(edg_root, 'edge')
            edge_out.set('id', f'{appr_id}_out')
            edge_out.set('from', nodes['to'])
            edge_out.set('to', nodes['out'])
            edge_out.set('priority', '13')
            edge_out.set('numLanes', '1')
            edge_out.set('speed', '13.89')
    
    # Write .edg file
    edg_tree = ET.ElementTree(edg_root)
    with open('intersection.edg.xml', 'wb') as f:
        edg_tree.write(f, encoding='utf-8', xml_declaration=True)
    
    print(f"  Created: intersection.nod.xml, intersection.edg.xml")
    return present_approaches


def create_sumo_network(signal_plans: List[Dict] = None, output_path: str = 'sumo_network.net.xml'):
    """
    Create intersection network in SUMO format using netconvert (supports 3-way T-junction or 4-way).
    Automatically detects which approaches are present from signal plans.
    
    Approaches: North (NB), South (SB), East (EB), West (WB)
    """
    print(f"Creating SUMO network: {output_path}")
    
    # Create .nod and .edg files first
    present_approaches = create_sumo_network_files(signal_plans)
    
    # Use netconvert to generate proper .net.xml file
    try:
        import subprocess
        cmd = [
            'netconvert',
            '--node-files=intersection.nod.xml',
            '--edge-files=intersection.edg.xml',
            '--output-file=' + output_path,
            '--no-turnarounds'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"Network file created using netconvert: {output_path}")
            print(f"  Nodes: {len(present_approaches) + 1}")
            print(f"  Edges: {len(present_approaches) * 2}")
            return
        else:
            print(f"Warning: netconvert failed: {result.stderr}")
            print("Falling back to manual XML generation...")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Warning: netconvert not available ({e})")
        print("Falling back to manual XML generation...")
    
    # Fallback: Manual XML generation (simplified, may have issues)
    
    root = ET.Element('net')
    root.set('version', '1.0')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/net_file.xsd')
    
    # Location
    location = ET.SubElement(root, 'location')
    location.set('netOffset', '0.00,0.00')
    location.set('convBoundary', f'0.00,0.00,{NETWORK_SIZE},{NETWORK_SIZE}')
    location.set('origBoundary', f'0.00,0.00,{NETWORK_SIZE},{NETWORK_SIZE}')
    location.set('projParameter', '!')
    
    center_x = NETWORK_SIZE / 2
    center_y = NETWORK_SIZE / 2
    
    # Define all possible nodes (we'll only use the ones we need)
    all_nodes = {
        'north': {'x': center_x, 'y': 0, 'type': 'priority'},
        'south': {'x': center_x, 'y': NETWORK_SIZE, 'type': 'priority'},
        'east': {'x': NETWORK_SIZE, 'y': center_y, 'type': 'priority'},
        'west': {'x': 0, 'y': center_y, 'type': 'priority'},
        'center': {'x': center_x, 'y': center_y, 'type': 'traffic_light'},
    }
    
    # Map approaches to nodes
    approach_to_node = {
        'NB': 'north',
        'SB': 'south',
        'EB': 'east',
        'WB': 'west',
    }
    
    # Determine which nodes we need
    needed_nodes = {'center'}  # Always need center
    for appr_id in present_approaches:
        if appr_id in approach_to_node:
            needed_nodes.add(approach_to_node[appr_id])
            # Also need the opposite node for outgoing traffic
            if appr_id == 'NB':
                needed_nodes.add('south')
            elif appr_id == 'SB':
                needed_nodes.add('north')
            elif appr_id == 'EB':
                needed_nodes.add('west')
            elif appr_id == 'WB':
                needed_nodes.add('east')
    
    # Create only needed nodes
    nodes = {k: v for k, v in all_nodes.items() if k in needed_nodes}
    
    # Create nodes
    inc_lanes = []
    for node_id, node_data in nodes.items():
        node = ET.SubElement(root, 'junction')
        node.set('id', node_id)
        node.set('type', node_data['type'])
        node.set('x', str(node_data['x']))
        node.set('y', str(node_data['y']))
        if node_id == 'center':
            # Will set incLanes after creating edges
            node.set('intLanes', '')
            node.set('shape', f'{center_x-5},{center_y-5} '
                             f'{center_x+5},{center_y-5} '
                             f'{center_x+5},{center_y+5} '
                             f'{center_x-5},{center_y+5}')
        else:
            node.set('incLanes', '')
            node.set('intLanes', '')
            node.set('shape', f'{node_data["x"]},{node_data["y"]}')
    
    # Define edges and lanes for each present approach
    approach_definitions = {
        'NB': {'from_node': 'north', 'to_node': 'center', 'out_node': 'south'},
        'SB': {'from_node': 'south', 'to_node': 'center', 'out_node': 'north'},
        'EB': {'from_node': 'east', 'to_node': 'center', 'out_node': 'west'},
        'WB': {'from_node': 'west', 'to_node': 'center', 'out_node': 'east'},
    }
    
    # Create edges and lanes only for present approaches
    for appr_id in present_approaches:
        if appr_id not in approach_definitions:
            continue
        
        appr_data = approach_definitions[appr_id]
        from_node = nodes[appr_data['from_node']]
        to_node = nodes[appr_data['to_node']]
        out_node = nodes[appr_data['out_node']]
        
        # Incoming edge (toward center)
        edge_in = ET.SubElement(root, 'edge')
        edge_in.set('id', f'{appr_id}_in')
        edge_in.set('from', appr_data['from_node'])
        edge_in.set('to', appr_data['to_node'])
        edge_in.set('priority', '13')
        
        lane_in = ET.SubElement(edge_in, 'lane')
        lane_id = f'{appr_id}_in_0'
        lane_in.set('id', lane_id)
        lane_in.set('index', '0')
        lane_in.set('speed', '13.89')  # 50 km/h = 13.89 m/s
        lane_in.set('length', str(LANE_LENGTH))
        lane_in.set('width', str(LANE_WIDTH))
        # Shape: from outer node to center
        lane_in.set('shape', f'{from_node["x"]},{from_node["y"]} {to_node["x"]},{to_node["y"]}')
        
        inc_lanes.append(lane_id)
        
        # Outgoing edge (from center)
        edge_out = ET.SubElement(root, 'edge')
        edge_out.set('id', f'{appr_id}_out')
        edge_out.set('from', appr_data['to_node'])
        edge_out.set('to', appr_data['out_node'])
        edge_out.set('priority', '13')
        
        lane_out = ET.SubElement(edge_out, 'lane')
        lane_out.set('id', f'{appr_id}_out_0')
        lane_out.set('index', '0')
        lane_out.set('speed', '13.89')
        lane_out.set('length', str(LANE_LENGTH))
        lane_out.set('width', str(LANE_WIDTH))
        # Shape: from center to outer node
        lane_out.set('shape', f'{to_node["x"]},{to_node["y"]} {out_node["x"]},{out_node["y"]}')
    
    # Update center junction with actual incoming lanes
    for node in root.findall('junction'):
        if node.get('id') == 'center':
            node.set('incLanes', ' '.join(inc_lanes))
            break
    
    # Write network file
    tree = ET.ElementTree(root)
    with open(output_path, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    
    print(f"Network file created: {output_path}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(present_approaches) * 2} ({len(present_approaches)} incoming + {len(present_approaches)} outgoing)")
    print(f"  Center junction type: traffic_light")
    print(f"  Incoming lanes: {', '.join(inc_lanes)}")


def create_traffic_lights(signal_plan: Dict, output_path: str, plan_name: str, network_file: str = 'sumo_network.net.xml'):
    """
    Convert signal plan timings to SUMO traffic light phases.
    
    SUMO phase format: 'r'=red, 'y'=yellow, 'g'=green
    Our plan uses: NS phase (NB+SB) and EW phase (EB+WB)
    
    Reads actual lane order from network file to match phase states correctly.
    """
    print(f"Creating traffic lights for {plan_name}: {output_path}")
    
    # Read connection order from network file (SUMO phase size = number of connections, not lanes)
    connections = []
    try:
        if os.path.exists(network_file):
            tree = ET.parse(network_file)
            root = tree.getroot()
            # Get all connections controlled by center traffic light
            for conn in root.findall('connection'):
                if conn.get('tl') == 'center':
                    from_edge = conn.get('from', '')
                    to_edge = conn.get('to', '')
                    link_index = int(conn.get('linkIndex', '0'))
                    connections.append({
                        'from': from_edge,
                        'to': to_edge,
                        'linkIndex': link_index,
                        'fromLane': conn.get('fromLane', '0')
                    })
            # Sort by linkIndex to get correct order
            connections.sort(key=lambda x: x['linkIndex'])
    except Exception as e:
        print(f"  Warning: Could not read connections from network: {e}")
    
    if not connections:
        print(f"  Warning: No connections found. Using simplified lane-based approach.")
        connections = []
    
    print(f"  Connections controlled by TL: {len(connections)}")
    if len(connections) > 0:
        conn_strs = [f"{c['from']}->{c['to']}" for c in connections[:5]]
        print(f"  Connection order: {', '.join(conn_strs)}...")
    
    root = ET.Element('additional')
    
    tl_logic = ET.SubElement(root, 'tlLogic')
    tl_logic.set('id', 'center')
    tl_logic.set('type', 'static')
    tl_logic.set('programID', plan_name)
    tl_logic.set('offset', '0')
    
    cycle_length = signal_plan['cycle_length']
    approaches = signal_plan['approaches']
    
    # Get timings for each approach
    nb = approaches.get('NB', {})
    sb = approaches.get('SB', {})
    eb = approaches.get('EB', {})
    wb = approaches.get('WB', {})
    
    # Determine which approaches are present
    present = []
    if nb and nb.get('green', 0) > 0:
        present.append('NB')
    if sb and sb.get('green', 0) > 0:
        present.append('SB')
    if eb and eb.get('green', 0) > 0:
        present.append('EB')
    if wb and wb.get('green', 0) > 0:
        present.append('WB')
    
    # Calculate phase timings
    # Phase 1: NS green (NB + SB)
    g_ns = 0.0
    if 'NB' in present:
        g_ns += nb.get('green', 0)
    if 'SB' in present:
        g_ns += sb.get('green', 0)
    
    y_ns = nb.get('amber', 3.0) if 'NB' in present else (sb.get('amber', 3.0) if 'SB' in present else 3.0)
    all_red = 2.0  # All-red between phases
    
    # Phase 2: EW green (EB + WB)
    g_ew = 0.0
    if 'EB' in present:
        g_ew += eb.get('green', 0)
    if 'WB' in present:
        g_ew += wb.get('green', 0)
    
    y_ew = eb.get('amber', 3.0) if 'EB' in present else (wb.get('amber', 3.0) if 'WB' in present else 3.0)
    
    # Build phase state string based on connection order from network
    def get_state_by_connections(nb_state, sb_state, eb_state, wb_state):
        """Build state string matching the connection order from network file"""
        if not connections:
            # Fallback: use lane-based (simplified, may not work)
            present_approaches = []
            for appr_id in ['NB', 'SB', 'EB', 'WB']:
                if appr_id in signal_plan.get('approaches', {}):
                    if signal_plan['approaches'][appr_id].get('green', 0) > 0:
                        present_approaches.append(appr_id)
            # Return state for each approach (simplified)
            state = ''
            for appr in present_approaches:
                if appr == 'NB':
                    state += nb_state
                elif appr == 'SB':
                    state += sb_state
                elif appr == 'EB':
                    state += eb_state
                elif appr == 'WB':
                    state += wb_state
            return state
        
        state = ''
        for conn in connections:
            from_edge = conn['from']
            # Determine which approach this connection belongs to
            if from_edge.startswith('NB'):
                state += nb_state
            elif from_edge.startswith('SB'):
                state += sb_state
            elif from_edge.startswith('EB'):
                state += eb_state
            elif from_edge.startswith('WB'):
                state += wb_state
            else:
                state += 'r'  # Default to red if unknown
        return state
    
    phases = []
    total_phase_time = 0.0
    
    # Phase 1: NS green, EW red
    if g_ns > 0:
        phase1 = ET.SubElement(tl_logic, 'phase')
        phase1.set('duration', str(int(round(g_ns))))
        state1 = get_state_by_connections('G', 'G', 'r', 'r')  # NS green, EW red
        phase1.set('state', state1)
        phases.append(('NS Green', g_ns, state1))
        total_phase_time += g_ns
    
    # Phase 2: NS yellow, EW red
    if y_ns > 0 and g_ns > 0:
        phase2 = ET.SubElement(tl_logic, 'phase')
        phase2.set('duration', str(int(round(y_ns))))
        state2 = get_state_by_connections('y', 'y', 'r', 'r')  # NS yellow, EW red
        phase2.set('state', state2)
        phases.append(('NS Yellow', y_ns, state2))
        total_phase_time += y_ns
    
    # Phase 3: All red
    if all_red > 0 and (g_ns > 0 or g_ew > 0):
        phase3 = ET.SubElement(tl_logic, 'phase')
        phase3.set('duration', str(int(round(all_red))))
        state3 = get_state_by_connections('r', 'r', 'r', 'r')  # All red
        phase3.set('state', state3)
        phases.append(('All Red', all_red, state3))
        total_phase_time += all_red
    
    # Phase 4: EW green, NS red
    if g_ew > 0:
        phase4 = ET.SubElement(tl_logic, 'phase')
        phase4.set('duration', str(int(round(g_ew))))
        state4 = get_state_by_connections('r', 'r', 'r', 'G')  # EW green (WB only), NS red
        phase4.set('state', state4)
        phases.append(('EW Green', g_ew, state4))
        total_phase_time += g_ew
    
    # Phase 5: EW yellow, NS red
    if y_ew > 0 and g_ew > 0:
        phase5 = ET.SubElement(tl_logic, 'phase')
        phase5.set('duration', str(int(round(y_ew))))
        state5 = get_state_by_connections('r', 'r', 'r', 'y')  # EW yellow (WB only), NS red
        phase5.set('state', state5)
        phases.append(('EW Yellow', y_ew, state5))
        total_phase_time += y_ew
    
    # Phase 6: All red (before cycle repeats)
    if all_red > 0 and (g_ns > 0 or g_ew > 0):
        phase6 = ET.SubElement(tl_logic, 'phase')
        phase6.set('duration', str(int(round(all_red))))
        state6 = get_state_by_connections('r', 'r', 'r', 'r')  # All red
        phase6.set('state', state6)
        phases.append(('All Red', all_red, state6))
        total_phase_time += all_red
    
    # Adjust last phase to match cycle length exactly
    if phases and abs(total_phase_time - cycle_length) > 0.5:
        last_phase = phases[-1]
        adjustment = cycle_length - total_phase_time
        new_duration = max(1, int(round(last_phase[1] + adjustment)))
        # Update the last phase duration
        for phase_elem in tl_logic.findall('phase'):
            pass  # Find last one
        last_elem = list(tl_logic.findall('phase'))[-1]
        last_elem.set('duration', str(new_duration))
        phases[-1] = (phases[-1][0], new_duration, phases[-1][2])
    
    # Write file
    tree = ET.ElementTree(root)
    with open(output_path, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    
    print(f"Traffic lights created: {output_path}")
    print(f"  Cycle length: {cycle_length:.1f}s")
    print(f"  Approaches: {', '.join(present)}")
    print(f"  Phases: {len(phases)}")
    for name, duration, state in phases:
        print(f"    {name}: {duration:.1f}s ({state})")


def create_routes(pcu_data: Dict, present_approaches: set, output_path: str = 'routes.rou.xml', sim_time: int = SIMULATION_TIME):
    """
    Generate vehicle routes based on PCU values.
    Converts PCU/hour to vehicles/second for SUMO.
    Only creates routes for approaches that exist in the network.
    """
    print(f"Creating routes: {output_path}")
    print(f"  Present approaches: {', '.join(sorted(present_approaches))}")
    
    root = ET.Element('routes')
    
    # Vehicle types
    vtype_default = ET.SubElement(root, 'vType')
    vtype_default.set('id', 'default')
    vtype_default.set('accel', '2.6')
    vtype_default.set('decel', '4.5')
    vtype_default.set('sigma', '0.5')
    vtype_default.set('length', '5.0')
    vtype_default.set('minGap', '2.5')
    vtype_default.set('maxSpeed', '13.89')  # 50 km/h
    
    # Dynamically determine route destinations based on T-junction geometry
    # In a T-junction (3-way), not all "through" movements exist
    approach_destinations = {}
    
    # Define valid connections based on T-junction configuration
    # For NB,SB,WB (missing EB): 
    #   - NB can only turn (no through to south without EB)
    #   - SB can go through north or turn east
    #   - WB can turn north or south
    if present_approaches == {'NB', 'SB', 'WB'}:
        # T-junction: missing east approach
        # Based on actual connections from netconvert:
        # NB_in -> WB_out, NB_in -> NB_out
        # SB_in -> SB_out, SB_in -> WB_out
        # WB_in -> SB_out, WB_in -> WB_out, WB_in -> NB_out
        approach_destinations = {
            'NB': 'WB_out',   # NB turns right (east)
            'SB': 'SB_out',   # SB goes through (northward via SB_out)
            'WB': 'NB_out'    # WB turns south
        }
    elif present_approaches == {'NB', 'SB', 'EB'}:
        # T-junction: missing west approach
        approach_destinations = {
            'NB': 'SB_out',   # NB goes through
            'SB': 'NB_out',   # SB goes through
            'EB': 'NB_out'    # EB turns
        }
    elif present_approaches == {'NB', 'EB', 'WB'}:
        # T-junction: missing south approach
        approach_destinations = {
            'NB': 'WB_out',   # NB turns
            'EB': 'WB_out',   # EB goes through
            'WB': 'EB_out'    # WB goes through
        }
    elif present_approaches == {'SB', 'EB', 'WB'}:
        # T-junction: missing north approach
        approach_destinations = {
            'SB': 'WB_out',   # SB turns
            'EB': 'WB_out',   # EB goes through
            'WB': 'EB_out'    # WB goes through
        }
    else:
        # 4-way intersection - use connections that netconvert creates
        # Netconvert creates turn movements (left/right) but not always straight through
        # For a standard 4-way, use right-turn movements which are always valid
        turn_right_map = {
            'NB': 'EB_out',  # NB turns right to go east
            'EB': 'SB_out',  # EB turns right to go south  
            'SB': 'WB_out',  # SB turns right to go west
            'WB': 'NB_out'   # WB turns right to go north
        }
        for approach in present_approaches:
            # Try right turn first (most reliable)
            if approach in turn_right_map:
                approach_destinations[approach] = turn_right_map[approach]
            else:
                # Fallback to any available output
                print(f"Fallback to any available output for {approach}")
                available = [f'{appr}_out' for appr in present_approaches if appr != approach]
                print(f"Available outputs: {available}")
                approach_destinations[approach] = available[0] if available else f'{approach}_out'
    
    print(f"  Route destinations: {approach_destinations}")
    
    for approach, data in pcu_data.items():
        # Only create routes for approaches that exist in the network
        if approach in present_approaches:
            total_pcu = data.get('total_pcu', 0.0)
            if total_pcu > 0:
                # Convert PCU/hour to vehicles/second
                # Assuming average PCU factor of 1.0 (mostly cars)
                veh_per_hour = total_pcu  # Simplified: 1 PCU = 1 vehicle
                veh_per_sec = veh_per_hour / 3600.0
                
                # Create flow
                flow = ET.SubElement(root, 'flow')
                flow.set('id', f'flow_{approach}')
                flow.set('type', 'default')
                flow.set('from', f'{approach}_in')
                flow.set('to', approach_destinations[approach])
                flow.set('begin', '0')
                flow.set('end', str(sim_time))
                flow.set('vehsPerHour', str(int(veh_per_hour)))
                flow.set('departLane', 'best')
                flow.set('departSpeed', 'max')
    
    # Write file
    tree = ET.ElementTree(root)
    with open(output_path, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    
    print(f"Routes created: {output_path}")
    total_flow = sum(data.get('total_pcu', 0) for data in pcu_data.values())
    print(f"  Total flow: {total_flow:.1f} PCU/hr ({total_flow/3600:.3f} veh/s)")


def create_sumo_config(net_file: str, route_file: str, add_file: str, 
                      output_file: str, plan_name: str):
    """Create SUMO configuration file"""
    print(f"Creating SUMO config: {output_file}")
    
    root = ET.Element('configuration')
    
    input_elem = ET.SubElement(root, 'input')
    net_elem = ET.SubElement(input_elem, 'net-file')
    net_elem.set('value', net_file)
    route_elem = ET.SubElement(input_elem, 'route-files')
    route_elem.set('value', route_file)
    add_elem = ET.SubElement(input_elem, 'additional-files')
    add_elem.set('value', add_file)
    
    time_elem = ET.SubElement(root, 'time')
    begin_elem = ET.SubElement(time_elem, 'begin')
    begin_elem.set('value', '0')
    end_elem = ET.SubElement(time_elem, 'end')
    end_elem.set('value', str(SIMULATION_TIME))
    
    output_elem = ET.SubElement(root, 'output')
    summary_elem = ET.SubElement(output_elem, 'summary-output')
    summary_elem.set('value', f'sumo_output_{plan_name}.xml')
    tripinfo_elem = ET.SubElement(output_elem, 'tripinfo-output')
    tripinfo_elem.set('value', f'tripinfo_{plan_name}.xml')
    
    # Write file
    tree = ET.ElementTree(root)
    with open(output_file, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    
    print(f"Config created: {output_file}")


def run_sumo_simulation(config_file: str, plan_name: str, use_gui: bool = False):
    """Run SUMO simulation"""
    print(f"\nRunning SUMO simulation for {plan_name}...")
    
    sumo_cmd = 'sumo-gui' if use_gui else 'sumo'
    cmd = [sumo_cmd, '-c', config_file, '--no-step-log', '--no-warnings']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"Simulation completed: {plan_name}")
            return True
        else:
            print(f"Simulation error for {plan_name}:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print(f"ERROR: SUMO not found. Please install SUMO and add it to PATH.")
        print("Download from: https://sumo.dlr.de/docs/Downloads.php")
        return False
    except subprocess.TimeoutExpired:
        print(f"Simulation timeout for {plan_name}")
        return False


def extract_sumo_metrics(tripinfo_file: str) -> Dict:
    """Extract performance metrics from SUMO tripinfo output"""
    if not os.path.exists(tripinfo_file):
        return None
    
    print(f"Extracting metrics from: {tripinfo_file}")
    
    tree = ET.parse(tripinfo_file)
    root = tree.getroot()
    
    trips = root.findall('.//tripinfo')
    
    if not trips:
        return None
    
    total_delay = 0.0
    total_waiting_time = 0.0
    total_travel_time = 0.0
    total_time_loss = 0.0
    total_depart_delay = 0.0
    vehicle_count = len(trips)
    
    for trip in trips:
        delay = float(trip.get('waitingTime', 0))
        waiting = float(trip.get('waitingTime', 0))
        travel = float(trip.get('duration', 0))
        time_loss = float(trip.get('timeLoss', 0))
        depart_delay = float(trip.get('departDelay', 0))
        
        total_delay += delay
        total_waiting_time += waiting
        total_travel_time += travel
        total_time_loss += time_loss
        total_depart_delay += depart_delay
    
    metrics = {
        'vehicle_count': vehicle_count,
        'avg_delay': total_delay / vehicle_count if vehicle_count > 0 else 0.0,
        'avg_waiting_time': total_waiting_time / vehicle_count if vehicle_count > 0 else 0.0,
        'avg_travel_time': total_travel_time / vehicle_count if vehicle_count > 0 else 0.0,
        'avg_time_loss': total_time_loss / vehicle_count if vehicle_count > 0 else 0.0,
        'avg_depart_delay': total_depart_delay / vehicle_count if vehicle_count > 0 else 0.0,
        'total_delay': total_delay,
        'total_waiting_time': total_waiting_time,
    }
    
    return metrics


def print_sumo_metrics(metrics: Dict) -> None:
    """Pretty-print SUMO metrics for a single plan."""
    if not metrics:
        print("No SUMO metrics available.")
        return

    print("\n" + "=" * 80)
    print("SUMO PERFORMANCE SUMMARY (Webster)")
    print("=" * 80)
    for key in [
        ('vehicle_count', 'Total Vehicles Processed'),
        ('avg_delay', 'Average Delay (s/veh)'),
        ('avg_waiting_time', 'Average Waiting Time (s/veh)'),
        ('avg_travel_time', 'Average Travel Time (s/veh)'),
        ('avg_time_loss', 'Average Time Loss (s/veh)'),
        ('avg_depart_delay', 'Average Depart Delay (s/veh)'),
        ('total_delay', 'Total Delay (s)'),
        ('total_waiting_time', 'Total Waiting Time (s)')
    ]:
        metric_key, label = key
        value = metrics.get(metric_key)
        if value is None:
            continue
        if isinstance(value, float):
            print(f"{label:<40}: {value:>10.2f}")
        else:
            print(f"{label:<40}: {value}")

    print("=" * 80)


def main():
    """Run SUMO simulation for Webster timing plan only."""
    print("=" * 80)
    print("SUMO SIMULATION FOR WEBSTER PLAN")
    print("=" * 80)

    if not SUMO_AVAILABLE:
        print("\nWarning: SUMO Python libraries not found.")
        print("The script will generate SUMO files but cannot run simulations.")
        print("Install SUMO from: https://sumo.dlr.de/docs/Downloads.php")
        print("Ensure sumolib and traci are in your PYTHONPATH.")

    # Step 0: generate Webster plan
    print("\n" + "=" * 80)
    print("STEP 0: Generating Webster Signal Plan")
    print("=" * 80)
    webster.main()

    plan_path = Path('outputs/webster_signal_plan.json')
    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found after running webster.py.")
        return

    with plan_path.open('r', encoding='utf-8') as f:
        webster_plan = json.load(f)
    print(f"\nLoaded Webster plan: {plan_path}")

    # Load intersection summary for PCU data
    intersection_summary_path = Path('outputs/intersection_summary.json')
    if intersection_summary_path.exists():
        with intersection_summary_path.open('r', encoding='utf-8') as f:
            intersection_data = json.load(f)
        print(f"Loaded intersection demands: {intersection_summary_path}")
    else:
        print(f"Warning: {intersection_summary_path} not found. Deriving demand from signal plan.")
        intersection_data = {}
        for appr, data in webster_plan.get('approaches', {}).items():
            intersection_data[appr] = {'total_pcu': data.get('arrival_flow_rate', 0)}

    # Create SUMO network
    print("\n" + "=" * 80)
    print("STEP 1: Creating SUMO Network")
    print("=" * 80)
    create_sumo_network([webster_plan], 'sumo_network.net.xml')

    # Create Webster traffic lights
    print("\n" + "=" * 80)
    print("STEP 2: Creating Traffic Light Definition")
    print("=" * 80)
    create_traffic_lights(webster_plan, 'webster_traffic_lights.add.xml', 'Webster', 'sumo_network.net.xml')

    # Detect present approaches
    present_approaches = set()
    for appr_id, data in webster_plan.get('approaches', {}).items():
        if data.get('green', 0) > 0:
            present_approaches.add(appr_id)
    if not present_approaches:
        present_approaches = {'NB', 'SB', 'EB', 'WB'}

    # Create routes
    print("\n" + "=" * 80)
    print("STEP 3: Creating Vehicle Routes")
    print("=" * 80)
    create_routes(intersection_data, present_approaches, 'routes.rou.xml')

    # Create SUMO config
    print("\n" + "=" * 80)
    print("STEP 4: Creating SUMO Configuration")
    print("=" * 80)
    create_sumo_config(
        'sumo_network.net.xml',
        'routes.rou.xml',
        'webster_traffic_lights.add.xml',
        'webster_config.sumocfg',
        'Webster'
    )

    if not SUMO_AVAILABLE:
        print("\nSUMO not available; skipping simulation. Config and network files are ready.")
        return

    # Run simulation
    print("\n" + "=" * 80)
    print("STEP 5: Running SUMO Simulation")
    print("=" * 80)
    print("Note: This requires SUMO to be installed and in PATH.")

    success = run_sumo_simulation('webster_config.sumocfg', 'Webster', use_gui=False)

    if success:
        print("\n" + "=" * 80)
        print("STEP 6: Extracting Results")
        print("=" * 80)
        metrics = extract_sumo_metrics('tripinfo_Webster.xml')
        if metrics:
            print_sumo_metrics(metrics)
            os.makedirs('outputs', exist_ok=True)
            output_path = Path('outputs/sumo_metrics_webster.json')
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"\nSaved SUMO metrics to {output_path}")
        else:
            print("ERROR: Could not extract metrics from tripinfo output.")
    else:
        print("\nSimulation did not complete successfully.")
        print("Generated files are available for manual execution with SUMO GUI/CLI.")

    print("\n" + "=" * 80)
    print("SUMO SIMULATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - sumo_network.net.xml")
    print("  - webster_traffic_lights.add.xml")
    print("  - routes.rou.xml")
    print("  - webster_config.sumocfg")
    if success:
        print("  - tripinfo_Webster.xml (simulation output)")


if __name__ == "__main__":
    main()

