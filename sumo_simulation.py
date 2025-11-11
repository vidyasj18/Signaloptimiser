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
from typing import Any, Dict, List, Optional, Tuple
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
            edge_in.set('numLanes', '3')
            edge_in.set('speed', '13.89')
            
            # Outgoing edge
            edge_out = ET.SubElement(edg_root, 'edge')
            edge_out.set('id', f'{appr_id}_out')
            edge_out.set('from', nodes['to'])
            edge_out.set('to', nodes['out'])
            edge_out.set('priority', '13')
            edge_out.set('numLanes', '3')
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
            '--no-turnarounds',
            '--lefthand'  # India: left-hand traffic geometry and priorities
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
    
    # Helpers to classify L/T/R for left-hand traffic
    def classify_movement(from_edge: str, to_edge: str) -> str:
        fm = 'NB' if from_edge.startswith('NB') else 'SB' if from_edge.startswith('SB') else 'EB' if from_edge.startswith('EB') else 'WB' if from_edge.startswith('WB') else ''
        to = 'NB' if to_edge.startswith('NB') else 'SB' if to_edge.startswith('SB') else 'EB' if to_edge.startswith('EB') else 'WB' if to_edge.startswith('WB') else ''
        # Canonical mapping for left-hand traffic
        mapping = {
            'NB': {'L': 'WB', 'T': 'SB', 'R': 'EB'},
            'SB': {'L': 'EB', 'T': 'NB', 'R': 'WB'},
            'EB': {'L': 'NB', 'T': 'WB', 'R': 'SB'},
            'WB': {'L': 'SB', 'T': 'EB', 'R': 'NB'},
        }
        if fm in mapping:
            for mv, dest in mapping[fm].items():
                if to == dest:
                    return mv  # 'L', 'T', or 'R'
        return ''  # unknown

    def build_state_ns_green() -> str:
        # NB/SB: T,R => 'G'; L => 'g'; EB/WB => 'r'
        state = ''
        for conn in connections:
            from_edge = conn['from']; to_edge = conn['to']
            if from_edge.startswith('NB') or from_edge.startswith('SB'):
                mv = classify_movement(from_edge, to_edge)
                state += 'g' if mv == 'L' else 'G'
            elif from_edge.startswith('EB') or from_edge.startswith('WB'):
                state += 'r'
            else:
                state += 'r'
        return state

    def build_state_ns_yellow() -> str:
        # NB/SB: T,R => 'y'; L => 'g' (left remains permissive); EB/WB => 'r'
        state = ''
        for conn in connections:
            from_edge = conn['from']; to_edge = conn['to']
            if from_edge.startswith('NB') or from_edge.startswith('SB'):
                mv = classify_movement(from_edge, to_edge)
                state += 'g' if mv == 'L' else 'y'
            elif from_edge.startswith('EB') or from_edge.startswith('WB'):
                state += 'r'
            else:
                state += 'r'
        return state

    def build_state_ew_green() -> str:
        # EB/WB: T,R => 'G'; L => 'g'; NB/SB => 'r'
        state = ''
        for conn in connections:
            from_edge = conn['from']; to_edge = conn['to']
            if from_edge.startswith('EB') or from_edge.startswith('WB'):
                mv = classify_movement(from_edge, to_edge)
                state += 'g' if mv == 'L' else 'G'
            elif from_edge.startswith('NB') or from_edge.startswith('SB'):
                state += 'r'
            else:
                state += 'r'
        return state

    def build_state_ew_yellow() -> str:
        # EB/WB: T,R => 'y'; L => 'g'; NB/SB => 'r'
        state = ''
        for conn in connections:
            from_edge = conn['from']; to_edge = conn['to']
            if from_edge.startswith('EB') or from_edge.startswith('WB'):
                mv = classify_movement(from_edge, to_edge)
                state += 'g' if mv == 'L' else 'y'
            elif from_edge.startswith('NB') or from_edge.startswith('SB'):
                state += 'r'
            else:
                state += 'r'
        return state
    
    phases = []
    total_phase_time = 0.0
    
    # Phase 1: NS green, EW red
    if g_ns > 0:
        phase1 = ET.SubElement(tl_logic, 'phase')
        phase1.set('duration', str(int(round(g_ns))))
        state1 = build_state_ns_green()
        phase1.set('state', state1)
        phases.append(('NS Green', g_ns, state1))
        total_phase_time += g_ns
    
    # Phase 2: NS yellow, EW red
    if y_ns > 0 and g_ns > 0:
        phase2 = ET.SubElement(tl_logic, 'phase')
        phase2.set('duration', str(int(round(y_ns))))
        state2 = build_state_ns_yellow()
        phase2.set('state', state2)
        phases.append(('NS Yellow', y_ns, state2))
        total_phase_time += y_ns
    
    # Phase 3: All red
    if all_red > 0 and (g_ns > 0 or g_ew > 0):
        phase3 = ET.SubElement(tl_logic, 'phase')
        phase3.set('duration', str(int(round(all_red))))
        # All red: everyone red (including lefts)
        state3 = 'r' * len(connections) if connections else 'r'
        phase3.set('state', state3)
        phases.append(('All Red', all_red, state3))
        total_phase_time += all_red
    
    # Phase 4: EW green, NS red
    if g_ew > 0:
        phase4 = ET.SubElement(tl_logic, 'phase')
        phase4.set('duration', str(int(round(g_ew))))
        state4 = build_state_ew_green()
        phase4.set('state', state4)
        phases.append(('EW Green', g_ew, state4))
        total_phase_time += g_ew
    
    # Phase 5: EW yellow, NS red
    if y_ew > 0 and g_ew > 0:
        phase5 = ET.SubElement(tl_logic, 'phase')
        phase5.set('duration', str(int(round(y_ew))))
        state5 = build_state_ew_yellow()
        phase5.set('state', state5)
        phases.append(('EW Yellow', y_ew, state5))
        total_phase_time += y_ew
    
    # Phase 6: All red (before cycle repeats)
    if all_red > 0 and (g_ns > 0 or g_ew > 0):
        phase6 = ET.SubElement(tl_logic, 'phase')
        phase6.set('duration', str(int(round(all_red))))
        state6 = 'r' * len(connections) if connections else 'r'
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


def create_routes(
    pcu_data: Dict,
    present_approaches: set,
    output_path: str = 'routes.rou.xml',
    sim_time: int = SIMULATION_TIME,
    turn_splits: Optional[Dict[str, Tuple[float, float, float]]] = None,
    network_file: str = 'sumo_network.net.xml',
) -> None:
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
    
    # Read actual connection availability from the generated network (from-edge -> allowed to-edges)
    available_to_by_from: Dict[str, set] = {}
    try:
        if os.path.exists(network_file):
            net_tree = ET.parse(network_file)
            net_root = net_tree.getroot()
            for conn in net_root.findall('connection'):
                if conn.get('tl') == 'center':
                    f = conn.get('from', '')
                    t = conn.get('to', '')
                    if f and t:
                        available_to_by_from.setdefault(f, set()).add(t)
    except Exception as e:
        print(f"  Warning: could not read connections for routing: {e}")

    # Default per-approach turn splits (Left, Through, Right) for left-hand traffic (India)
    # Use field values if available by passing turn_splits like {'NB': (0.15, 0.7, 0.15), ...}
    default_split = (0.15, 0.70, 0.15)
    if turn_splits is None:
        turn_splits = {appr: default_split for appr in ['NB', 'SB', 'EB', 'WB']}

    # Define canonical destinations for L/T/R under left-hand traffic
    # NB: L->WB_out, T->SB_out, R->EB_out
    canonical_dest = {
        # For Northbound approach: straight goes to NB_out (center -> south)
        'NB': {'L': 'WB_out', 'T': 'NB_out', 'R': 'EB_out'},
        # For Southbound approach: straight goes to SB_out (center -> north)
        'SB': {'L': 'EB_out', 'T': 'SB_out', 'R': 'WB_out'},
        # For Eastbound approach: straight goes to EB_out (center -> west)
        'EB': {'L': 'NB_out', 'T': 'EB_out', 'R': 'SB_out'},
        # For Westbound approach: straight goes to EB_out (center -> east)
        'WB': {'L': 'SB_out', 'T': 'EB_out', 'R': 'NB_out'},
    }
    
    # Remove movements that go to missing approaches and those not present in the actual network connections
    def valid_movements(appr: str) -> Dict[str, str]:
        mv = canonical_dest[appr].copy()
        # Filter out destinations whose approach is not present
        def dest_ok(dest: str) -> bool:
            # dest like 'WB_out' -> 'WB'
            return dest.split('_')[0] in present_approaches
        # Filter by actual network connections from this approach's incoming edge
        from_edge = f'{appr}_in'
        allowed_to = available_to_by_from.get(from_edge, None)
        def conn_ok(dest: str) -> bool:
            return (allowed_to is None) or (dest in allowed_to)
        return {k: v for k, v in mv.items() if dest_ok(v) and conn_ok(v)}
    
    for approach, data in pcu_data.items():
        # Only create routes for approaches that exist in the network
        if approach in present_approaches:
            total_pcu = data.get('total_pcu', 0.0)
            if total_pcu > 0:
                left_ratio, thru_ratio, right_ratio = turn_splits.get(approach, default_split)
                movement_map = valid_movements(approach)
                # Renormalize ratios if some movements are missing (e.g., no through in a T)
                ratios = {'L': left_ratio, 'T': thru_ratio, 'R': right_ratio}
                kept = {k: ratios[k] for k in movement_map.keys()}
                total_ratio = sum(kept.values()) or 1.0
                kept = {k: v / total_ratio for k, v in kept.items()}

                for mv_key, dest in movement_map.items():
                    mv_ratio = kept[mv_key]
                    vehs_per_hour = int(round(total_pcu * mv_ratio))
                    if vehs_per_hour <= 0:
                        continue
                    flow = ET.SubElement(root, 'flow')
                    flow.set('id', f'flow_{approach}_{mv_key}')
                    flow.set('type', 'default')
                    flow.set('from', f'{approach}_in')
                    flow.set('to', dest)
                    flow.set('begin', '0')
                    flow.set('end', str(sim_time))
                    flow.set('vehsPerHour', str(vehs_per_hour))
                    flow.set('departLane', 'best')
                    flow.set('departSpeed', 'max')
                    # Make insertion robust
                    flow.set('departPos', 'random_free')
    
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
    cmd = [
        sumo_cmd, '-c', config_file,
        '--no-step-log', '--no-warnings',
        '--max-depart-delay', '-1',         # never drop vehicles due to spawn delay
        '--time-to-teleport', '-1',         # do not teleport stuck vehicles
        '--collision.action', 'none'        # avoid auto-teleports on collision
    ]
    
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


def run_webster_pipeline(
    pcu_override: Optional[Dict[str, float]] = None,
    run_sumo: bool = True,
    use_gui: bool = False,
    plan_output: str = 'outputs/webster_signal_plan.json',
    net_output: str = 'sumo_network.net.xml',
    add_output: str = 'webster_traffic_lights.add.xml',
    route_output: str = 'generated_routes.rou.xml',
    config_output: str = 'webster_config.sumocfg',
    metrics_output: str = 'outputs/sumo_metrics_webster.json',
) -> Dict[str, Any]:
    """
    Run the Webster-to-SUMO pipeline.

    Returns a dictionary containing the plan, generated files, SUMO availability and metrics.
    """
    results: Dict[str, Any] = {}

    # Step 0: Generate Webster plan (optionally using override PCUs)
    plan = webster.compute_webster_plan(
        pcu_override=pcu_override,
        output_path=plan_output,
        save_json=True,
        ensure_outputs_dir=True,
    )
    results['plan'] = plan

    plan_path = Path(plan_output)
    if not plan_path.exists():
        raise FileNotFoundError(f"{plan_output} not found after computing Webster plan.")

    with plan_path.open('r', encoding='utf-8') as f:
        webster_plan = json.load(f)

    # Load intersection summary for PCU data
    intersection_summary_path = Path('outputs/intersection_summary.json')
    if intersection_summary_path.exists():
        with intersection_summary_path.open('r', encoding='utf-8') as f:
            intersection_data = json.load(f)
    else:
        intersection_data = {}
        for appr, data in webster_plan.get('approaches', {}).items():
            intersection_data[appr] = {'total_pcu': data.get('arrival_flow_rate', 0)}

    # Create SUMO network
    create_sumo_network([webster_plan], net_output)

    # Create traffic lights
    create_traffic_lights(webster_plan, add_output, 'Webster', net_output)

    # Detect present approaches
    present_approaches = set()
    for appr_id, data in webster_plan.get('approaches', {}).items():
        if data.get('green', 0) > 0:
            present_approaches.add(appr_id)
    if not present_approaches:
        present_approaches = {'NB', 'SB', 'EB', 'WB'}

    # Create routes
    create_routes(
        intersection_data,
        present_approaches,
        route_output,
        SIMULATION_TIME,
        None,
        net_output
    )

    # Create SUMO config
    create_sumo_config(
        net_output,
        route_output,
        add_output,
        config_output,
        'Webster'
    )

    results['files'] = {
        'plan': plan_output,
        'network': net_output,
        'routes': route_output,
        'additional': add_output,
        'config': config_output,
        'tripinfo': 'tripinfo_Webster.xml',
        'metrics': metrics_output,
    }
    results['present_approaches'] = sorted(present_approaches)
    results['sumo_available'] = SUMO_AVAILABLE

    metrics = None
    success = False

    if run_sumo and SUMO_AVAILABLE:
        success = run_sumo_simulation(config_output, 'Webster', use_gui=use_gui)
        if success:
            metrics = extract_sumo_metrics('tripinfo_Webster.xml')
            if metrics:
                os.makedirs(Path(metrics_output).parent, exist_ok=True)
                with open(metrics_output, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
            else:
                print("ERROR: Could not extract metrics from tripinfo output.")
        else:
            print("Simulation did not complete successfully.")
    else:
        if run_sumo and not SUMO_AVAILABLE:
            print("SUMO Python libraries not available; skipping simulation.")

    results['sumo_success'] = success
    results['metrics'] = metrics
    return results


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

    results = run_webster_pipeline()

    plan = results['plan']
    print("\n" + "=" * 80)
    print("WEBSTER PLAN SUMMARY")
    print("=" * 80)
    print(f"Cycle length: {plan['cycle_length']:.2f}s")
    for appr, data in plan['approaches'].items():
        g = data.get('green', 0.0)
        a = data.get('amber', 0.0)
        r = data.get('red', 0.0)
        print(f"{appr}: green={g:.2f}, amber={a:.2f}, red={r:.2f}")

    if results['metrics']:
        print_sumo_metrics(results['metrics'])
        print(f"\nSaved SUMO metrics to {results['files']['metrics']}")
    elif results['sumo_available']:
        print("No SUMO metrics could be extracted.")
    else:
        print("SUMO run skipped; generated files ready for manual execution.")

    print("\nGenerated files:")
    for label, path in results['files'].items():
        print(f"  - {label}: {path}")


if __name__ == "__main__":
    main()

