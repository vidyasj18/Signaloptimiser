import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from typing import Optional, Tuple, Dict
import io

# PCU factors based on IRC 106-1990
PCU_FACTORS = {
    'car': 1.0,
    'truck': 3.0,
    'bus': 3.0,
    'motorcycle': 0.5,
    'bicycle': 0.5,
    'person': 0.0,  # Not counted for PCU
    'train': 4.0,
    'boat': 0.0,    # Not applicable for road traffic
    'traffic light': 0.0,  # Not a vehicle
    'fire hydrant': 0.0,   # Not a vehicle
    'stop sign': 0.0,      # Not a vehicle
    'parking meter': 0.0,  # Not a vehicle
    'bench': 0.0,          # Not a vehicle
    'bird': 0.0,           # Not a vehicle
    'cat': 0.0,            # Not a vehicle
    'dog': 0.0,            # Not a vehicle
    'horse': 0.0,          # Not a vehicle
    'sheep': 0.0,          # Not a vehicle
    'cow': 0.0,            # Not a vehicle
    'elephant': 0.0,       # Not a vehicle
    'bear': 0.0,           # Not a vehicle
    'zebra': 0.0,          # Not a vehicle
    'giraffe': 0.0,        # Not a vehicle
    'backpack': 0.0,       # Not a vehicle
    'umbrella': 0.0,       # Not a vehicle
    'handbag': 0.0,        # Not a vehicle
    'tie': 0.0,            # Not a vehicle
    'suitcase': 0.0,       # Not a vehicle
    'frisbee': 0.0,        # Not a vehicle
    'skis': 0.0,           # Not a vehicle
    'snowboard': 0.0,      # Not a vehicle
    'sports ball': 0.0,    # Not a vehicle
    'kite': 0.0,           # Not a vehicle
    'baseball bat': 0.0,   # Not a vehicle
    'baseball glove': 0.0, # Not a vehicle
    'skateboard': 0.0,     # Not a vehicle
    'surfboard': 0.0,      # Not a vehicle
    'tennis racket': 0.0,  # Not a vehicle
    'bottle': 0.0,         # Not a vehicle
    'wine glass': 0.0,     # Not a vehicle
    'cup': 0.0,            # Not a vehicle
    'fork': 0.0,           # Not a vehicle
    'knife': 0.0,          # Not a vehicle
    'spoon': 0.0,          # Not a vehicle
    'bowl': 0.0,           # Not a vehicle
    'banana': 0.0,         # Not a vehicle
    'apple': 0.0,          # Not a vehicle
    'sandwich': 0.0,       # Not a vehicle
    'orange': 0.0,         # Not a vehicle
    'broccoli': 0.0,       # Not a vehicle
    'carrot': 0.0,         # Not a vehicle
    'hot dog': 0.0,        # Not a vehicle
    'pizza': 0.0,          # Not a vehicle
    'donut': 0.0,          # Not a vehicle
    'cake': 0.0,           # Not a vehicle
    'chair': 0.0,          # Not a vehicle
    'couch': 0.0,          # Not a vehicle
    'potted plant': 0.0,   # Not a vehicle
    'bed': 0.0,            # Not a vehicle
    'dining table': 0.0,   # Not a vehicle
    'toilet': 0.0,         # Not a vehicle
    'tv': 0.0,             # Not a vehicle
    'laptop': 0.0,         # Not a vehicle
    'mouse': 0.0,          # Not a vehicle
    'remote': 0.0,         # Not a vehicle
    'keyboard': 0.0,       # Not a vehicle
    'cell phone': 0.0,     # Not a vehicle
    'microwave': 0.0,      # Not a vehicle
    'oven': 0.0,           # Not a vehicle
    'toaster': 0.0,        # Not a vehicle
    'sink': 0.0,           # Not a vehicle
    'refrigerator': 0.0,   # Not a vehicle
    'book': 0.0,           # Not a vehicle
    'clock': 0.0,          # Not a vehicle
    'vase': 0.0,           # Not a vehicle
    'scissors': 0.0,       # Not a vehicle
    'teddy bear': 0.0,     # Not a vehicle
    'hair drier': 0.0,     # Not a vehicle
    'toothbrush': 0.0,     # Not a vehicle
}

# Vehicle class mapping for YOLO
VEHICLE_CLASSES = {
    2: 'car',      # car
    3: 'motorcycle', # motorcycle
    5: 'bus',      # bus
    7: 'truck',    # truck
    1: 'bicycle',  # bicycle
}

class VehicleTracker:
    """Simple vehicle tracking to avoid double counting"""
    
    def __init__(self, max_disappeared=30, min_distance=100, min_frames=5):
        self.trackers = {}  # {track_id: {'bbox': (x1,y1,x2,y2), 'type': str, 'frames_missing': int, 'frames_seen': int, 'history': [(cx,cy)], 'counted': bool}}
        self.next_id = 0
        self.max_disappeared = max_disappeared  # frames before considering vehicle gone
        self.min_distance = min_distance  # minimum distance to consider same vehicle
        self.min_frames = min_frames  # minimum frames a vehicle must be seen to be counted
        
    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of (bbox, vehicle_type, confidence)
        """
        # If no existing trackers, create new ones
        if not self.trackers:
            for bbox, vehicle_type, conf in detections:
                center = self._get_center(bbox)
                self.trackers[self.next_id] = {
                    'bbox': bbox,
                    'type': vehicle_type,
                    'frames_missing': 0,
                    'frames_seen': 1,
                    'confidence': conf,
                    'history': [center],
                    'counted': False,
                }
                self.next_id += 1
            return list(self.trackers.keys())
        
        # Calculate distances between existing trackers and new detections
        matched_trackers = set()
        matched_detections = set()
        
        for track_id, track_info in self.trackers.items():
            if track_info['frames_missing'] >= self.max_disappeared:
                continue
                
            track_center = self._get_center(track_info['bbox'])
            best_distance = float('inf')
            best_detection_idx = -1
            
            for i, (bbox, vehicle_type, conf) in enumerate(detections):
                if i in matched_detections:
                    continue
                    
                if vehicle_type != track_info['type']:
                    continue
                
                # Calculate both distance and overlap
                detection_center = self._get_center(bbox)
                distance = self._calculate_distance(track_center, detection_center)
                overlap = self._calculate_bbox_overlap(track_info['bbox'], bbox)
                
                # Use both distance and overlap for better matching
                # Prefer overlap over distance for close objects
                if overlap > 0.1:  # Significant overlap
                    if overlap > best_distance:  # Use overlap as score
                        best_distance = overlap
                        best_detection_idx = i
                elif distance < best_distance and distance < self.min_distance:
                    best_distance = distance
                    best_detection_idx = i
            
            if best_detection_idx != -1:
                # Update existing tracker
                self.trackers[track_id]['bbox'] = detections[best_detection_idx][0]
                self.trackers[track_id]['frames_missing'] = 0
                self.trackers[track_id]['frames_seen'] += 1
                self.trackers[track_id]['confidence'] = detections[best_detection_idx][2]
                # Append center history
                new_center = self._get_center(self.trackers[track_id]['bbox'])
                self.trackers[track_id]['history'].append(new_center)
                matched_trackers.add(track_id)
                matched_detections.add(best_detection_idx)
            else:
                # Increment missing frames
                self.trackers[track_id]['frames_missing'] += 1
        
        # Add new trackers for unmatched detections
        for i, (bbox, vehicle_type, conf) in enumerate(detections):
            if i not in matched_detections:
                center = self._get_center(bbox)
                self.trackers[self.next_id] = {
                    'bbox': bbox,
                    'type': vehicle_type,
                    'frames_missing': 0,
                    'frames_seen': 1,
                    'confidence': conf,
                    'history': [center],
                    'counted': False,
                }
                self.next_id += 1
        
        # Return active trackers (not missing for too long)
        active_trackers = {track_id: info for track_id, info in self.trackers.items() 
                          if info['frames_missing'] < self.max_disappeared}
        return list(active_trackers.keys())
    
    def get_counts(self):
        """Get total vehicle counts (only count vehicles that appeared for minimum frames)"""
        counts = defaultdict(int)
        for track_info in self.trackers.values():
            if track_info['frames_seen'] >= self.min_frames:
                counts[track_info['type']] += 1
        return dict(counts)
    
    def get_frame_counts(self):
        """Get current frame vehicle counts (for frame-by-frame analysis)"""
        counts = defaultdict(int)
        for track_info in self.trackers.values():
            if track_info['frames_missing'] < self.max_disappeared:
                counts[track_info['type']] += 1
        return dict(counts)
    
    def _get_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def load_mask(mask_path: Optional[str], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load a grayscale mask image and resize to frame (w,h). Returns binary mask (0/255)."""
    if not mask_path or not os.path.exists(mask_path):
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    h, w = frame_shape[1], frame_shape[0]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def point_in_mask(mask: np.ndarray, point: Tuple[int, int]) -> bool:
    """Check if a point (x,y) lies in white area of the mask."""
    x, y = point
    h, w = mask.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return mask[y, x] > 0


def signed_side(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
    """Signed side of point p relative to line a->b (cross product sign)."""
    ax, ay = a
    bx, by = b
    px, py = p
    return np.sign((bx - ax) * (py - ay) - (by - ay) * (px - ax))


def line_signed_distance(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
    """Signed perpendicular distance from point p to infinite line a->b (pixels)."""
    ax, ay = a
    bx, by = b
    px, py = p
    denom = np.hypot(bx - ax, by - ay)
    if denom == 0:
        return 0.0
    # Cross product magnitude divided by line length with sign
    return ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / denom


def extract_first_frame_from_bytes(video_bytes: bytes) -> Optional[np.ndarray]:
    """Return first frame (BGR) from raw video bytes using a temp file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            temp_path = tmp.name
        cap = cv2.VideoCapture(temp_path)
        ok, frame = cap.read()
        cap.release()
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        if ok:
            return frame
        return None
    except Exception:
        return None

@st.cache_resource
def load_model():
    """Load YOLO model with caching"""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_video(
    video_file,
    model,
    create_annotated_video=False,
    min_frames=5,
    min_distance=100,
    max_disappeared=30,
    approach_name: Optional[str] = None,
    stopline: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    mask_path: Optional[str] = None,
    invert_stopline: bool = False,
    start_frame: int = 0,
    end_frame: int = 0,
    crossing_tolerance: int = 5,
    min_normal_motion: int = 4,
    frame_stride: int = 1,
):
    """Process video and return vehicle counts and PCU data"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        video_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Normalize frame bounds
        start_frame = max(0, int(start_frame))
        end_frame = int(end_frame) if int(end_frame) > 0 else total_frames
        end_frame = min(end_frame, total_frames)
        frames_to_process = max(0, end_frame - start_frame)
        duration = frames_to_process / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.info(f"Video Info: {total_frames} frames, {fps} FPS, Duration: {duration:.2f} seconds, Resolution: {width}x{height}")

        # Prepare mask if provided
        mask = load_mask(mask_path, (width, height)) if mask_path else None
        
        # Initialize tracker and counters
        tracker = VehicleTracker(max_disappeared=max_disappeared, min_distance=min_distance, min_frames=min_frames)
        vehicle_counts: Dict[str, int] = {}
        frame_data = []
        frame_count = 0
        current_frame_index = -1
        inbound_counts: Dict[str, int] = defaultdict(int)
        
        # Video writer for annotated video
        annotated_video_path = None
        video_writer = None
        if create_annotated_video:
            annotated_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            # Use a more compatible codec for HTML5 players
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(annotated_video_path, fourcc, max(fps, 1), (width, height))
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame_index += 1
            if current_frame_index < start_frame:
                continue
            frame_count += 1
            
            # Update progress
            progress_den = frames_to_process if frames_to_process > 0 else max(1, total_frames)
            progress = min(1.0, frame_count / progress_den)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {start_frame + frame_count}/{end_frame}")
            if start_frame + frame_count >= end_frame:
                break
            
            # Apply frame stride (process every Nth frame)
            if frame_stride and frame_stride > 1:
                if ((current_frame_index - start_frame) % frame_stride) != 0:
                    continue

            # Run YOLO detection (optionally masked for visualization only; detection still on full frame)
            results = model(frame, verbose=False)
            
            # Process detections
            frame_vehicles = {}
            annotated_frame = frame.copy() if create_annotated_video else None
            
            # Collect detections for tracking
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Only process vehicles with confidence > 0.5
                        if conf > 0.5 and cls in VEHICLE_CLASSES:
                            vehicle_type = VEHICLE_CLASSES[cls]
                            bbox = tuple(box.xyxy[0].cpu().numpy())
                            # If mask supplied, keep only detections whose center lies in white region
                            if mask is not None:
                                cx, cy = map(int, ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
                                if not point_in_mask(mask, (cx, cy)):
                                    continue
                            detections.append((bbox, vehicle_type, conf))
            
            # Update tracker
            active_trackers = tracker.update(detections)
            
            # Count vehicles in current frame (for frame-by-frame analysis)
            frame_vehicles = tracker.get_frame_counts()
            
            # Draw annotations on frame
            if create_annotated_video:
                # Draw all current detections
                for bbox, vehicle_type, conf in detections:
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Calculate center for circle
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    radius = int(min(x2 - x1, y2 - y1) / 2)
                    
                    # Color coding based on vehicle type
                    color_map = {
                        'car': (0, 255, 0),      # Green
                        'truck': (255, 0, 0),    # Blue
                        'bus': (0, 0, 255),      # Red
                        'motorcycle': (255, 255, 0),  # Cyan
                        'bicycle': (255, 0, 255)      # Magenta
                    }
                    color = color_map.get(vehicle_type, (255, 255, 255))
                    
                    # Draw circle around vehicle
                    cv2.circle(annotated_frame, (center_x, center_y), radius, color, 2)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{vehicle_type.upper()}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw stopline if provided
                if stopline is not None:
                    (ax, ay), (bx, by) = stopline
                    cv2.line(annotated_frame, (int(ax), int(ay)), (int(bx), int(by)), (0, 165, 255), 3)
                # Overlay mask outline (optional): draw its contours faintly
                if mask is not None:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_frame, contours, -1, (0, 255, 255), 2)
            
            # Write annotated frame to video
            if create_annotated_video and video_writer:
                video_writer.write(annotated_frame)
            
            # If using inbound stopline, check crossing events and count once per track
            if stopline is not None:
                a, b = stopline
                for tid, info in tracker.trackers.items():
                    if info.get('counted', False):
                        continue
                    hist = info.get('history', [])
                    if len(hist) < 2 or info['frames_seen'] < min_frames:
                        continue
                    # Robust crossing test using signed distance and tolerance band
                    d_prev = line_signed_distance(a, b, hist[-2])
                    d_curr = line_signed_distance(a, b, hist[-1])
                    # Must move across the line with minimum normal motion
                    crossed_band = (abs(d_prev) <= crossing_tolerance and abs(d_curr) <= crossing_tolerance)
                    crossed_sign = (d_prev * d_curr) < 0 and abs(d_curr - d_prev) >= min_normal_motion
                    s_prev = np.sign(d_prev)
                    s_curr = np.sign(d_curr)
                    crossed_default = (s_prev > 0 and s_curr < 0) and (crossed_sign or crossed_band)
                    crossed_inverted = (s_prev < 0 and s_curr > 0) and (crossed_sign or crossed_band)
                    crossed = crossed_inverted if invert_stopline else crossed_default
                    if crossed:
                        inbound_counts[info['type']] += 1
                        info['counted'] = True

            # Store frame data with zero-filled vehicle columns for stability
            row = {
                'frame': frame_count,
                'time': frame_count / max(fps, 1),
            }
            for vtype in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                row[vtype] = frame_vehicles.get(vtype, 0)
            frame_data.append(row)
        
        cap.release()
        if video_writer:
            video_writer.release()
        progress_bar.empty()
        status_text.empty()
        
        # Final counts
        if stopline is not None:
            vehicle_counts = dict(inbound_counts)
            # Defensive fallback: if nothing crossed but there were detections/tracks, fall back to persistence counts
            if sum(vehicle_counts.values()) == 0:
                fallback_counts = tracker.get_counts()
                if sum(fallback_counts.values()) > 0:
                    st.warning("No inbound crossings detected; falling back to unique tracker counts. Check stopline/invert.")
                    vehicle_counts = fallback_counts
        else:
            # Fallback: use tracker persistence-based counts
            vehicle_counts = tracker.get_counts()
        
        # Debug information
        st.info(f"🔍 **Tracking Debug:** Total trackers created: {tracker.next_id}, Final unique vehicles: {sum(vehicle_counts.values())}")
        
        # Calculate PCU
        total_pcu = sum(vehicle_counts.get(vehicle_type, 0) * PCU_FACTORS.get(vehicle_type, 0) 
                       for vehicle_type in vehicle_counts)
        
        result_obj = {
            'vehicle_counts': vehicle_counts,
            'total_pcu': total_pcu,
            'frame_data': frame_data,
            'duration': duration,
            'total_frames': total_frames,
            'fps': fps,
            'annotated_video_path': annotated_video_path,
            'approach': approach_name or 'UNKNOWN'
        }

        # Write per-approach CSV summary if approach is specified
        try:
            approach_tag = (approach_name or 'UNKNOWN').upper()
            os.makedirs('outputs', exist_ok=True)
            summary_rows = []
            for vtype, count in vehicle_counts.items():
                summary_rows.append({
                    'Approach': approach_tag,
                    'Vehicle Type': vtype,
                    'Count': count,
                    'PCU Factor': PCU_FACTORS.get(vtype, 0),
                    'PCU Value': count * PCU_FACTORS.get(vtype, 0)
                })
            summary_rows.append({
                'Approach': approach_tag,
                'Vehicle Type': 'TOTAL',
                'Count': sum(vehicle_counts.values()),
                'PCU Factor': '-',
                'PCU Value': total_pcu
            })
            pd.DataFrame(summary_rows).to_csv(f'outputs/{approach_tag}_summary.csv', index=False)
        except Exception:
            pass

        return result_obj
        
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception:
                # On Windows the file may still be locked briefly; ignore
                pass

def create_download_data(vehicle_counts, total_pcu, frame_data):
    """Create downloadable data"""
    # Summary data
    summary_data = []
    for vehicle_type, count in vehicle_counts.items():
        pcu_factor = PCU_FACTORS.get(vehicle_type, 0)
        pcu_value = count * pcu_factor
        summary_data.append({
            'Vehicle Type': vehicle_type,
            'Count': count,
            'PCU Factor': pcu_factor,
            'PCU Value': pcu_value
        })
    
    summary_data.append({
        'Vehicle Type': 'TOTAL PCU',
        'Count': sum(vehicle_counts.values()),
        'PCU Factor': '-',
        'PCU Value': total_pcu
    })
    
    return pd.DataFrame(summary_data), pd.DataFrame(frame_data)

def main():
    st.set_page_config(
        page_title="Vehicle Counting & PCU Calculator",
        page_icon="🚗",
        layout="wide"
    )
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'annotated_video_path' not in st.session_state:
        st.session_state.annotated_video_path = None
    if 'annotated_video_bytes' not in st.session_state:
        st.session_state.annotated_video_bytes = None
    if 'summary_df' not in st.session_state:
        st.session_state.summary_df = None
    if 'frame_df' not in st.session_state:
        st.session_state.frame_df = None
    
    st.title("🚗 Vehicle Counting & PCU Calculator")
    st.markdown("Upload a video to detect vehicles and calculate Passenger Car Units (PCU) based on IRC 106-1990 standards")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load YOLO model. Please check your installation.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze vehicle traffic"
    )

    # Approach selector and inbound gating options (per-video)
    approach = st.selectbox("Approach for this video", ["NB", "SB", "EB", "WB"], index=0, help="All counts/PCU will be attributed to this approach")
    use_stopline = st.checkbox("Count inbound only (use stopline crossing)", value=True)
    mask_file = st.file_uploader("Optional ROI mask (grayscale; white=analyze, black=ignore)", type=["png", "jpg", "jpeg"], help="ATLAS-style mask to focus on inbound lanes")
    col_sl1, col_sl2, col_sl3, col_sl4 = st.columns(4)
    stopline_coords = None
    invert_stopline = False
    if use_stopline:
        with col_sl1:
            ax = st.number_input("Stopline Ax", min_value=0, value=0)
        with col_sl2:
            ay = st.number_input("Stopline Ay", min_value=0, value=480, help="For 1080p, a horizontal line at y=480 is typical")
        with col_sl3:
            bx = st.number_input("Stopline Bx", min_value=0, value=1919)
        with col_sl4:
            by = st.number_input("Stopline By", min_value=0, value=480)
        stopline_coords = ((int(ax), int(ay)), (int(bx), int(by)))
        invert_stopline = st.checkbox("Invert inbound direction", value=False, help="Enable if totals stay 0 but vehicles cross the line")
        col_tol1, col_tol2 = st.columns(2)
        with col_tol1:
            crossing_tolerance = st.slider("Crossing tolerance (px)", 0, 20, 5, help="Band around the line for robust crossing detection")
        with col_tol2:
            min_normal_motion = st.slider("Min normal motion (px)", 0, 30, 4, help="Required motion across the line between frames")
    else:
        crossing_tolerance = 5
        min_normal_motion = 4
    
    # Optional frame range selection
    col_fr1, col_fr2 = st.columns(2)
    with col_fr1:
        start_frame = st.number_input("Start frame (inclusive)", min_value=0, value=0, help="0 means from the beginning")
    with col_fr2:
        end_frame = st.number_input("End frame (exclusive, 0=till end)", min_value=0, value=0, help="0 processes till last frame")

    if uploaded_file is not None:
        # Read bytes once for preview and later processing
        file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file.read()
        st.video(file_bytes)
        
        # Processing options
        st.subheader("⚙️ Processing Options")
        col1, col2 = st.columns(2)
        
        with col1:
            create_annotated_video = st.checkbox(
                "🎯 Create annotated video with vehicle detection",
                value=True,
                help="Generate a new video with circles and bounding boxes around detected vehicles"
            )
        
        with col2:
            if create_annotated_video:
                st.info("✅ Annotated video will be created with colored circles and labels around detected vehicles")
            else:
                st.info("ℹ️ Only counting analysis will be performed (faster processing)")
        
        # Advanced tracking options
        with st.expander("🔧 Advanced Tracking Options"):
            col1, col2, col3 = st.columns(3)
            with col1:
                min_frames = st.slider("Min frames to count vehicle", 1, 20, 5, 
                                     help="Minimum frames a vehicle must appear to be counted")
            with col2:
                min_distance = st.slider("Min distance for tracking", 50, 200, 100, 
                                       help="Minimum distance to consider same vehicle")
            with col3:
                max_disappeared = st.slider("Max frames disappeared", 10, 60, 30, 
                                          help="Frames before considering vehicle gone")
        
        # Live preview of stopline/mask on first frame
        with st.expander("👁️ Preview stopline on first frame", expanded=True):
            first_frame = extract_first_frame_from_bytes(file_bytes)
            if first_frame is not None:
                preview = first_frame.copy()
                # Overlay mask contour if any
                mask_preview_path = None
                if mask_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_mask:
                        tmp_mask.write(mask_file.getvalue() if hasattr(mask_file, 'getvalue') else mask_file.read())
                        mask_preview_path = tmp_mask.name
                    mask_img = load_mask(mask_preview_path, (preview.shape[1], preview.shape[0]))
                    if mask_img is not None:
                        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(preview, contours, -1, (0, 255, 255), 2)
                    try:
                        os.unlink(mask_preview_path)
                    except Exception:
                        pass
                # Draw stopline if enabled
                if use_stopline:
                    (axp, ayp), (bxp, byp) = stopline_coords
                    cv2.line(preview, (int(axp), int(ayp)), (int(bxp), int(byp)), (0, 165, 255), 3)
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="First frame with stopline/mask", use_container_width=True)

        if st.button("🚀 Process Video", type="primary"):
            with st.spinner("Processing video... This may take a while depending on video length."):
                # Persist mask to temp if provided
                mask_path = None
                if mask_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_mask:
                        tmp_mask.write(mask_file.getvalue() if hasattr(mask_file, 'getvalue') else mask_file.read())
                        mask_path = tmp_mask.name
                
                # Recreate an in-memory file object for processing
                video_file_like = io.BytesIO(file_bytes)
                
                results = process_video(
                    video_file_like,
                    model,
                    create_annotated_video,
                    min_frames,
                    min_distance,
                    max_disappeared,
                    approach_name=approach,
                    stopline=stopline_coords if use_stopline else None,
                    mask_path=mask_path,
                    invert_stopline=invert_stopline,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    crossing_tolerance=crossing_tolerance,
                    min_normal_motion=min_normal_motion,
                )
            
            if results:
                # Store results in session state
                st.session_state.results = results
                
                # Store annotated video path and bytes if available
                if results.get('annotated_video_path') and os.path.exists(results['annotated_video_path']):
                    st.session_state.annotated_video_path = results['annotated_video_path']
                    with open(results['annotated_video_path'], 'rb') as video_file:
                        st.session_state.annotated_video_bytes = video_file.read()
                
                # Store dataframes
                summary_df, frame_df = create_download_data(results['vehicle_counts'], results['total_pcu'], results['frame_data'])
                st.session_state.summary_df = summary_df
                st.session_state.frame_df = frame_df
                vehicle_counts = results['vehicle_counts']
                total_pcu = results['total_pcu']
                frame_data = results['frame_data']
                duration = results['duration']
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Vehicles", sum(vehicle_counts.values()))
                
                with col2:
                    st.metric("Total PCU", f"{total_pcu:.2f}")
                
                with col3:
                    st.metric("Video Duration", f"{duration:.2f}s")
                
                # Show tracking info
                st.info("🔄 **Tracking System:** Vehicles are tracked across frames to prevent double-counting. Each unique vehicle is counted only once.")
                
                # Vehicle counts table
                st.subheader("📊 Vehicle Counts by Type")
                if vehicle_counts:
                    count_df = pd.DataFrame([
                        {
                            'Vehicle Type': vehicle_type,
                            'Count': count,
                            'PCU Factor': PCU_FACTORS.get(vehicle_type, 0),
                            'PCU Value': count * PCU_FACTORS.get(vehicle_type, 0)
                        }
                        for vehicle_type, count in vehicle_counts.items()
                    ])
                    st.dataframe(count_df, use_container_width=True)
                    
                    # Bar chart
                    fig = px.bar(
                        count_df,
                        x='Vehicle Type',
                        y='Count',
                        title='Vehicle Counts by Type',
                        color='Vehicle Type'
                    )
                    st.plotly_chart(fig, use_container_width=True, key="main_bar_chart")
                else:
                    st.warning("No vehicles detected in the video.")
                
                # Frame-by-frame analysis
                if frame_data:
                    st.subheader("📈 Frame-by-Frame Analysis")
                    
                    # Create time series chart
                    frame_df = pd.DataFrame(frame_data)
                    vehicle_columns = [col for col in frame_df.columns if col not in ['frame', 'time']]
                    
                    if vehicle_columns:
                        fig = go.Figure()
                        for vehicle_type in vehicle_columns:
                            fig.add_trace(go.Scatter(
                                x=frame_df['time'],
                                y=frame_df[vehicle_type],
                                mode='lines',
                                name=vehicle_type,
                                stackgroup='one'
                            ))
                        
                        fig.update_layout(
                            title='Vehicle Counts Over Time',
                            xaxis_title='Time (seconds)',
                            yaxis_title='Vehicle Count',
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True, key="main_time_series")
                
                # Annotated video section
                if results.get('annotated_video_path') and os.path.exists(results['annotated_video_path']):
                    st.subheader("🎯 Annotated Video")
                    st.success("✅ Annotated video created successfully!")
                    
                    # Display annotated video (prefer bytes for browser compatibility)
                    with open(results['annotated_video_path'], 'rb') as video_file:
                        video_bytes = video_file.read()
                    st.session_state.annotated_video_bytes = video_bytes
                    st.video(video_bytes)
                    
                    # Download annotated video
                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=video_bytes,
                        file_name=f"annotated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4",
                        key="dl_annot_current"
                    )
                    
                    # Don't delete the file immediately - keep it for display
                    # It will be cleaned up when session state is cleared
                
                # Download section
                st.subheader("💾 Download Results")
                summary_df, frame_df = create_download_data(vehicle_counts, total_pcu, frame_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=csv_summary,
                        file_name=f"vehicle_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="dl_summary_current"
                    )
                
                with col2:
                    csv_frames = frame_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Frame Data CSV",
                        data=csv_frames,
                        file_name=f"frame_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="dl_frames_current"
                    )

                # Per-approach CSV was also saved under outputs/<APPROACH>_summary.csv
                st.info(f"Per-approach summary saved to outputs/{approach.upper()}_summary.csv")
    
    # ===========================
    # 🧠 Adaptive Simulator (beta)
    # ===========================
    with st.expander("🧠 Adaptive Simulator (beta)", expanded=False):
        st.markdown("Simulate cycle-by-cycle plans. Warm-up 20s; stride defaults to 3.")
        # Videos per approach
        sim_inputs = {}
        cols = st.columns(4)
        for i, appr in enumerate(["N", "S", "E", "W"]):
            with cols[i]:
                sim_inputs[appr] = st.file_uploader(f"{appr} video", type=["mp4","avi","mov","mkv"], key=f"sim_vid_{appr}")
        warmup_s = st.number_input("Warm-up seconds", min_value=5, max_value=60, value=20)
        frame_stride_sim = st.slider("Frame stride", 1, 5, 3)
        brain = st.selectbox("Brain", ["ML (default)", "Webster"], index=0)
        # One stopline per video
        st.markdown("Define stoplines per approach (ignored if no video).")
        sl = {}
        for appr in ["N","S","E","W"]:
            with st.expander(f"Stopline for {appr}"):
                c1, c2, c3, c4 = st.columns(4)
                ax_ = c1.number_input(f"{appr} Ax", min_value=0, value=0, key=f"{appr}_ax")
                ay_ = c2.number_input(f"{appr} Ay", min_value=0, value=480, key=f"{appr}_ay")
                bx_ = c3.number_input(f"{appr} Bx", min_value=0, value=1919, key=f"{appr}_bx")
                by_ = c4.number_input(f"{appr} By", min_value=0, value=480, key=f"{appr}_by")
                inv_ = st.checkbox(f"Invert {appr}", value=False, key=f"{appr}_inv")
                sl[appr] = {"line": ((int(ax_), int(ay_)), (int(bx_), int(by_))), "invert": inv_}

        # Helper: count PCU in [start_s, end_s] for one video
        def count_slice(file_obj, start_s, end_s, line, invert):
            if file_obj is None:
                return 0.0
            fb = file_obj.getvalue() if hasattr(file_obj, 'getvalue') else file_obj.read()
            # Probe fps
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(fb)
                p = tmp.name
            cap = cv2.VideoCapture(p)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            try:
                os.unlink(p)
            except Exception:
                pass
            start_f = int(float(start_s) * fps)
            end_f = int(float(end_s) * fps)
            # Local defaults to avoid dependency on other UI controls
            min_frames_sim = 5
            min_distance_sim = 100
            max_disappeared_sim = 30
            res = process_video(
                io.BytesIO(fb), model, False, min_frames_sim, min_distance_sim, max_disappeared_sim,
                approach_name=None,
                stopline=line,
                mask_path=None,
                invert_stopline=invert,
                start_frame=start_f,
                end_frame=end_f,
                crossing_tolerance=5,
                min_normal_motion=4,
                frame_stride=frame_stride_sim,
            )
            return res['total_pcu'] if res else 0.0

        # Brain: Webster fallback (ML wrapper can plug in later)
        def plan_from_pcu(pcu_dict):
            N, S, E, W = [pcu_dict.get(x, 0.0) for x in ['N','S','E','W']]
            NS, EW = N+S, E+W
            num_present = sum(1 for v in [N,S,E,W] if v > 0)
            capacity = max(1, num_present) * DEFAULT_LANES * SAT_PER_LANE
            Y = min((NS+EW)/capacity if capacity>0 else 0.0, 0.95)
            C = (1.5*DEFAULT_LOST_TIME + 5)/(1-Y) if Y < 0.95 else 180.0
            C = float(np.clip(C, 60.0, 180.0))
            eff = max(0.0, C-DEFAULT_LOST_TIME)
            g_NS = eff * (NS/(NS+EW)) if (NS+EW) > 0 else 0.0
            g_EW = eff - g_NS
            split = {}
            split['NB'] = g_NS * (N/NS) if NS>0 and N>0 else 0.0
            split['SB'] = g_NS * (S/NS) if NS>0 and S>0 else 0.0
            split['EB'] = g_EW * (E/EW) if EW>0 and E>0 else 0.0
            split['WB'] = g_EW * (W/EW) if EW>0 and W>0 else 0.0
            return {"cycle": C, "g_NS": g_NS, "g_EW": g_EW, **split}

        if st.button("▶️ Start Simulation", type="primary", key="btn_start_sim"):
            with st.spinner("Running warm-up slice..."):
                pcu = {}
                for appr in ["N","S","E","W"]:
                    pcu[appr] = count_slice(sim_inputs[appr], 0.0, float(warmup_s), sl[appr]["line"], sl[appr]["invert"]) if sim_inputs.get(appr) else 0.0
                # Convert to per-second rate
                rate = {k: (v/float(warmup_s)) for k,v in pcu.items()}
                # Use rate directly as planning load (Webster scales internally)
                plan = plan_from_pcu(rate)
            st.success("Warm-up complete. Showing next cycle plan (preview):")
            st.write({k: (round(v,2) if isinstance(v,float) else v) for k,v in plan.items()})
            # Preview phase diagram
            try:
                fig = go.Figure()
                C = plan['cycle']
                gNS = plan['g_NS']; gEW = plan['g_EW']
                t = 0.0
                if gNS>0:
                    fig.add_trace(go.Bar(x=[gNS], y=["NS"], orientation='h', base=t, marker_color="#2ecc71", name='green'))
                    t += gNS
                    fig.add_trace(go.Bar(x=[YELLOW], y=["NS"], orientation='h', base=t, marker_color="#f1c40f", name='amber'))
                    t += YELLOW
                if gNS>0 and gEW>0:
                    fig.add_trace(go.Bar(x=[ALL_RED], y=["NS"], orientation='h', base=t, marker_color="#e74c3c", name='red'))
                    fig.add_trace(go.Bar(x=[ALL_RED], y=["EW"], orientation='h', base=t, marker_color="#e74c3c", showlegend=False))
                    t += ALL_RED
                if gEW>0:
                    fig.add_trace(go.Bar(x=[gEW], y=["EW"], orientation='h', base=t, marker_color="#2ecc71", showlegend=False))
                    t += gEW
                    fig.add_trace(go.Bar(x=[YELLOW], y=["EW"], orientation='h', base=t, marker_color="#f1c40f", showlegend=False))
                    t += YELLOW
                if t < C:
                    tail = C - t
                    fig.add_trace(go.Bar(x=[tail], y=["NS"], orientation='h', base=t, marker_color="#e74c3c", showlegend=False))
                    fig.add_trace(go.Bar(x=[tail], y=["EW"], orientation='h', base=t, marker_color="#e74c3c", showlegend=False))
                fig.update_layout(barmode='stack', title=f"Next Cycle Preview — C={C:.1f}s", xaxis_title='Time (s)', yaxis_title='Phase Group', height=300)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        # Full run loop
        if st.button("⏩ Run Full Simulation", type="secondary", key="btn_run_full_sim"):
            # Probe durations (min duration defines horizon)
            durations = {}
            for appr in ["N","S","E","W"]:
                f = sim_inputs.get(appr)
                if not f:
                    durations[appr] = 0.0
                    continue
                fb = f.getvalue() if hasattr(f, 'getvalue') else f.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(fb)
                    p = tmp.name
                cap = cv2.VideoCapture(p)
                fps_ = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frames_ = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                cap.release()
                try:
                    os.unlink(p)
                except Exception:
                    pass
                durations[appr] = float(frames_/max(fps_,1.0))
            sim_horizon = max(0.0, min([d for d in durations.values() if d>0] + [0.0]))

            # Warm-up first slice
            logs = []
            T = 0.0
            pcu1 = {}
            for appr in ["N","S","E","W"]:
                pcu1[appr] = count_slice(sim_inputs[appr], T, min(T+float(warmup_s), sim_horizon), sl[appr]["line"], sl[appr]["invert"]) if sim_inputs.get(appr) else 0.0
            dur1 = float(min(warmup_s, sim_horizon - T)) if sim_horizon > T else 0.0
            rate1 = {k: (pcu1[k]/dur1 if dur1>0 else 0.0) for k in ["N","S","E","W"]}
            plan1 = plan_from_pcu(rate1)
            # Commit cycle 1
            T_end = T + plan1['cycle']
            pcu_cycle1 = {}
            for appr in ["N","S","E","W"]:
                pcu_cycle1[appr] = count_slice(sim_inputs[appr], T, min(T_end, sim_horizon), sl[appr]["line"], sl[appr]["invert"]) if sim_inputs.get(appr) else 0.0
            logs.append({"t_start": T, "t_end": min(T_end, sim_horizon), "pcu": pcu_cycle1, "plan": plan1})
            T = T_end

            # Prepare placeholders for charts
            os.makedirs('outputs', exist_ok=True)
            ts_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = f'outputs/sim_{ts_name}.json'
            import json
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            cycle_hist = [plan1['cycle']]
            gns_hist = [plan1['g_NS']]
            gew_hist = [plan1['g_EW']]
            line_ph = st.empty()
            bar_ph = st.empty()
            next_ph = st.empty()

            # Main loop
            last_rates = [rate1]
            while T < sim_horizon:
                # Average of last two cycle rates (per-second)
                if len(last_rates) >= 2:
                    avg_rate = {k: (last_rates[-1][k] + last_rates[-2][k]) / 2.0 for k in ["N","S","E","W"]}
                else:
                    avg_rate = last_rates[-1]
                plan_next = plan_from_pcu(avg_rate)

                # Preview next plan
                with next_ph.container():
                    try:
                        fig2 = go.Figure()
                        Cn = plan_next['cycle']
                        gNSn = plan_next['g_NS']; gEWn = plan_next['g_EW']
                        t2 = 0.0
                        if gNSn>0:
                            fig2.add_trace(go.Bar(x=[gNSn], y=["NS"], orientation='h', base=t2, marker_color="#2ecc71", name='green'))
                            t2 += gNSn
                            fig2.add_trace(go.Bar(x=[YELLOW], y=["NS"], orientation='h', base=t2, marker_color="#f1c40f", name='amber'))
                            t2 += YELLOW
                        if gNSn>0 and gEWn>0:
                            fig2.add_trace(go.Bar(x=[ALL_RED], y=["NS"], orientation='h', base=t2, marker_color="#e74c3c", name='red'))
                            fig2.add_trace(go.Bar(x=[ALL_RED], y=["EW"], orientation='h', base=t2, marker_color="#e74c3c", showlegend=False))
                            t2 += ALL_RED
                        if gEWn>0:
                            fig2.add_trace(go.Bar(x=[gEWn], y=["EW"], orientation='h', base=t2, marker_color="#2ecc71", showlegend=False))
                            t2 += gEWn
                            fig2.add_trace(go.Bar(x=[YELLOW], y=["EW"], orientation='h', base=t2, marker_color="#f1c40f", showlegend=False))
                            t2 += YELLOW
                        if t2 < Cn:
                            tail2 = Cn - t2
                            fig2.add_trace(go.Bar(x=[tail2], y=["NS"], orientation='h', base=t2, marker_color="#e74c3c", showlegend=False))
                            fig2.add_trace(go.Bar(x=[tail2], y=["EW"], orientation='h', base=t2, marker_color="#e74c3c", showlegend=False))
                        fig2.update_layout(barmode='stack', title=f"Next Cycle Preview — C={Cn:.1f}s", xaxis_title='Time (s)', yaxis_title='Phase Group', height=300)
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception:
                        pass

                # Execute this plan cycle
                T_end2 = T + plan_next['cycle']
                pcu_cycle = {}
                for appr in ["N","S","E","W"]:
                    pcu_cycle[appr] = count_slice(sim_inputs[appr], T, min(T_end2, sim_horizon), sl[appr]["line"], sl[appr]["invert"]) if sim_inputs.get(appr) else 0.0
                dur_k = float(max(0.0, min(T_end2, sim_horizon) - T))
                rate_k = {k: (pcu_cycle[k]/dur_k if dur_k>0 else 0.0) for k in ["N","S","E","W"]}
                last_rates.append(rate_k)
                logs.append({"t_start": T, "t_end": min(T_end2, sim_horizon), "pcu": pcu_cycle, "plan": plan_next})
                T = T_end2

                # Persist logs and update charts
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)
                cycle_hist.append(plan_next['cycle'])
                gns_hist.append(plan_next['g_NS'])
                gew_hist.append(plan_next['g_EW'])
                # Charts
                with line_ph.container():
                    try:
                        figL = go.Figure()
                        figL.add_trace(go.Scatter(y=cycle_hist, x=list(range(1, len(cycle_hist)+1)), mode='lines+markers', name='Cycle'))
                        figL.update_layout(title='Cycle Length per Cycle', xaxis_title='Cycle #', yaxis_title='Seconds', height=280)
                        st.plotly_chart(figL, use_container_width=True)
                    except Exception:
                        pass
                with bar_ph.container():
                    try:
                        figB = go.Figure()
                        figB.add_trace(go.Bar(y=gns_hist, x=list(range(1, len(gns_hist)+1)), name='g_NS'))
                        figB.add_trace(go.Bar(y=gew_hist, x=list(range(1, len(gew_hist)+1)), name='g_EW'))
                        figB.update_layout(barmode='stack', title='NS/EW Greens per Cycle', xaxis_title='Cycle #', yaxis_title='Seconds', height=280)
                        st.plotly_chart(figB, use_container_width=True)
                    except Exception:
                        pass

            st.success(f"Simulation complete. Log saved to {log_path}")
    # Batch processing for four approaches (simplified)
    with st.expander("📦 Batch: Process four approach videos (NB/SB/EB/WB)"):
        st.markdown("Upload up to four videos below. We'll process them one by one and save per-approach CSVs plus a combined summary map for your notebook.")
        batch_inputs = {}
        for appr in ["NB", "SB", "EB", "WB"]:
            st.markdown(f"##### {appr}")
            file_obj = st.file_uploader(
                f"Video for {appr}", type=['mp4','avi','mov','mkv'], key=f"batch_file_{appr}"
            )
            invert_b = st.checkbox(f"Invert inbound direction ({appr})", value=False, key=f"batch_invert_{appr}")
            colb1, colb2 = st.columns(2)
            start_frame_b = colb1.number_input(f"{appr} Start frame", min_value=0, value=0, key=f"batch_start_{appr}")
            end_frame_b = colb2.number_input(f"{appr} End frame (0=end)", min_value=0, value=0, key=f"batch_end_{appr}")
            batch_inputs[appr] = {
                'file': file_obj,
                'invert': invert_b,
                'start': int(start_frame_b),
                'end': int(end_frame_b),
            }

        if st.button("🚀 Process All Four", type="primary", key="btn_process_all_four"):
            batch_results = {}
            os.makedirs('outputs', exist_ok=True)
            for appr, cfg in batch_inputs.items():
                if cfg['file'] is None:
                    continue
                file_bytes = cfg['file'].getvalue() if hasattr(cfg['file'], 'getvalue') else cfg['file'].read()
                video_like = io.BytesIO(file_bytes)
                res = process_video(
                    video_like,
                    model,
                    create_annotated_video=False,
                    min_frames=min_frames,
                    min_distance=min_distance,
                    max_disappeared=max_disappeared,
                    approach_name=appr,
                    stopline=stopline_coords if use_stopline else None,
                    mask_path=None,
                    invert_stopline=cfg['invert'],
                    start_frame=cfg['start'],
                    end_frame=cfg['end'],
                    crossing_tolerance=crossing_tolerance,
                    min_normal_motion=min_normal_motion,
                )
                if res:
                    batch_results[appr] = {
                        'vehicle_counts': res['vehicle_counts'],
                        'total_pcu': res['total_pcu'],
                        'duration': res['duration']
                    }
            if batch_results:
                import json
                with open('outputs/intersection_summary.json', 'w', encoding='utf-8') as f:
                    json.dump(batch_results, f, ensure_ascii=False, indent=2)
                rows = []
                for appr, data in batch_results.items():
                    rows.append({'Approach': appr, 'Total PCU': data['total_pcu']})
                pd.DataFrame(rows).to_csv('outputs/intersection_summary.csv', index=False)
                st.success("Batch completed. Saved outputs/intersection_summary.json and outputs/intersection_summary.csv")

    # Display results from session state if available (for when app re-runs after download)
    if st.session_state.results is not None:
        st.markdown("---")
        st.subheader("📊 Previous Analysis Results")
        
        # Clear results button
        if st.button("🗑️ Clear Previous Results"):
            # Clean up video file if it exists
            if st.session_state.annotated_video_path and os.path.exists(st.session_state.annotated_video_path):
                try:
                    os.unlink(st.session_state.annotated_video_path)
                except:
                    pass
            
            st.session_state.results = None
            st.session_state.annotated_video_path = None
            st.session_state.annotated_video_bytes = None
            st.session_state.summary_df = None
            st.session_state.frame_df = None
            st.rerun()
        
        results = st.session_state.results
        vehicle_counts = results['vehicle_counts']
        total_pcu = results['total_pcu']
        duration = results['duration']
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Vehicles", sum(vehicle_counts.values()))
        
        with col2:
            st.metric("Total PCU", f"{total_pcu:.2f}")
        
        with col3:
            st.metric("Video Duration", f"{duration:.2f}s")
        
        # Show tracking info
        st.info("🔄 **Tracking System:** Vehicles are tracked across frames to prevent double-counting. Each unique vehicle is counted only once.")
        
        # Vehicle counts table
        st.subheader("📊 Vehicle Counts by Type")
        if vehicle_counts:
            count_df = pd.DataFrame([
                {
                    'Vehicle Type': vehicle_type,
                    'Count': count,
                    'PCU Factor': PCU_FACTORS.get(vehicle_type, 0),
                    'PCU Value': count * PCU_FACTORS.get(vehicle_type, 0)
                }
                for vehicle_type, count in vehicle_counts.items()
            ])
            st.dataframe(count_df, use_container_width=True)
            
            # Bar chart
            fig = px.bar(
                count_df,
                x='Vehicle Type',
                y='Count',
                title='Vehicle Counts by Type',
                color='Vehicle Type'
            )
            st.plotly_chart(fig, use_container_width=True, key="session_bar_chart")
        
        # Frame-by-frame analysis
        if results['frame_data']:
            st.subheader("📈 Frame-by-Frame Analysis")
            
            # Create time series chart
            frame_df = pd.DataFrame(results['frame_data'])
            vehicle_columns = [col for col in frame_df.columns if col not in ['frame', 'time']]
            
            if vehicle_columns:
                fig = go.Figure()
                for vehicle_type in vehicle_columns:
                    fig.add_trace(go.Scatter(
                        x=frame_df['time'],
                        y=frame_df[vehicle_type],
                        mode='lines',
                        name=vehicle_type,
                        stackgroup='one'
                    ))
                
                fig.update_layout(
                    title='Vehicle Counts Over Time',
                    xaxis_title='Time (seconds)',
                    yaxis_title='Vehicle Count',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True, key="session_time_series")
        
        # Annotated video section
        if st.session_state.annotated_video_path and os.path.exists(st.session_state.annotated_video_path):
            st.subheader("🎯 Annotated Video")
            st.success("✅ Annotated video created successfully!")
            
            # Display annotated video using stored bytes for compatibility
            st.video(st.session_state.annotated_video_bytes)
            
            # Download annotated video
            st.download_button(
                label="📥 Download Annotated Video",
                data=st.session_state.annotated_video_bytes,
                file_name=f"annotated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                mime="video/mp4",
                key="dl_annot_prev"
            )
        
        # Download section
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_summary = st.session_state.summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary CSV",
                data=csv_summary,
                file_name=f"vehicle_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_summary_prev"
            )
        
        with col2:
            csv_frames = st.session_state.frame_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Frame Data CSV",
                data=csv_frames,
                file_name=f"frame_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_frames_prev"
            )
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses YOLOv8 to detect vehicles in video footage and calculates Passenger Car Units (PCU) based on IRC 106-1990 standards.
        
        **PCU Factors:**
        - Car: 1.0
        - Truck/Bus: 3.0
        - Motorcycle/Bicycle: 0.5
        
        **Supported video formats:** MP4, AVI, MOV, MKV
        """)
        
        st.header("🔧 Technical Details")
        st.markdown("""
        - **Model:** YOLOv8n (nano)
        - **Detection Confidence:** >50%
        - **Processing:** Frame-by-frame analysis with tracking
        - **Tracking:** Prevents double-counting of same vehicle
        - **Annotations:** Colored circles and bounding boxes
        """)
        
        st.header("🎨 Annotation Colors")
        st.markdown("""
        - 🟢 **Green:** Cars
        - 🔵 **Blue:** Trucks  
        - 🔴 **Red:** Buses
        - 🟡 **Yellow:** Motorcycles
        - 🟣 **Magenta:** Bicycles
        """)

if __name__ == "__main__":
    main() 