import cv2
import numpy as np
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import time
from collections import defaultdict
import math

@dataclass
class InsectDetection:
    """Data class for insect detection results"""
    frame_number: int
    timestamp: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[float, float]
    area: float
    velocity: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    track_id: Optional[int] = None
    
    def to_dict(self):
        return asdict(self)

class InsectTracker:
    """Simple tracker for following insects across frames"""
    
    def __init__(self, max_distance=50, max_frames_missing=5):
        self.tracks = {}
        self.next_track_id = 0
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        
    def update(self, detections: List[InsectDetection], frame_number: int) -> List[InsectDetection]:
        """Update tracks with new detections"""
        
        # Remove old tracks that haven't been updated
        self.tracks = {
            tid: track for tid, track in self.tracks.items() 
            if frame_number - track['last_seen'] <= self.max_frames_missing
        }
        
        # Match detections to existing tracks
        matched_detections = []
        
        for detection in detections:
            best_track_id = None
            min_distance = float('inf')
            
            # Find closest existing track
            for track_id, track in self.tracks.items():
                distance = math.sqrt(
                    (detection.center[0] - track['last_position'][0])**2 +
                    (detection.center[1] - track['last_position'][1])**2
                )
                
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_track_id = track_id
            
            # Assign to existing track or create new one
            if best_track_id is not None:
                # Update existing track
                track = self.tracks[best_track_id]
                detection.track_id = best_track_id
                
                # Calculate velocity
                dt = frame_number - track['last_frame']
                if dt > 0:
                    detection.velocity = (
                        (detection.center[0] - track['last_position'][0]) / dt,
                        (detection.center[1] - track['last_position'][1]) / dt
                    )
                
                # Update track info
                track['last_position'] = detection.center
                track['last_frame'] = frame_number
                track['last_seen'] = frame_number
                track['total_detections'] += 1
                
            else:
                # Create new track
                detection.track_id = self.next_track_id
                self.tracks[self.next_track_id] = {
                    'last_position': detection.center,
                    'last_frame': frame_number,
                    'last_seen': frame_number,
                    'total_detections': 1
                }
                self.next_track_id += 1
            
            matched_detections.append(detection)
        
        return matched_detections

class InsectDetector:
    """Main class for detecting flying insects in video"""
    
    def __init__(self, 
                 min_area=20, 
                 max_area=500,
                 min_aspect_ratio=0.3,
                 max_aspect_ratio=3.0,
                 motion_threshold=10,
                 min_confidence=0.6,
                 min_track_length=3):
        
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.motion_threshold = motion_threshold
        self.min_confidence = min_confidence
        self.min_track_length = min_track_length
        
        # Background subtractor with more conservative settings
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True  # Increased threshold
        )
        
        # Optical flow for motion analysis
        self.prev_gray = None
        
        # Tracker
        self.tracker = InsectTracker()
        
        # Results storage
        self.all_detections = []
        self.frame_detections = defaultdict(list)
    
    def detect_insects_in_frame(self, frame, frame_number, timestamp, motion_frame=None):
        """Detect insects in a single frame using motion data"""
        
        if motion_frame is not None:
            # Use pre-processed motion frame (much better!)
            motion_mask = motion_frame.copy()
            
            # Apply additional thresholding to focus on significant motion
            # Motion videos are already normalized 0-255, so we can threshold directly
            _, motion_mask = cv2.threshold(motion_mask, 50, 255, cv2.THRESH_BINARY)  # Higher threshold
            
        else:
            # Fallback to background subtraction (less preferred)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_mask = self.bg_subtractor.apply(frame)
        
        # Additional noise reduction and center filling for motion data
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        # First: Remove small noise with opening
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Second: Fill gaps and holes with aggressive closing and dilation
        # This helps with ping pong balls and objects with uniform centers
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_large)
        motion_mask = cv2.dilate(motion_mask, kernel_medium, iterations=3)
        
        # Third: Clean up the shape with another closing operation
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Fourth: Slight erosion to bring back to more reasonable size
        motion_mask = cv2.erode(motion_mask, kernel_small, iterations=1)
        
        # Fifth: Fill remaining holes using floodFill technique
        # This is particularly effective for objects with hollow centers
        motion_mask_filled = motion_mask.copy()
        h, w = motion_mask.shape[:2]
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        
        # Find and fill holes by flood filling from the edges
        cv2.floodFill(motion_mask_filled, flood_mask, (0,0), 255)
        motion_mask_filled_inv = cv2.bitwise_not(motion_mask_filled)
        motion_mask = motion_mask | motion_mask_filled_inv
        
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (insects tend to be somewhat elongated)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Calculate additional features
            center = (x + w/2, y + h/2)
            
            # Create detection
            detection = InsectDetection(
                frame_number=frame_number,
                timestamp=timestamp,
                bbox=(x, y, w, h),
                center=center,
                area=area,
                confidence=self._calculate_confidence(contour, area, aspect_ratio)
            )
            
            detections.append(detection)
        
        # Update tracker
        tracked_detections = self.tracker.update(detections, frame_number)
        
        # Store results
        self.frame_detections[frame_number] = tracked_detections
        self.all_detections.extend(tracked_detections)
        
        return tracked_detections
    
    def _calculate_confidence(self, contour, area, aspect_ratio):
        """Calculate confidence score for detection"""
        score = 0.5  # Base score
        
        # Bonus for moderate size
        if 50 <= area <= 200:
            score += 0.2
        
        # Bonus for insect-like aspect ratio
        if 0.5 <= aspect_ratio <= 2.0:
            score += 0.2
        
        # Bonus for relatively smooth contour (insects vs noise)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if 0.1 <= circularity <= 0.9:  # Not too circular, not too irregular
                score += 0.1
        
        return min(score, 1.0)
    
    def process_video(self, video_path, output_path=None, max_frames=None, motion_video_path=None, progress_callback=None):
        """Process entire video for insect detection"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Open motion video if provided
        motion_cap = None
        if motion_video_path and os.path.exists(motion_video_path):
            motion_cap = cv2.VideoCapture(motion_video_path)
            if not motion_cap.isOpened():
                print(f"Warning: Could not open motion video: {motion_video_path}")
                motion_cap = None
            else:
                print(f"Using motion video: {motion_video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Processing {total_frames} frames at {fps:.2f} FPS")
        if motion_cap:
            print("Using pre-processed motion data for better accuracy")
        
        # Setup output video if requested
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        start_time = time.time()
        
        while frame_number < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Read corresponding motion frame if available
            motion_frame = None
            if motion_cap:
                ret_motion, motion_frame = motion_cap.read()
                if ret_motion:
                    # Convert to grayscale if needed
                    if len(motion_frame.shape) == 3:
                        motion_frame = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
                else:
                    motion_frame = None
            
            timestamp = frame_number / fps
            
            # Detect insects
            detections = self.detect_insects_in_frame(frame, frame_number, timestamp, motion_frame)
            
            # Draw detections on frame if creating output video
            if out is not None:
                annotated_frame = self.draw_detections(frame, detections)
                out.write(annotated_frame)
            
            frame_number += 1
            
            # Call progress callback
            if progress_callback:
                unique_tracks = len(set(d.track_id for d in self.all_detections if d.track_id is not None))
                progress_callback(frame_number, total_frames, len(self.all_detections), unique_tracks)
            
            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_number / elapsed
                print(f"Processed: {frame_number}/{total_frames} ({fps_processing:.1f} FPS)", end='\r')
        
        cap.release()
        if motion_cap:
            motion_cap.release()
        if out:
            out.release()
        
        print(f"\nProcessing complete. Found {len(self.all_detections)} total detections")
        return self.get_detection_summary()
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and info on frame"""
        annotated = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Choose color based on track_id
            if detection.track_id is not None:
                color_idx = detection.track_id % 10
                colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255),
                         (0,255,255), (128,255,0), (255,128,0), (128,0,255), (255,255,128)]
                color = colors[color_idx]
            else:
                color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            center_x, center_y = int(detection.center[0]), int(detection.center[1])
            cv2.circle(annotated, (center_x, center_y), 3, color, -1)
            
            # Draw track ID and confidence
            text = f"ID:{detection.track_id} C:{detection.confidence:.2f}"
            cv2.putText(annotated, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw velocity vector if available
            if detection.velocity[0] != 0 or detection.velocity[1] != 0:
                vel_x, vel_y = detection.velocity
                end_x = int(center_x + vel_x * 10)  # Scale velocity for visibility
                end_y = int(center_y + vel_y * 10)
                cv2.arrowedLine(annotated, (center_x, center_y), (end_x, end_y), color, 1)
        
        return annotated
    
    def get_detection_summary(self):
        """Get summary statistics for Swift UI integration"""
        if not self.all_detections:
            return {}
        
        # Track statistics
        track_stats = defaultdict(list)
        for detection in self.all_detections:
            if detection.track_id is not None:
                track_stats[detection.track_id].append(detection)
        
        summary = {
            'total_detections': len(self.all_detections),
            'unique_tracks': len(track_stats),
            'frames_with_detections': len(self.frame_detections),
            'tracks_info': []
        }
        
        # Per-track information
        for track_id, track_detections in track_stats.items():
            track_info = {
                'track_id': track_id,
                'detection_count': len(track_detections),
                'first_frame': min(d.frame_number for d in track_detections),
                'last_frame': max(d.frame_number for d in track_detections),
                'avg_confidence': sum(d.confidence for d in track_detections) / len(track_detections),
                'max_velocity': max(math.sqrt(d.velocity[0]**2 + d.velocity[1]**2) for d in track_detections),
                'path': [(d.center[0], d.center[1]) for d in track_detections]
            }
            summary['tracks_info'].append(track_info)
        
        return summary
    
    def export_results_for_swift(self, output_path):
        """Export results in JSON format for Swift UI consumption"""
        
        # Prepare data for export
        export_data = {
            'metadata': {
                'total_detections': len(self.all_detections),
                'unique_tracks': len(set(d.track_id for d in self.all_detections if d.track_id is not None)),
                'processed_frames': len(self.frame_detections)
            },
            'detections': [detection.to_dict() for detection in self.all_detections],
            'summary': self.get_detection_summary()
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to: {output_path}")
    
    def extract_insect_crops(self, video_path, output_folder, padding=10):
        """Extract cropped images of detected insects"""
        
        os.makedirs(output_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        crops_saved = 0
        
        for frame_num, detections in self.frame_detections.items():
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Extract crops for each detection in this frame
            for i, detection in enumerate(detections):
                x, y, w, h = detection.bbox
                
                # Add padding
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                # Extract crop
                crop = frame[y1:y2, x1:x2]
                
                # Save crop
                crop_filename = f"insect_track{detection.track_id}_frame{frame_num}_{i}.jpg"
                crop_path = os.path.join(output_folder, crop_filename)
                cv2.imwrite(crop_path, crop)
                crops_saved += 1
        
        cap.release()
        print(f"Extracted {crops_saved} insect crops to: {output_folder}")

def main():
    """Main function for testing"""
    
    # Example usage
    video_path = "/Users/jame/Downloads/IMG_2219.mov"  # Use your video path
    
    # Create detector with stricter parameters to reduce false positives
    detector = InsectDetector(
        min_area=400,         # Even larger minimum area
        max_area=800,         # Smaller maximum area for typical pollinators
        min_aspect_ratio=0.4, # More restrictive aspect ratio
        max_aspect_ratio=3.0, # Less elongation allowed
        motion_threshold=20   # Higher threshold for motion sensitivity
    )
    
    # Process video
    input_path = Path(video_path)
    output_video = f"{input_path.stem}_insect_detection.mp4"
    output_json = f"{input_path.stem}_insect_data.json"
    output_crops = f"{input_path.stem}_insect_crops"
    
    # Look for corresponding motion video
    motion_video_path = f"{input_path.stem}_motion_normalized.mp4"
    if not os.path.exists(motion_video_path):
        print(f"Motion video not found: {motion_video_path}")
        print("Creating motion video first...")
        
        # Import and use your motion creation function
        import sys
        sys.path.append(os.path.dirname(__file__))
        from createmotionvideo import create_motion_video_from_video_streaming
        
        # Create motion video
        create_motion_video_from_video_streaming(
            video_path,
            motion_video_path,
            fps=None  # Use original FPS
        )
    
    print("Starting insect detection with motion video...")
    summary = detector.process_video(
        video_path, 
        output_path=output_video,
        max_frames=36000,  # Around 10 minute limit for testing
        motion_video_path=motion_video_path  # Use motion video!
    )
    
    # Export results for Swift UI
    detector.export_results_for_swift(output_json)
    
    # Extract individual insect images
    detector.extract_insect_crops(video_path, output_crops)
    
    print("\nDetection Summary:")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Unique tracks: {summary['unique_tracks']}")
    print(f"Frames with detections: {summary['frames_with_detections']}")
    
    # Additional filtering based on track quality
    quality_tracks = [track for track in summary['tracks_info'] 
                     if track['detection_count'] >= 3 and track['avg_confidence'] > 0.5]
    
    # Even stricter filtering for likely real pollinators
    pollinator_tracks = [track for track in summary['tracks_info'] 
                        if track['detection_count'] >= 5 and   # Must be seen in at least 5 frames
                           track['avg_confidence'] > 0.7 and   # High confidence
                           track['max_velocity'] > 1.0]        # Must be moving (not static noise)
    
    print(f"High-quality tracks (3+ detections, >0.5 confidence): {len(quality_tracks)}")
    print(f"Likely pollinators (5+ detections, >0.7 confidence, moving): {len(pollinator_tracks)}")
    
    if len(pollinator_tracks) > 0:
        print("\nLikely pollinator tracks:")
        for track in sorted(pollinator_tracks, key=lambda x: x['detection_count'], reverse=True)[:10]:
            print(f"  Track {track['track_id']}: {track['detection_count']} detections, "
                  f"confidence: {track['avg_confidence']:.2f}, "
                  f"max velocity: {track['max_velocity']:.1f}, "
                  f"frames: {track['first_frame']}-{track['last_frame']}")
    
    if len(quality_tracks) > 0:
        print(f"\nAll high-quality tracks (showing top 10):")
        for track in sorted(quality_tracks, key=lambda x: x['detection_count'], reverse=True)[:10]:
            print(f"  Track {track['track_id']}: {track['detection_count']} detections, "
                  f"confidence: {track['avg_confidence']:.2f}, "
                  f"frames: {track['first_frame']}-{track['last_frame']}")

if __name__ == "__main__":
    main()