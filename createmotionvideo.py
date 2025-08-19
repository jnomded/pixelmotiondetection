import cv2
import numpy as np
import os
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frames_from_video(video_path, output_folder, max_frames=None):
    """
    Extract frames directly from video file using OpenCV.
    
    Parameters:
    - video_path: Path to input video file
    - output_folder: Folder to save extracted frames
    - max_frames: Maximum number of frames to extract (None for all)
    
    Returns:
    - List of frame file paths
    """
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames at {fps:.2f} FPS")
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
        
        # Save frame
        frame_path = os.path.join(output_folder, f"frame_{frame_count+1:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Extracted {frame_count}/{total_frames} frames", end='\r')
    
    cap.release()
    print(f"\nExtracted {frame_count} frames to {output_folder}")
    return frame_paths, fps

def process_frame_pair(args):
    """
    Process a single frame pair for motion detection.
    Used for parallel processing.
    """
    prev_path, curr_path, i = args
    
    prev_frame = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
    curr_frame = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
    
    if prev_frame is None or curr_frame is None:
        return i, None
    
    diff = cv2.absdiff(prev_frame, curr_frame)
    return i, diff

def create_motion_video_from_video_direct(video_path, output_video_path, fps=None, num_workers=None):
    """
    Create normalized motion video directly from input video file with proper frame ordering.
    
    Parameters:
    - video_path: Input video file path
    - output_video_path: Output motion video path
    - fps: Output FPS (uses input video FPS if None)
    - num_workers: Number of parallel workers (unused in sequential version)
    """
    
    print(f"Processing video with sequential frame processing...")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps is None:
        fps = input_fps
    
    print(f"Input: {total_frames} frames, {width}x{height}, {input_fps:.2f} FPS")
    print(f"Output: {fps:.2f} FPS")
    
    # First pass: collect motion data for normalization
    print("First pass: analyzing motion range...")
    
    all_diffs = []
    frame_count = 0
    prev_gray = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, curr_gray)
            all_diffs.append(diff)
        
        prev_gray = curr_gray
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Analyzed: {frame_count}/{total_frames}", end='\r')
    
    print(f"\nAnalyzed {len(all_diffs)} frame differences")
    
    if not all_diffs:
        print("No motion data found")
        cap.release()
        return
    
    # Find global min/max for normalization
    print("Computing normalization range...")
    all_diffs_array = np.array(all_diffs)
    global_min = np.min(all_diffs_array)
    global_max = np.max(all_diffs_array)
    
    print(f"Motion range: {global_min} to {global_max}")
    
    # Reset video capture for second pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Second pass: create normalized video
    print("Second pass: creating normalized motion video...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)
    
    if not out.isOpened():
        raise ValueError(f"Could not create output video: {output_video_path}")
    
    frame_count = 0
    diff_index = 0
    prev_gray = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is None:
            # First frame - write black frame
            motion_frame = np.zeros((height, width), dtype=np.uint8)
        else:
            # Use pre-computed difference and normalize
            if diff_index < len(all_diffs):
                diff = all_diffs[diff_index]
                if global_max > global_min:
                    motion_frame = ((diff - global_min) / (global_max - global_min) * 255).astype(np.uint8)
                else:
                    motion_frame = np.zeros_like(diff, dtype=np.uint8)
                diff_index += 1
            else:
                motion_frame = np.zeros((height, width), dtype=np.uint8)
        
        out.write(motion_frame)
        prev_gray = curr_gray
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Writing: {frame_count}/{total_frames}", end='\r')
    
    cap.release()
    out.release()
    print(f"\nNormalized motion video saved to: {output_video_path}")

def create_motion_video_from_video_streaming(video_path, output_video_path, fps=None):
    """
    Create motion video with single-pass streaming approach for better performance.
    """
    
    print(f"Processing video with streaming approach...")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps is None:
        fps = input_fps
    
    print(f"Input: {total_frames} frames, {width}x{height}, {input_fps:.2f} FPS")
    print(f"Output: {fps:.2f} FPS")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)
    
    if not out.isOpened():
        raise ValueError(f"Could not create output video: {output_video_path}")
    
    # Process frames sequentially
    frame_count = 0
    prev_gray = None
    motion_buffer = []
    buffer_size = 100  # Buffer for adaptive normalization
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is None:
            # First frame - write black frame
            motion_frame = np.zeros((height, width), dtype=np.uint8)
        else:
            # Compute motion
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Add to buffer for local normalization
            motion_buffer.append(diff)
            if len(motion_buffer) > buffer_size:
                motion_buffer.pop(0)
            
            # Normalize based on recent motion history
            if len(motion_buffer) > 1:
                buffer_array = np.array(motion_buffer)
                local_min = np.min(buffer_array)
                local_max = np.max(buffer_array)
                
                if local_max > local_min:
                    motion_frame = ((diff - local_min) / (local_max - local_min) * 255).astype(np.uint8)
                else:
                    motion_frame = np.zeros_like(diff, dtype=np.uint8)
            else:
                motion_frame = np.clip(diff, 0, 255).astype(np.uint8)
        
        out.write(motion_frame)
        prev_gray = curr_gray
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processing: {frame_count}/{total_frames}", end='\r')
    
    cap.release()
    out.release()
    print(f"\nMotion video saved to: {output_video_path}")

def create_motion_video(input_folder, output_video_path, motion_threshold=10, fps=30):
    """
    Create a motion visualization video from a sequence of images.
    
    Parameters:
    - input_folder: Path to folder containing frame images
    - output_video_path: Output video file path
    - motion_threshold: Minimum pixel difference to consider as motion
    - fps: Frames per second for output video
    """
    
    # Find all image files
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.jpg")) + 
                        glob.glob(os.path.join(input_folder, "*.png")))
    
    if len(image_files) < 2:
        print(f"Need at least 2 images, found {len(image_files)}")
        return
    
    print(f"Processing {len(image_files)} frames...")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(image_files[0])
    height, width = first_frame.shape[:2]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)
    
    # Process each pair of consecutive frames
    prev_frame = None
    
    for i, img_path in enumerate(image_files):
        print(f"Processing frame {i+1}/{len(image_files)}", end='\r')
        
        # Load current frame in grayscale
        curr_frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if prev_frame is None:
            # First frame - create a black motion frame
            motion_frame = np.zeros_like(curr_frame, dtype=np.uint8)
        else:
            # Compute absolute difference between frames
            diff = cv2.absdiff(prev_frame, curr_frame)
            
            # Apply threshold to reduce noise
            _, motion_frame = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Optional: Apply morphological operations to clean up
            kernel = np.ones((3,3), np.uint8)
            motion_frame = cv2.morphologyEx(motion_frame, cv2.MORPH_CLOSE, kernel)
            motion_frame = cv2.morphologyEx(motion_frame, cv2.MORPH_OPEN, kernel)
        
        # Write frame to video
        out.write(motion_frame)
        
        # Update previous frame
        prev_frame = curr_frame.copy()
    
    # Clean up
    out.release()
    print(f"\nMotion video saved to: {output_video_path}")

def create_enhanced_motion_video(input_folder, output_video_path, fps=30):
    """
    Create an enhanced motion visualization with multiple visualization modes.
    """
    
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.jpg")) + 
                        glob.glob(os.path.join(input_folder, "*.png")))
    
    if len(image_files) < 2:
        print(f"Need at least 2 images, found {len(image_files)}")
        return
    
    print(f"Processing {len(image_files)} frames with enhanced motion detection...")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(image_files[0])
    height, width = first_frame.shape[:2]
    
    # Setup video writer (color output for enhanced visualization)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)
    
    prev_frame = None
    
    for i, img_path in enumerate(image_files):
        print(f"Processing frame {i+1}/{len(image_files)}", end='\r')
        
        # Load current frame
        curr_frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if prev_frame is None:
            # First frame - create a black motion frame
            motion_viz = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Compute frame difference
            diff = cv2.absdiff(prev_frame, curr_frame)
            
            # Create different intensity levels for different amounts of motion
            motion_viz = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Low motion (green)
            low_motion = (diff > 5) & (diff <= 15)
            motion_viz[low_motion] = [0, 255, 0]
            
            # Medium motion (yellow)
            med_motion = (diff > 15) & (diff <= 30)
            motion_viz[med_motion] = [0, 255, 255]
            
            # High motion (red)
            high_motion = diff > 30
            motion_viz[high_motion] = [0, 0, 255]
            
            # Very high motion (white)
            very_high_motion = diff > 60
            motion_viz[very_high_motion] = [255, 255, 255]
        
        # Write frame to video
        out.write(motion_viz)
        
        # Update previous frame
        prev_frame = curr_frame.copy()
    
    # Clean up
    out.release()
    print(f"\nEnhanced motion video saved to: {output_video_path}")

def create_normalized_motion_video_parallel(input_folder, output_video_path, fps=30, num_workers=None):
    """
    Create a normalized (0-1) grayscale motion video with parallel processing.
    """
    
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.jpg")) + 
                        glob.glob(os.path.join(input_folder, "*.png")))
    
    if len(image_files) < 2:
        print(f"Need at least 2 images, found {len(image_files)}")
        return
    
    print(f"Creating normalized motion video from {len(image_files)} frames with {num_workers} workers...")
    
    # Prepare frame pairs for parallel processing
    frame_pairs = [(image_files[i], image_files[i+1], i) for i in range(len(image_files)-1)]
    
    # First pass: parallel motion detection
    print("First pass: analyzing motion ranges...")
    
    all_diffs = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_frame_pair, pair) for pair in frame_pairs]
        
        # Collect results in order
        results = [None] * len(frame_pairs)
        for future in as_completed(futures):
            i, diff = future.result()
            if diff is not None:
                results[i] = diff
            
            completed = sum(1 for r in results if r is not None)
            print(f"Analyzed: {completed}/{len(frame_pairs)}", end='\r')
    
    # Filter out None results and create array
    all_diffs = [diff for diff in results if diff is not None]
    
    if not all_diffs:
        print("No valid motion data found")
        return
    
    # Find global min/max for normalization
    all_diffs_array = np.array(all_diffs)
    global_min = np.min(all_diffs_array)
    global_max = np.max(all_diffs_array)
    
    print(f"\nMotion range: {global_min} to {global_max}")
    
    # Second pass: create normalized video
    first_frame = cv2.imread(image_files[0])
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)
    
    # Write first frame (black)
    first_motion_frame = np.zeros((height, width), dtype=np.uint8)
    out.write(first_motion_frame)
    
    # Write normalized frames
    for i, diff in enumerate(all_diffs):
        if global_max > global_min:
            normalized = ((diff - global_min) / (global_max - global_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(diff, dtype=np.uint8)
        
        out.write(normalized)
        
        if (i + 1) % 50 == 0:
            print(f"Second pass: {i+1}/{len(all_diffs)}", end='\r')
    
    out.release()
    print(f"\nNormalized motion video saved to: {output_video_path}")

def quick_motion_from_video(video_path, output_path, max_frames=None):
    """
    Quick motion video creation directly from video file.
    """
    print(f"Creating quick motion video from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Processing {total_frames} frames at {fps:.2f} FPS")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Could not read first frame")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Write black first frame
    out.write(np.zeros_like(prev_gray))
    
    frame_count = 1
    while frame_count < total_frames:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute motion
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, motion = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        
        out.write(motion)
        
        prev_gray = curr_gray
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed: {frame_count}/{total_frames}", end='\r')
    
    cap.release()
    out.release()
    print(f"\nQuick motion video saved to: {output_path}")

if __name__ == "__main__":
    # Process video directly to create normalized motion video
    video_path = "/Users/jame/Downloads/IMG_4527.mov"
    
    # Generate output filename based on input filename
    input_path = Path(video_path)
    output_filename = f"{input_path.stem}_motion_normalized.mp4"
    output_path = output_filename
    
    print(f"Processing: {video_path}")
    print(f"Output: {output_path}")
    
    # Use streaming approach for better performance and no stuttering
    create_motion_video_from_video_streaming(
        video_path,
        output_path,
        fps=30
    )
    
    print(f"\nDone! Motion video saved as: {output_path}")