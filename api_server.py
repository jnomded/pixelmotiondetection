# Enhanced API server with macOS optimizations
import threading
import os
import time
import uuid
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging

from macos_integration import MacOSIntegration, create_installer_script, create_xcode_project_template

# Add the project directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

try:
    from insect_detection import InsectDetector
except ImportError as e:
    print(f"Warning: Could not import InsectDetector: {e}")
    InsectDetector = None

# Global dictionary to track jobs
jobs = {}

class JobTracker:
    """Track processing jobs with live updates"""
    
    def __init__(self, job_id):
        self.job_id = job_id
        self.status = "queued"
        self.progress = 0.0
        self.message = "Initializing..."
        self.created_at = time.time()
        self.results = None
        self.error = None
        self.output_files = {}
        self.frames_processed = 0
        self.total_frames = 0
        self.detections_count = 0
        self.unique_tracks = 0
        
    def update(self, status=None, progress=None, message=None, **kwargs):
        """Update job status"""
        if status:
            self.status = status
        if progress is not None:
            self.progress = progress
        if message:
            self.message = message
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def to_dict(self):
        """Convert to dictionary for JSON response"""
        return {
            'job_id': self.job_id,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'created_at': self.created_at,
            'results': self.results,
            'error': self.error,
            'output_files': self.output_files,
            'frames_processed': self.frames_processed,
            'total_frames': self.total_frames,
            'detections_count': self.detections_count,
            'unique_tracks': self.unique_tracks
        }

class EnhancedMacOSAPIServer:
    """Enhanced API server with macOS-specific optimizations"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # macOS integration
        self.macos = MacOSIntegration()
        self.macos.setup_app_directories()
        
        # Configure logging to use Application Support directory
        log_file = self.macos.app_support_dir / "Logs" / "api_server.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Set up API routes with macOS optimizations"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'platform': 'macOS',
                'app_support_dir': str(self.macos.app_support_dir),
                'dependencies_ok': self.macos.check_dependencies()
            })
        
        @self.app.route('/api/system-info', methods=['GET'])
        def system_info():
            """Get macOS system information"""
            try:
                # Get system info
                system_info = {
                    'platform': 'macOS',
                    'python_version': os.sys.version,
                    'opencv_available': self._check_opencv(),
                    'app_directories': {
                        'app_support': str(self.macos.app_support_dir),
                        'cache': str(self.macos.app_support_dir / "Cache"),
                        'results': str(self.macos.app_support_dir / "Results")
                    }
                }
                
                return jsonify(system_info)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # Add all existing routes from the original API server here...
        # (The upload, status, results, download routes)
        
        @self.app.route('/api/upload', methods=['POST'])
        def upload_video():
            """Upload video file for processing"""
            try:
                if 'video' not in request.files:
                    return jsonify({'error': 'No video file provided'}), 400
                
                file = request.files['video']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Generate unique job ID
                job_id = str(uuid.uuid4())
                
                # Create job tracker
                job_tracker = JobTracker(job_id)
                jobs[job_id] = job_tracker
                
                # Save uploaded file
                uploads_dir = self.macos.app_support_dir / "uploads"
                uploads_dir.mkdir(exist_ok=True)
                
                filename = f"{job_id}_{file.filename}"
                filepath = uploads_dir / filename
                file.save(str(filepath))
                
                job_tracker.update(status="uploaded", message="Video uploaded, starting processing...")
                
                # Start background processing
                processing_thread = threading.Thread(
                    target=self._process_video_background,
                    args=(job_id, str(filepath))
                )
                processing_thread.daemon = True
                processing_thread.start()
                
                return jsonify({
                    'job_id': job_id,
                    'status': 'uploaded',
                    'message': 'Video uploaded successfully, processing started'
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/status/<job_id>', methods=['GET'])
        def get_status(job_id):
            """Get processing status for a job"""
            try:
                if job_id not in jobs:
                    return jsonify({'error': 'Job not found'}), 404
                
                job_tracker = jobs[job_id]
                return jsonify(job_tracker.to_dict())
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/results/<job_id>', methods=['GET'])
        def get_results(job_id):
            """Get results for a completed job"""
            try:
                if job_id not in jobs:
                    return jsonify({'error': 'Job not found'}), 404
                
                job_tracker = jobs[job_id]
                
                if job_tracker.status != 'completed':
                    return jsonify({'error': 'Job not completed yet'}), 400
                
                return jsonify(job_tracker.results or {
                    'metadata': {
                        'total_detections': 0,
                        'unique_tracks': 0,
                        'processed_frames': 0
                    },
                    'detections': [],
                    'summary': {
                        'total_detections': 0,
                        'unique_tracks': 0,
                        'frames_with_detections': 0,
                        'tracks_info': []
                    }
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _process_video_background(self, job_id, video_path):
        """Process video in background thread with live updates"""
        try:
            job_tracker = jobs[job_id]
            job_tracker.update(status="processing", message="Starting video analysis...")
            
            if InsectDetector is None:
                job_tracker.update(
                    status="error", 
                    error="InsectDetector not available",
                    message="Failed to load insect detection module"
                )
                return
            
            # Initialize detector
            detector = InsectDetector()
            
            # Setup output directories
            results_dir = self.macos.app_support_dir / "results" / job_id
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Custom progress callback
            def progress_callback(frame_num, total_frames, detections_so_far=0, tracks_so_far=0):
                progress = frame_num / total_frames if total_frames > 0 else 0
                job_tracker.update(
                    progress=progress,
                    frames_processed=frame_num,
                    total_frames=total_frames,
                    detections_count=detections_so_far,
                    unique_tracks=tracks_so_far,
                    message=f"Processing frame {frame_num}/{total_frames} - {detections_so_far} detections, {tracks_so_far} tracks"
                )
            
            # Process video with live updates
            results = detector.process_video(
                video_path, 
                output_path=str(results_dir / "output.mp4"),
                progress_callback=progress_callback
            )
            
            # Get all detections from the detector
            all_detections = [detection.to_dict() for detection in detector.all_detections]
            print(f"Processing completed. Found {len(all_detections)} total detections")
            print(f"Summary results: {results}")
            
            # Get summary from results (which is what get_detection_summary returns)
            summary_stats = results if results else {}
            
            # Format results for API
            formatted_results = {
                'metadata': {
                    'total_detections': len(all_detections),
                    'unique_tracks': summary_stats.get('unique_tracks', 0),
                    'processed_frames': job_tracker.frames_processed
                },
                'detections': all_detections,
                'summary': {
                    'total_detections': len(all_detections),
                    'unique_tracks': summary_stats.get('unique_tracks', 0),
                    'frames_with_detections': summary_stats.get('frames_with_detections', 0),
                    'tracks_info': summary_stats.get('tracks_info', [])
                }
            }
            
            print(f"Formatted results: {formatted_results['metadata']}")
            print(f"Summary: {formatted_results['summary']}")
            
            job_tracker.update(
                status="completed",
                progress=1.0,
                message="Processing completed successfully",
                results=formatted_results
            )
            
        except Exception as e:
            print(f"Error processing video for job {job_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            job_tracker = jobs.get(job_id)
            if job_tracker:
                job_tracker.update(
                    status="error",
                    error=str(e),
                    message=f"Processing failed: {str(e)}"
                )
    
    def _check_opencv(self):
        """Check if OpenCV is available and working"""
        try:
            import cv2
            return True
        except ImportError:
            return False
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the Flask server"""
        self.logger.info(f"Starting macOS-optimized API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def setup_macos_environment():
    """Complete macOS environment setup"""
    print("🐛 Setting up Insect Detection for macOS...")
    
    # Initialize macOS integration
    macos = MacOSIntegration()
    
    # Setup directories
    macos.setup_app_directories()
    print("✓ Application directories created")
    
    # Check dependencies
    if macos.check_dependencies():
        print("✓ All dependencies satisfied")
    else:
        print("❌ Some dependencies missing - please install them")
        return False
    
    # Create installer and templates
    create_installer_script()
    create_xcode_project_template()
    
    print("✓ Created installer script and Xcode templates")
    
    # Optionally install as service
    install_service = input("Install API server as background service? (y/n): ").lower().startswith('y')
    if install_service:
        if macos.install_service():
            print("✓ Background service installed")
        else:
            print("❌ Service installation failed")
    
    print("\n🎉 macOS setup complete!")
    print("\nNext steps:")
    print("1. Open Xcode and create a new macOS app")
    print("2. Replace the default SwiftUI code with the provided dashboard code")
    print("3. Build and run the app")
    print("4. The Python backend will handle video processing automatically")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        setup_macos_environment()
    elif len(sys.argv) > 1 and sys.argv[1] == 'server':
        server = EnhancedMacOSAPIServer()
        server.run(debug=True)
    else:
        print("Usage:")
        print("  python macos_integration.py setup  - Set up macOS environment")
        print("  python macos_integration.py server - Run enhanced API server")