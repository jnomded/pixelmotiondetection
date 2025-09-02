# config.py - Configuration management for insect detection pipeline

import os
from pathlib import Path
import json

class Config:
    """Configuration class for insect detection pipeline"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    OUTPUT_DIR = BASE_DIR / "outputs"
    TEMP_DIR = BASE_DIR / "temp"
    
    # macOS-specific directories
    if os.name == 'posix' and os.uname().sysname == 'Darwin':  # macOS
        APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "InsectDetection"
        CACHE_DIR = APP_SUPPORT_DIR / "Cache"
        LOGS_DIR = APP_SUPPORT_DIR / "Logs"
        RESULTS_DIR = APP_SUPPORT_DIR / "Results"
        VIDEOS_DIR = APP_SUPPORT_DIR / "Videos"
    else:
        APP_SUPPORT_DIR = BASE_DIR / "app_data"
        CACHE_DIR = APP_SUPPORT_DIR / "cache"
        LOGS_DIR = APP_SUPPORT_DIR / "logs"
        RESULTS_DIR = APP_SUPPORT_DIR / "results"
        VIDEOS_DIR = APP_SUPPORT_DIR / "videos"
    
    # Detection parameters - optimized for flying insects
    DETECTION_PARAMS = {
        # Size filtering (in pixels²)
        'min_area': int(os.getenv('INSECT_MIN_AREA', '15')),          # Small insects
        'max_area': int(os.getenv('INSECT_MAX_AREA', '400')),         # Medium insects
        
        # Shape filtering (width/height ratio)
        'min_aspect_ratio': float(os.getenv('INSECT_MIN_ASPECT', '0.2')),   # Very elongated
        'max_aspect_ratio': float(os.getenv('INSECT_MAX_ASPECT', '5.0')),   # Very round
        
        # Motion sensitivity
        'motion_threshold': int(os.getenv('INSECT_MOTION_THRESHOLD', '8')),
        
        # Background subtraction parameters
        'bg_history': int(os.getenv('BG_HISTORY', '500')),
        'bg_var_threshold': int(os.getenv('BG_VAR_THRESHOLD', '16')),
        'bg_detect_shadows': os.getenv('BG_DETECT_SHADOWS', 'True').lower() == 'true',
        
        # Morphological operations
        'morph_kernel_size': int(os.getenv('MORPH_KERNEL_SIZE', '3')),
        
        # Confidence scoring weights
        'size_weight': float(os.getenv('SIZE_WEIGHT', '0.2')),
        'aspect_weight': float(os.getenv('ASPECT_WEIGHT', '0.2')),
        'circularity_weight': float(os.getenv('CIRCULARITY_WEIGHT', '0.1')),
        'base_confidence': float(os.getenv('BASE_CONFIDENCE', '0.5'))
    }
    
    # Tracking parameters
    TRACKING_PARAMS = {
        'max_distance': int(os.getenv('TRACK_MAX_DISTANCE', '50')),          # Max pixels between detections
        'max_frames_missing': int(os.getenv('TRACK_MAX_MISSING', '5')),      # Frames before losing track
        'min_track_length': int(os.getenv('TRACK_MIN_LENGTH', '3')),         # Minimum detections per track
        'velocity_smoothing': float(os.getenv('VELOCITY_SMOOTHING', '0.3'))   # Velocity calculation smoothing
    }
    
    # Video processing parameters
    PROCESSING_PARAMS = {
        'max_video_duration': int(os.getenv('MAX_VIDEO_DURATION', '600')),   # seconds
        'max_file_size': int(os.getenv('MAX_FILE_SIZE', '500')) * 1024 * 1024,  # MB to bytes
        'default_fps': int(os.getenv('DEFAULT_FPS', '30')),
        'max_frames_limit': int(os.getenv('MAX_FRAMES_LIMIT', '18000')),     # 10 minutes at 30fps
        'frame_skip': int(os.getenv('FRAME_SKIP', '1')),                     # Process every Nth frame
        'resize_factor': float(os.getenv('RESIZE_FACTOR', '1.0')),           # Resize input for processing
        'supported_formats': ['mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv']
    }
    
    # API server settings
    API_SETTINGS = {
        'host': os.getenv('API_HOST', '127.0.0.1'),
        'port': int(os.getenv('API_PORT', '5000')),
        'debug': os.getenv('DEBUG', 'False').lower() == 'true',
        'threaded': True,
        'max_content_length': int(os.getenv('MAX_UPLOAD_SIZE', '500')) * 1024 * 1024  # 500MB default
    }
    
    # Cleanup and maintenance
    MAINTENANCE = {
        'cleanup_interval_hours': int(os.getenv('CLEANUP_INTERVAL_HOURS', '1')),
        'max_job_age_hours': int(os.getenv('MAX_JOB_AGE_HOURS', '24')),
        'max_storage_gb': int(os.getenv('MAX_STORAGE_GB', '10')),
        'auto_cleanup': os.getenv('AUTO_CLEANUP', 'True').lower() == 'true'
    }
    
    # Logging configuration
    LOGGING = {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'max_log_size_mb': int(os.getenv('MAX_LOG_SIZE_MB', '10')),
        'backup_count': int(os.getenv('LOG_BACKUP_COUNT', '5')),
        'console_logging': os.getenv('CONSOLE_LOGGING', 'True').lower() == 'true'
    }
    
    # Export settings
    EXPORT_SETTINGS = {
        'video_codec': os.getenv('VIDEO_CODEC', 'mp4v'),
        'video_quality': int(os.getenv('VIDEO_QUALITY', '80')),  # 0-100
        'crop_padding': int(os.getenv('CROP_PADDING', '10')),    # pixels around detection
        'json_indent': int(os.getenv('JSON_INDENT', '2')),
        'include_debug_info': os.getenv('INCLUDE_DEBUG_INFO', 'False').lower() == 'true'
    }
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.BASE_DIR,
            cls.UPLOAD_DIR,
            cls.OUTPUT_DIR,
            cls.TEMP_DIR,
            cls.APP_SUPPORT_DIR,
            cls.CACHE_DIR,
            cls.LOGS_DIR,
            cls.RESULTS_DIR,
            cls.VIDEOS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep files for empty directories
        for directory in [cls.UPLOAD_DIR, cls.OUTPUT_DIR, cls.TEMP_DIR, cls.CACHE_DIR]:
            gitkeep_file = directory / ".gitkeep"
            if not gitkeep_file.exists():
                gitkeep_file.touch()
    
    @classmethod
    def save_config(cls, filepath=None):
        """Save current configuration to JSON file"""
        if filepath is None:
            filepath = cls.BASE_DIR / "current_config.json"
        
        config_data = {
            'detection_params': cls.DETECTION_PARAMS,
            'tracking_params': cls.TRACKING_PARAMS,
            'processing_params': cls.PROCESSING_PARAMS,
            'api_settings': cls.API_SETTINGS,
            'maintenance': cls.MAINTENANCE,
            'logging': cls.LOGGING,
            'export_settings': cls.EXPORT_SETTINGS
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        print(f"Configuration saved to: {filepath}")
    
    @classmethod
    def load_config(cls, filepath):
        """Load configuration from JSON file"""
        if not Path(filepath).exists():
            print(f"Config file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # Update class attributes
            if 'detection_params' in config_data:
                cls.DETECTION_PARAMS.update(config_data['detection_params'])
            if 'tracking_params' in config_data:
                cls.TRACKING_PARAMS.update(config_data['tracking_params'])
            if 'processing_params' in config_data:
                cls.PROCESSING_PARAMS.update(config_data['processing_params'])
            
            print(f"Configuration loaded from: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    @classmethod
    def get_detection_config(cls):
        """Get detection parameters as a dictionary for InsectDetector"""
        return {
            'min_area': cls.DETECTION_PARAMS['min_area'],
            'max_area': cls.DETECTION_PARAMS['max_area'],
            'min_aspect_ratio': cls.DETECTION_PARAMS['min_aspect_ratio'],
            'max_aspect_ratio': cls.DETECTION_PARAMS['max_aspect_ratio'],
            'motion_threshold': cls.DETECTION_PARAMS['motion_threshold']
        }
    
    @classmethod
    def get_tracking_config(cls):
        """Get tracking parameters as a dictionary for InsectTracker"""
        return {
            'max_distance': cls.TRACKING_PARAMS['max_distance'],
            'max_frames_missing': cls.TRACKING_PARAMS['max_frames_missing']
        }
    
    @classmethod
    def validate_config(cls):
        """Validate configuration parameters"""
        errors = []
        
        # Validate detection parameters
        if cls.DETECTION_PARAMS['min_area'] >= cls.DETECTION_PARAMS['max_area']:
            errors.append("min_area must be less than max_area")
        
        if cls.DETECTION_PARAMS['min_aspect_ratio'] >= cls.DETECTION_PARAMS['max_aspect_ratio']:
            errors.append("min_aspect_ratio must be less than max_aspect_ratio")
        
        if cls.DETECTION_PARAMS['motion_threshold'] < 0:
            errors.append("motion_threshold must be positive")
        
        # Validate tracking parameters
        if cls.TRACKING_PARAMS['max_distance'] <= 0:
            errors.append("max_distance must be positive")
        
        if cls.TRACKING_PARAMS['max_frames_missing'] <= 0:
            errors.append("max_frames_missing must be positive")
        
        # Validate processing parameters
        if cls.PROCESSING_PARAMS['max_video_duration'] <= 0:
            errors.append("max_video_duration must be positive")
        
        if cls.API_SETTINGS['port'] < 1 or cls.API_SETTINGS['port'] > 65535:
            errors.append("API port must be between 1 and 65535")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("Configuration validation passed")
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=== Insect Detection Configuration ===\n")
        
        print("Detection Parameters:")
        for key, value in cls.DETECTION_PARAMS.items():
            print(f"  {key}: {value}")
        
        print("\nTracking Parameters:")
        for key, value in cls.TRACKING_PARAMS.items():
            print(f"  {key}: {value}")
        
        print("\nProcessing Parameters:")
        for key, value in cls.PROCESSING_PARAMS.items():
            print(f"  {key}: {value}")
        
        print("\nAPI Settings:")
        for key, value in cls.API_SETTINGS.items():
            print(f"  {key}: {value}")
        
        print(f"\nDirectories:")
        print(f"  Base: {cls.BASE_DIR}")
        print(f"  App Support: {cls.APP_SUPPORT_DIR}")
        print(f"  Uploads: {cls.UPLOAD_DIR}")
        print(f"  Outputs: {cls.OUTPUT_DIR}")

# Preset configurations for different use cases
class PresetConfigs:
    """Predefined configurations for different scenarios"""
    
    @staticmethod
    def small_insects():
        """Configuration optimized for small insects (gnats, fruit flies)"""
        return {
            'detection_params': {
                'min_area': 5,
                'max_area': 50,
                'min_aspect_ratio': 0.3,
                'max_aspect_ratio': 3.0,
                'motion_threshold': 5
            }
        }
    
    @staticmethod
    def large_insects():
        """Configuration optimized for larger insects (butterflies, dragonflies)"""
        return {
            'detection_params': {
                'min_area': 100,
                'max_area': 1000,
                'min_aspect_ratio': 0.2,
                'max_aspect_ratio': 5.0,
                'motion_threshold': 12
            }
        }
    
    @staticmethod
    def high_sensitivity():
        """Configuration for maximum detection sensitivity"""
        return {
            'detection_params': {
                'min_area': 8,
                'max_area': 800,
                'min_aspect_ratio': 0.1,
                'max_aspect_ratio': 8.0,
                'motion_threshold': 3
            },
            'tracking_params': {
                'max_distance': 80,
                'max_frames_missing': 8
            }
        }
    
    @staticmethod
    def fast_processing():
        """Configuration optimized for speed over accuracy"""
        return {
            'detection_params': {
                'min_area': 20,
                'max_area': 300,
                'motion_threshold': 15
            },
            'processing_params': {
                'frame_skip': 2,
                'resize_factor': 0.5
            }
        }
    
    @staticmethod
    def apply_preset(preset_name):
        """Apply a preset configuration"""
        presets = {
            'small_insects': PresetConfigs.small_insects(),
            'large_insects': PresetConfigs.large_insects(),
            'high_sensitivity': PresetConfigs.high_sensitivity(),
            'fast_processing': PresetConfigs.fast_processing()
        }
        
        if preset_name not in presets:
            print(f"Unknown preset: {preset_name}")
            print(f"Available presets: {list(presets.keys())}")
            return False
        
        preset_config = presets[preset_name]
        
        # Apply preset to Config class
        if 'detection_params' in preset_config:
            Config.DETECTION_PARAMS.update(preset_config['detection_params'])
        if 'tracking_params' in preset_config:
            Config.TRACKING_PARAMS.update(preset_config['tracking_params'])
        if 'processing_params' in preset_config:
            Config.PROCESSING_PARAMS.update(preset_config['processing_params'])
        
        print(f"Applied preset: {preset_name}")
        return True

# Initialize directories on import
Config.create_directories()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "print":
            Config.print_config()
        elif command == "validate":
            Config.validate_config()
        elif command == "save":
            filepath = sys.argv[2] if len(sys.argv) > 2 else None
            Config.save_config(filepath)
        elif command == "load":
            if len(sys.argv) > 2:
                Config.load_config(sys.argv[2])
            else:
                print("Usage: python config.py load <filepath>")
        elif command == "preset":
            if len(sys.argv) > 2:
                PresetConfigs.apply_preset(sys.argv[2])
                Config.print_config()
            else:
                print("Available presets: small_insects, large_insects, high_sensitivity, fast_processing")
                print("Usage: python config.py preset <preset_name>")
        else:
            print("Unknown command. Available commands: print, validate, save, load, preset")
    else:
        print("Insect Detection Configuration")
        print("Commands: print, validate, save, load, preset")
        Config.print_config()