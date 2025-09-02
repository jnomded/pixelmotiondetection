# macos_integration.py - macOS-specific utilities and improvements

import os
import subprocess
import plistlib
import json
from pathlib import Path
import shutil

class MacOSIntegration:
    """Utilities for better macOS integration"""
    
    def __init__(self):
        self.app_name = "InsectDetection"
        self.bundle_id = "com.research.insectdetection"
        self.app_support_dir = Path.home() / "Library" / "Application Support" / self.app_name
        self.preferences_dir = Path.home() / "Library" / "Preferences"
        
    def setup_app_directories(self):
        """Create necessary app directories on macOS"""
        directories = [
            self.app_support_dir,
            self.app_support_dir / "Videos",
            self.app_support_dir / "Results",
            self.app_support_dir / "Cache",
            self.app_support_dir / "Logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Created app directories in: {self.app_support_dir}")
    
    def create_launch_agent(self, python_path="/usr/local/bin/python3"):
        """Create a Launch Agent plist for the API server"""
        
        script_path = Path(__file__).parent / "api_server.py"
        
        plist_content = {
            'Label': f'{self.bundle_id}.api-server',
            'ProgramArguments': [python_path, str(script_path)],
            'RunAtLoad': True,
            'KeepAlive': True,
            'WorkingDirectory': str(Path(__file__).parent),
            'StandardOutPath': str(self.app_support_dir / 'Logs' / 'api_server.log'),
            'StandardErrorPath': str(self.app_support_dir / 'Logs' / 'api_server_error.log'),
            'EnvironmentVariables': {
                'PYTHONPATH': str(Path(__file__).parent),
                'PATH': '/usr/local/bin:/usr/bin:/bin'
            }
        }
        
        plist_path = Path.home() / "Library" / "LaunchAgents" / f"{self.bundle_id}.api-server.plist"
        
        with open(plist_path, 'wb') as f:
            plistlib.dump(plist_content, f)
        
        print(f"Created Launch Agent at: {plist_path}")
        return plist_path
    
    def install_service(self):
        """Install the API server as a macOS service"""
        plist_path = self.create_launch_agent()
        
        # Load the service
        try:
            subprocess.run(['launchctl', 'load', str(plist_path)], check=True)
            print("API server service installed and started")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install service: {e}")
            return False
    
    def uninstall_service(self):
        """Uninstall the API server service"""
        plist_path = Path.home() / "Library" / "LaunchAgents" / f"{self.bundle_id}.api-server.plist"
        
        if plist_path.exists():
            try:
                subprocess.run(['launchctl', 'unload', str(plist_path)], check=True)
                plist_path.unlink()
                print("Service uninstalled successfully")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Failed to uninstall service: {e}")
                return False
        else:
            print("Service not found")
            return False
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        dependencies = {
            'opencv-python': 'cv2',
            'numpy': 'numpy',
            'flask': 'flask'
        }
        
        missing = []
        for package, import_name in dependencies.items():
            try:
                __import__(import_name)
                print(f"✓ {package} is installed")
            except ImportError:
                missing.append(package)
                print(f"✗ {package} is missing")
        
        if missing:
            print(f"\nTo install missing dependencies:")
            print(f"pip3 install {' '.join(missing)}")
            return False
        
        return True
    
    def create_app_bundle_structure(self):
        """Create a basic app bundle structure for distribution"""
        bundle_path = Path.cwd() / f"{self.app_name}.app"
        contents_path = bundle_path / "Contents"
        macos_path = contents_path / "MacOS"
        resources_path = contents_path / "Resources"
        
        # Create directories
        for path in [bundle_path, contents_path, macos_path, resources_path]:
            path.mkdir(exist_ok=True)
        
        # Create Info.plist
        info_plist = {
            'CFBundleName': self.app_name,
            'CFBundleIdentifier': self.bundle_id,
            'CFBundleVersion': '1.0',
            'CFBundleShortVersionString': '1.0',
            'CFBundleExecutable': self.app_name,
            'CFBundleIconFile': 'AppIcon',
            'LSMinimumSystemVersion': '11.0',
            'NSRequiresAquaSystemAppearance': False,
            'NSHighResolutionCapable': True,
            'CFBundleDocumentTypes': [{
                'CFBundleTypeName': 'Video File',
                'CFBundleTypeExtensions': ['mp4', 'mov', 'avi', 'mkv'],
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Alternate'
            }]
        }
        
        with open(contents_path / "Info.plist", 'wb') as f:
            plistlib.dump(info_plist, f)
        
        # Copy Python backend
        backend_path = resources_path / "backend"
        backend_path.mkdir(exist_ok=True)
        
        # Copy all Python files
        python_files = [
            "api_server.py",
            "insect_detector.py",
            "createmotionvideo.py",
            "macos_integration.py",
            "config.py",
            "requirements.txt"
        ]
        
        for file in python_files:
            if Path(file).exists():
                shutil.copy2(file, backend_path)
        
        print(f"App bundle structure created at: {bundle_path}")
        return bundle_path

def create_installer_script():
    """Create an installer script for easy setup"""
    installer_content = '''#!/bin/bash

# Insect Detection macOS Installer
set -e

echo "🐛 Installing Insect Detection System for macOS..."

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3 from https://python.org or using Homebrew:"
    echo "brew install python"
    exit 1
fi

echo "✓ Python 3 found"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install --user opencv-python numpy flask flask-cors pillow

# Set up directories and service
echo "🔧 Setting up application structure..."
python3 -c "
from macos_integration import MacOSIntegration
integration = MacOSIntegration()
integration.setup_app_directories()
if integration.check_dependencies():
    print('✓ All dependencies satisfied')
    if integration.install_service():
        print('✓ Background service installed')
    else:
        print('⚠️  Service installation failed, you can run manually')
else:
    print('❌ Some dependencies are missing')
"

echo "🎉 Installation complete!"
echo ""
echo "The API server is now running in the background."
echo "You can now build and run the SwiftUI app."
echo ""
echo "Useful commands:"
echo "  • Check service status: launchctl list | grep insectdetection"
echo "  • Stop service: launchctl unload ~/Library/LaunchAgents/com.research.insectdetection.api-server.plist"
echo "  • Start service: launchctl load ~/Library/LaunchAgents/com.research.insectdetection.api-server.plist"
echo "  • View logs: tail -f ~/Library/Application\\ Support/InsectDetection/Logs/api_server.log"
'''
    
    with open('install_macos.sh', 'w') as f:
        f.write(installer_content)
    
    # Make executable
    os.chmod('install_macos.sh', 0o755)
    print("Created installer script: install_macos.sh")

def create_xcode_project_template():
    """Create Xcode project configuration templates"""
    
    # Package.swift for Swift Package Manager
    package_swift = '''// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "InsectDetection",
    platforms: [
        .macOS(.v11)
    ],
    products: [
        .executable(name: "InsectDetection", targets: ["InsectDetection"])
    ],
    dependencies: [
        // Add any external dependencies here
    ],
    targets: [
        .executableTarget(
            name: "InsectDetection",
            dependencies: [],
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "InsectDetectionTests",
            dependencies: ["InsectDetection"]
        )
    ]
)
'''
    
    # Build configuration
    build_settings = {
        "MACOSX_DEPLOYMENT_TARGET": "11.0",
        "SWIFT_VERSION": "5.0",
        "PRODUCT_BUNDLE_IDENTIFIER": "com.research.insectdetection",
        "PRODUCT_NAME": "Insect Detection",
        "MARKETING_VERSION": "1.0",
        "CURRENT_PROJECT_VERSION": "1"
    }
    
    # Create project structure
    project_structure = {
        "Package.swift": package_swift,
        "Sources/InsectDetection/main.swift": "// Main app entry point\n@main\nstruct InsectDetectionApp: App { /* ... */ }",
        "Sources/InsectDetection/Models/": "// Data models directory",
        "Sources/InsectDetection/Views/": "// SwiftUI views directory",
        "Sources/InsectDetection/Services/": "// API services directory",
        "Resources/": "// App resources directory",
        "Tests/InsectDetectionTests/": "// Unit tests directory"
    }
    
    # Save configuration
    with open('xcode_project_template.json', 'w') as f:
        json.dump({
            "package_swift": package_swift,
            "build_settings": build_settings,
            "project_structure": project_structure,
            "instructions": [
                "1. Create new macOS app in Xcode",
                "2. Replace ContentView.swift with the provided SwiftUI code",
                "3. Add the data models to a separate Models group",
                "4. Create an API service class in Services group",
                "5. Update Info.plist with document types for video files",
                "6. Add App Sandbox entitlements for file access",
                "7. Test with the Python backend running locally"
            ]
        }, f, indent=2)
    
    print("Created Xcode project template: xcode_project_template.json")
