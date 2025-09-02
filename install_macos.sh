#!/bin/bash

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
echo "  • View logs: tail -f ~/Library/Application\ Support/InsectDetection/Logs/api_server.log"
