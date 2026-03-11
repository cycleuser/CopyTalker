#!/bin/bash
# CopyTalker Installation Script for macOS and Linux
# This script installs all required dependencies automatically

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_command() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
print_header"Detecting Operating System"

if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    print_success "Detected macOS"
elif [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS="linux"
    DISTRO=$ID
    print_success "Detected Linux ($DISTRO)"
else
    print_error "Unsupported operating system"
    exit 1
fi

echo

# Install system dependencies
print_header"Installing System Dependencies"

if [[ "$OS" == "macos" ]]; then
    # Check if Homebrew is installed
    if ! check_command brew; then
        print_warning "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        print_success "Homebrew is installed"
    fi
    
    print_header "Installing Audio Dependencies via Homebrew"
    brew update
    brew install ffmpeg portaudio libsndfile
    print_success "Audio dependencies installed"
    
elif [[ "$OS" == "linux" ]]; then
    # Detect package manager
    if check_command apt; then
        PKG_MANAGER="apt"
        UPDATE_CMD="sudo apt update"
        INSTALL_CMD="sudo apt install -y"
        
        print_header "Installing System Dependencies (Debian/Ubuntu)"
        eval $UPDATE_CMD
        
        $INSTALL_CMD ffmpeg portaudio19-dev libsndfile1 python3-dev \
                        libmecab-dev mecab mecab-ipadic-utf8 \
                        espeak-ng libespeak1 build-essential
        
    elif check_command dnf; then
        PKG_MANAGER="dnf"
        UPDATE_CMD="sudo dnf update -y"
        INSTALL_CMD="sudo dnf install -y"
        
        print_header"Installing System Dependencies (Fedora)"
        eval $UPDATE_CMD
        
        $INSTALL_CMD ffmpeg portaudio-devel python3-devel libsndfile \
                        mecab mecab-devel mecab-ipadic \
                        espeak-ng gcc gcc-c++
                        
    elif check_command pacman; then
        PKG_MANAGER="pacman"
        UPDATE_CMD="sudo pacman -Syu --noconfirm"
        INSTALL_CMD="sudo pacman -S --noconfirm"
        
        print_header "Installing System Dependencies (Arch Linux)"
        eval $UPDATE_CMD
        
        $INSTALL_CMD ffmpeg portaudio libsndfile \
                        mecab mecab-ipadic \
                        espeak-ng base-devel
                        
    else
        print_error "Unsupported package manager. Please install dependencies manually."
        exit 1
    fi
    
    print_success "System dependencies installed"
fi

echo

# Check Python version
print_header "Checking Python Version"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python $PYTHON_VERSION detected"

# Check if version is >= 3.9
REQUIRED_VERSION="3.9"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python 3.9 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo

# Create virtual environment (optional)
print_header "Setting Up Virtual Environment (Optional)"
read -p "Create a virtual environment? (recommended) [Y/n]: " create_venv
create_venv=${create_venv:-y}

if [[ $create_venv =~ ^[Yy]$ ]]; then
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
        
        # Activate virtual environment
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_warning "Virtual environment already exists, activating..."
        source venv/bin/activate
    fi
else
    print_warning "Skipping virtual environment (not recommended)"
fi

echo

# Upgrade pip
print_header "Upgrading pip"
pip install --upgrade pip
print_success "pip upgraded"

echo

# Install CopyTalker
print_header "Installing CopyTalker"
echo "Choose installation type:"
echo "1) Basic (core functionality only)"
echo "2) Full (all TTS engines + CJK support) - RECOMMENDED"
echo "3) Minimal (no TTS engines)"
echo

read -p "Enter choice [1-3]: " install_choice
install_choice=${install_choice:-2}

case $install_choice in
    1)
        print_header "Installing Basic CopyTalker"
        pip install copytalker[full,cjk]
        print_success "Basic installation complete"
        ;;
    2)
        print_header "Installing Full CopyTalker (Recommended)"
        pip install copytalker[full,cjk,indextts,fish-speech]
        print_success "Full installation complete"
        ;;
    3)
        print_header "Installing Minimal CopyTalker"
        pip install copytalker
        print_success "Minimal installation complete"
        print_warning "You will need to install TTS engines separately"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo

# Verify installation
print_header "Verifying Installation"

if copytalker --help >/dev/null 2>&1; then
    print_success "CopyTalker CLI is working"
else
    print_error "CopyTalker CLI verification failed"
fi

echo

# Run TTS checker
print_header "Checking TTS Engines"
if [ -f "check_tts.py" ]; then
    python3 check_tts.py
else
    print_warning "TTS checker not found. You can run it later with: python check_tts.py"
fi

echo

# Summary
print_header "Installation Complete!"

echo -e "${GREEN}CopyTalker has been successfully installed!${NC}"
echo
echo "Next steps:"
echo "  1. Download models: copytalker download-models --all"
echo "  2. Start GUI: copytalker --gui"
echo "  3. Or use CLI: copytalker translate --target zh"
echo
echo "Documentation:"
echo "  - README.md / README_CN.md"
echo "  - INSTALL.md(detailed installation guide)"
echo
echo "Troubleshooting:"
echo "  - Run: python check_tts.py"
echo "  - Check: INSTALL.md for common issues"
echo

if [[ "$OS" == "macos" ]]; then
    print_warning "macOS users: You may need to grant microphone permissions in System Preferences"
fi

if [[ "$OS" == "linux" ]]; then
    print_warning "Linux users: If you encounter audio issues, check that your user is in the 'audio' group"
    echo "  sudo usermod -aG audio \$USER"
    echo "  (Then log out and log back in)"
fi

echo
print_success "Enjoy using CopyTalker! 🎙️"
