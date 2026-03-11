# CopyTalker Installation Script for Windows
# This script installs all required dependencies automatically

#Requires -Version 5.0

Write-Host "========================================" -ForegroundColor Blue
Write-Host "CopyTalker Installation Script (Windows)" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion detected" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.9 or higher from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    exit 1
}

# Check Python version (must be 3.9+)
$versionMatch = $pythonVersion -match 'Python (\d+)\.(\d+)'
if ($versionMatch) {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
        Write-Host "✗ Python 3.9 or higher is required. Found: $pythonVersion" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Check if FFmpeg is installed
Write-Host "Checking FFmpeg..." -ForegroundColor Yellow
try {
    $ffmpegVersion= ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "✓ FFmpeg is installed" -ForegroundColor Green
    Write-Host "  $ffmpegVersion" -ForegroundColor Gray
} catch {
    Write-Host "⚠ FFmpeg not found" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "FFmpeg is required for audio processing." -ForegroundColor Yellow
    Write-Host "You can install it using winget:" -ForegroundColor Cyan
    Write-Host "  winget install Gyan.FFmpeg" -ForegroundColor White
    Write-Host ""
    Write-Host "Or download from: https://ffmpeg.org/download.html" -ForegroundColor Cyan
    Write-Host ""
    
    $installFfmpeg = Read-Host "Install FFmpeg now using winget? (y/n)"
    if ($installFfmpeg -eq 'y' -or $installFfmpeg -eq 'Y') {
        try {
            Write-Host "Installing FFmpeg..." -ForegroundColor Yellow
            winget install Gyan.FFmpeg
            Write-Host "✓ FFmpeg installed" -ForegroundColor Green
            Write-Host "  Please restart your terminal for FFmpeg to be available." -ForegroundColor Yellow
        } catch {
            Write-Host "⚠ Failed to install FFmpeg via winget" -ForegroundColor Yellow
            Write-Host "  You can install it manually later." -ForegroundColor Yellow
        }
    }
}

Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python-m pip install --upgrade pip
Write-Host "✓ pip upgraded" -ForegroundColor Green

Write-Host ""

# Installation type selection
Write-Host "Choose installation type:" -ForegroundColor Yellow
Write-Host "1) Basic (core functionality + all TTS engines)" -ForegroundColor White
Write-Host "2) Full (all features including voice cloning)" -ForegroundColor White
Write-Host "3) Minimal (no TTS engines - not recommended)" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter choice [1-3] (default: 1)"
$choice = if ([string]::IsNullOrEmpty($choice)) { "1" } else { $choice }

Write-Host ""

switch ($choice) {
    "1" {
        Write-Host "Installing CopyTalker (Basic with TTS)..." -ForegroundColor Yellow
        pip install copytalker[full,cjk]
        Write-Host "✓ Basic installation complete" -ForegroundColor Green
    }
    "2" {
        Write-Host "Installing CopyTalker (Full with all features)..." -ForegroundColor Yellow
        pip install copytalker[full,cjk,indextts,fish-speech]
        Write-Host "✓ Full installation complete" -ForegroundColor Green
    }
    "3" {
        Write-Host "Installing CopyTalker (Minimal)..." -ForegroundColor Yellow
        pip install copytalker
        Write-Host "✓ Minimal installation complete" -ForegroundColor Green
        Write-Host "⚠ You will need to install TTS engines separately" -ForegroundColor Yellow
    }
    default {
        Write-Host "✗ Invalid choice" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Yellow
try {
    $testCli = copytalker --help 2>&1
    Write-Host "✓ CopyTalker CLI is working" -ForegroundColor Green
} catch {
    Write-Host "✗ CopyTalker CLI verification failed" -ForegroundColor Red
    Write-Host "  Please check the error messages above." -ForegroundColor Yellow
}

Write-Host ""

# Run TTS checker if it exists
if (Test-Path "check_tts.py") {
    Write-Host "Checking TTS engines..." -ForegroundColor Yellow
    python check_tts.py
} else {
    Write-Host "⚠ TTS checker not found in current directory" -ForegroundColor Yellow
    Write-Host "  You can run it later with: python check_tts.py" -ForegroundColor Gray
}

Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Download models: copytalker download-models --all" -ForegroundColor White
Write-Host "  2. Start GUI: copytalker --gui" -ForegroundColor White
Write-Host "  3. Or use CLI: copytalker translate --target zh" -ForegroundColor White
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Cyan
Write-Host "  - README.md / README_CN.md" -ForegroundColor Gray
Write-Host "  - INSTALL.md (detailed installation guide)" -ForegroundColor Gray
Write-Host ""
Write-Host "Troubleshooting:" -ForegroundColor Cyan
Write-Host "  - Run: python check_tts.py" -ForegroundColor Gray
Write-Host "  - Check: INSTALL.md for common issues" -ForegroundColor Gray
Write-Host ""

# Windows-specific notes
Write-Host "Windows-Specific Notes:" -ForegroundColor Yellow
Write-Host "  - FFmpeg may require a terminal restart to be available" -ForegroundColor Gray
Write-Host "  - Some antivirus software may block microphone access" -ForegroundColor Gray
Write-Host "  - Windows Defender may ask for permissions on first run" -ForegroundColor Gray
Write-Host ""

Write-Host "Enjoy using CopyTalker!" -ForegroundColor Green
