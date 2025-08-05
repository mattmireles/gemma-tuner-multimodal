#!/bin/bash
# Setup script for Apple Silicon (MPS) environment

echo "🍎 Whisper Fine-Tuner - Apple Silicon Setup"
echo "=========================================="

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "❌ This script requires Apple Silicon (ARM64)"
    echo "   Detected architecture: $ARCH"
    exit 1
fi

echo "✅ Running on Apple Silicon"

# Check Python architecture
PYTHON_ARCH=$(python3 -c "import platform; print(platform.machine())")
if [[ "$PYTHON_ARCH" != "arm64" ]]; then
    echo "❌ Python is running under Rosetta emulation!"
    echo "   Please install native ARM64 Python"
    echo "   Recommended: Install Miniforge or native Homebrew Python"
    exit 1
fi

echo "✅ Python is native ARM64"

# Set environment variables
echo ""
echo "Setting MPS environment variables..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
export SDPA_ALLOW_FLASH_ATTN=1

echo "✅ Environment variables set:"
echo "   PYTORCH_ENABLE_MPS_FALLBACK=1 (for compatibility)"
echo "   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 (80% memory limit)"
echo "   SDPA_ALLOW_FLASH_ATTN=1 (Flash Attention 2 enabled - reduces memory by ~28%)"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify MPS support
echo ""
echo "Verifying MPS support..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS is available!')
    print(f'   PyTorch version: {torch.__version__}')
else:
    print('❌ MPS is not available')
    print('   Please reinstall PyTorch')
    exit(1)
"

# Run system check
echo ""
echo "Running system check..."
python3 scripts/system_check.py

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review the system check output above"
echo "2. Configure your training in config.ini"
echo "3. Run: python main.py finetune <your-profile>"
echo ""
echo "Tips:"
echo "- Batch sizes: M1/M2 Pro (2-4), Max (4-6), Ultra (4-6)"
echo "- Flash Attention 2 is enabled (28% less memory usage)"
echo "- Monitor Activity Monitor for memory usage"
echo "- Remove PYTORCH_ENABLE_MPS_FALLBACK=1 after testing"
echo "- Use fp16 (not bf16) for best MPS performance"