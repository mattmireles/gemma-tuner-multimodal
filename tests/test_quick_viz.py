#!/usr/bin/env python3
"""Quick test to verify visualizer works"""

import sys
import time
from whisper_tuner.visualizer import start_visualization_server, TrainingVisualizer
import torch

print("🎆 Testing Whisper Training Visualizer")
print("=" * 50)

# Start server without opening browser
try:
    server_thread = start_visualization_server(
        host='127.0.0.1',
        port=8080,
        open_browser=False
    )
    print("✅ Visualization server started successfully!")
    print("📊 Server running at http://localhost:8080")
    
    # Test creating visualizer
    device = torch.device('cpu')
    viz = TrainingVisualizer(None, device)
    print("✅ Visualizer instance created successfully!")
    
    # Send a test update
    viz.update_training_step(
        loss=2.5,
        learning_rate=1e-4,
        batch=None,
        outputs=None,
        optimizer=None
    )
    print("✅ Test data sent successfully!")
    
    print("\n🎉 All tests passed! The visualizer is working!")
    print("You can now use it with the wizard by enabling visualization.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
