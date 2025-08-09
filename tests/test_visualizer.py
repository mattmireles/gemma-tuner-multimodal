#!/usr/bin/env python3
"""
Test script for the Whisper Training Visualizer

This script simulates training data to test the visualization system
without actually running a full training session.
"""

import time
import random
import math
import numpy as np
from visualizer import TrainingVisualizer, start_visualization_server
import torch

def simulate_training():
    """Simulate a training run with fake data for testing the visualizer."""
    
    print("🎆 Starting Whisper Training Visualizer Test")
    print("=" * 50)
    
    # Create a fake model structure for testing
    class FakeModel:
        def __init__(self):
            self.config = type('Config', (), {
                'encoder_layers': 12,
                'decoder_layers': 12,
                'encoder_attention_heads': 12,
                'd_model': 768,
                'vocab_size': 51865
            })()
            
        def parameters(self):
            # Return some fake parameters
            return [torch.randn(768, 768) for _ in range(10)]
    
    # Initialize visualizer with fake model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = FakeModel()
    viz = TrainingVisualizer(model, device)
    
    # Start the visualization server
    print("\n🚀 Starting visualization server...")
    server_thread = start_visualization_server(
        host='127.0.0.1',
        port=8080,
        open_browser=True
    )
    
    print("✨ Visualizer running at http://localhost:8080")
    print("\n📊 Simulating training data...")
    print("Press Ctrl+C to stop\n")
    
    # Simulate training loop
    step = 0
    epoch = 0
    base_loss = 5.0
    base_lr = 1e-4
    
    try:
        while True:
            step += 1
            
            # Update epoch every 100 steps
            if step % 100 == 0:
                epoch += 1
                viz.update_epoch(epoch)
                print(f"📈 Epoch {epoch} started")
            
            # Simulate loss (decreasing with noise)
            loss = base_loss * math.exp(-step * 0.01) + random.random() * 0.2
            base_loss *= 0.999  # Gradual improvement
            
            # Simulate learning rate (cosine schedule)
            lr = base_lr * (0.5 * (1 + math.cos(step * 0.01)))
            
            # Simulate batch data
            batch = {
                'input_features': torch.randn(4, 80, 3000)  # Fake mel spectrogram
            }
            
            # Simulate model outputs
            class FakeOutputs:
                def __init__(self):
                    self.loss = torch.tensor(loss)
                    self.logits = torch.randn(4, 100, 51865)  # Batch, seq_len, vocab
                    # Simulate attention weights
                    self.attentions = [torch.randn(4, 12, 20, 20) for _ in range(3)]
            
            outputs = FakeOutputs()
            
            # Create fake optimizer for gradient simulation
            class FakeOptimizer:
                def __init__(self):
                    self.param_groups = [{'lr': lr}]
            
            optimizer = FakeOptimizer()
            
            # Update visualizer
            viz.update_training_step(
                loss=loss,
                learning_rate=lr,
                batch=batch,
                outputs=outputs,
                optimizer=optimizer
            )
            
            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"Step {step:4d} | Loss: {loss:.4f} | LR: {lr:.6f}")
            
            # Simulate validation every 50 steps
            if step % 50 == 0:
                val_loss = loss * 1.1  # Validation loss slightly higher
                val_metrics = {
                    'accuracy': random.uniform(0.8, 0.95),
                    'wer': random.uniform(0.05, 0.15)
                }
                viz.update_validation(val_loss, val_metrics)
                print(f"📊 Validation - Loss: {val_loss:.4f} | Accuracy: {val_metrics['accuracy']:.2%}")
            
            # Control simulation speed
            time.sleep(0.1)  # 10 updates per second
            
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation stopped by user")
        print("✅ Visualization test completed successfully!")
        print("\nThe visualizer will continue running. Close the browser tab or press Ctrl+C again to exit.")
        
        # Keep server running for inspection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    simulate_training()