#!/usr/bin/env python3

"""
Whisper Training Visualizer - The Most Beautiful Way to Watch AI Learn

This module creates a mesmerizing real-time visualization of the Whisper training process,
streaming live data to a web interface with stunning 3D graphics and particle effects.

Architecture:
- Flask server with SocketIO for real-time data streaming
- Hooks into PyTorch training loop to extract metrics
- Efficient buffering to prevent performance impact
- WebGL/Three.js frontend for GPU-accelerated graphics

Called by:
- scripts/finetune.py when --visualize flag is set
- wizard.py when visualization mode is enabled

Visualization includes:
- 3D neural network with flowing gradients
- Real-time loss landscape
- Attention weight heatmaps
- Audio spectrogram waterfalls
- Token generation particles
- Memory usage waves
"""

import os
import json
import time
import threading
import logging
import webbrowser
from collections import deque
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import torch
import torch.nn as nn

# Initialize Flask app and SocketIO
app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')
app.config['SECRET_KEY'] = 'whisper-training-viz-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """
    Core visualization engine that extracts and streams training data.
    
    This class hooks into the training process to capture real-time metrics
    and streams them to connected web clients for visualization.
    """
    
    def __init__(self, model: Optional[nn.Module] = None, device: Optional[torch.device] = None):
        """
        Initialize the visualizer with model and device information.
        
        Args:
            model: The Whisper model being trained
            device: The device (cuda/mps/cpu) being used
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.socketio = socketio
        
        # Data buffers with max length to prevent memory overflow
        self.buffer_size = 1000
        self.loss_history = deque(maxlen=self.buffer_size)
        self.grad_history = deque(maxlen=self.buffer_size)
        self.lr_history = deque(maxlen=self.buffer_size)
        self.memory_history = deque(maxlen=self.buffer_size)
        self.attention_buffer = deque(maxlen=10)  # Keep last 10 attention maps
        self.token_buffer = deque(maxlen=100)  # Last 100 generated tokens
        
        # Performance metrics
        self.step_count = 0
        self.epoch = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.update_frequency = 10  # Update visualization every N steps
        
        # Model layer information for visualization
        self.layer_info = self._extract_model_architecture() if model else {}
        
        # Training state
        self.is_training = False
        self.current_batch_size = 0
        self.total_params = 0
        self.trainable_params = 0
        
        if model:
            self._calculate_param_stats()
            self._register_hooks()
    
    def _extract_model_architecture(self) -> Dict[str, Any]:
        """Extract model architecture for visualization."""
        if not self.model:
            return {}
        
        architecture = {
            'encoder_layers': 0,
            'decoder_layers': 0,
            'attention_heads': 0,
            'hidden_size': 0,
            'vocab_size': 0
        }
        
        try:
            if hasattr(self.model, 'config'):
                config = self.model.config
                architecture['encoder_layers'] = getattr(config, 'encoder_layers', 12)
                architecture['decoder_layers'] = getattr(config, 'decoder_layers', 12)
                architecture['attention_heads'] = getattr(config, 'encoder_attention_heads', 12)
                architecture['hidden_size'] = getattr(config, 'd_model', 768)
                architecture['vocab_size'] = getattr(config, 'vocab_size', 51865)
        except Exception as e:
            print(f"Could not extract model architecture: {e}")
        
        return architecture
    
    def _calculate_param_stats(self):
        """Calculate total and trainable parameters."""
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        self.activation_handles = []
        self.gradient_handles = []
        
        # Hook into encoder layers for attention visualization
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'encoder'):
            encoder = self.model.model.encoder
            if hasattr(encoder, 'layers'):
                for idx, layer in enumerate(encoder.layers[:3]):  # Hook first 3 layers for performance
                    handle = layer.register_forward_hook(self._attention_hook(f'encoder_{idx}'))
                    self.activation_handles.append(handle)
    
    def _attention_hook(self, layer_name: str):
        """Create a hook function to capture attention weights."""
        def hook(module, input, output):
            if hasattr(output, 'attentions') and output.attentions is not None:
                # Store attention weights for visualization
                attention = output.attentions.detach().cpu().numpy()
                # Average over heads and batch for 2D visualization
                if len(attention.shape) >= 4:
                    attention_2d = attention.mean(axis=(0, 1))  # Average over batch and heads
                    self.attention_buffer.append({
                        'layer': layer_name,
                        'attention': attention_2d.tolist()[:20, :20]  # Limit size for performance
                    })
        return hook
    
    def update_training_step(self, 
                           loss: float,
                           learning_rate: float,
                           batch: Optional[Dict] = None,
                           outputs: Optional[Any] = None,
                           optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Update visualizer with data from current training step.
        
        Called from the training loop after each batch.
        
        Args:
            loss: Current batch loss value
            learning_rate: Current learning rate
            batch: Input batch data (for audio visualization)
            outputs: Model outputs (for attention/logits)
            optimizer: Optimizer (for gradient stats)
        """
        self.step_count += 1
        current_time = time.time()
        
        # Update buffers
        self.loss_history.append(loss)
        self.lr_history.append(learning_rate)
        
        # Calculate gradient norm
        if self.model and optimizer:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.grad_history.append(total_norm)
        
        # Get memory usage
        memory_gb = 0.0
        if self.device.type == 'cuda':
            memory_gb = torch.cuda.memory_allocated() / 1024**3
        elif self.device.type == 'mps':
            memory_gb = torch.mps.current_allocated_memory() / 1024**3
        self.memory_history.append(memory_gb)
        
        # Extract attention and token probabilities from outputs
        attention_data = None
        token_probs = None
        
        if outputs and hasattr(outputs, 'attentions') and outputs.attentions:
            # Get last layer attention
            last_attention = outputs.attentions[-1]
            if last_attention is not None:
                # Average over heads for 2D visualization
                avg_attention = last_attention.mean(dim=1).detach().cpu().numpy()
                attention_data = avg_attention[0, :20, :20].tolist()  # First in batch, limited size
        
        if outputs and hasattr(outputs, 'logits'):
            # Get top 5 token probabilities
            logits = outputs.logits[:, -1, :]  # Last position
            probs = torch.softmax(logits, dim=-1)
            top5 = torch.topk(probs[0], k=5)
            token_probs = {
                'values': top5.values.detach().cpu().numpy().tolist(),
                'indices': top5.indices.detach().cpu().numpy().tolist()
            }
        
        # Extract audio features if available
        mel_spectrogram = None
        if batch and 'input_features' in batch:
            # Get first item in batch, subsample for performance
            mel = batch['input_features'][0].detach().cpu().numpy()
            # Subsample to reduce data size (every 10th frame and frequency)
            mel_spectrogram = mel[::10, ::10].tolist()
        
        # Send update to frontend every N steps
        if self.step_count % self.update_frequency == 0:
            self._emit_update({
                'step': self.step_count,
                'epoch': self.epoch,
                'loss': loss,
                'gradient_norm': self.grad_history[-1] if self.grad_history else 0,
                'learning_rate': learning_rate,
                'memory_gb': memory_gb,
                'attention': attention_data,
                'token_probs': token_probs,
                'mel_spectrogram': mel_spectrogram,
                'steps_per_second': self.update_frequency / (current_time - self.last_update_time),
                'total_time': current_time - self.start_time,
                'architecture': self.layer_info
            })
            self.last_update_time = current_time
    
    def _emit_update(self, data: Dict[str, Any]):
        """Emit update to all connected clients."""
        try:
            self.socketio.emit('training_update', data)
        except Exception as e:
            logger.debug(f"Visualizer emit failed: {e}")
    
    def update_epoch(self, epoch: int):
        """Update current epoch number."""
        self.epoch = epoch
        self._emit_update({'epoch': epoch, 'event': 'epoch_change'})
    
    def update_validation(self, val_loss: float, val_metrics: Dict[str, float]):
        """Update validation metrics."""
        self._emit_update({
            'event': 'validation',
            'val_loss': val_loss,
            'val_metrics': val_metrics
        })
    
    def set_training_state(self, is_training: bool):
        """Set training state (training/stopped)."""
        self.is_training = is_training
        self._emit_update({'event': 'training_state', 'is_training': is_training})

# Global visualizer instance
visualizer: Optional[TrainingVisualizer] = None

def init_visualizer(model: nn.Module, device: torch.device) -> TrainingVisualizer:
    """Initialize the global visualizer instance."""
    global visualizer
    visualizer = TrainingVisualizer(model, device)
    return visualizer

def get_visualizer() -> Optional[TrainingVisualizer]:
    """Get the global visualizer instance."""
    return visualizer

# Flask routes
@app.route('/')
def index():
    """Serve the main visualization page."""
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    if visualizer:
        # Send initial state to new client
        emit('initial_state', {
            'architecture': visualizer.layer_info,
            'total_params': visualizer.total_params,
            'trainable_params': visualizer.trainable_params,
            'device': str(visualizer.device),
            'is_training': visualizer.is_training
        })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Visualizer client disconnected')

@socketio.on('request_history')
def handle_history_request():
    """Send historical data to client."""
    if visualizer:
        emit('history_data', {
            'loss_history': list(visualizer.loss_history),
            'grad_history': list(visualizer.grad_history),
            'lr_history': list(visualizer.lr_history),
            'memory_history': list(visualizer.memory_history)
        })

def _find_free_port(preferred_port: int) -> int:
    import socket as _socket
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred_port))
            return preferred_port
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

def start_visualization_server(host='127.0.0.1', port=8080, open_browser=False):
    """
    Start the visualization server in a separate thread.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        open_browser: Whether to automatically open the browser
    """
    port = _find_free_port(port)

    def run_server():
        # Allow unsafe werkzeug only if explicitly in dev mode
        allow_unsafe = os.environ.get("VIZ_ALLOW_UNSAFE_WERKZEUG", "0") == "1"
        socketio.run(
            app,
            host=host,
            port=port,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=allow_unsafe,
        )
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Open browser after a short delay
    if open_browser:
        time.sleep(2)  # Give server time to start
        url = f'http://{host}:{port}'
        print(f"\n🎆 Opening visualization at {url}")
        webbrowser.open(url)
    
    return server_thread

if __name__ == '__main__':
    # Test mode - run server directly
    print("Starting Whisper Training Visualizer in test mode...")
    print("Open http://localhost:8080 in your browser")
    socketio.run(app, host='0.0.0.0', port=8080, debug=True, allow_unsafe_werkzeug=True)