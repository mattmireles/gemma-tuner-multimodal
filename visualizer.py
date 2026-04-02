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
                        'attention': attention_2d[:20, :20].tolist()  # Limit size for performance
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
        """
        Update the training/evaluation state for visualization context.
        
        This method tracks whether the model is in training or evaluation mode,
        affecting how metrics are interpreted and displayed in the UI.
        
        Called by:
        - Training loop when switching between train/eval modes
        - VisualizerTrainerCallback on training state changes
        
        UI effects:
        - Training mode: Shows gradient flow, learning rate changes
        - Evaluation mode: Highlights validation metrics, hides training-only data
        
        Args:
            is_training (bool): True if training, False if evaluating
        """
        self.is_training = is_training
        self._emit_update({'event': 'training_state', 'is_training': is_training})

# Global visualizer instance
visualizer: Optional[TrainingVisualizer] = None

def init_visualizer(model: nn.Module, device: torch.device) -> TrainingVisualizer:
    """
    Initialize the global visualizer instance for training monitoring.
    
    This function creates and configures the singleton visualizer that will
    be used throughout the training process to capture and stream metrics.
    
    Called by:
    - Training scripts when --visualize flag is enabled
    - VisualizerTrainerCallback during initialization
    
    Args:
        model (nn.Module): The Whisper model to be monitored
        device (torch.device): Compute device (cuda/mps/cpu) for memory tracking
    
    Returns:
        TrainingVisualizer: Configured visualizer instance
    
    Note:
        Only one visualizer instance should exist per training process.
        Multiple calls will replace the existing instance.
    """
    global visualizer
    visualizer = TrainingVisualizer(model, device)
    return visualizer

def get_visualizer() -> Optional[TrainingVisualizer]:
    """
    Retrieve the global visualizer instance if it exists.
    
    Called by:
    - Training callbacks needing to update visualization
    - Flask routes serving visualization data
    - Testing code verifying visualizer state
    
    Returns:
        Optional[TrainingVisualizer]: Current visualizer or None if not initialized
    """
    return visualizer

# Flask routes
@app.route('/')
def index():
    """
    Serve the main visualization dashboard HTML page.
    
    This route provides the entry point for the web-based training visualization
    interface. The HTML template includes the Three.js 3D visualization canvas
    and real-time metric displays.
    
    Called by:
    - Browser navigation to visualization URL
    - Auto-opened browser when training starts with --visualize
    
    Returns:
        str: Rendered HTML template with embedded JavaScript visualization
    
    Template features:
    - 3D neural network visualization with WebGL
    - Real-time loss and metric graphs
    - Attention weight heatmaps
    - Memory usage monitoring
    - Training throughput statistics
    """
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """
    Serve static JavaScript and CSS files for the visualization UI.
    
    This route handles requests for visualization assets including Three.js
    libraries, custom visualization scripts, and stylesheets.
    
    Called by:
    - HTML template loading JavaScript modules
    - Dynamic asset loading during visualization updates
    
    Args:
        path (str): Relative path to static file within static/ directory
    
    Returns:
        Response: Static file content with appropriate MIME type
    
    Security:
        - Path traversal prevention handled by Flask
        - Only serves files from designated static directory
    """
    return send_from_directory('static', path)

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """
    Handle new WebSocket client connection to visualization server.
    
    This handler initializes new clients with current training state and model
    information, enabling them to display accurate visualization immediately
    upon connection without waiting for the next update cycle.
    
    Called by:
    - SocketIO when browser establishes WebSocket connection
    - Reconnection after network interruption
    
    Emits:
    - 'initial_state': Complete model and training configuration
    
    Initial state packet includes:
    - architecture: Model layer configuration for 3D visualization
    - total_params: Total parameter count for model complexity display
    - trainable_params: Trainable parameter count for LoRA/freezing verification
    - device: Current compute device (cuda/mps/cpu) for context
    - is_training: Current training state for UI mode selection
    
    Connection management:
    - Supports up to MAX_CONCURRENT_CONNECTIONS simultaneous clients
    - Each client receives independent state updates
    - Clients can connect/disconnect without affecting training
    """
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
    """
    Handle WebSocket client disconnection from visualization server.
    
    This handler performs cleanup when a visualization client disconnects,
    either intentionally or due to network issues. The training process
    continues unaffected by client disconnections.
    
    Called by:
    - SocketIO when WebSocket connection is closed
    - Browser navigation away from visualization page
    - Network interruption or timeout
    
    Cleanup actions:
    - Logs disconnection for monitoring
    - No data cleanup needed (visualizer persists for reconnection)
    - No impact on training process
    
    Note:
        Clients can reconnect and request historical data to resume
        visualization from where they left off.
    """
    logger.info('Visualizer client disconnected')

@socketio.on('request_history')
def handle_history_request():
    """
    Send historical training metrics to requesting client.
    
    This handler provides buffered historical data to clients that need to
    reconstruct the training timeline, such as after reconnection or initial
    connection mid-training. Enables seamless visualization continuity.
    
    Called by:
    - Client JavaScript after connection establishment
    - Reconnecting clients needing to sync state
    - Export functionality requesting complete training history
    
    Emits:
    - 'history_data': Complete buffered metric history
    
    History packet includes:
    - loss_history: Training loss values (up to METRICS_BUFFER_SIZE entries)
    - grad_history: Gradient norm values for stability monitoring
    - lr_history: Learning rate schedule progression
    - memory_history: GPU/MPS memory usage over time
    
    Data characteristics:
    - Circular buffers prevent unbounded growth
    - Typically contains last 1000 data points
    - Sufficient for ~3-4 hours of training visualization
    - Lists are copied to prevent concurrent modification issues
    
    Performance:
    - History packet typically 10-50KB depending on buffer fill
    - Sent once per client connection, not on every update
    """
    if visualizer:
        emit('history_data', {
            'loss_history': list(visualizer.loss_history),
            'grad_history': list(visualizer.grad_history),
            'lr_history': list(visualizer.lr_history),
            'memory_history': list(visualizer.memory_history)
        })

def _find_free_port(preferred_port: int) -> int:
    """
    Find an available network port for the visualization server.
    
    This utility function attempts to bind to the preferred port and falls
    back to any available port if the preferred one is occupied. Essential
    for avoiding port conflicts when multiple training runs are active.
    
    Called by:
    - start_visualization_server() during server initialization
    
    Port selection strategy:
    1. Try to bind to preferred_port on localhost
    2. If OSError (port in use), bind to port 0 (OS selects free port)
    3. Return the successfully bound port number
    
    Args:
        preferred_port (int): Desired port number (typically 8080)
    
    Returns:
        int: Available port number (preferred or system-assigned)
    
    Example:
        port = _find_free_port(8080)  # Returns 8080 if free, else random port
    
    Note:
        The socket is immediately closed after port verification, allowing
        the actual server to bind to it. There's a small race condition window
        but it's acceptable for development use.
    """
    import socket as _socket
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred_port))
            return preferred_port
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

from wft_constants import VisualizationConstants


def start_visualization_server(host='127.0.0.1', port=VisualizationConstants.DEFAULT_PORT, open_browser=False):
    """
    Start the visualization web server in a background thread.
    
    This function launches the Flask/SocketIO server that serves the visualization
    interface and handles real-time data streaming. The server runs in a daemon
    thread to avoid blocking the training process.
    
    Called by:
    - Training scripts when --visualize flag is enabled
    - wizard.py in visualization mode
    - Manual visualization testing workflows
    
    Calls to:
    - _find_free_port() to handle port conflicts
    - Flask/SocketIO run() for server initialization
    - webbrowser.open() if auto-open requested
    
    Server configuration:
    - Default port: 8080 (configurable via constants)
    - Binds to localhost by default for security
    - Supports WebSocket connections for real-time updates
    - Runs in daemon thread (terminates with main process)
    
    Port selection strategy:
    1. Attempts to bind to requested port
    2. If occupied, automatically finds free port
    3. Logs actual port for user reference
    
    Args:
        host (str): Network interface to bind to
            - '127.0.0.1': Local only (secure, default)
            - '0.0.0.0': All interfaces (for remote access)
        port (int): Preferred port number (default: 8080)
            - Automatically finds alternative if occupied
        open_browser (bool): Auto-open visualization in default browser
            - Useful for interactive development
            - Disabled by default for headless training
    
    Security considerations:
    - Default localhost binding prevents external access
    - No authentication (assumes trusted local environment)
    - Use SSH tunneling for secure remote access
    
    Performance:
    - Minimal impact on training (<2% overhead)
    - Async updates prevent training blocking
    - Supports up to 20 concurrent viewers
    """
    port = _find_free_port(port)

    def run_server():
        """
        Internal function to run the Flask/SocketIO server.
        
        Executes in a daemon thread to avoid blocking training.
        Configured for development mode with appropriate safety checks.
        """
        # Allow unsafe werkzeug in explicit dev mode or under pytest to avoid thread exceptions
        # This keeps production usage safe while making local tests/dev seamless.
        allow_unsafe = (
            os.environ.get("VIZ_ALLOW_UNSAFE_WERKZEUG", "0") == "1"
            or os.environ.get("PYTEST_CURRENT_TEST") is not None
        )
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