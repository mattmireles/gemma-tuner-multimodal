# 🎆 Whisper Training Visualizer

## The Most Beautiful Way to Watch AI Learn!

Turn your boring training logs into a **mesmerizing light show** with real-time 3D neural networks, flowing gradients, and particle explosions!

![Training Visualizer](https://img.shields.io/badge/Status-Epic-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Web](https://img.shields.io/badge/Web-Three.js-orange)

## ✨ Features

### Real-Time Visualizations
- **3D Neural Network Galaxy** - Watch neurons fire and connections glow
- **Loss Landscape** - See your model descend into optimal valleys
- **Gradient Flow** - Flowing rivers of backpropagation
- **Attention Heatmaps** - Visualize what your model is focusing on
- **Audio Spectrograms** - See audio transform into text
- **Token Probability Clouds** - Watch words emerge from probability distributions
- **Memory Pulse** - System heartbeat that shows GPU/MPS usage
- **Learning Rate Dance** - Watch the optimizer's rhythm

### Interactive Controls
- 🖱️ **Mouse rotation** - Explore the 3D neural network
- ⏸️ **Pause/Resume** - Control the data flow
- 🔊 **Sound effects** - Optional audio feedback (beeps and bloops!)
- 🖼️ **Fullscreen mode** - Immersive training experience

### Performance Features
- 60 FPS smooth animations
- WebSocket real-time streaming
- Efficient data buffering
- No impact on training performance

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask flask-socketio python-socketio
```

### 2. Use with the Wizard
```bash
whisper-tuner wizard
# When asked "Enable live training visualization?" → Say YES! 🎆
```

### 3. Or Test It Standalone
```bash
# Start a custom training flow with `visualize_training=True` in your trainer hook.
```

## 🎮 How to Use

### With Training Wizard
1. Run `whisper-tuner wizard`
2. Configure your training as normal
3. When asked about visualization, choose **Yes**
4. Browser opens automatically when training starts
5. Watch your AI learn in real-time!

### Manual Integration
```python
from whisper_tuner.visualizer import init_visualizer, get_visualizer

# In your training script
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
device = torch.device("mps")

# Initialize visualizer
viz = init_visualizer(model, device)

# Start the server
from whisper_tuner.visualizer import start_visualization_server
start_visualization_server(open_browser=True)

# In your training loop
for batch in dataloader:
    outputs = model(batch)
    loss = outputs.loss
    
    # Update visualizer
    viz.update_training_step(
        loss=loss.item(),
        learning_rate=optimizer.param_groups[0]['lr'],
        batch=batch,
        outputs=outputs,
        optimizer=optimizer
    )
```

## 🎨 What You'll See

### Loss Landscape
- **Green line** = Training loss over time
- **Smooth curves** = Model is learning well
- **Particle explosions** = Low loss celebration! 🎉

### Neural Network Galaxy
- **Green neurons** = Encoder layers
- **Yellow neurons** = Decoder layers
- **Glowing connections** = Active weights
- **Pulsing** = Gradient flow intensity

### Attention Heatmap
- **Red areas** = High attention
- **Blue areas** = Low attention
- Shows what the model "looks at"

### Token Probability Cloud
- **Larger tokens** = Higher probability
- **Brighter** = More confident
- Top 5 most likely next tokens

### Memory Wave
- **Green** = Safe memory usage (<50%)
- **Yellow** = Getting full (50-80%)
- **Red** = Danger zone (>80%)

## 🛠️ Customization

### Change Port
```python
start_visualization_server(host='0.0.0.0', port=9000)
```

### Disable Browser Auto-Open
```python
start_visualization_server(open_browser=False)
```

### Adjust Update Frequency
```python
viz.update_frequency = 5  # Update every 5 steps instead of 10
```

## 🔧 Troubleshooting

### Server Won't Start
- Check port 8080 is free: `lsof -i :8080`
- Try different port: `port=9000`

### No Data Showing
- Verify training is running
- Check browser console for errors (F12)
- Ensure WebSocket connection established

### Performance Issues
- Reduce update frequency
- Close other browser tabs
- Disable sound effects

## 🎯 Pro Tips

1. **Best viewed in Chrome/Safari** for WebGL performance
2. **Dark room + fullscreen** = Maximum immersion
3. **Sound on** for the full experience
4. **Multiple monitors** = Training on one, visualizer on another
5. **Screenshot** the best moments for your papers!

## 🏗️ Architecture

```
visualizer.py          # Backend server + data extraction
├── templates/
│   └── index.html    # Main UI with Three.js setup
└── static/
    └── visualizer.js # 3D animations and real-time updates
```

### Data Flow
1. Training loop → `viz.update_training_step()`
2. Data extraction → WebSocket emission
3. Browser receives → Updates visualizations
4. 60 FPS render loop → Smooth animations

## 📊 Real Data Being Visualized

Everything you see is **REAL training data**:
- ✅ Actual loss values
- ✅ Real gradient norms
- ✅ True attention weights
- ✅ Actual token probabilities
- ✅ Real memory usage
- ✅ True learning rate
- ✅ Actual mel spectrograms

## 🚦 System Requirements

- Python 3.8+
- Modern browser with WebGL support
- 4GB+ RAM for smooth performance
- Apple Silicon (M1/M2/M3) or NVIDIA GPU recommended

## 🎬 Demo

Run a training flow with visualization enabled to see it in action:
```bash
# (No standalone test script is shipped.)
# Start your training workflow as usual and enable visualization.
```

The visualizer server opens at http://localhost:8080 by default.

## 🤝 Contributing

Want to make it even more epic? Ideas welcome:
- More particle effects
- Additional visualizations
- VR support (why not?)
- Music generation from loss curves

## 📝 License

Part of the Whisper Fine-Tuner for Apple Silicon project.

---

**Remember**: Training AI doesn't have to be boring. Make it beautiful! 🎆✨

*"The best way to predict the future is to train it beautifully."* - Probably Steve Jobs
