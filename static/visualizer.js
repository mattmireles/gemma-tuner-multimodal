/**
 * Whisper Training Visualizer - The Magic Happens Here
 * 
 * This is where we turn boring training data into a mesmerizing light show!
 * Real-time 3D neural networks, flowing gradients, and particle explosions.
 */

// Global state
let socket = null;
let reconnectDelay = 1000; // Start with 1s
let reconnectTimer = null;
let isPaused = false;
let soundEnabled = false;
let animationFrameId = null;

// Feature flags (URL params)
const urlParams = new URLSearchParams(window.location.search);
const lightMode = urlParams.get('viz') === 'light';
let enable3D = urlParams.get('show3D') !== '0' && !lightMode;
let enableAttention = urlParams.get('showAttention') !== '0' && !lightMode;
let enableTokens = urlParams.get('showTokens') !== '0' && !lightMode;
let enableSpectrogram = urlParams.get('showSpectrogram') !== '0' && !lightMode;

// Three.js objects
let scene, camera, renderer;
let neuralNetwork = null;
let particles = [];

// Chart.js objects
let lossChart, gradientChart, memoryChart, lrChart;

// Data buffers
const dataBuffers = {
    loss: [],
    gradients: [],
    memory: [],
    learningRate: [],
    attention: null,
    tokens: [],
    spectrogram: null
};

// Performance tracking
let lastFrameTime = Date.now();
let frameCount = 0;
let fps = 60;

// Audio context for sound effects (optional)
let audioContext = null;
let oscillator = null;

/**
 * Initialize everything when page loads
 */
window.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Whisper Training Visualizer...');
    
    initSocket();
    initBackgroundParticles();
    if (enable3D) {
        init3DNeuralNetwork();
    } else {
        const panel = document.getElementById('neural-network-3d');
        if (panel && panel.parentElement) panel.parentElement.style.display = 'none';
    }
    initCharts();
    initEventListeners();
    
    // Start animation loop
    animate();
    
    // Hide loading indicator
    setTimeout(() => {
        document.getElementById('loading').style.display = 'none';
    }, 1000);

    // Wire feature toggle buttons
    const toggle = (id, stateVarName, onToggle) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('click', () => {
            window[stateVarName] = !window[stateVarName];
            el.classList.toggle('active', window[stateVarName]);
            if (typeof onToggle === 'function') onToggle(window[stateVarName]);
        });
    };

    toggle('toggle-3d', 'enable3D', (on) => {
        const panel = document.getElementById('neural-network-3d').parentElement;
        panel.style.display = on ? '' : 'none';
        if (on && !renderer) init3DNeuralNetwork();
    });
    toggle('toggle-attn', 'enableAttention');
    toggle('toggle-tokens', 'enableTokens');
    toggle('toggle-spec', 'enableSpectrogram');
});

/**
 * Initialize WebSocket connection
 */
function initSocket() {
    socket = io({
        reconnectionAttempts: Infinity,
        reconnectionDelay: 500,
        reconnectionDelayMax: 5000,
    });
    
    socket.on('connect', () => {
        console.log('✅ Connected to training server');
        socket.emit('request_history');
    });
    
    socket.on('disconnect', () => {
        console.log('❌ Disconnected from training server');
    });

    socket.on('connect_error', (err) => {
        console.log('⚠️ Connection error:', err.message);
    });
    
    socket.on('initial_state', (data) => {
        console.log('📊 Received initial state:', data);
        updateStats(data);
    });
    
    socket.on('training_update', (data) => {
        if (!isPaused) {
            handleTrainingUpdate(data);
        }
    });
    
    socket.on('history_data', (data) => {
        console.log('📈 Received history data');
        loadHistoricalData(data);
    });
}

/**
 * Initialize animated background particles
 */
function initBackgroundParticles() {
    const canvas = document.getElementById('particles-bg');
    const ctx = canvas.getContext('2d');
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const particles = [];
    const particleCount = 100;
    
    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            size: Math.random() * 2
        });
    }
    
    function drawParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        particles.forEach((p, i) => {
            // Create gradient color based on particle position
            const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 3);
            const t = i / particleCount;
            if (t < 0.4) {
                gradient.addColorStop(0, 'rgba(255, 0, 204, 0.8)'); // Pink
                gradient.addColorStop(1, 'rgba(255, 0, 204, 0)');
            } else if (t < 0.7) {
                gradient.addColorStop(0, 'rgba(153, 51, 204, 0.8)'); // Purple
                gradient.addColorStop(1, 'rgba(153, 51, 204, 0)');
            } else {
                gradient.addColorStop(0, 'rgba(0, 153, 255, 0.8)'); // Blue
                gradient.addColorStop(1, 'rgba(0, 153, 255, 0)');
            }
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
            
            // Update position
            p.x += p.vx;
            p.y += p.vy;
            
            // Wrap around screen
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
        });
        
        requestAnimationFrame(drawParticles);
    }
    
    drawParticles();
    
    // Handle window resize
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}

/**
 * Initialize 3D Neural Network visualization
 */
function init3DNeuralNetwork() {
    const container = document.getElementById('neural-network-3d');
    const width = container.clientWidth;
    const height = container.clientHeight || 150; // Use container's actual height
    
    // Create scene
    scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x000000, 1, 100);
    
    // Create camera - adjusted for smaller viewport
    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 20;
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Create neural network structure
    createNeuralNetworkMesh();
    
    // Add mouse controls
    let mouseX = 0, mouseY = 0;
    container.addEventListener('mousemove', (event) => {
        const rect = container.getBoundingClientRect();
        mouseX = ((event.clientX - rect.left) / width) * 2 - 1;
        mouseY = -((event.clientY - rect.top) / height) * 2 + 1;
    });
    
    // Mouse rotation
    function updateCameraPosition() {
        if (neuralNetwork) {
            neuralNetwork.rotation.y = mouseX * 0.5;
            neuralNetwork.rotation.x = mouseY * 0.3;
        }
    }
    
    // Add to animation loop
    const animateNetwork = () => {
        updateCameraPosition();
        renderer.render(scene, camera);
    };
    
    // Store animation function
    window.animateNetwork = animateNetwork;
}

/**
 * Create the 3D neural network mesh
 */
function createNeuralNetworkMesh() {
    const group = new THREE.Group();
    
    // Create layers (encoder, decoder)
    const layers = [];
    const layerCount = 6;
    const neuronsPerLayer = 8;
    const layerSpacing = 5;
    
    for (let l = 0; l < layerCount; l++) {
        const layer = new THREE.Group();
        const isEncoder = l < layerCount / 2;
        
        for (let n = 0; n < neuronsPerLayer; n++) {
            // Create neuron
            const geometry = new THREE.SphereGeometry(0.3, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color: isEncoder ? 0xFF00CC : 0x0099FF,  // Pink for encoder, Blue for decoder
                emissive: isEncoder ? 0xFF00CC : 0x0099FF,
                emissiveIntensity: 0.2
            });
            const neuron = new THREE.Mesh(geometry, material);
            
            // Position in circle
            const angle = (n / neuronsPerLayer) * Math.PI * 2;
            const radius = 3;
            neuron.position.x = Math.cos(angle) * radius;
            neuron.position.y = Math.sin(angle) * radius;
            neuron.position.z = (l - layerCount / 2) * layerSpacing;
            
            // Store reference
            neuron.userData = { layer: l, index: n };
            layer.add(neuron);
        }
        
        layers.push(layer);
        group.add(layer);
    }
    
    // Create connections between layers
    const connectionMaterial = new THREE.LineBasicMaterial({
        color: 0x9933CC,  // Purple for connections
        opacity: 0.3,
        transparent: true
    });
    
    for (let l = 0; l < layers.length - 1; l++) {
        const currentLayer = layers[l];
        const nextLayer = layers[l + 1];
        
        currentLayer.children.forEach((neuron1, i) => {
            nextLayer.children.forEach((neuron2, j) => {
                // Create connection with some probability for visual clarity
                if (Math.random() > 0.7) {
                    const points = [];
                    points.push(neuron1.position.clone());
                    points.push(neuron2.position.clone());
                    
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const line = new THREE.Line(geometry, connectionMaterial);
                    group.add(line);
                }
            });
        });
    }
    
    neuralNetwork = group;
    scene.add(neuralNetwork);
}

/**
 * Initialize Chart.js charts
 */
function initCharts() {
    // Loss chart
    const lossCtx = document.getElementById('loss-chart').getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: '#FF00CC',
                backgroundColor: 'rgba(255, 0, 204, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: { 
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF' }
                },
                y: { 
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF' }
                }
            },
            plugins: {
                legend: { labels: { color: '#FFFFFF' } }
            }
        }
    });
    
    // Gradient chart
    const gradCtx = document.getElementById('gradient-chart').getContext('2d');
    gradientChart = new Chart(gradCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Gradient Norm',
                data: [],
                borderColor: '#9933CC',
                backgroundColor: 'rgba(153, 51, 204, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: { 
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF' }
                },
                y: { 
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF' }
                }
            },
            plugins: {
                legend: { labels: { color: '#FFFFFF' } }
            }
        }
    });
    
    // Memory chart
    const memCtx = document.getElementById('memory-chart').getContext('2d');
    memoryChart = new Chart(memCtx, {
        type: 'bar',
        data: {
            labels: ['GPU Memory'],
            datasets: [{
                label: 'Memory (GB)',
                data: [0],
                backgroundColor: 'rgba(0, 153, 255, 0.6)',
                borderColor: '#0099FF',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            scales: {
                y: { 
                    beginAtZero: true,
                    max: 16,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
    
    // Learning rate chart
    const lrCtx = document.getElementById('lr-chart').getContext('2d');
    lrChart = new Chart(lrCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Learning Rate',
                data: [],
                borderColor: '#FF00CC',
                backgroundColor: 'rgba(255, 0, 204, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: { 
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF' }
                },
                y: { 
                    type: 'logarithmic',
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#FFFFFF' }
                }
            },
            plugins: {
                legend: { labels: { color: '#FFFFFF' } }
            }
        }
    });
}

/**
 * Handle training update from server
 */
function handleTrainingUpdate(data) {
    // Update stats
    if (data.step !== undefined) {
        document.getElementById('step-count').textContent = data.step;
    }
    if (data.epoch !== undefined) {
        document.getElementById('epoch-count').textContent = data.epoch;
    }
    if (data.steps_per_second !== undefined) {
        document.getElementById('speed').textContent = data.steps_per_second.toFixed(1);
    }
    
    // Update loss chart
    if (data.loss !== undefined) {
        updateLossChart(data.loss, data.step);
        
        // Create particle explosion on low loss
        if (data.loss < 0.1) {
            createParticleExplosion();
        }
    }
    
    // Update gradient chart
    if (data.gradient_norm !== undefined) {
        updateGradientChart(data.gradient_norm, data.step);
        updateNeuralNetworkGradients(data.gradient_norm);
    }
    
    // Update memory
    if (data.memory_gb !== undefined) {
        updateMemoryChart(data.memory_gb);
    }
    
    // Update learning rate
    if (data.learning_rate !== undefined) {
        updateLearningRateChart(data.learning_rate, data.step);
    }
    
    // Update attention heatmap
    if (enableAttention && data.attention) {
        updateAttentionHeatmap(data.attention);
    }
    
    // Update token probabilities
    if (enableTokens && data.token_probs) {
        updateTokenCloud(data.token_probs);
    }
    
    // Update spectrogram
    if (enableSpectrogram && data.mel_spectrogram) {
        updateSpectrogram(data.mel_spectrogram);
    }
    
    // Sound effects
    if (soundEnabled && data.loss !== undefined) {
        playLossSound(data.loss);
    }
}

/**
 * Update loss chart
 */
function updateLossChart(loss, step) {
    const maxPoints = 100;
    
    lossChart.data.labels.push(step || lossChart.data.labels.length);
    lossChart.data.datasets[0].data.push(loss);
    
    // Keep only last N points
    if (lossChart.data.labels.length > maxPoints) {
        lossChart.data.labels.shift();
        lossChart.data.datasets[0].data.shift();
    }
    
    lossChart.update('none');
}

/**
 * Update gradient chart
 */
function updateGradientChart(gradNorm, step) {
    const maxPoints = 100;
    
    gradientChart.data.labels.push(step || gradientChart.data.labels.length);
    gradientChart.data.datasets[0].data.push(gradNorm);
    
    // Keep only last N points
    if (gradientChart.data.labels.length > maxPoints) {
        gradientChart.data.labels.shift();
        gradientChart.data.datasets[0].data.shift();
    }
    
    gradientChart.update('none');
}

/**
 * Update memory chart
 */
function updateMemoryChart(memoryGB) {
    memoryChart.data.datasets[0].data[0] = memoryGB;
    
    // Change color based on memory usage
    const percentage = memoryGB / 16;
    let color;
    if (percentage < 0.5) {
        color = 'rgba(0, 153, 255, 0.6)';  // Blue for low
    } else if (percentage < 0.8) {
        color = 'rgba(153, 51, 204, 0.6)';  // Purple for medium
    } else {
        color = 'rgba(255, 0, 204, 0.6)';  // Pink for high
    }
    
    memoryChart.data.datasets[0].backgroundColor = color;
    memoryChart.update('none');
}

/**
 * Update learning rate chart
 */
function updateLearningRateChart(lr, step) {
    const maxPoints = 100;
    
    lrChart.data.labels.push(step || lrChart.data.labels.length);
    lrChart.data.datasets[0].data.push(lr);
    
    // Keep only last N points
    if (lrChart.data.labels.length > maxPoints) {
        lrChart.data.labels.shift();
        lrChart.data.datasets[0].data.shift();
    }
    
    lrChart.update('none');
}

/**
 * Update neural network based on gradients
 */
function updateNeuralNetworkGradients(gradNorm) {
    if (!neuralNetwork) return;
    
    // Pulse neurons based on gradient magnitude
    const intensity = Math.min(gradNorm / 10, 1);
    
    neuralNetwork.children.forEach((layer) => {
        if (layer.children) {
            layer.children.forEach((neuron) => {
                if (neuron.material) {
                    // Animate emissive intensity
                    neuron.material.emissiveIntensity = 0.2 + intensity * 0.8;
                    
                    // Scale based on gradient
                    const scale = 1 + intensity * 0.2;
                    neuron.scale.set(scale, scale, scale);
                }
            });
        }
    });
}

/**
 * Update attention heatmap
 */
function updateAttentionHeatmap(attentionData) {
    const canvas = document.getElementById('attention-canvas');
    const ctx = canvas.getContext('2d');
    
    if (!enableAttention || !attentionData || !attentionData.length) return;
    
    const size = attentionData.length;
    const container = canvas.parentElement;
    canvas.width = container.clientWidth || 200;
    canvas.height = container.clientHeight || 100;
    
    const cellSize = canvas.width / size;
    
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const value = attentionData[i][j] || 0;
            const intensity = Math.min(value * 255, 255);
            
            // Create gradient from blue to red
            const r = intensity;
            const g = 255 - intensity;
            const b = 0;
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
    }
}

/**
 * Update token probability cloud
 */
function updateTokenCloud(tokenProbs) {
    const container = document.getElementById('token-cloud');
    container.innerHTML = '';
    
    if (!enableTokens || !tokenProbs || !tokenProbs.indices) return;
    
    tokenProbs.indices.forEach((tokenId, i) => {
        const prob = tokenProbs.values[i];
        const tokenDiv = document.createElement('div');
        tokenDiv.className = 'token';
        tokenDiv.textContent = `Token ${tokenId}: ${(prob * 100).toFixed(1)}%`;
        tokenDiv.style.opacity = 0.3 + prob * 0.7;
        tokenDiv.style.transform = `scale(${0.8 + prob * 0.4})`;
        container.appendChild(tokenDiv);
    });
}

/**
 * Update audio spectrogram
 */
function updateSpectrogram(spectrogramData) {
    const canvas = document.getElementById('spectrogram-canvas');
    const ctx = canvas.getContext('2d');
    
    if (!enableSpectrogram || !spectrogramData || !spectrogramData.length) return;
    
    const height = spectrogramData.length;
    const width = spectrogramData[0].length;
    
    const container = canvas.parentElement;
    canvas.width = container.clientWidth || 300;
    canvas.height = container.clientHeight || 100;
    
    const cellWidth = canvas.width / width;
    const cellHeight = canvas.height / height;
    
    for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
            const value = spectrogramData[i][j] || 0;
            const intensity = Math.min(Math.abs(value) * 50, 255);
            
            ctx.fillStyle = `hsl(${240 - intensity}, 100%, ${intensity / 255 * 50}%)`;
            ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
        }
    }
}

/**
 * Create particle explosion effect
 */
function createParticleExplosion() {
    const container = document.body;
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = '50%';
        particle.style.top = '50%';
        
        const angle = (Math.PI * 2 * i) / particleCount;
        const velocity = 200 + Math.random() * 200;
        
        container.appendChild(particle);
        
        // Animate particle
        let opacity = 1;
        let x = 0;
        let y = 0;
        
        const animateParticle = () => {
            x += Math.cos(angle) * velocity * 0.02;
            y += Math.sin(angle) * velocity * 0.02;
            opacity -= 0.02;
            
            particle.style.transform = `translate(${x}px, ${y}px)`;
            particle.style.opacity = opacity;
            
            if (opacity > 0) {
                requestAnimationFrame(animateParticle);
            } else {
                particle.remove();
            }
        };
        
        requestAnimationFrame(animateParticle);
    }
}

/**
 * Play sound based on loss value
 */
function playLossSound(loss) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    // Map loss to frequency (lower loss = higher pitch)
    const frequency = 200 + (1 - Math.min(loss, 1)) * 600;
    oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
    
    // Quick beep
    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.1);
}

/**
 * Initialize event listeners
 */
function initEventListeners() {
    // Pause button
    document.getElementById('pause-btn').addEventListener('click', () => {
        isPaused = !isPaused;
        const btn = document.getElementById('pause-btn');
        btn.innerHTML = isPaused ? '<i class="fas fa-play"></i> Resume' : '<i class="fas fa-pause"></i> Pause';
        btn.classList.toggle('active', isPaused);
    });
    
    // Sound button
    document.getElementById('sound-btn').addEventListener('click', () => {
        soundEnabled = !soundEnabled;
        const btn = document.getElementById('sound-btn');
        btn.innerHTML = soundEnabled ? '<i class="fas fa-volume-up"></i> Sound' : '<i class="fas fa-volume-mute"></i> Sound';
        btn.classList.toggle('active', soundEnabled);
    });
    
    // Fullscreen button
    document.getElementById('fullscreen-btn').addEventListener('click', () => {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    });
}

/**
 * Update statistics display
 */
function updateStats(data) {
    if (data.device) {
        document.getElementById('device-name').textContent = data.device;
    }
    if (data.total_params) {
        const millions = (data.total_params / 1e6).toFixed(1);
        document.getElementById('param-count').textContent = `${millions}M`;
    }
}

/**
 * Load historical data
 */
function loadHistoricalData(data) {
    // Load loss history
    if (data.loss_history) {
        data.loss_history.forEach((loss, i) => {
            updateLossChart(loss, i);
        });
    }
    
    // Load gradient history
    if (data.grad_history) {
        data.grad_history.forEach((grad, i) => {
            updateGradientChart(grad, i);
        });
    }
    
    // Load learning rate history
    if (data.lr_history) {
        data.lr_history.forEach((lr, i) => {
            updateLearningRateChart(lr, i);
        });
    }
}

/**
 * Main animation loop
 */
function animate() {
    animationFrameId = requestAnimationFrame(animate);
    
    // Calculate FPS
    const currentTime = Date.now();
    const deltaTime = currentTime - lastFrameTime;
    frameCount++;
    
    if (frameCount % 30 === 0) {
        fps = Math.round(1000 / deltaTime);
        document.getElementById('fps-value').textContent = fps;
    }
    
    lastFrameTime = currentTime;
    
    // Animate 3D neural network
    if (window.animateNetwork) {
        window.animateNetwork();
    }
    
    // Rotate neural network
    if (neuralNetwork && !isPaused) {
        neuralNetwork.rotation.y += 0.002;
        
        // Pulse effect
        const pulse = Math.sin(currentTime * 0.001) * 0.1 + 1;
        neuralNetwork.scale.set(pulse, pulse, pulse);
    }
}

console.log('✨ Whisper Training Visualizer loaded and ready!');