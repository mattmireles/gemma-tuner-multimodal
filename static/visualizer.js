/**
 * Gemma Training Visualizer - The Magic Happens Here
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
/** @type {THREE.Mesh[]} Neuron spheres for gradient pulsing */
let galaxyNeuronMeshes = [];
let galaxyLastFingerprint = '';
const GALAXY_MAX_NODES = 340;
const GOLDEN_ANGLE = 2.39996322972865332;
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

// Honor prefers-reduced-motion. Read once at module load — the OS-level
// preference rarely changes mid-session, and a live MediaQueryList listener
// would just complicate the animate() hot path.
const reducedMotion =
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Hero loss display (single source of truth — no duplicate io() in the template) */
let lastHeroLoss = null;

function formatHeroLoss(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return '—';
    if (n >= 100) return n.toFixed(1);
    if (n >= 10) return n.toFixed(2);
    return n.toFixed(3);
}

function updateHeroLoss(n) {
    const el = document.getElementById('loss-value');
    if (!el) return;
    if (n === null || n === undefined || Number.isNaN(n)) return;
    el.textContent = formatHeroLoss(n);
    if (lastHeroLoss !== null && n > lastHeroLoss * 1.02) {
        el.classList.add('is-rising');
        setTimeout(() => el.classList.remove('is-rising'), 600);
    }
    lastHeroLoss = n;
}

// Audio context for sound effects (optional)
let audioContext = null;
let oscillator = null;

/**
 * Initialize everything when page loads
 */
window.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Gemma Training Visualizer...');
    
    initSocket();
    try {
        initCharts();
    } catch (e) {
        console.error('Chart init failed:', e);
    }
    if (enable3D) {
        try {
            init3DNeuralNetwork();
        } catch (e) {
            console.error('3D visualizer init failed (charts still work):', e);
        }
    } else {
        // Hide the whole panel card (title + body), not just the canvas slot,
        // so we don't leave an orphaned title.
        const card = document.getElementById('neural-network-3d')?.closest('.panel');
        if (card) card.style.display = 'none';
    }
    initEventListeners();
    
    // Start animation loop
    animate();
    
    // Hide loading indicator
    setTimeout(() => {
        document.getElementById('loading').style.display = 'none';
    }, 1000);

    // Wire feature toggle buttons.
    //
    // The previous implementation was `window[stateVarName] = !window[...]`,
    // which silently failed: top-level `let` bindings (enable3D, etc.) do not
    // become window properties, so the toggles wrote to window while the rest
    // of the code read the closure binding. Buttons appeared to do nothing.
    //
    // Fix: pass an explicit getter/setter pair so the click handler can read
    // and write the actual let binding via closures.
    const toggle = (id, getter, setter, onToggle) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.toggle('active', getter());
        el.addEventListener('click', () => {
            const next = !getter();
            setter(next);
            el.classList.toggle('active', next);
            if (typeof onToggle === 'function') onToggle(next);
        });
    };

    toggle(
        'toggle-3d',
        () => enable3D,
        (v) => { enable3D = v; },
        (on) => {
            const card = document.getElementById('neural-network-3d')?.closest('.panel');
            if (card) card.style.display = on ? '' : 'none';
            if (on && !renderer) {
                try {
                    init3DNeuralNetwork();
                } catch (e) {
                    console.error('3D visualizer init failed:', e);
                }
            }
        }
    );
    toggle(
        'toggle-attn',
        () => enableAttention,
        (v) => { enableAttention = v; },
        (on) => {
            const card = document.getElementById('attention-canvas')?.closest('.panel');
            if (card) card.style.display = on ? '' : 'none';
        }
    );
    toggle(
        'toggle-tokens',
        () => enableTokens,
        (v) => { enableTokens = v; },
        (on) => {
            const sec = document.getElementById('token-cloud')?.closest('.saying');
            if (sec) sec.style.display = on ? '' : 'none';
        }
    );
    toggle(
        'toggle-spec',
        () => enableSpectrogram,
        (v) => { enableSpectrogram = v; },
        (on) => {
            const card = document.getElementById('spectrogram-canvas')?.closest('.panel');
            if (card) card.style.display = on ? '' : 'none';
        }
    );
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
    
    // Placeholder spiral until ``initial_state`` delivers HF architecture
    createNeuralNetworkMesh(null);
    
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

function _hashString(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return h >>> 0;
}

function _mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
        let t = (a += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function mergeArchDefaults(arch) {
    const a = arch || {};
    return {
        encoder_layers: a.encoder_layers || 0,
        decoder_layers: a.decoder_layers || 0,
        num_hidden_layers: a.num_hidden_layers || 0,
        hidden_size: a.hidden_size || 2048,
        attention_heads: a.attention_heads || 8,
        vocab_size: a.vocab_size || 256000,
        total_params: a.total_params || 0,
        trainable_params: a.trainable_params || 0,
        model_type: a.model_type || 'unknown',
    };
}

function estimateRingCount(merged) {
    let n = merged.num_hidden_layers;
    if (!n) n = merged.encoder_layers + merged.decoder_layers;
    if (!n && merged.total_params) {
        const tp = Math.max(10, merged.total_params);
        n = Math.max(4, Math.min(48, Math.round(Math.log10(tp) * 8)));
    }
    if (!n) n = 14;
    return Math.max(4, Math.min(48, n));
}

function disposeGalaxyContents() {
    galaxyNeuronMeshes = [];
    if (!scene || !neuralNetwork) return;
    scene.remove(neuralNetwork);
    neuralNetwork.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
            else obj.material.dispose();
        }
    });
    neuralNetwork = null;
}

/**
 * Build a generic barred-spiral "galaxy" from HF architecture fields (any Gemma / modality).
 * @param {object|null} arch — from ``initial_state.architecture`` or training payload
 */
function createNeuralNetworkMesh(arch) {
    if (!scene) return;

    const container = document.getElementById('neural-network-3d');
    disposeGalaxyContents();

    const merged = mergeArchDefaults(arch);
    const rng = _mulberry32(_hashString(JSON.stringify(merged)) || 0xcafebabe);

    const ringCount = estimateRingCount(merged);
    const hidden = Math.max(256, merged.hidden_size);
    let nodesPerRing = Math.round(Math.sqrt(hidden / 96) * 3.5);
    nodesPerRing = Math.max(6, Math.min(22, nodesPerRing));
    let totalNodes = ringCount * nodesPerRing;
    if (totalNodes > GALAXY_MAX_NODES) {
        nodesPerRing = Math.max(4, Math.floor(GALAXY_MAX_NODES / ringCount));
    }

    const group = new THREE.Group();
    group.userData = { arch: merged, neuronMeshes: [] };

    const arms = Math.max(3, Math.min(12, merged.attention_heads || 6));
    const positions = [];

    for (let ring = 0; ring < ringCount; ring++) {
        const t = ring / Math.max(1, ringCount - 1);
        const spiralR = 0.9 + t * 7.2;
        const z = (ring - (ringCount - 1) / 2) * 1.02;
        // warm sweep: deep ember (inner rings) → bright gold (outer rings).
        // stays inside the amber signature; depth reads as warmth, not hue.
        const hue = 0.08 + 0.055 * t;

        for (let n = 0; n < nodesPerRing; n++) {
            const arm = n % arms;
            const ang =
                (2 * Math.PI * (n / nodesPerRing)) +
                ring * GOLDEN_ANGLE +
                (arm / arms) * 0.55;
            const wobble = (rng() - 0.5) * 0.35;
            const radiusJ = spiralR + wobble + Math.sin(ring * 0.45 + arm) * 0.12;
            const x = Math.cos(ang) * radiusJ;
            const y = Math.sin(ang) * radiusJ;

            positions.push({ x, y, z, ring, hue });
        }
    }

    const sphereR = 0.11 + Math.min(0.14, Math.log10(Math.max(merged.vocab_size, 1000)) * 0.018);
    const matForHue = (hue, emissiveScale) => {
        // Lightness rides the t ring-index so the galaxy reads as depth,
        // not color cycling. Saturation stays high to keep the amber warm.
        const l = 0.42 + (hue - 0.08) * 2.4;           // 0.42 → 0.55 ish
        const c = new THREE.Color().setHSL(hue % 1, 0.92, Math.min(0.62, l));
        return new THREE.MeshStandardMaterial({
            color: c,
            emissive: c.clone().multiplyScalar(0.4),
            emissiveIntensity: emissiveScale,
            metalness: 0.3,
            roughness: 0.35,
        });
    };

    positions.forEach((p, idx) => {
        const geo = new THREE.SphereGeometry(sphereR, 10, 10);
        const mat = matForHue(p.hue, 0.45 + rng() * 0.25);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(p.x, p.y, p.z);
        mesh.userData = { ring: p.ring, idx };
        group.add(mesh);
        group.userData.neuronMeshes.push(mesh);
        galaxyNeuronMeshes.push(mesh);
    });

    const lineMat = new THREE.LineBasicMaterial({
        color: 0xff8800,           // warm ember, additive — reads as a glow
        transparent: true,
        opacity: 0.20,
        blending: THREE.AdditiveBlending,
    });
    const linePoints = [];
    for (let ring = 0; ring < ringCount - 1; ring++) {
        const base = ring * nodesPerRing;
        const nextBase = (ring + 1) * nodesPerRing;
        for (let n = 0; n < nodesPerRing; n++) {
            const a = positions[base + n];
            const b = positions[nextBase + ((n + Math.floor(rng() * 3)) % nodesPerRing)];
            if (!a || !b) continue;
            if (rng() > 0.42) continue;
            linePoints.push(
                new THREE.Vector3(a.x, a.y, a.z),
                new THREE.Vector3(b.x, b.y, b.z)
            );
        }
    }
    if (linePoints.length) {
        const lg = new THREE.BufferGeometry().setFromPoints(linePoints);
        const lines = new THREE.LineSegments(lg, lineMat);
        group.add(lines);
    }

    const coreR = 0.55 + 0.08 * Math.log10(Math.max(merged.vocab_size, 1024));
    const coreGeo = new THREE.IcosahedronGeometry(coreR, 1);
    const coreMat = new THREE.MeshStandardMaterial({
        color: 0xffb000,           // the signature amber — this is the sun
        emissive: 0x663300,
        emissiveIntensity: 0.95,
        metalness: 0.2,
        roughness: 0.25,
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    core.userData.isCore = true;
    group.add(core);

    const dustN = Math.min(6000, Math.max(1200, Math.floor((merged.total_params || 5e8) / 1.2e6)));
    const dustPos = new Float32Array(dustN * 3);
    const dustCol = new Float32Array(dustN * 3);
    for (let i = 0; i < dustN; i++) {
        const u = rng();
        const v = rng();
        const theta = 2 * Math.PI * u;
        const phi = Math.acos(2 * v - 1);
        const rr = 3 + rng() * 11;
        dustPos[i * 3] = rr * Math.sin(phi) * Math.cos(theta);
        dustPos[i * 3 + 1] = rr * Math.sin(phi) * Math.sin(theta);
        dustPos[i * 3 + 2] = rr * Math.cos(phi) * 0.35 + (rng() - 0.5) * 4;
        // dust = warm starfield, scattered across the amber range with a few
        // near-white motes for highlights. no cool tones — the room is warm.
        const cool = rng() < 0.18;
        const col = cool
            ? new THREE.Color().setHSL(0.11, 0.15, 0.78 + rng() * 0.12)   // pale warm white
            : new THREE.Color().setHSL(0.07 + rng() * 0.08, 0.75, 0.55 + rng() * 0.18);
        dustCol[i * 3] = col.r;
        dustCol[i * 3 + 1] = col.g;
        dustCol[i * 3 + 2] = col.b;
    }
    const dustGeo = new THREE.BufferGeometry();
    dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
    dustGeo.setAttribute('color', new THREE.BufferAttribute(dustCol, 3));
    const dustMat = new THREE.PointsMaterial({
        size: 0.045,
        vertexColors: true,
        transparent: true,
        opacity: 0.55,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
    });
    const dust = new THREE.Points(dustGeo, dustMat);
    group.add(dust);

    // rim = warm cream, fill = deep ember. all-warm three-point lighting:
    // contrast comes from intensity and direction, not from cool/warm split.
    const rimLight = new THREE.PointLight(0xffe8b8, 1.15, 80);
    rimLight.position.set(14, 8, 10);
    group.add(rimLight);
    const fillLight = new THREE.PointLight(0xff7a1a, 0.65, 80);
    fillLight.position.set(-12, -6, -8);
    group.add(fillLight);

    group.userData.label =
        merged.model_type +
        ' · ' +
        ringCount +
        ' rings × ' +
        nodesPerRing +
        ' · ' +
        (merged.total_params ? (merged.total_params / 1e6).toFixed(1) + 'M params' : '');

    neuralNetwork = group;
    scene.add(neuralNetwork);

    // Replace the static panel subtitle with a quiet model fingerprint
    // (e.g. "whisper · 244M params") so a first-time user sees what their
    // machine is actually training. The panel title — "inside the model" —
    // stays put; this updates the line beneath it.
    const panelEl = document.querySelector('#neural-network-3d')?.closest('.panel');
    const subtitleEl = panelEl?.querySelector('.panel-subtitle');
    if (subtitleEl && merged.model_type && merged.model_type !== 'unknown') {
        const parts = [merged.model_type];
        if (merged.total_params) {
            parts.push((merged.total_params / 1e6).toFixed(0) + 'M params');
        }
        subtitleEl.textContent = parts.join(' · ');
    }
}

function galaxyFingerprint(arch, totalParams, trainableParams) {
    const a = arch || {};
    return [
        a.model_type,
        a.num_hidden_layers,
        a.encoder_layers,
        a.decoder_layers,
        a.hidden_size,
        a.attention_heads,
        a.vocab_size,
        totalParams || 0,
        trainableParams || 0,
    ].join('|');
}

function maybeRebuildGalaxyFromArchitecture(arch, totalParams, trainableParams) {
    if (!enable3D || !scene || !arch) return;
    const tp = totalParams ?? arch.total_params ?? 0;
    const trp = trainableParams ?? arch.trainable_params ?? 0;
    const fp = galaxyFingerprint(arch, tp, trp);
    if (fp === galaxyLastFingerprint) return;
    galaxyLastFingerprint = fp;
    const merged = mergeArchDefaults(arch);
    merged.total_params = tp || merged.total_params;
    merged.trainable_params = trp || merged.trainable_params;
    createNeuralNetworkMesh(merged);
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
                label: 'loss',
                data: [],
                borderColor: '#FFB000',
                backgroundColor: 'rgba(255, 176, 0, 0.08)',
                borderWidth: 1.5,
                tension: 0.35,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    display: false,
                    grid: { display: false },
                    ticks: { display: false }
                },
                y: {
                    // The heartbeat number above the chart is the value
                    // reference. The curve is the shape of the story.
                    display: false,
                    grid: { display: false },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
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
                label: 'signal',
                data: [],
                borderColor: '#FFB000',
                backgroundColor: 'rgba(255, 176, 0, 0.06)',
                borderWidth: 1,
                tension: 0.35,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: { display: false, grid: { display: false }, ticks: { display: false } },
                y: { display: false, grid: { display: false }, ticks: { display: false } }
            },
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
        }
    });

    // Memory chart
    const memCtx = document.getElementById('memory-chart').getContext('2d');
    memoryChart = new Chart(memCtx, {
        type: 'bar',
        data: {
            labels: [''],
            datasets: [{
                label: 'memory',
                data: [0],
                backgroundColor: 'rgba(255, 176, 0, 0.55)',
                borderColor: '#FFB000',
                borderWidth: 0,
                borderRadius: 0,
                maxBarThickness: 22
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 240 },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 16,
                    grid: { display: false },
                    ticks: {
                        color: '#3A3A38',
                        font: { family: 'ui-monospace, "SF Mono", monospace', size: 10 }
                    }
                },
                y: { display: false, grid: { display: false } }
            },
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
        }
    });

    // Learning rate chart
    const lrCtx = document.getElementById('lr-chart').getContext('2d');
    lrChart = new Chart(lrCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'step size',
                data: [],
                borderColor: '#FFB000',
                backgroundColor: 'rgba(255, 176, 0, 0.06)',
                borderWidth: 1,
                tension: 0.35,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: { display: false, grid: { display: false }, ticks: { display: false } },
                y: { type: 'logarithmic', display: false, grid: { display: false }, ticks: { display: false } }
            },
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
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
    
    // Update loss chart + hero
    if (data.loss !== undefined) {
        updateHeroLoss(data.loss);
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

    if (data.architecture) {
        maybeRebuildGalaxyFromArchitecture(
            data.architecture,
            data.total_params ?? data.architecture.total_params,
            data.trainable_params ?? data.architecture.trainable_params
        );
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
    if (!lossChart) return;
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
    if (!gradientChart) return;
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
    if (!memoryChart) return;
    memoryChart.data.datasets[0].data[0] = memoryGB;
    
    // Calm amber until memory is in danger; rose only when nearing the cap.
    const percentage = memoryGB / 16;
    let color;
    if (percentage < 0.85) {
        color = 'rgba(255, 176, 0, 0.55)';  // amber
    } else if (percentage < 0.95) {
        color = 'rgba(255, 176, 0, 0.85)';  // amber, intensified
    } else {
        color = 'rgba(255, 77, 109, 0.85)'; // rose — the only danger signal
    }

    memoryChart.data.datasets[0].backgroundColor = color;
    memoryChart.update('none');
}

/**
 * Update learning rate chart
 */
function updateLearningRateChart(lr, step) {
    if (!lrChart) return;
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
    const intensity = Math.min(gradNorm / 10, 1);
    const baseEmissive = 0.35;
    galaxyNeuronMeshes.forEach((neuron) => {
        if (neuron.material) {
            neuron.material.emissiveIntensity = baseEmissive + intensity * 0.95;
            const scale = 1 + intensity * 0.22;
            neuron.scale.set(scale, scale, scale);
        }
    });
    if (neuralNetwork) {
        neuralNetwork.traverse((obj) => {
            if (obj.userData && obj.userData.isCore && obj.material) {
                obj.material.emissiveIntensity = 0.75 + intensity * 0.55;
            }
        });
    }
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
    // Pause button — clean lowercase label, no injected icon markup.
    document.getElementById('pause-btn').addEventListener('click', () => {
        isPaused = !isPaused;
        const btn = document.getElementById('pause-btn');
        btn.textContent = isPaused ? 'resume' : 'pause';
        btn.classList.toggle('active', isPaused);
    });

    // Sound button — opt-in beep oscillator from playLossSound().
    document.getElementById('sound-btn').addEventListener('click', () => {
        soundEnabled = !soundEnabled;
        const btn = document.getElementById('sound-btn');
        btn.textContent = soundEnabled ? 'sound on' : 'sound';
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
    if (data.architecture) {
        maybeRebuildGalaxyFromArchitecture(data.architecture, data.total_params, data.trainable_params);
    }
}

/**
 * Load historical data
 */
function loadHistoricalData(data) {
    // Load loss history
    if (data.loss_history && data.loss_history.length) {
        data.loss_history.forEach((loss, i) => {
            updateLossChart(loss, i);
        });
        updateHeroLoss(data.loss_history[data.loss_history.length - 1]);
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

    if (data.memory_history && data.memory_history.length) {
        updateMemoryChart(data.memory_history[data.memory_history.length - 1]);
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
    
    // Slow rotation, slow breathing pulse, slow core spin — the lava-lamp
    // ambient life that says "the room is alive even between training steps."
    // Suppressed entirely under prefers-reduced-motion.
    if (neuralNetwork && !isPaused && !reducedMotion) {
        neuralNetwork.rotation.y += 0.002;
        const pulse = Math.sin(currentTime * 0.001) * 0.08 + 1;
        neuralNetwork.scale.set(pulse, pulse, pulse);
        neuralNetwork.traverse((obj) => {
            if (obj.userData && obj.userData.isCore) {
                obj.rotation.x += 0.006;
                obj.rotation.y += 0.009;
            }
        });
    }
}

console.log('✨ Gemma Training Visualizer loaded and ready!');