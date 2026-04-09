/**
 * Socket.IO, training handlers, DOM, animation loop (window.GemmaViz).
 */
(function (V) {
/** Hero caption: "Your machine is <verb>." — lifecycle only (lowercase). */
const HERO_VERBS = { READY: 'getting ready', LEARN: 'learning', DONE: 'done learning' };

function setHeroVerb(phrase) {
    const el = document.getElementById('hero-verb');
    if (!el) return;
    el.textContent = phrase;
}

function initSocket() {
    V.socket = io({
        reconnectionAttempts: Infinity,
        reconnectionDelay: 500,
        reconnectionDelayMax: 5000,
    });
    
    V.socket.on('connect', () => {
        console.log('✅ Connected to training server');
        if (window.__vizStatus && window.__vizStatus.setConnected) {
            window.__vizStatus.setConnected(true);
        }
        V.socket.emit('request_history');
    });
    
    V.socket.on('disconnect', () => {
        console.log('❌ Disconnected from training server');
        if (window.__vizStatus && window.__vizStatus.setDisconnected) {
            window.__vizStatus.setDisconnected();
        }
    });

    V.socket.on('connect_error', (err) => {
        console.log('⚠️ Connection error:', err.message);
    });
    
    V.socket.on('initial_state', (data) => {
        console.log('📊 Received initial state:', data);
        updateStats(data);
        if (data && data.is_training === true) {
            setHeroVerb(HERO_VERBS.LEARN);
        }
    });

    V.socket.on('training_update', (data) => {
        if (data && data.event === 'training_finished') {
            setHeroVerb(HERO_VERBS.DONE);
            // Auto-stop recording when training ends so the recap captures
            // the complete run without requiring the user to remember to stop.
            if (V.recording && V.recording.isActive()) {
                V.recordingAutoStopped = true;
                V.recording.stop();
            }
            return;
        }
        if (!V.isPaused) {
            handleTrainingUpdate(data);
        }
    });
    
    V.socket.on('history_data', (data) => {
        console.log('📈 Received history data');
        loadHistoricalData(data);
    });
}

function handleTrainingUpdate(data) {
    if (window.__vizStatus && window.__vizStatus.markTrainingData) {
        window.__vizStatus.markTrainingData();
    }

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
        V.hasSeenTrainingLoss = true;
        setHeroVerb(HERO_VERBS.LEARN);
        V.updateHeroLoss(data.loss);
        V.updateLossChart(data.loss, data.step);
        
        // Create particle explosion on low loss
        if (data.loss < 0.1) {
            V.createParticleExplosion();
        }
    }
    
    // Update gradient chart
    if (data.gradient_norm !== undefined) {
        V.updateGradientChart(data.gradient_norm, data.step);
        V.updateNeuralNetworkGradients(data.gradient_norm);
    }
    
    // Update memory
    if (data.memory_gb !== undefined) {
        V.updateMemoryChart(data.memory_gb);
    }
    
    // Update learning rate
    if (data.learning_rate !== undefined) {
        V.updateLearningRateChart(data.learning_rate, data.step);
    }
    
    // Update attention heatmap
    if (V.enableAttention && data.attention) {
        V.updateAttentionHeatmap(data.attention);
    }
    
    // Update token probabilities
    if (V.enableTokens && data.token_probs) {
        V.updateTokenCloud(data.token_probs);
    }
    
    // Audio: auto-reveal the "listening" panel the first time a real
    // spectrogram arrives in the training stream. Text-only runs never
    // see this panel; audio runs see it the moment there's real data.
    // After the auto-reveal, the user's explicit dock toggle wins (so
    // they can turn it off again and it stays off for the rest of the
    // session).
    if (data.mel_spectrogram && !V.autoShowedSpectrogram) {
        V.autoShowedSpectrogram = true;
        V.enableSpectrogram = true;
        const card = document.getElementById('spectrogram-canvas')?.closest('.panel');
        if (card) card.style.display = '';
        const btn = document.getElementById('toggle-spec');
        if (btn) btn.classList.add('active');
    }
    if (V.enableSpectrogram && data.mel_spectrogram) {
        V.updateSpectrogram(data.mel_spectrogram);
    }

    if (data.architecture) {
        V.maybeRebuildGalaxyFromArchitecture(
            data.architecture,
            data.total_params ?? data.architecture.total_params,
            data.trainable_params ?? data.architecture.trainable_params
        );
    }
    
    // Sound effects
    if (V.soundEnabled && data.loss !== undefined) {
        V.playLossSound(data.loss);
    }
}

function initEventListeners() {
    // Pause button — clean lowercase label, no injected icon markup.
    document.getElementById('pause-btn').addEventListener('click', () => {
        V.isPaused = !V.isPaused;
        const btn = document.getElementById('pause-btn');
        btn.textContent = V.isPaused ? 'resume' : 'pause';
        btn.classList.toggle('active', V.isPaused);
    });

    // Sound button — opt-in beep oscillator from V.playLossSound().
    //
    // Labels are imperative so the off switch is findable: "sound" when
    // off (i.e. "click to turn sound on"), "mute" when on (click to
    // stop). A user frantically hunting for the off switch scans for
    // the word that describes the action they want, which is "mute" —
    // not "sound on," which reads as a status and hides the affordance.
    document.getElementById('sound-btn').addEventListener('click', () => {
        V.soundEnabled = !V.soundEnabled;
        applySoundButtonState();
    });

    // Global ESC mute — the fastest possible off switch. If the beeps
    // surprise you, Escape kills them from anywhere on the page without
    // needing to find the dock.
    document.addEventListener('keydown', (ev) => {
        if (ev.key === 'Escape' && V.soundEnabled) {
            V.soundEnabled = false;
            applySoundButtonState();
        }
    });

    // Fullscreen button
    document.getElementById('fullscreen-btn').addEventListener('click', () => {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    });

    // Record button — start or stop the tab recording.
    // The recording module (gemma-viz-recording.js) may be absent (script
    // error, custom embed) so guard with a V.recording existence check.
    const recBtn = document.getElementById('record-btn');
    if (recBtn && V.recording) {
        recBtn.addEventListener('click', () => {
            if (V.recording.isActive()) {
                V.recording.stop();
            } else {
                V.recording.start();
            }
        });
    }
}

/**
 * Sync the sound dock button label + active state to V.soundEnabled.
 *
 * Kept as a small helper because both the click handler and the ESC-key
 * handler need to rewrite the same DOM in the same way — without this
 * the ESC path leaves the button reading "mute" while the state is off.
 */
function applySoundButtonState() {
    const btn = document.getElementById('sound-btn');
    if (!btn) return;
    btn.textContent = V.soundEnabled ? 'mute' : 'sound';
    btn.classList.toggle('active', V.soundEnabled);
}

function updateStats(data) {
    if (data.device) {
        document.getElementById('device-name').textContent = data.device;
    }
    if (data.total_params) {
        const millions = (data.total_params / 1e6).toFixed(1);
        document.getElementById('param-count').textContent = `${millions}M`;
    }
    if (data.architecture) {
        V.maybeRebuildGalaxyFromArchitecture(data.architecture, data.total_params, data.trainable_params);
    }
}

/**
 * Load historical data
 */
function loadHistoricalData(data) {
    if (data.loss_history && data.loss_history.length && window.__vizStatus && window.__vizStatus.markTrainingData) {
        window.__vizStatus.markTrainingData();
    }
    if (data.loss_history && data.loss_history.length) {
        V.hasSeenTrainingLoss = true;
        setHeroVerb(HERO_VERBS.LEARN);
    }
    // Load loss history
    if (data.loss_history && data.loss_history.length) {
        data.loss_history.forEach((loss, i) => {
            V.updateLossChart(loss, i);
        });
        V.updateHeroLoss(data.loss_history[data.loss_history.length - 1]);
    }
    
    // Load gradient history
    if (data.grad_history) {
        data.grad_history.forEach((grad, i) => {
            V.updateGradientChart(grad, i);
        });
    }
    
    // Load learning rate history
    if (data.lr_history) {
        data.lr_history.forEach((lr, i) => {
            V.updateLearningRateChart(lr, i);
        });
    }

    if (data.memory_history && data.memory_history.length) {
        V.updateMemoryChart(data.memory_history[data.memory_history.length - 1]);
    }
}

function animate() {
    V.animationFrameId = requestAnimationFrame(animate);
    
    // Calculate FPS
    const currentTime = Date.now();
    const deltaTime = currentTime - V.lastFrameTime;
    V.frameCount++;
    
    if (V.frameCount % 30 === 0) {
        V.fps = Math.round(1000 / deltaTime);
        document.getElementById('fps-value').textContent = V.fps;
    }
    
    V.lastFrameTime = currentTime;
    
    // Decay the galaxy's emissive level toward 0 every frame, then re-render
    // the scene. The ONLY thing that drives motion in the galaxy now is the
    // training stream itself: every gradient update brightens it via
    // updateNeuralNetworkGradients(), and the decay here pulls it back down
    // between updates. A stalled run goes quiet within a couple of seconds;
    // an active run shimmers continuously. Mouse rotation still works inside
    // window.animateNetwork().
    //
    // Removed in this pass: the constant rotation.y += 0.002, the
    // sin(time)-based breathing pulse on .scale, and the slow core spin.
    // All three were ambient motion uncoupled from training data — the kind
    // of "screensaver" motion that makes a panel feel salient without saying
    // anything. The user explicitly flagged this as distracting.
    if (V.neuralNetwork && !V.isPaused && !V.reducedMotion) {
        V.applyGalaxyEmissive();
    }

    if (window.animateNetwork) {
        window.animateNetwork();
    }
}

V.initSocket = initSocket;
V.handleTrainingUpdate = handleTrainingUpdate;
V.initEventListeners = initEventListeners;
V.updateStats = updateStats;
V.loadHistoricalData = loadHistoricalData;
V.animate = animate;

function boot() {
    console.log('🚀 Initializing Gemma Training Visualizer...');

    V.hasSeenTrainingLoss = false;
    setHeroVerb(HERO_VERBS.READY);

    V.initSocket();
    try {
        V.initCharts();
    } catch (e) {
        console.error('Chart init failed:', e);
    }
    if (V.enable3D) {
        try {
            V.init3DNeuralNetwork();
        } catch (e) {
            console.error('3D visualizer init failed (charts still work):', e);
        }
    } else {
        const card = document.getElementById('neural-network-3d')?.closest('.panel');
        if (card) card.style.display = 'none';
    }
    V.initEventListeners();

    animate();

    setTimeout(() => {
        const loading = document.getElementById('loading');
        if (loading) loading.style.display = 'none';
    }, 1000);

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
        () => V.enable3D,
        (v) => { V.enable3D = v; },
        (on) => {
            const card = document.getElementById('neural-network-3d')?.closest('.panel');
            if (card) card.style.display = on ? '' : 'none';
            if (on && !V.renderer) {
                try {
                    V.init3DNeuralNetwork();
                } catch (e) {
                    console.error('3D visualizer init failed:', e);
                }
            }
        }
    );
    toggle(
        'toggle-attn',
        () => V.enableAttention,
        (v) => { V.enableAttention = v; },
        (on) => {
            const card = document.getElementById('attention-canvas')?.closest('.panel');
            if (card) card.style.display = on ? '' : 'none';
        }
    );
    toggle(
        'toggle-tokens',
        () => V.enableTokens,
        (v) => { V.enableTokens = v; },
        (on) => {
            const sec = document.getElementById('token-cloud')?.closest('.saying');
            if (sec) sec.style.display = on ? '' : 'none';
        }
    );
    toggle(
        'toggle-spec',
        () => V.enableSpectrogram,
        (v) => { V.enableSpectrogram = v; },
        (on) => {
            const card = document.getElementById('spectrogram-canvas')?.closest('.panel');
            if (card) card.style.display = on ? '' : 'none';
        }
    );
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
} else {
    boot();
}

})(window.GemmaViz);

console.log('✨ Gemma Training Visualizer modules ready');
