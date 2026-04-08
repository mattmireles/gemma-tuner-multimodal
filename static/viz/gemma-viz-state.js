/**
 * Shared dashboard state (namespace: window.GemmaViz).
 * Loaded first; other gemma-viz-*.js modules attach functions.
 */
(function () {
    const urlParams = new URLSearchParams(window.location.search);
    const lightMode = urlParams.get('viz') === 'light';

    window.GemmaViz = {
        socket: null,
        isPaused: false,
        soundEnabled: false,
        animationFrameId: null,

        lightMode: lightMode,
        // The 3D galaxy is opt-in via the dock "model" toggle (or ?show3D=1).
        // Reason: the panel is visually salient but the only training signal
        // it actually receives is the *global* gradient norm, which it then
        // pulses uniformly across every neuron — there is no per-layer signal
        // available, so the galaxy cannot honestly "show" learning happening
        // anywhere specific. Hidden by default; opt in for spectacle.
        enable3D: urlParams.get('show3D') === '1' && !lightMode,
        enableAttention: urlParams.get('showAttention') !== '0' && !lightMode,
        enableTokens: urlParams.get('showTokens') !== '0' && !lightMode,
        // Audio panel: always starts off, regardless of URL params.
        // Auto-revealed the first time a training update carries a
        // mel_spectrogram payload (see handleTrainingUpdate). Text-only
        // runs never see the panel; audio runs see it the moment real
        // spectrogram data starts flowing. The dock "audio" button still
        // lets the user force-show or hide it after the fact.
        //
        // Note: the URL flag (?showSpectrogram=...) is intentionally
        // removed. Earlier versions used it to *disable* a default-on
        // panel; now that the panel is default-off in the HTML, a URL
        // "enable" flag would also need a boot-time DOM unhide to avoid
        // drift between the state var and the panel visibility. Simpler
        // to drop the override entirely and let data drive visibility.
        enableSpectrogram: false,
        autoShowedSpectrogram: false,

        scene: null,
        camera: null,
        renderer: null,
        neuralNetwork: null,
        galaxyNeuronMeshes: [],
        galaxyLastFingerprint: '',
        GALAXY_MAX_NODES: 340,
        GOLDEN_ANGLE: 2.39996322972865332,

        // Galaxy emissive level — written by updateNeuralNetworkGradients on
        // every training step, decayed each frame in animate(). The result is
        // that the galaxy brightens on each step and fades between them, so
        // its visible rhythm literally is the training step rate. A stalled
        // run quiets down; an active run shimmers.
        galaxyEmissive: 0,

        lossChart: null,
        gradientChart: null,
        memoryChart: null,
        lrChart: null,

        lastFrameTime: Date.now(),
        frameCount: 0,
        fps: 60,

        lastHeroLoss: null,

        reducedMotion:
            typeof window !== 'undefined' &&
            typeof window.matchMedia === 'function' &&
            window.matchMedia('(prefers-reduced-motion: reduce)').matches,

        audioContext: null,

        formatHeroLoss: function (n) {
            if (n === null || n === undefined || Number.isNaN(n)) return '—';
            if (n >= 100) return n.toFixed(1);
            if (n >= 10) return n.toFixed(2);
            return n.toFixed(3);
        },

        updateHeroLoss: function (n) {
            const V = window.GemmaViz;
            const el = document.getElementById('loss-value');
            if (!el) return;
            if (n === null || n === undefined || Number.isNaN(n)) return;
            el.textContent = V.formatHeroLoss(n);
            if (V.lastHeroLoss !== null && n > V.lastHeroLoss * 1.02) {
                el.classList.add('is-rising');
                setTimeout(function () {
                    el.classList.remove('is-rising');
                }, 600);
            }
            V.lastHeroLoss = n;
        },
    };
})();
