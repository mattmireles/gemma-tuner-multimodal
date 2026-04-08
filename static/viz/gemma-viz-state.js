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
        enable3D: urlParams.get('show3D') !== '0' && !lightMode,
        enableAttention: urlParams.get('showAttention') !== '0' && !lightMode,
        enableTokens: urlParams.get('showTokens') !== '0' && !lightMode,
        enableSpectrogram: urlParams.get('showSpectrogram') !== '0' && !lightMode,

        scene: null,
        camera: null,
        renderer: null,
        neuralNetwork: null,
        galaxyNeuronMeshes: [],
        galaxyLastFingerprint: '',
        GALAXY_MAX_NODES: 340,
        GOLDEN_ANGLE: 2.39996322972865332,

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
