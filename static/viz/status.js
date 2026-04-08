/**
 * Connection UX: waiting + stall hints. Loaded before gemma-viz-*.js.
 */
(function () {
    var stallMs = 90000;
    var pollMs = 12000;
    var lastTrainingAt = 0;
    var connectAt = 0;
    var connected = false;

    function ensureBanner() {
        var el = document.getElementById('viz-status-banner');
        if (el) return el;
        el = document.createElement('div');
        el.id = 'viz-status-banner';
        el.setAttribute('role', 'status');
        el.style.cssText =
            'display:none;position:fixed;bottom:0;left:0;right:0;padding:10px 18px;' +
            'font:12px/1.4 ui-monospace,Menlo,monospace;color:#F5F5F0;' +
            'border-top:1px solid #3A3A38;z-index:99999;text-align:center';
        document.body.appendChild(el);
        return el;
    }

    function styleBanner(kind) {
        var el = ensureBanner();
        if (kind === 'warn') {
            el.style.background = 'rgba(255, 176, 0, 0.22)';
        } else if (kind === 'error') {
            el.style.background = 'rgba(255, 77, 109, 0.92)';
        } else {
            el.style.background = 'rgba(255, 176, 0, 0.15)';
        }
    }

    function show(msg, kind) {
        var el = ensureBanner();
        styleBanner(kind || 'wait');
        el.textContent = msg;
        el.style.display = 'block';
    }

    function hide() {
        var el = document.getElementById('viz-status-banner');
        if (el) el.style.display = 'none';
    }

    window.__vizStatus = {
        markTrainingData: function () {
            lastTrainingAt = Date.now();
            hide();
        },
        setConnected: function (c) {
            connected = !!c;
            if (connected) {
                connectAt = Date.now();
                show(
                    'Connected — waiting for training metrics (first point follows your logging_steps).',
                    'wait'
                );
            }
        },
        setDisconnected: function () {
            connected = false;
            show('Disconnected from training server — refresh or check the terminal.', 'error');
        }
    };

    setInterval(function () {
        if (!connected) return;
        var ref = lastTrainingAt || connectAt;
        if (!ref) return;
        if (Date.now() - ref > stallMs) {
            show(
                'No training metrics for ' +
                    Math.round(stallMs / 1000) +
                    's. Confirm training is running and logging_steps is not huge.',
                'warn'
            );
        }
    }, pollMs);

    fetch('/healthz')
        .then(function (r) {
            return r.ok ? r.json() : null;
        })
        .then(function (j) {
            if (j && j.viz) window.__vizHealth = j;
        })
        .catch(function () {});
})();
