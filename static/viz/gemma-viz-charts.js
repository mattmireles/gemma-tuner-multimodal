/**
 * Chart.js loss / grad / memory / lr (window.GemmaViz).
 */
(function (V) {
function initCharts() {
    // Loss chart
    const lossCtx = document.getElementById('loss-chart').getContext('2d');
    V.lossChart = new Chart(lossCtx, {
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
    V.gradientChart = new Chart(gradCtx, {
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

    // Memory chart — a line sparkline of memory_gb over steps. Was a
    // horizontal bar against a hardcoded 16 GB max, which meant on a 64 GB
    // machine the bar barely moved and the user couldn't tell anything was
    // changing. A sparkline makes the trajectory (growing, stable, leaking)
    // legible, and the numeric value above the chart gives the absolute
    // reading.
    //
    // CRITICAL: `beginAtZero: true` on the Y axis. Without it, Chart.js
    // auto-fits the Y range to [min, max] of the sliding window — and
    // because GPU/MPS memory is typically stable with nanogram-of-a-GB
    // float noise, a flat line ends up *normalized* to full-height swings
    // across the panel. The graph reads as "changing wildly" while the
    // rounded number next to it reads as static. Anchoring the Y axis at
    // zero means a 0.001 GB wiggle is a fraction of a pixel and a stable
    // run looks stable. This is the single correct reading.
    const memCtx = document.getElementById('memory-chart').getContext('2d');
    V.memoryChart = new Chart(memCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'memory',
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
                y: {
                    display: false,
                    grid: { display: false },
                    ticks: { display: false },
                    beginAtZero: true
                }
            },
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
        }
    });

    // Learning rate chart
    const lrCtx = document.getElementById('lr-chart').getContext('2d');
    V.lrChart = new Chart(lrCtx, {
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
 * Update loss chart
 */
function updateLossChart(loss, step) {
    if (!V.lossChart) return;
    const maxPoints = 100;
    
    V.lossChart.data.labels.push(step || V.lossChart.data.labels.length);
    V.lossChart.data.datasets[0].data.push(loss);
    
    // Keep only last N points
    if (V.lossChart.data.labels.length > maxPoints) {
        V.lossChart.data.labels.shift();
        V.lossChart.data.datasets[0].data.shift();
    }
    
    V.lossChart.update('none');
}

/**
 * Format a gradient norm as a short tabular-figure string.
 *
 * The gradient norm is dimensionless and typically lives in the 0.1..10
 * range for a healthy run. We print 3 significant figures at small values
 * and 2 decimal places at larger ones, so a first-time user sees a stable
 * number that actually moves when training moves.
 */
function formatGradientNorm(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return '—';
    if (n >= 100) return n.toFixed(0);
    if (n >= 10) return n.toFixed(1);
    return n.toFixed(2);
}

/**
 * Update gradient chart + headline value.
 */
function updateGradientChart(gradNorm, step) {
    if (!V.gradientChart) return;
    const maxPoints = 100;

    V.gradientChart.data.labels.push(step || V.gradientChart.data.labels.length);
    V.gradientChart.data.datasets[0].data.push(gradNorm);

    // Keep only last N points
    if (V.gradientChart.data.labels.length > maxPoints) {
        V.gradientChart.data.labels.shift();
        V.gradientChart.data.datasets[0].data.shift();
    }

    V.gradientChart.update('none');

    // Write the headline number above the sparkline. No unit — gradient
    // norm is dimensionless.
    writeDxValue('gradient-value', formatGradientNorm(gradNorm), '');
}

/**
 * Format a memory reading in GB, tabular-figure-ready.
 *
 * Below 10 GB we print one decimal (5.3 GB); above, we drop the decimal
 * to keep the number width stable (12 GB, not 12.0 GB). The unit lives
 * in a separate <span class="unit"> so CSS can color it tertiary.
 */
function formatMemoryGB(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return '—';
    if (n >= 100) return n.toFixed(0);
    if (n >= 10) return n.toFixed(1);
    return n.toFixed(2);
}

/**
 * Update memory sparkline + headline value.
 */
function updateMemoryChart(memoryGB) {
    if (!V.memoryChart) return;

    const maxPoints = 100;
    const ds = V.memoryChart.data.datasets[0];
    V.memoryChart.data.labels.push(V.memoryChart.data.labels.length);
    ds.data.push(memoryGB);
    if (V.memoryChart.data.labels.length > maxPoints) {
        V.memoryChart.data.labels.shift();
        ds.data.shift();
    }

    // The escalation rule used to assume a 16 GB ceiling; without that
    // ceiling we infer one from the peak ever seen. If the current reading
    // is within 5% of the largest so far AND we've seen at least ~1 GB of
    // growth, we treat it as "near the cap" and show the rose warning.
    // Otherwise the sparkline is quiet amber.
    const peak = Math.max(...ds.data);
    const nearCap = peak > 1 && memoryGB > peak * 0.95;
    ds.borderColor = nearCap ? '#FF4D6D' : '#FFB000';
    ds.backgroundColor = nearCap
        ? 'rgba(255, 77, 109, 0.10)'
        : 'rgba(255, 176, 0, 0.06)';

    V.memoryChart.update('none');

    // Write the headline number above the sparkline. We mutate a stable
    // text node (cached on first call) instead of rebuilding the whole
    // subtree every update — avoids recreating the <span class="unit">
    // five times a second and keeps the DOM quiet.
    writeDxValue('memory-value', formatMemoryGB(memoryGB), 'gb');
}

/**
 * Write a headline number into a .dx-value element while preserving the
 * static " gb" / "—" unit span that lives inside it.
 *
 * Two jobs: (1) the element's first child becomes a text node we own and
 * mutate in place; (2) the unit span stays untouched across updates, so
 * Chart.js never sees the parent's box change dimensions mid-frame.
 *
 * Passing an empty `unitText` skips the unit entirely (for unitless values
 * like gradient norm).
 */
function writeDxValue(elId, text, unitText) {
    const el = document.getElementById(elId);
    if (!el) return;
    // First call: wipe the placeholder ("— gb") and install a stable text
    // node + an optional unit span. Subsequent calls just update the text
    // node's data — no DOM churn.
    if (!el.__vizTextNode) {
        el.textContent = '';
        const tn = document.createTextNode(text);
        el.appendChild(tn);
        el.__vizTextNode = tn;
        if (unitText) {
            const span = document.createElement('span');
            span.className = 'unit';
            span.textContent = ' ' + unitText;
            el.appendChild(span);
        }
    } else {
        el.__vizTextNode.data = text;
    }
}

/**
 * Format a learning rate for the step-size headline. LR values in fine-
 * tuning typically live in the 1e-5..1e-3 range, where a naive .toFixed()
 * would print noise. We format the mantissa to one decimal and exponent
 * in plain digits: 5.0e-5, 2.3e-4, 1.0e-3. No superscript (would demand
 * a richer DOM write) and no Unicode minus (stays searchable).
 */
function formatLearningRate(lr) {
    if (lr === null || lr === undefined || Number.isNaN(lr)) return '—';
    if (lr === 0) return '0';
    if (lr >= 0.01) return lr.toFixed(3);
    // Use toExponential and normalize the shape: "5.0e-5", not "5.000e-5".
    const parts = lr.toExponential(1).split('e');
    const mantissa = parts[0];
    // exponent may be "-5" or "+05"; strip the plus and leading zero.
    let exp = parts[1].replace('+', '');
    exp = exp.replace(/^(-?)0+(?=\d)/, '$1');
    return mantissa + 'e' + exp;
}

/**
 * Update learning rate sparkline + headline value.
 */
function updateLearningRateChart(lr, step) {
    if (!V.lrChart) return;
    const maxPoints = 100;

    V.lrChart.data.labels.push(step || V.lrChart.data.labels.length);
    V.lrChart.data.datasets[0].data.push(lr);

    // Keep only last N points
    if (V.lrChart.data.labels.length > maxPoints) {
        V.lrChart.data.labels.shift();
        V.lrChart.data.datasets[0].data.shift();
    }

    V.lrChart.update('none');

    // Write the headline number above the sparkline.
    writeDxValue('lr-value', formatLearningRate(lr), '');
}

V.initCharts = initCharts;
V.updateLossChart = updateLossChart;
V.updateGradientChart = updateGradientChart;
V.updateMemoryChart = updateMemoryChart;
V.updateLearningRateChart = updateLearningRateChart;

})(window.GemmaViz);
