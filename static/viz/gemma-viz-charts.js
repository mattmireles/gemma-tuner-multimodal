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

    // Memory chart
    const memCtx = document.getElementById('memory-chart').getContext('2d');
    V.memoryChart = new Chart(memCtx, {
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
 * Handle training update from server
 */
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
 * Update gradient chart
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
}

/**
 * Update memory chart
 */
function updateMemoryChart(memoryGB) {
    if (!V.memoryChart) return;
    V.memoryChart.data.datasets[0].data[0] = memoryGB;
    
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

    V.memoryChart.data.datasets[0].backgroundColor = color;
    V.memoryChart.update('none');
}

/**
 * Update learning rate chart
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
}

V.initCharts = initCharts;
V.updateLossChart = updateLossChart;
V.updateGradientChart = updateGradientChart;
V.updateMemoryChart = updateMemoryChart;
V.updateLearningRateChart = updateLearningRateChart;

})(window.GemmaViz);
