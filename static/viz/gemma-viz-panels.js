/**
 * Attention, tokens, spectrogram, particles, loss beep (window.GemmaViz).
 */
(function (V) {
function updateAttentionHeatmap(attentionData) {
    const canvas = document.getElementById('attention-canvas');
    const ctx = canvas.getContext('2d');
    
    if (!V.enableAttention || !attentionData || !attentionData.length) return;
    
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
    
    if (!V.enableTokens || !tokenProbs || !tokenProbs.indices) return;
    
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
    
    if (!V.enableSpectrogram || !spectrogramData || !spectrogramData.length) return;
    
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
    if (!V.audioContext) {
        V.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    const oscillator = V.audioContext.createOscillator();
    const gainNode = V.audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(V.audioContext.destination);
    
    // Map loss to frequency (lower loss = higher pitch)
    const frequency = 200 + (1 - Math.min(loss, 1)) * 600;
    oscillator.frequency.setValueAtTime(frequency, V.audioContext.currentTime);
    
    // Quick beep
    gainNode.gain.setValueAtTime(0.1, V.audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, V.audioContext.currentTime + 0.1);
    
    oscillator.start(V.audioContext.currentTime);
    oscillator.stop(V.audioContext.currentTime + 0.1);
}

V.updateAttentionHeatmap = updateAttentionHeatmap;
V.updateTokenCloud = updateTokenCloud;
V.updateSpectrogram = updateSpectrogram;
V.createParticleExplosion = createParticleExplosion;
V.playLossSound = playLossSound;

})(window.GemmaViz);
