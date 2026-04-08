/**
 * Three.js galaxy scene + gradient pulse (window.GemmaViz).
 */
(function (V) {
function init3DNeuralNetwork() {
    const container = document.getElementById('neural-network-3d');
    const width = container.clientWidth;
    const height = container.clientHeight || 150; // Use container's actual height
    
    // Create scene
    V.scene = new THREE.Scene();
    V.scene.fog = new THREE.Fog(0x000000, 1, 100);
    
    // Create camera — adjusted for smaller viewport
    V.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    V.camera.position.z = 20;
    
    // Create renderer
    V.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    V.renderer.setSize(width, height);
    V.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(V.renderer.domElement);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    V.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1);
    V.scene.add(directionalLight);
    
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
        if (V.neuralNetwork) {
            V.neuralNetwork.rotation.y = mouseX * 0.5;
            V.neuralNetwork.rotation.x = mouseY * 0.3;
        }
    }
    
    // Add to animation loop
    const animateNetwork = () => {
        updateCameraPosition();
        V.renderer.render(V.scene, V.camera);
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
    V.galaxyNeuronMeshes = [];
    if (!V.scene || !V.neuralNetwork) return;
    V.scene.remove(V.neuralNetwork);
    V.neuralNetwork.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
            else obj.material.dispose();
        }
    });
    V.neuralNetwork = null;
}

/**
 * Build a generic barred-spiral "galaxy" from HF architecture fields (any Gemma / modality).
 * @param {object|null} arch — from ``initial_state.architecture`` or training payload
 */
function createNeuralNetworkMesh(arch) {
    if (!V.scene) return;

    const container = document.getElementById('neural-network-3d');
    disposeGalaxyContents();

    const merged = mergeArchDefaults(arch);
    const rng = _mulberry32(_hashString(JSON.stringify(merged)) || 0xcafebabe);

    const ringCount = estimateRingCount(merged);
    const hidden = Math.max(256, merged.hidden_size);
    let nodesPerRing = Math.round(Math.sqrt(hidden / 96) * 3.5);
    nodesPerRing = Math.max(6, Math.min(22, nodesPerRing));
    let totalNodes = ringCount * nodesPerRing;
    if (totalNodes > V.GALAXY_MAX_NODES) {
        nodesPerRing = Math.max(4, Math.floor(V.GALAXY_MAX_NODES / ringCount));
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
                ring * V.GOLDEN_ANGLE +
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
        V.galaxyNeuronMeshes.push(mesh);
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

    V.neuralNetwork = group;
    V.scene.add(V.neuralNetwork);

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
    if (!V.enable3D || !V.scene || !arch) return;
    const tp = totalParams ?? arch.total_params ?? 0;
    const trp = trainableParams ?? arch.trainable_params ?? 0;
    const fp = galaxyFingerprint(arch, tp, trp);
    if (fp === V.galaxyLastFingerprint) return;
    V.galaxyLastFingerprint = fp;
    const merged = mergeArchDefaults(arch);
    merged.total_params = tp || merged.total_params;
    merged.trainable_params = trp || merged.trainable_params;
    createNeuralNetworkMesh(merged);
}


function updateNeuralNetworkGradients(gradNorm) {
    const intensity = Math.min(gradNorm / 10, 1);
    const baseEmissive = 0.35;
    V.galaxyNeuronMeshes.forEach((neuron) => {
        if (neuron.material) {
            neuron.material.emissiveIntensity = baseEmissive + intensity * 0.95;
            const scale = 1 + intensity * 0.22;
            neuron.scale.set(scale, scale, scale);
        }
    });
    if (V.neuralNetwork) {
        V.neuralNetwork.traverse((obj) => {
            if (obj.userData && obj.userData.isCore && obj.material) {
                obj.material.emissiveIntensity = 0.75 + intensity * 0.55;
            }
        });
    }
}

V.init3DNeuralNetwork = init3DNeuralNetwork;
V.maybeRebuildGalaxyFromArchitecture = maybeRebuildGalaxyFromArchitecture;
V.updateNeuralNetworkGradients = updateNeuralNetworkGradients;

})(window.GemmaViz);
