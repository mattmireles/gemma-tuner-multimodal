/**
 * Tab recording → 60-second speedup download (window.GemmaViz).
 *
 * Phase 1 — Capture: getDisplayMedia() tab stream → MediaRecorder chunks.
 * Phase 2 — Speedup: seek through the raw blob frame-by-frame → re-encode at
 *   1800 frames (30 fps × 60 s) so any training run compresses to one minute.
 *
 * No HTML required in index.html — the processing modal is built lazily here.
 */
(function (V) {

// ── internal state ──────────────────────────────────────────────────────────
var _mediaStream = null;  // MediaStream from getDisplayMedia
var _recorder    = null;  // MediaRecorder instance
var _chunks      = [];    // Blob array accumulating raw frames

// ── modal DOM refs (lazily populated) ──────────────────────────────────────
var _modal       = null;
var _progressBar = null;
var _frameCount  = null;

// ── codec selection ─────────────────────────────────────────────────────────
// VP9 gives the best compression for a training visualization. Cascade to
// VP8 and then browser-default for Firefox/Safari compatibility. Safari
// MediaRecorder produces MP4/H.264 regardless of mimeType; the cascade
// stops before an error and lets the browser pick its own container.
function pickMimeType() {
    var candidates = [
        'video/webm;codecs=vp9',
        'video/webm;codecs=vp8',
        'video/webm'
    ];
    for (var i = 0; i < candidates.length; i++) {
        if (MediaRecorder.isTypeSupported(candidates[i])) return candidates[i];
    }
    return '';
}

// ── start recording ──────────────────────────────────────────────────────────
async function startRecording() {
    if (_recorder && _recorder.state === 'recording') return;
    _chunks = [];

    var stream;
    try {
        stream = await navigator.mediaDevices.getDisplayMedia({
            video: { frameRate: 30 },
            audio: false,
            // Chrome 107+ hints to pre-select the current tab in the picker.
            // Ignored silently on other browsers — still shows picker.
            preferCurrentTab: true
        });
    } catch (e) {
        // User cancelled picker or permission denied — silently reset button.
        _setButtonState('idle');
        return;
    }

    _mediaStream = stream;

    // If the user clicks the browser's native "Stop sharing" bar, treat it
    // the same as pressing our Stop button.
    stream.getVideoTracks()[0].addEventListener('ended', function () {
        if (_recorder && _recorder.state === 'recording') stopAndProcess();
    });

    // Warn before navigating away with an active recording.
    window.addEventListener('beforeunload', _warnBeforeUnload);

    var mimeType = pickMimeType();
    var options  = mimeType ? { mimeType: mimeType } : {};
    try {
        _recorder = new MediaRecorder(stream, options);
    } catch (e) {
        _recorder = new MediaRecorder(stream);
    }

    _recorder.ondataavailable = function (e) {
        if (e.data && e.data.size > 0) _chunks.push(e.data);
    };

    // 1-second timeslices: ondataavailable fires regularly rather than once
    // at stop(), keeping peak memory flat over long training runs.
    _recorder.start(1000);
    V.isRecording = true;
    _setButtonState('recording');
}

// ── stop + process ────────────────────────────────────────────────────────────
async function stopAndProcess() {
    if (!_recorder) return;
    // Guard against double-call (manual stop + training_finished auto-stop).
    if (_recorder.state === 'inactive') return;

    _setButtonState('processing');
    window.removeEventListener('beforeunload', _warnBeforeUnload);

    // Stop the recorder and collect the final chunk.
    var rawBlob = await new Promise(function (resolve) {
        _recorder.onstop = function () {
            resolve(new Blob(_chunks, { type: 'video/webm' }));
        };
        _recorder.stop();
    });

    // Release the tab-sharing indicator immediately.
    if (_mediaStream) {
        _mediaStream.getTracks().forEach(function (t) { t.stop(); });
        _mediaStream = null;
    }

    _showModal();

    var outBlob;
    try {
        outBlob = await _speedUpTo60s(rawBlob);
    } catch (e) {
        console.error('[recording] speedup failed:', e);
        _hideModal();
        V.isRecording = false;
        _setButtonState('idle');
        return;
    }

    _hideModal();
    V.isRecording = false;
    _triggerDownload(outBlob);
    _setButtonState('idle');
}

// ── seek-and-capture speedup ──────────────────────────────────────────────────
// Loads the raw blob into a hidden <video>, seeks to 1800 evenly-spaced
// timestamps, draws each frame to a canvas, and pushes to a new MediaRecorder.
// Result: exactly 60 seconds of output at 30 fps regardless of input duration.
async function _speedUpTo60s(srcBlob) {
    var TARGET_FRAMES = 1800;  // 60 s × 30 fps

    var video = document.createElement('video');
    video.src  = URL.createObjectURL(srcBlob);
    video.muted = true;
    await new Promise(function (r) { video.onloadedmetadata = r; });

    var duration = video.duration;
    var dt       = duration / TARGET_FRAMES;

    var canvas   = document.createElement('canvas');
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    var ctx = canvas.getContext('2d');

    // captureStream(0) = manual frame push via track.requestFrame().
    // This gives precise control over the output frame rate during the seek loop.
    var outStream = canvas.captureStream(0);
    var track     = outStream.getVideoTracks()[0];

    var outChunks = [];
    var mimeType  = pickMimeType();
    var outOptions = mimeType ? { mimeType: mimeType } : {};
    var mr;
    try {
        mr = new MediaRecorder(outStream, outOptions);
    } catch (e) {
        mr = new MediaRecorder(outStream);
    }
    mr.ondataavailable = function (e) {
        if (e.data && e.data.size > 0) outChunks.push(e.data);
    };
    mr.start();

    for (var i = 0; i < TARGET_FRAMES; i++) {
        video.currentTime = i * dt;
        await new Promise(function (r) { video.onseeked = r; });
        ctx.drawImage(video, 0, 0);
        track.requestFrame();

        // Update progress modal.
        if (_progressBar) {
            _progressBar.style.width = (((i + 1) / TARGET_FRAMES) * 100).toFixed(1) + '%';
        }
        if (_frameCount) {
            _frameCount.textContent = (i + 1) + ' / ' + TARGET_FRAMES;
        }

        // Yield every 60 frames so the browser stays responsive and doesn't
        // show an "unresponsive page" warning during the 1800-frame loop.
        if (i % 60 === 59) {
            await new Promise(function (r) { setTimeout(r, 0); });
        }
    }

    mr.stop();
    await new Promise(function (r) { mr.onstop = r; });

    URL.revokeObjectURL(video.src);
    return new Blob(outChunks, { type: 'video/webm' });
}

// ── download trigger ──────────────────────────────────────────────────────────
function _triggerDownload(blob) {
    var now = new Date();
    var pad = function (n) { return String(n).padStart(2, '0'); };
    var date = now.getFullYear() + '-' + pad(now.getMonth() + 1) + '-' + pad(now.getDate());
    var fname = 'gemma-training-' + date + '.webm';

    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Revoke after a delay so the download has time to start.
    setTimeout(function () { URL.revokeObjectURL(url); }, 10000);
}

// ── button state machine ──────────────────────────────────────────────────────
function _setButtonState(state) {
    var btn = document.getElementById('record-btn');
    if (!btn) return;
    if (state === 'idle') {
        btn.textContent = 'record';
        btn.classList.remove('active');
        btn.disabled = false;
    } else if (state === 'recording') {
        btn.textContent = 'stop';
        btn.classList.add('active');
        btn.disabled = false;
    } else if (state === 'processing') {
        btn.textContent = 'record';
        btn.classList.remove('active');
        btn.disabled = true;
    }
}

// ── processing modal ──────────────────────────────────────────────────────────
// Built lazily on first use — keeps index.html free of hidden markup for a
// feature that most page loads will never invoke.
function _showModal() {
    var el = document.getElementById('rec-modal');
    if (el) {
        el.style.display = 'flex';
        return;
    }

    el = document.createElement('div');
    el.id = 'rec-modal';
    el.style.cssText =
        'display:flex;position:fixed;inset:0;z-index:9000;' +
        'align-items:center;justify-content:center;' +
        'background:rgba(0,0,0,0.82);';

    var inner = document.createElement('div');
    inner.style.cssText =
        'display:flex;flex-direction:column;align-items:center;gap:20px;';

    var label = document.createElement('div');
    label.style.cssText =
        'font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display",' +
        '"Helvetica Neue",system-ui,sans-serif;' +
        'font-size:13px;letter-spacing:0.18em;text-transform:lowercase;color:#F5F5F0;';
    label.textContent = 'preparing your recap';

    var track = document.createElement('div');
    track.style.cssText =
        'width:240px;height:1px;background:#3A3A38;position:relative;';

    var fill = document.createElement('div');
    fill.style.cssText =
        'position:absolute;left:0;top:0;height:100%;width:0%;background:#FFB000;' +
        'transition:width 80ms linear;';
    track.appendChild(fill);
    _progressBar = fill;

    var counter = document.createElement('div');
    counter.style.cssText =
        'font-family:ui-monospace,"SF Mono",Menlo,monospace;' +
        'font-size:11px;letter-spacing:0.12em;color:#8A8A85;' +
        'font-variant-numeric:tabular-nums;';
    counter.textContent = '0 / 1800';
    _frameCount = counter;

    inner.appendChild(label);
    inner.appendChild(track);
    inner.appendChild(counter);
    el.appendChild(inner);
    document.body.appendChild(el);
    _modal = el;
}

function _hideModal() {
    var el = document.getElementById('rec-modal');
    if (el) el.style.display = 'none';
    _progressBar = null;
    _frameCount  = null;
}

// ── beforeunload guard ────────────────────────────────────────────────────────
function _warnBeforeUnload(e) {
    if (_recorder && _recorder.state === 'recording') {
        e.preventDefault();
        e.returnValue = '';  // Required for Chrome to show the dialog.
    }
}

// ── public API ────────────────────────────────────────────────────────────────
V.recording = {
    start:    startRecording,
    stop:     stopAndProcess,
    isActive: function () { return !!(_recorder && _recorder.state === 'recording'); }
};

})(window.GemmaViz);
