/**
 * Tab recording → shareable recap (window.GemmaViz).
 *
 * Phase 1 — Capture: getDisplayMedia() tab stream → MediaRecorder chunks.
 * Phase 2 — Encode: seek through the raw blob frame-by-frame and produce two
 *   outputs in a single decode pass:
 *     • a 60 s WebM (1800 frames @ 30 fps) — primary artifact
 *     • a 30 s GIF  (300 frames  @ 10 fps, max 720 px wide) — sidecar for
 *       surfaces that don't embed video (GitHub README, Slack, email)
 *
 * Vendored deps: static/vendor/gifenc/gifenc.global.js (gifenc 1.0.3 from
 * https://unpkg.com/gifenc@1.0.3/dist/gifenc.js, IIFE-wrapped).
 *
 * No HTML required in index.html — the processing modal is built lazily here.
 */
(function (V) {

// ── tunables (module scope so the modal counter can read WEBM_FRAMES) ───────
var WEBM_FRAMES = 1800;  // 60 s × 30 fps — output WebM length
var GIF_STRIDE  = 6;     // every Nth WebM frame → 1800/6 = 300 GIF frames
var GIF_DELAY   = 100;   // ms per GIF frame → 10 fps
var GIF_MAX_W   = 720;   // GIF downscale ceiling for shareable file size

// ── internal state ──────────────────────────────────────────────────────────
var _mediaStream = null;  // MediaStream from getDisplayMedia
var _recorder    = null;  // MediaRecorder instance
var _chunks      = [];    // Blob array accumulating raw frames
var _cachedMimeType = null;  // Lazy result of pickMimeType()

// ── modal DOM refs (lazily populated) ──────────────────────────────────────
var _progressBar = null;
var _frameCount  = null;

// ── codec selection ─────────────────────────────────────────────────────────
// VP9 gives the best compression for a training visualization. Cascade to
// VP8 and then browser-default for Firefox/Safari compatibility. Safari
// MediaRecorder produces MP4/H.264 regardless of mimeType; the cascade
// stops before an error and lets the browser pick its own container.
//
// Cached: this is browser-static, so the first call is the only call.
function pickMimeType() {
    if (_cachedMimeType !== null) return _cachedMimeType;
    var candidates = [
        'video/webm;codecs=vp9',
        'video/webm;codecs=vp8',
        'video/webm'
    ];
    for (var i = 0; i < candidates.length; i++) {
        if (MediaRecorder.isTypeSupported(candidates[i])) {
            _cachedMimeType = candidates[i];
            return _cachedMimeType;
        }
    }
    _cachedMimeType = '';
    return _cachedMimeType;
}

// ── _awaitEvent: bounded promise wrapper around a media event ───────────────
// Every `await new Promise((r) => target.onevent = r)` in this file is a
// hang waiting to happen — if `error` fires instead, or the event simply
// never comes (codec quirk, OOM, corrupt blob), the modal sticks forever
// and the button is permanently `processing`. This helper races the success
// event against an `error` listener and a timeout watchdog so every await
// either succeeds, rejects with an error, or rejects with a timeout — never
// hangs. Always cleans up its listeners and timer to avoid leaks.
function _awaitEvent(target, eventName, timeoutMs) {
    return new Promise(function (resolve, reject) {
        var done = false;
        // `timer` is declared with `var` so the hoisted binding is visible
        // inside cleanup() — necessary because cleanup is defined before the
        // setTimeout call assigns it. cleanup() only ever runs after the
        // timer is assigned (via either the success listener, the error
        // listener, or the timer itself), so the read is always defined.
        var timer;
        var cleanup = function () {
            target.removeEventListener(eventName, onEvent);
            target.removeEventListener('error', onError);
            if (timer) clearTimeout(timer);
        };
        var onEvent = function () {
            if (done) return;
            done = true;
            cleanup();
            resolve();
        };
        var onError = function (e) {
            if (done) return;
            done = true;
            cleanup();
            // Pull the underlying media error if the target exposes one;
            // otherwise fall back to the event itself for debug context.
            var detail = (target && target.error && target.error.message)
                || (e && e.message)
                || 'unknown';
            reject(new Error('media error during ' + eventName + ': ' + detail));
        };
        target.addEventListener(eventName, onEvent);
        target.addEventListener('error', onError);
        timer = setTimeout(function () {
            if (done) return;
            done = true;
            cleanup();
            reject(new Error(eventName + ' did not fire within ' + timeoutMs + 'ms'));
        }, timeoutMs);
    });
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

    // Build the MediaRecorder before wiring listeners so a constructor
    // failure can't strand the stream + beforeunload handler. pickMimeType
    // already filtered through isTypeSupported, so this should not throw,
    // but a malicious browser extension could mess with the constructor —
    // catch defensively and tear down the stream if it does.
    var mimeType = pickMimeType();
    var recorder;
    try {
        recorder = mimeType
            ? new MediaRecorder(stream, { mimeType: mimeType })
            : new MediaRecorder(stream);
    } catch (e) {
        console.error('[recording] MediaRecorder unavailable:', e);
        stream.getTracks().forEach(function (t) { t.stop(); });
        _setButtonState('idle');
        return;
    }

    _mediaStream = stream;
    _recorder    = recorder;

    // If the user clicks the browser's native "Stop sharing" bar, treat it
    // the same as pressing our Stop button.
    stream.getVideoTracks()[0].addEventListener('ended', function () {
        if (_recorder && _recorder.state === 'recording') stopAndProcess();
    });

    // Warn before navigating away with an active recording.
    window.addEventListener('beforeunload', _warnBeforeUnload);

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

    // Show the modal BEFORE awaiting the recorder drain, so the user sees
    // immediate feedback that work has begun. The drain itself can take
    // 100-500ms; without this, the button just goes grey for that window.
    _showModal();

    // Stop the recorder and collect the final chunk. Bounded by _awaitEvent
    // so a recorder that fails to fire `stop` doesn't strand the modal.
    var rawBlob;
    try {
        var stopPromise = _awaitEvent(_recorder, 'stop', 10000);
        _recorder.stop();
        await stopPromise;
        rawBlob = new Blob(_chunks, { type: 'video/webm' });
    } catch (e) {
        console.error('[recording] recorder stop failed:', e);
        _hideModal();
        V.isRecording = false;
        _setButtonState('idle');
        if (_mediaStream) {
            _mediaStream.getTracks().forEach(function (t) { t.stop(); });
            _mediaStream = null;
        }
        return;
    }

    // Release the tab-sharing indicator immediately.
    if (_mediaStream) {
        _mediaStream.getTracks().forEach(function (t) { t.stop(); });
        _mediaStream = null;
    }

    var outputs;
    try {
        outputs = await _encodeOutputs(rawBlob);
    } catch (e) {
        console.error('[recording] encode failed:', e);
        _hideModal();
        V.isRecording = false;
        _setButtonState('idle');
        return;
    }

    _hideModal();
    V.isRecording = false;
    // Download WebM first (primary artifact), then GIF (sidecar) with a
    // brief delay so both `a.click()` calls are honored by the browser.
    // The size guard rejects empty blobs (gifenc finishes with no frames).
    _triggerDownload(outputs.webm, 'webm');
    if (outputs.gif && outputs.gif.size > 0) {
        setTimeout(function () {
            _triggerDownload(outputs.gif, 'gif');
        }, 500);
    }
    _setButtonState('idle');
}

// ── resolve real video duration ──────────────────────────────────────────────
// MediaRecorder-produced WebM blobs in Chrome routinely lack a Duration
// element in the EBML header. When you load such a blob into a <video>,
// `video.duration` returns Infinity until the browser has actually scanned
// to the end of the stream. The standard workaround: seek to a huge value,
// wait for `seeked` to fire (which forces the scan), then read duration —
// `video.duration` is now the real value. We seek back to 0 before returning
// so the caller starts from a known position.
//
// Both seeks use _awaitEvent so a corrupt blob or a browser that simply
// fails to fire `seeked` rejects with a timeout instead of hanging the UI.
async function _resolveDuration(video) {
    if (Number.isFinite(video.duration) && video.duration > 0) {
        return video.duration;
    }
    var seekedPromise = _awaitEvent(video, 'seeked', 10000);
    // Number.MAX_SAFE_INTEGER is the documented sentinel for this trick;
    // the browser clamps to the real end and fires `seeked`.
    video.currentTime = Number.MAX_SAFE_INTEGER;
    await seekedPromise;
    var duration = video.duration;
    // Seek back to the start so the caller starts from frame 0.
    var rewindPromise = _awaitEvent(video, 'seeked', 10000);
    video.currentTime = 0;
    await rewindPromise;
    return duration;
}

// ── seek-and-capture: one pass, two outputs ──────────────────────────────────
// Loads the raw blob into a hidden <video>, seeks to WEBM_FRAMES evenly-spaced
// timestamps, draws each frame to a canvas, and pushes to a new MediaRecorder
// for the WebM output. Every GIF_STRIDE-th frame, a downscaled copy is also
// quantized and pushed to gifenc for the GIF sidecar.
// Doing both in one seek loop avoids paying the video decode cost twice.
// Result: { webm, gif } — exactly 60 s WebM, 30 s GIF, from the same run.
//
// Every async wait uses _awaitEvent so the encode either completes, throws,
// or times out — never hangs. Inner try/finally guarantees the blob URL,
// inner MediaRecorder, and capture-stream track are all released even if
// the seek loop throws partway through.
async function _encodeOutputs(srcBlob) {
    var video = document.createElement('video');
    var videoUrl = URL.createObjectURL(srcBlob);
    video.src   = videoUrl;
    video.muted = true;

    var mr = null;          // inner MediaRecorder for the speedup output
    var outStream = null;   // canvas captureStream — needs explicit teardown
    try {
        await _awaitEvent(video, 'loadedmetadata', 10000);

        // Resolve the real duration (works around Chrome's Infinity bug).
        var duration = await _resolveDuration(video);
        if (!Number.isFinite(duration) || duration <= 0) {
            throw new Error('recording has no valid duration: ' + duration);
        }
        var dt = duration / WEBM_FRAMES;

        // WebM canvas + MediaRecorder (full resolution).
        var canvas   = document.createElement('canvas');
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        var ctx = canvas.getContext('2d');

        // captureStream(0) = manual frame push via track.requestFrame(); gives
        // precise control over the output frame rate during the seek loop.
        outStream = canvas.captureStream(0);

        var outChunks = [];
        var mimeType  = pickMimeType();
        // pickMimeType returns either a supported MIME or '' (browser default).
        // No need for a separate fallback constructor — the empty string and
        // an absent options arg are equivalent to the spec.
        mr = mimeType
            ? new MediaRecorder(outStream, { mimeType: mimeType })
            : new MediaRecorder(outStream);
        mr.ondataavailable = function (e) {
            if (e.data && e.data.size > 0) outChunks.push(e.data);
        };
        mr.start();

        // GIF setup — gracefully skip if gifenc is not loaded (script missing,
        // older browser, CSP blocking, etc.). The WebM still ships.
        var gifEnabled = typeof window.gifenc === 'object' &&
                         typeof window.gifenc.GIFEncoder === 'function';
        var gif, gifCanvas, gifCtx, gifWidth, gifHeight, quantize, applyPalette;
        if (gifEnabled) {
            quantize     = window.gifenc.quantize;
            applyPalette = window.gifenc.applyPalette;
            gif          = window.gifenc.GIFEncoder();
            gifWidth  = Math.min(video.videoWidth, GIF_MAX_W);
            gifHeight = Math.round(video.videoHeight * (gifWidth / video.videoWidth));
            gifCanvas = document.createElement('canvas');
            gifCanvas.width  = gifWidth;
            gifCanvas.height = gifHeight;
            gifCtx = gifCanvas.getContext('2d');
        }

        // Frozen GIF palette — built once from a representative mid-run frame
        // (the first GIF frame the loop encodes, see below) and reused for
        // every subsequent frame. Quantizing per-frame would (a) make the
        // colors flicker as adjacent frames pick subtly different palette
        // indices, and (b) burn ~10× the encode time on the slowest call in
        // the loop. The training visualization is a low-color-diversity
        // amber-on-black scene, so a single palette holds up well across
        // the whole recap.
        var gifPalette = null;

        // The first iteration is special: the video is already at currentTime
        // = 0 from _resolveDuration, so setting currentTime to 0 again may
        // not fire `seeked` in all browsers (the spec only says it "may").
        // Skip the seek for i === 0; subsequent iterations always seek to a
        // different timestamp.
        for (var i = 0; i < WEBM_FRAMES; i++) {
            if (i > 0) {
                var seekPromise = _awaitEvent(video, 'seeked', 10000);
                video.currentTime = i * dt;
                await seekPromise;
            }

            // WebM frame (every iteration).
            ctx.drawImage(video, 0, 0);
            outStream.getVideoTracks()[0].requestFrame();

            // GIF frame (every GIF_STRIDE iterations). The palette is built
            // from the first GIF frame and reused for the rest.
            if (gifEnabled && i % GIF_STRIDE === 0) {
                gifCtx.drawImage(video, 0, 0, gifWidth, gifHeight);
                var data = gifCtx.getImageData(0, 0, gifWidth, gifHeight).data;
                if (gifPalette === null) {
                    gifPalette = quantize(data, 256, { format: 'rgb444' });
                }
                var indexed = applyPalette(data, gifPalette, 'rgb444');
                gif.writeFrame(indexed, gifWidth, gifHeight, { palette: gifPalette, delay: GIF_DELAY });
            }

            // Update progress modal — reflects seek progress through the loop.
            if (_progressBar) {
                _progressBar.style.width = (((i + 1) / WEBM_FRAMES) * 100).toFixed(1) + '%';
            }
            if (_frameCount) {
                _frameCount.textContent = (i + 1) + ' / ' + WEBM_FRAMES;
            }

            // Yield every 60 frames so the browser stays responsive and doesn't
            // show an "unresponsive page" warning during the long seek loop.
            if (i % 60 === 59) {
                await new Promise(function (r) { setTimeout(r, 0); });
            }
        }

        var stopPromise = _awaitEvent(mr, 'stop', 10000);
        mr.stop();
        await stopPromise;
        mr = null;  // signal to finally: cleanup already complete

        var webmBlob = new Blob(outChunks, { type: 'video/webm' });
        var gifBlob  = null;
        if (gifEnabled) {
            gif.finish();
            gifBlob = new Blob([gif.bytes()], { type: 'image/gif' });
        }
        return { webm: webmBlob, gif: gifBlob };
    } finally {
        // Stop the inner MediaRecorder if the loop threw before reaching the
        // normal stop path. We can't await `mr.stop()` here (finally is sync),
        // so we just call stop() and let the recorder garbage-collect — its
        // chunks are already lost on a failure path.
        if (mr && mr.state !== 'inactive') {
            try { mr.stop(); } catch (e) { /* nothing to do */ }
        }
        // Stop the captureStream track so the canvas can be released.
        if (outStream) {
            outStream.getTracks().forEach(function (t) { t.stop(); });
        }
        URL.revokeObjectURL(videoUrl);
        video.removeAttribute('src');
        video.load();
    }
}

// ── download trigger ──────────────────────────────────────────────────────────
function _triggerDownload(blob, ext) {
    var now = new Date();
    var pad = function (n) { return String(n).padStart(2, '0'); };
    var date = now.getFullYear() + '-' + pad(now.getMonth() + 1) + '-' + pad(now.getDate());
    var fname = 'gemma-training-' + date + '.' + (ext || 'webm');

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
//
// All colors and fonts come from the CSS variables defined on :root in
// index.html (--amber, --ink, --ink-2, --ink-3, --sans, --mono). The modal
// is appended to <body> so var() resolution finds them. Don't inline the
// hex values — that creates silent drift if the palette ever changes.
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
        'font-family:var(--sans);' +
        'font-size:13px;letter-spacing:0.18em;text-transform:lowercase;' +
        'color:var(--ink);';
    label.textContent = 'preparing your recap';

    var track = document.createElement('div');
    track.style.cssText =
        'width:240px;height:1px;background:var(--ink-3);position:relative;';

    var fill = document.createElement('div');
    fill.style.cssText =
        'position:absolute;left:0;top:0;height:100%;width:0%;' +
        'background:var(--amber);transition:width 80ms linear;';
    track.appendChild(fill);
    _progressBar = fill;

    var counter = document.createElement('div');
    counter.style.cssText =
        'font-family:var(--mono);' +
        'font-size:11px;letter-spacing:0.12em;color:var(--ink-2);' +
        'font-variant-numeric:tabular-nums;';
    counter.textContent = '0 / ' + WEBM_FRAMES;
    _frameCount = counter;

    inner.appendChild(label);
    inner.appendChild(track);
    inner.appendChild(counter);
    el.appendChild(inner);
    document.body.appendChild(el);
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
