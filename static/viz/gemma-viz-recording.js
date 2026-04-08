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

// ── internal state ──────────────────────────────────────────────────────────
var _mediaStream = null;  // MediaStream from getDisplayMedia
var _recorder    = null;  // MediaRecorder instance
var _chunks      = [];    // Blob array accumulating raw frames

// ── modal DOM refs (lazily populated) ──────────────────────────────────────
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

    // Build the MediaRecorder before wiring listeners so a constructor
    // failure can't strand the stream + beforeunload handler. If the
    // primary mime fails AND the bare fallback also fails, tear down
    // the freshly-acquired stream and reset the button.
    var mimeType = pickMimeType();
    var options  = mimeType ? { mimeType: mimeType } : {};
    var recorder;
    try {
        try {
            recorder = new MediaRecorder(stream, options);
        } catch (e) {
            recorder = new MediaRecorder(stream);
        }
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
    _triggerDownload(outputs.webm, 'webm');
    if (outputs.gif) {
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
async function _resolveDuration(video) {
    if (Number.isFinite(video.duration) && video.duration > 0) {
        return video.duration;
    }
    await new Promise(function (resolve) {
        var onSeeked = function () {
            video.removeEventListener('seeked', onSeeked);
            resolve();
        };
        video.addEventListener('seeked', onSeeked);
        // Number.MAX_SAFE_INTEGER is the documented sentinel for this trick;
        // the browser clamps to the real end and fires `seeked`.
        video.currentTime = Number.MAX_SAFE_INTEGER;
    });
    var duration = video.duration;
    // Seek back to the start so the caller starts from frame 0.
    await new Promise(function (resolve) {
        var onSeeked = function () {
            video.removeEventListener('seeked', onSeeked);
            resolve();
        };
        video.addEventListener('seeked', onSeeked);
        video.currentTime = 0;
    });
    return duration;
}

// ── seek-and-capture: one pass, two outputs ──────────────────────────────────
// Loads the raw blob into a hidden <video>, seeks to 1800 evenly-spaced
// timestamps, draws each frame to a canvas, and pushes to a new MediaRecorder
// for the WebM output. Every 6th frame, a downscaled copy is also quantized
// and pushed to gifenc for the GIF sidecar — 1800/6 = 300 frames = 30 s @ 10 fps.
// Doing both in one seek loop avoids paying the video decode cost twice.
// Result: { webm, gif } — exactly 60 s WebM, 30 s GIF, from the same run.
async function _encodeOutputs(srcBlob) {
    var WEBM_FRAMES = 1800;   // 60 s × 30 fps
    var GIF_STRIDE  = 6;      // every 6th WebM frame → 300 GIF frames
    var GIF_DELAY   = 100;    // ms per GIF frame → 10 fps
    var GIF_MAX_W   = 720;    // downscale to keep file size shareable

    var video = document.createElement('video');
    var videoUrl = URL.createObjectURL(srcBlob);
    video.src   = videoUrl;
    video.muted = true;

    // try/finally guarantees the blob URL is revoked even if encoding throws
    // partway through — otherwise a corrupt input or a failing GIF frame
    // would leave the URL pinned for the lifetime of the page.
    try {
        await new Promise(function (r) { video.onloadedmetadata = r; });

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

        for (var i = 0; i < WEBM_FRAMES; i++) {
            video.currentTime = i * dt;
            await new Promise(function (r) { video.onseeked = r; });

            // WebM frame (every iteration).
            ctx.drawImage(video, 0, 0);
            track.requestFrame();

            // GIF frame (every GIF_STRIDE iterations). Quantize per-frame with
            // rgb444 — slightly lower color fidelity than rgb565 but much faster,
            // and the amber-on-black palette doesn't stress the format.
            if (gifEnabled && i % GIF_STRIDE === 0) {
                gifCtx.drawImage(video, 0, 0, gifWidth, gifHeight);
                var data    = gifCtx.getImageData(0, 0, gifWidth, gifHeight).data;
                var palette = quantize(data, 256, { format: 'rgb444' });
                var indexed = applyPalette(data, palette, 'rgb444');
                gif.writeFrame(indexed, gifWidth, gifHeight, { palette: palette, delay: GIF_DELAY });
            }

            // Update progress modal — reflects seek progress through the loop.
            if (_progressBar) {
                _progressBar.style.width = (((i + 1) / WEBM_FRAMES) * 100).toFixed(1) + '%';
            }
            if (_frameCount) {
                _frameCount.textContent = (i + 1) + ' / ' + WEBM_FRAMES;
            }

            // Yield every 60 frames so the browser stays responsive and doesn't
            // show an "unresponsive page" warning during the 1800-frame loop.
            if (i % 60 === 59) {
                await new Promise(function (r) { setTimeout(r, 0); });
            }
        }

        mr.stop();
        await new Promise(function (r) { mr.onstop = r; });

        var webmBlob = new Blob(outChunks, { type: 'video/webm' });
        var gifBlob  = null;
        if (gifEnabled) {
            gif.finish();
            gifBlob = new Blob([gif.bytes()], { type: 'image/gif' });
        }
        return { webm: webmBlob, gif: gifBlob };
    } finally {
        URL.revokeObjectURL(videoUrl);
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
    counter.textContent = '0 / 1800';
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
