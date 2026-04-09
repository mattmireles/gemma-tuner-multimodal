/**
 * Browser global wrapper for gifenc (https://github.com/mattdesl/gifenc).
 * Version 1.0.3. The upstream dist is CommonJS; this shim wraps it in an
 * IIFE with a fake `exports` object and then assigns to window.gifenc.
 */
(function () {
  var exports = {};
var __defProp = Object.defineProperty;
var __markAsModule = (target) => __defProp(target, "__esModule", {value: true});
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, {get: all[name], enumerable: true});
};

// src/index.js
__markAsModule(exports);
__export(exports, {
  GIFEncoder: () => GIFEncoder,
  applyPalette: () => applyPalette,
  default: () => src_default,
  nearestColor: () => nearestColor,
  nearestColorIndex: () => nearestColorIndex,
  nearestColorIndexWithDistance: () => nearestColorIndexWithDistance,
  prequantize: () => prequantize,
  quantize: () => quantize,
  snapColorsToPalette: () => snapColorsToPalette
});

// src/constants.js
var constants_default = {
  signature: "GIF",
  version: "89a",
  trailer: 59,
  extensionIntroducer: 33,
  applicationExtensionLabel: 255,
  graphicControlExtensionLabel: 249,
  imageSeparator: 44,
  signatureSize: 3,
  versionSize: 3,
  globalColorTableFlagMask: 128,
  colorResolutionMask: 112,
  sortFlagMask: 8,
  globalColorTableSizeMask: 7,
  applicationIdentifierSize: 8,
  applicationAuthCodeSize: 3,
  disposalMethodMask: 28,
  userInputFlagMask: 2,
  transparentColorFlagMask: 1,
  localColorTableFlagMask: 128,
  interlaceFlagMask: 64,
  idSortFlagMask: 32,
  localColorTableSizeMask: 7
};

// src/stream.js
function createStream(initialCapacity = 256) {
  let cursor = 0;
  let contents = new Uint8Array(initialCapacity);
  return {
    get buffer() {
      return contents.buffer;
    },
    reset() {
      cursor = 0;
    },
    bytesView() {
      return contents.subarray(0, cursor);
    },
    bytes() {
      return contents.slice(0, cursor);
    },
    writeByte(byte) {
      expand(cursor + 1);
      contents[cursor] = byte;
      cursor++;
    },
    writeBytes(data, offset = 0, byteLength = data.length) {
      expand(cursor + byteLength);
      for (let i = 0; i < byteLength; i++) {
        contents[cursor++] = data[i + offset];
      }
    },
    writeBytesView(data, offset = 0, byteLength = data.byteLength) {
      expand(cursor + byteLength);
      contents.set(data.subarray(offset, offset + byteLength), cursor);
      cursor += byteLength;
    }
  };
  function expand(newCapacity) {
    var prevCapacity = contents.length;
    if (prevCapacity >= newCapacity)
      return;
    var CAPACITY_DOUBLING_MAX = 1024 * 1024;
    newCapacity = Math.max(newCapacity, prevCapacity * (prevCapacity < CAPACITY_DOUBLING_MAX ? 2 : 1.125) >>> 0);
    if (prevCapacity != 0)
      newCapacity = Math.max(newCapacity, 256);
    const oldContents = contents;
    contents = new Uint8Array(newCapacity);
    if (cursor > 0)
      contents.set(oldContents.subarray(0, cursor), 0);
  }
}

// src/lzwEncode.js
var BITS = 12;
var DEFAULT_HSIZE = 5003;
var MASKS = [
  0,
  1,
  3,
  7,
  15,
  31,
  63,
  127,
  255,
  511,
  1023,
  2047,
  4095,
  8191,
  16383,
  32767,
  65535
];
function lzwEncode(width, height, pixels, colorDepth, outStream = createStream(512), accum = new Uint8Array(256), htab = new Int32Array(DEFAULT_HSIZE), codetab = new Int32Array(DEFAULT_HSIZE)) {
  const hsize = htab.length;
  const initCodeSize = Math.max(2, colorDepth);
  accum.fill(0);
  codetab.fill(0);
  htab.fill(-1);
  let cur_accum = 0;
  let cur_bits = 0;
  const init_bits = initCodeSize + 1;
  const g_init_bits = init_bits;
  let clear_flg = false;
  let n_bits = g_init_bits;
  let maxcode = (1 << n_bits) - 1;
  const ClearCode = 1 << init_bits - 1;
  const EOFCode = ClearCode + 1;
  let free_ent = ClearCode + 2;
  let a_count = 0;
  let ent = pixels[0];
  let hshift = 0;
  for (let fcode = hsize; fcode < 65536; fcode *= 2) {
    ++hshift;
  }
  hshift = 8 - hshift;
  outStream.writeByte(initCodeSize);
  output(ClearCode);
  const length = pixels.length;
  for (let idx = 1; idx < length; idx++) {
    next_block: {
      const c = pixels[idx];
      const fcode = (c << BITS) + ent;
      let i = c << hshift ^ ent;
      if (htab[i] === fcode) {
        ent = codetab[i];
        break next_block;
      }
      const disp = i === 0 ? 1 : hsize - i;
      while (htab[i] >= 0) {
        i -= disp;
        if (i < 0)
          i += hsize;
        if (htab[i] === fcode) {
          ent = codetab[i];
          break next_block;
        }
      }
      output(ent);
      ent = c;
      if (free_ent < 1 << BITS) {
        codetab[i] = free_ent++;
        htab[i] = fcode;
      } else {
        htab.fill(-1);
        free_ent = ClearCode + 2;
        clear_flg = true;
        output(ClearCode);
      }
    }
  }
  output(ent);
  output(EOFCode);
  outStream.writeByte(0);
  return outStream.bytesView();
  function output(code) {
    cur_accum &= MASKS[cur_bits];
    if (cur_bits > 0)
      cur_accum |= code << cur_bits;
    else
      cur_accum = code;
    cur_bits += n_bits;
    while (cur_bits >= 8) {
      accum[a_count++] = cur_accum & 255;
      if (a_count >= 254) {
        outStream.writeByte(a_count);
        outStream.writeBytesView(accum, 0, a_count);
        a_count = 0;
      }
      cur_accum >>= 8;
      cur_bits -= 8;
    }
    if (free_ent > maxcode || clear_flg) {
      if (clear_flg) {
        n_bits = g_init_bits;
        maxcode = (1 << n_bits) - 1;
        clear_flg = false;
      } else {
        ++n_bits;
        maxcode = n_bits === BITS ? 1 << n_bits : (1 << n_bits) - 1;
      }
    }
    if (code == EOFCode) {
      while (cur_bits > 0) {
        accum[a_count++] = cur_accum & 255;
        if (a_count >= 254) {
          outStream.writeByte(a_count);
          outStream.writeBytesView(accum, 0, a_count);
          a_count = 0;
        }
        cur_accum >>= 8;
        cur_bits -= 8;
      }
      if (a_count > 0) {
        outStream.writeByte(a_count);
        outStream.writeBytesView(accum, 0, a_count);
        a_count = 0;
      }
    }
  }
}
var lzwEncode_default = lzwEncode;

// src/rgb-packing.js
function rgb888_to_rgb565(r, g, b) {
  return r << 8 & 63488 | g << 2 & 992 | b >> 3;
}
function rgba8888_to_rgba4444(r, g, b, a) {
  return r >> 4 | g & 240 | (b & 240) << 4 | (a & 240) << 8;
}
function rgb888_to_rgb444(r, g, b) {
  return r >> 4 << 8 | g & 240 | b >> 4;
}

// src/pnnquant2.js
function clamp(value, min, max) {
  return value < min ? min : value > max ? max : value;
}
function sqr(value) {
  return value * value;
}
function find_nn(bins, idx, hasAlpha) {
  var nn = 0;
  var err = 1e100;
  const bin1 = bins[idx];
  const n1 = bin1.cnt;
  const wa = bin1.ac;
  const wr = bin1.rc;
  const wg = bin1.gc;
  const wb = bin1.bc;
  for (var i = bin1.fw; i != 0; i = bins[i].fw) {
    const bin = bins[i];
    const n2 = bin.cnt;
    const nerr2 = n1 * n2 / (n1 + n2);
    if (nerr2 >= err)
      continue;
    var nerr = 0;
    if (hasAlpha) {
      nerr += nerr2 * sqr(bin.ac - wa);
      if (nerr >= err)
        continue;
    }
    nerr += nerr2 * sqr(bin.rc - wr);
    if (nerr >= err)
      continue;
    nerr += nerr2 * sqr(bin.gc - wg);
    if (nerr >= err)
      continue;
    nerr += nerr2 * sqr(bin.bc - wb);
    if (nerr >= err)
      continue;
    err = nerr;
    nn = i;
  }
  bin1.err = err;
  bin1.nn = nn;
}
function create_bin() {
  return {
    ac: 0,
    rc: 0,
    gc: 0,
    bc: 0,
    cnt: 0,
    nn: 0,
    fw: 0,
    bk: 0,
    tm: 0,
    mtm: 0,
    err: 0
  };
}
function create_bin_list(data, format) {
  const bincount = format === "rgb444" ? 4096 : 65536;
  const bins = new Array(bincount);
  const size = data.length;
  if (format === "rgba4444") {
    for (let i = 0; i < size; ++i) {
      const color = data[i];
      const a = color >> 24 & 255;
      const b = color >> 16 & 255;
      const g = color >> 8 & 255;
      const r = color & 255;
      const index = rgba8888_to_rgba4444(r, g, b, a);
      let bin = index in bins ? bins[index] : bins[index] = create_bin();
      bin.rc += r;
      bin.gc += g;
      bin.bc += b;
      bin.ac += a;
      bin.cnt++;
    }
  } else if (format === "rgb444") {
    for (let i = 0; i < size; ++i) {
      const color = data[i];
      const b = color >> 16 & 255;
      const g = color >> 8 & 255;
      const r = color & 255;
      const index = rgb888_to_rgb444(r, g, b);
      let bin = index in bins ? bins[index] : bins[index] = create_bin();
      bin.rc += r;
      bin.gc += g;
      bin.bc += b;
      bin.cnt++;
    }
  } else {
    for (let i = 0; i < size; ++i) {
      const color = data[i];
      const b = color >> 16 & 255;
      const g = color >> 8 & 255;
      const r = color & 255;
      const index = rgb888_to_rgb565(r, g, b);
      let bin = index in bins ? bins[index] : bins[index] = create_bin();
      bin.rc += r;
      bin.gc += g;
      bin.bc += b;
      bin.cnt++;
    }
  }
  return bins;
}
function quantize(rgba, maxColors, opts = {}) {
  const {
    format = "rgb565",
    clearAlpha = true,
    clearAlphaColor = 0,
    clearAlphaThreshold = 0,
    oneBitAlpha = false
  } = opts;
  if (!rgba || !rgba.buffer) {
    throw new Error("quantize() expected RGBA Uint8Array data");
  }
  if (!(rgba instanceof Uint8Array) && !(rgba instanceof Uint8ClampedArray)) {
    throw new Error("quantize() expected RGBA Uint8Array data");
  }
  const data = new Uint32Array(rgba.buffer);
  let useSqrt = opts.useSqrt !== false;
  const hasAlpha = format === "rgba4444";
  const bins = create_bin_list(data, format);
  const bincount = bins.length;
  const bincountMinusOne = bincount - 1;
  const heap = new Uint32Array(bincount + 1);
  var maxbins = 0;
  for (var i = 0; i < bincount; ++i) {
    const bin = bins[i];
    if (bin != null) {
      var d = 1 / bin.cnt;
      if (hasAlpha)
        bin.ac *= d;
      bin.rc *= d;
      bin.gc *= d;
      bin.bc *= d;
      bins[maxbins++] = bin;
    }
  }
  if (sqr(maxColors) / maxbins < 0.022) {
    useSqrt = false;
  }
  var i = 0;
  for (; i < maxbins - 1; ++i) {
    bins[i].fw = i + 1;
    bins[i + 1].bk = i;
    if (useSqrt)
      bins[i].cnt = Math.sqrt(bins[i].cnt);
  }
  if (useSqrt)
    bins[i].cnt = Math.sqrt(bins[i].cnt);
  var h, l, l2;
  for (i = 0; i < maxbins; ++i) {
    find_nn(bins, i, false);
    var err = bins[i].err;
    for (l = ++heap[0]; l > 1; l = l2) {
      l2 = l >> 1;
      if (bins[h = heap[l2]].err <= err)
        break;
      heap[l] = h;
    }
    heap[l] = i;
  }
  var extbins = maxbins - maxColors;
  for (i = 0; i < extbins; ) {
    var tb;
    for (; ; ) {
      var b1 = heap[1];
      tb = bins[b1];
      if (tb.tm >= tb.mtm && bins[tb.nn].mtm <= tb.tm)
        break;
      if (tb.mtm == bincountMinusOne)
        b1 = heap[1] = heap[heap[0]--];
      else {
        find_nn(bins, b1, false);
        tb.tm = i;
      }
      var err = bins[b1].err;
      for (l = 1; (l2 = l + l) <= heap[0]; l = l2) {
        if (l2 < heap[0] && bins[heap[l2]].err > bins[heap[l2 + 1]].err)
          l2++;
        if (err <= bins[h = heap[l2]].err)
          break;
        heap[l] = h;
      }
      heap[l] = b1;
    }
    var nb = bins[tb.nn];
    var n1 = tb.cnt;
    var n2 = nb.cnt;
    var d = 1 / (n1 + n2);
    if (hasAlpha)
      tb.ac = d * (n1 * tb.ac + n2 * nb.ac);
    tb.rc = d * (n1 * tb.rc + n2 * nb.rc);
    tb.gc = d * (n1 * tb.gc + n2 * nb.gc);
    tb.bc = d * (n1 * tb.bc + n2 * nb.bc);
    tb.cnt += nb.cnt;
    tb.mtm = ++i;
    bins[nb.bk].fw = nb.fw;
    bins[nb.fw].bk = nb.bk;
    nb.mtm = bincountMinusOne;
  }
  let palette = [];
  var k = 0;
  for (i = 0; ; ++k) {
    let r = clamp(Math.round(bins[i].rc), 0, 255);
    let g = clamp(Math.round(bins[i].gc), 0, 255);
    let b = clamp(Math.round(bins[i].bc), 0, 255);
    let a = 255;
    if (hasAlpha) {
      a = clamp(Math.round(bins[i].ac), 0, 255);
      if (oneBitAlpha) {
        const threshold = typeof oneBitAlpha === "number" ? oneBitAlpha : 127;
        a = a <= threshold ? 0 : 255;
      }
      if (clearAlpha && a <= clearAlphaThreshold) {
        r = g = b = clearAlphaColor;
        a = 0;
      }
    }
    const color = hasAlpha ? [r, g, b, a] : [r, g, b];
    const exists = existsInPalette(palette, color);
    if (!exists)
      palette.push(color);
    if ((i = bins[i].fw) == 0)
      break;
  }
  return palette;
}
function existsInPalette(palette, color) {
  for (let i = 0; i < palette.length; i++) {
    const p = palette[i];
    let matchesRGB = p[0] === color[0] && p[1] === color[1] && p[2] === color[2];
    let matchesAlpha = p.length >= 4 && color.length >= 4 ? p[3] === color[3] : true;
    if (matchesRGB && matchesAlpha)
      return true;
  }
  return false;
}

// src/color.js
function euclideanDistanceSquared(a, b) {
  var sum = 0;
  var n;
  for (n = 0; n < a.length; n++) {
    const dx = a[n] - b[n];
    sum += dx * dx;
  }
  return sum;
}

// src/palettize.js
function roundStep(byte, step) {
  return step > 1 ? Math.round(byte / step) * step : byte;
}
function prequantize(rgba, {roundRGB = 5, roundAlpha = 10, oneBitAlpha = null} = {}) {
  const data = new Uint32Array(rgba.buffer);
  for (let i = 0; i < data.length; i++) {
    const color = data[i];
    let a = color >> 24 & 255;
    let b = color >> 16 & 255;
    let g = color >> 8 & 255;
    let r = color & 255;
    a = roundStep(a, roundAlpha);
    if (oneBitAlpha) {
      const threshold = typeof oneBitAlpha === "number" ? oneBitAlpha : 127;
      a = a <= threshold ? 0 : 255;
    }
    r = roundStep(r, roundRGB);
    g = roundStep(g, roundRGB);
    b = roundStep(b, roundRGB);
    data[i] = a << 24 | b << 16 | g << 8 | r << 0;
  }
}
function applyPalette(rgba, palette, format = "rgb565") {
  if (!rgba || !rgba.buffer) {
    throw new Error("quantize() expected RGBA Uint8Array data");
  }
  if (!(rgba instanceof Uint8Array) && !(rgba instanceof Uint8ClampedArray)) {
    throw new Error("quantize() expected RGBA Uint8Array data");
  }
  if (palette.length > 256) {
    throw new Error("applyPalette() only works with 256 colors or less");
  }
  const data = new Uint32Array(rgba.buffer);
  const length = data.length;
  const bincount = format === "rgb444" ? 4096 : 65536;
  const index = new Uint8Array(length);
  const cache = new Array(bincount);
  const hasAlpha = format === "rgba4444";
  if (format === "rgba4444") {
    for (let i = 0; i < length; i++) {
      const color = data[i];
      const a = color >> 24 & 255;
      const b = color >> 16 & 255;
      const g = color >> 8 & 255;
      const r = color & 255;
      const key = rgba8888_to_rgba4444(r, g, b, a);
      const idx = key in cache ? cache[key] : cache[key] = nearestColorIndexRGBA(r, g, b, a, palette);
      index[i] = idx;
    }
  } else {
    const rgb888_to_key = format === "rgb444" ? rgb888_to_rgb444 : rgb888_to_rgb565;
    for (let i = 0; i < length; i++) {
      const color = data[i];
      const b = color >> 16 & 255;
      const g = color >> 8 & 255;
      const r = color & 255;
      const key = rgb888_to_key(r, g, b);
      const idx = key in cache ? cache[key] : cache[key] = nearestColorIndexRGB(r, g, b, palette);
      index[i] = idx;
    }
  }
  return index;
}
function nearestColorIndexRGBA(r, g, b, a, palette) {
  let k = 0;
  let mindist = 1e100;
  for (let i = 0; i < palette.length; i++) {
    const px2 = palette[i];
    const a2 = px2[3];
    let curdist = sqr2(a2 - a);
    if (curdist > mindist)
      continue;
    const r2 = px2[0];
    curdist += sqr2(r2 - r);
    if (curdist > mindist)
      continue;
    const g2 = px2[1];
    curdist += sqr2(g2 - g);
    if (curdist > mindist)
      continue;
    const b2 = px2[2];
    curdist += sqr2(b2 - b);
    if (curdist > mindist)
      continue;
    mindist = curdist;
    k = i;
  }
  return k;
}
function nearestColorIndexRGB(r, g, b, palette) {
  let k = 0;
  let mindist = 1e100;
  for (let i = 0; i < palette.length; i++) {
    const px2 = palette[i];
    const r2 = px2[0];
    let curdist = sqr2(r2 - r);
    if (curdist > mindist)
      continue;
    const g2 = px2[1];
    curdist += sqr2(g2 - g);
    if (curdist > mindist)
      continue;
    const b2 = px2[2];
    curdist += sqr2(b2 - b);
    if (curdist > mindist)
      continue;
    mindist = curdist;
    k = i;
  }
  return k;
}
function snapColorsToPalette(palette, knownColors, threshold = 5) {
  if (!palette.length || !knownColors.length)
    return;
  const paletteRGB = palette.map((p) => p.slice(0, 3));
  const thresholdSq = threshold * threshold;
  const dim = palette[0].length;
  for (let i = 0; i < knownColors.length; i++) {
    let color = knownColors[i];
    if (color.length < dim) {
      color = [color[0], color[1], color[2], 255];
    } else if (color.length > dim) {
      color = color.slice(0, 3);
    } else {
      color = color.slice();
    }
    const r = nearestColorIndexWithDistance(paletteRGB, color.slice(0, 3), euclideanDistanceSquared);
    const idx = r[0];
    const distanceSq = r[1];
    if (distanceSq > 0 && distanceSq <= thresholdSq) {
      palette[idx] = color;
    }
  }
}
function sqr2(a) {
  return a * a;
}
function nearestColorIndex(colors, pixel, distanceFn = euclideanDistanceSquared) {
  let minDist = Infinity;
  let minDistIndex = -1;
  for (let j = 0; j < colors.length; j++) {
    const paletteColor = colors[j];
    const dist = distanceFn(pixel, paletteColor);
    if (dist < minDist) {
      minDist = dist;
      minDistIndex = j;
    }
  }
  return minDistIndex;
}
function nearestColorIndexWithDistance(colors, pixel, distanceFn = euclideanDistanceSquared) {
  let minDist = Infinity;
  let minDistIndex = -1;
  for (let j = 0; j < colors.length; j++) {
    const paletteColor = colors[j];
    const dist = distanceFn(pixel, paletteColor);
    if (dist < minDist) {
      minDist = dist;
      minDistIndex = j;
    }
  }
  return [minDistIndex, minDist];
}
function nearestColor(colors, pixel, distanceFn = euclideanDistanceSquared) {
  return colors[nearestColorIndex(colors, pixel, distanceFn)];
}

// src/index.js
function GIFEncoder(opt = {}) {
  const {initialCapacity = 4096, auto = true} = opt;
  const stream = createStream(initialCapacity);
  const HSIZE = 5003;
  const accum = new Uint8Array(256);
  const htab = new Int32Array(HSIZE);
  const codetab = new Int32Array(HSIZE);
  let hasInit = false;
  return {
    reset() {
      stream.reset();
      hasInit = false;
    },
    finish() {
      stream.writeByte(constants_default.trailer);
    },
    bytes() {
      return stream.bytes();
    },
    bytesView() {
      return stream.bytesView();
    },
    get buffer() {
      return stream.buffer;
    },
    get stream() {
      return stream;
    },
    writeHeader,
    writeFrame(index, width, height, opts = {}) {
      const {
        transparent = false,
        transparentIndex = 0,
        delay = 0,
        palette = null,
        repeat = 0,
        colorDepth = 8,
        dispose = -1
      } = opts;
      let first = false;
      if (auto) {
        if (!hasInit) {
          first = true;
          writeHeader();
          hasInit = true;
        }
      } else {
        first = Boolean(opts.first);
      }
      width = Math.max(0, Math.floor(width));
      height = Math.max(0, Math.floor(height));
      if (first) {
        if (!palette) {
          throw new Error("First frame must include a { palette } option");
        }
        encodeLogicalScreenDescriptor(stream, width, height, palette, colorDepth);
        encodeColorTable(stream, palette);
        if (repeat >= 0) {
          encodeNetscapeExt(stream, repeat);
        }
      }
      const delayTime = Math.round(delay / 10);
      encodeGraphicControlExt(stream, dispose, delayTime, transparent, transparentIndex);
      const useLocalColorTable = Boolean(palette) && !first;
      encodeImageDescriptor(stream, width, height, useLocalColorTable ? palette : null);
      if (useLocalColorTable)
        encodeColorTable(stream, palette);
      encodePixels(stream, index, width, height, colorDepth, accum, htab, codetab);
    }
  };
  function writeHeader() {
    writeUTFBytes(stream, "GIF89a");
  }
}
function encodeGraphicControlExt(stream, dispose, delay, transparent, transparentIndex) {
  stream.writeByte(33);
  stream.writeByte(249);
  stream.writeByte(4);
  if (transparentIndex < 0) {
    transparentIndex = 0;
    transparent = false;
  }
  var transp, disp;
  if (!transparent) {
    transp = 0;
    disp = 0;
  } else {
    transp = 1;
    disp = 2;
  }
  if (dispose >= 0) {
    disp = dispose & 7;
  }
  disp <<= 2;
  const userInput = 0;
  stream.writeByte(0 | disp | userInput | transp);
  writeUInt16(stream, delay);
  stream.writeByte(transparentIndex || 0);
  stream.writeByte(0);
}
function encodeLogicalScreenDescriptor(stream, width, height, palette, colorDepth = 8) {
  const globalColorTableFlag = 1;
  const sortFlag = 0;
  const globalColorTableSize = colorTableSize(palette.length) - 1;
  const fields = globalColorTableFlag << 7 | colorDepth - 1 << 4 | sortFlag << 3 | globalColorTableSize;
  const backgroundColorIndex = 0;
  const pixelAspectRatio = 0;
  writeUInt16(stream, width);
  writeUInt16(stream, height);
  stream.writeBytes([fields, backgroundColorIndex, pixelAspectRatio]);
}
function encodeNetscapeExt(stream, repeat) {
  stream.writeByte(33);
  stream.writeByte(255);
  stream.writeByte(11);
  writeUTFBytes(stream, "NETSCAPE2.0");
  stream.writeByte(3);
  stream.writeByte(1);
  writeUInt16(stream, repeat);
  stream.writeByte(0);
}
function encodeColorTable(stream, palette) {
  const colorTableLength = 1 << colorTableSize(palette.length);
  for (let i = 0; i < colorTableLength; i++) {
    let color = [0, 0, 0];
    if (i < palette.length) {
      color = palette[i];
    }
    stream.writeByte(color[0]);
    stream.writeByte(color[1]);
    stream.writeByte(color[2]);
  }
}
function encodeImageDescriptor(stream, width, height, localPalette) {
  stream.writeByte(44);
  writeUInt16(stream, 0);
  writeUInt16(stream, 0);
  writeUInt16(stream, width);
  writeUInt16(stream, height);
  if (localPalette) {
    const interlace = 0;
    const sorted = 0;
    const palSize = colorTableSize(localPalette.length) - 1;
    stream.writeByte(128 | interlace | sorted | 0 | palSize);
  } else {
    stream.writeByte(0);
  }
}
function encodePixels(stream, index, width, height, colorDepth = 8, accum, htab, codetab) {
  lzwEncode_default(width, height, index, colorDepth, stream, accum, htab, codetab);
}
function writeUInt16(stream, short) {
  stream.writeByte(short & 255);
  stream.writeByte(short >> 8 & 255);
}
function writeUTFBytes(stream, text) {
  for (var i = 0; i < text.length; i++) {
    stream.writeByte(text.charCodeAt(i));
  }
}
function colorTableSize(length) {
  return Math.max(Math.ceil(Math.log2(length)), 1);
}
var src_default = GIFEncoder;
//# sourceMappingURL=gifenc.js.map

  window.gifenc = exports;
})();
