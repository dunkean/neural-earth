// Generated from Magna Urbis rust/bridge/fantasyPainting.ts. GPL-3.0. Rebuild: node web/scripts/build_fantasy_painter.mjs
var FantasyPainting = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // ../rust/bridge/fantasyPainting.ts
  var fantasyPainting_exports = {};
  __export(fantasyPainting_exports, {
    ATLAS_GEOMETRY: () => ATLAS_GEOMETRY,
    DEFAULT_FANTASY_OPTIONS: () => DEFAULT_FANTASY_OPTIONS,
    FAMILY_NAMES: () => FAMILY_NAMES,
    atlasRect: () => atlasRect,
    brushFrame: () => brushFrame,
    createBrushLibrary: () => createBrushLibrary,
    hash: () => hash,
    isFantasyStyle: () => isFantasyStyle,
    paintSeed: () => paintSeed,
    paintingCanvas: () => paintingCanvas,
    paintingSvg: () => paintingSvg,
    planPainting: () => planPainting
  });
  var DEFAULT_FANTASY_OPTIONS = { density: 1, size: 1, spacing: 1 };
  var isFantasyStyle = (style) => style === "fantasy-ink" || style === "fantasy-color";
  var FAMILY_NAMES = ["pines", "broadleaf", "tropical", "cypress", "hills", "mountains", "snow-mountains", "rocks", "dunes", "wetlands", "crops", "orchards", "grasses"];
  var BRUSH_FAMILY = ["pines", "broadleaf", "tropical", "cypress", "hills", "mountains", "snow-mountains", "rocks", "dunes", "wetlands", "crops", "orchards", "pines", "broadleaf", "tropical", "grasses"];
  function createBrushLibrary(style, families) {
    const frames = [];
    for (let brush = 0; brush < 16; brush++) for (let v = 0; v < 8; v++) {
      const family = BRUSH_FAMILY[brush], atlas = families[family];
      if (!atlas) throw new Error(`Missing fantasy brush sheet: ${family}`);
      let variant = v;
      if (brush === 0 || brush === 1) variant %= 6;
      if (brush === 12 || brush === 13) variant = 6 + v % 2;
      if (brush === 2) variant = [0, 1, 2, 5][v % 4];
      if (brush === 14) variant = [3, 4, 6, 7][v % 4];
      const index = variant + (style === "fantasy-color" ? 8 : 0), rect = atlasRect(atlas, index);
      frames.push({ atlas, family, index, token: `${family}-${index}`, rect, aspect: Math.max(0.4, Math.min(1.8, rect[3] / rect[2])) });
    }
    return { kind: "families", style, families, frames };
  }
  function brushFrame(library, stamp) {
    return library.frames[stamp.brush * 8 + (stamp.variant ?? 0) % 8];
  }
  var ATLAS_GEOMETRY = {
    "fantasy-ink": { width: 1254, height: 1254, rows: [0, 0.28, 0.52, 0.735, 1], rects: [
      [35, 0, 265, 346],
      [323, 20, 305, 325],
      [651, 20, 286, 322],
      [1075, 0, 130, 346],
      [0, 414, 320, 230],
      [325, 350, 305, 294],
      [638, 350, 312, 294],
      [958, 399, 296, 245],
      [0, 692, 320, 225],
      [328, 665, 310, 250],
      [644, 680, 301, 236],
      [962, 647, 292, 272],
      [30, 928, 282, 326],
      [325, 928, 300, 326],
      [647, 929, 307, 325],
      [973, 975, 281, 279]
    ] },
    "fantasy-color": { width: 1295, height: 1214, rows: [0, 0.295, 0.54, 0.75, 1], rects: [
      [40, 0, 245, 358],
      [333, 20, 310, 338],
      [677, 15, 290, 343],
      [1039, 0, 222, 358],
      [0, 420, 326, 234],
      [328, 357, 325, 300],
      [660, 350, 320, 308],
      [993, 395, 302, 254],
      [0, 680, 333, 234],
      [338, 663, 310, 252],
      [653, 682, 329, 233],
      [995, 657, 300, 245],
      [30, 916, 292, 298],
      [328, 916, 312, 298],
      [660, 916, 320, 298],
      [999, 970, 266, 244]
    ] }
  };
  function hash(x, y, seed) {
    let h = Math.imul(x, 2654435769) ^ Math.imul(y, 2246822507) ^ seed;
    h = Math.imul(h ^ h >>> 16, 2146121005);
    h = Math.imul(h ^ h >>> 15, 2221713035);
    return ((h ^ h >>> 16) >>> 0) / 4294967296;
  }
  function paintSeed(value) {
    let seed = 2166136261;
    for (let i = 0; i < value.length; i++) seed = Math.imul(seed ^ value.charCodeAt(i), 16777619);
    return seed;
  }
  function noise(x, y, seed) {
    const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
    const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
    const a = hash(ix, iy, seed), b = hash(ix + 1, iy, seed), c = hash(ix, iy + 1, seed), d = hash(ix + 1, iy + 1, seed);
    return (a + (b - a) * u) * (1 - v) + (c + (d - c) * u) * v;
  }
  function planPainting(bounds, sample, options) {
    const pitch = options.pitch, density = Math.max(0, Math.min(2, options.density ?? 1)), size = Math.max(0.3, Math.min(2, options.size ?? 1));
    if (!Number.isFinite(pitch) || pitch <= 0 || ![bounds.x, bounds.y, bounds.width, bounds.height, density, size].every(Number.isFinite) || bounds.width <= 0 || bounds.height <= 0) return [];
    const seed = paintSeed(options.seed), result = [], budget = Number.isFinite(options.maxStamps ?? 4e4) ? Math.max(0, Math.min(8e4, Math.floor(options.maxStamps ?? 4e4))) : 4e4;
    const pad = pitch * 12 * size;
    for (const pass of ["summits", "ground", "forest"]) {
      const step = pitch * (pass === "summits" ? 2 : pass === "forest" ? 1 / 4 : 1);
      const x0 = Math.floor((bounds.x - pad) / step), x1 = Math.ceil((bounds.x + bounds.width + pad) / step);
      const y0 = Math.floor((bounds.y - pad) / step), y1 = Math.ceil((bounds.y + bounds.height + pad) / step);
      const stride = Math.max(1, Math.ceil(Math.sqrt((x1 - x0 + 1) * (y1 - y0 + 1) / 24e4)));
      if (![x0, x1, y0, y1, stride].every(Number.isSafeInteger)) return [];
      for (let gy = y0; gy <= y1; gy += stride) for (let gx = x0; gx <= x1; gx += stride) {
        const r = hash(gx, gy, seed), x = (gx + 0.18 + hash(gx, gy, seed ^ 71) * 0.64) * step;
        const y = (gy + 0.18 + hash(gx, gy, seed ^ 191) * 0.64) * step, s = sample(x, y);
        if (!s || ![s.height, s.slope, s.temperature, s.rain].every(Number.isFinite) || s.height <= 0 || s.water || s.occupied) continue;
        const patch = noise(x / (pitch * 11), y / (pitch * 11), seed ^ 891);
        const detail = noise(x / (pitch * 3), y / (pitch * 3), seed ^ 421);
        const temp = s.temperature - s.height * 6e-3, arid = s.rain < 350;
        let brush, cover, scale = 1;
        if (s.height > 1850 || s.slope > 0.48) {
          brush = temp < 0 || s.height > 2900 ? 6 : 5;
          cover = 0.46 + 0.25 * detail;
          scale = 5.6;
        } else if (s.slope > 0.22) {
          brush = s.slope > 0.36 ? 7 : 4;
          cover = 0.32 + 0.32 * patch;
          scale = 2.1;
        } else if (arid) {
          brush = 8;
          cover = 0.16 + 0.48 * patch;
          scale = 1.35;
        } else if (s.height < 70 && s.rain > 1600 && s.slope < 0.04 && patch > 0.66) {
          brush = 9;
          cover = 0.22 + patch * 0.36;
        } else {
          cover = Math.max(0, Math.min(1, (s.rain - 250) / 1300)) * Math.max(0, Math.min(1, (patch * 0.75 + detail * 0.25 - 0.4) * 4.5));
          const conifer = temp < 9 || s.height > 1e3, tropical = temp > 22 && s.rain > 1100;
          brush = tropical ? r < 0.18 ? 2 : 14 : conifer ? r < 0.22 ? 12 : 0 : r < 0.22 ? 13 : 1;
          if (!tropical && !conifer && temp > 14 && s.rain < 850 && r < 0.14) brush = 3;
          scale = 0.48;
          if (cover < 0.16) {
            brush = patch > 0.18 && patch < 0.28 && s.slope < 0.05 && temp > 5 && temp < 24 ? detail > 0.6 ? 11 : 10 : 15;
            cover = brush === 15 ? 0.035 : 0.13;
            scale = brush === 15 ? 0.65 : 1.05;
          }
        }
        const tree = brush < 4 || brush >= 12 && brush <= 14;
        if (pass !== (brush === 5 || brush === 6 ? "summits" : tree ? "forest" : "ground")) continue;
        if (hash(gx, gy, seed ^ 9811) > cover * density) continue;
        if (brush === 5 || brush === 6 || brush === 7) {
          let crowded = false;
          for (let oy = -1; oy <= 1 && !crowded; oy++) for (let ox = -1; ox <= 1; ox++) {
            if (!ox && !oy || hash(gx + ox, gy + oy, seed) >= r) continue;
            const neighbour = sample(x + ox * step, y + oy * step);
            if (neighbour && !neighbour.water && (neighbour.height > 1850 || neighbour.slope > 0.36)) {
              crowded = true;
              break;
            }
          }
          if (crowded) continue;
        }
        const w = pitch * scale * size * (0.85 + 0.3 * hash(gx, gy, seed ^ 319));
        const margin = (brush === 5 || brush === 6 ? Math.min(w, pitch * 2.8 * size) : w) * 0.36;
        if ([-1, 1].some((d) => [-1, 1].some((e) => {
          const edge = sample(x + d * margin, y + e * margin * 0.25);
          return !edge || edge.height <= 0 || edge.water || edge.occupied;
        }))) continue;
        result.push({ x, y, width: w, height: w * (brush >= 4 && brush <= 11 ? 0.78 : 1.15), brush, variant: Math.floor(hash(gx, gy, seed ^ 6317) * 8), flip: hash(gx, gy, seed ^ 97) > 0.5, opacity: brush === 15 ? 0.65 : 1 });
      }
    }
    result.sort((a, b) => a.y - b.y || a.x - b.x);
    if (result.length > budget) {
      const relief = result.filter((s) => s.brush >= 4 && s.brush <= 8);
      const ground = result.filter((s) => s.brush < 4 || s.brush > 8);
      const thin = (items, count) => items.filter((_, i) => Math.floor(i * count / items.length) !== Math.floor((i - 1) * count / items.length)).slice(0, count);
      const reliefBudget = Math.min(relief.length, budget);
      return [...thin(relief, reliefBudget), ...thin(ground, budget - reliefBudget)].sort((a, b) => a.y - b.y || a.x - b.x);
    }
    return result;
  }
  function atlasRect(atlas, brush) {
    if (atlas.rects?.[brush]) return atlas.rects[brush];
    const row = Math.floor(brush / 4), x = brush % 4 * atlas.width / 4, y = atlas.rows[row] * atlas.height;
    return [x, y, atlas.width / 4, (atlas.rows[row + 1] - atlas.rows[row]) * atlas.height];
  }
  function paintingSvg(stamps, atlas, namespace, ink = false) {
    if (!stamps.length) return "";
    const id = namespace.replace(/[^a-zA-Z0-9_-]/g, "-");
    if ("kind" in atlas) return familyPaintingSvg(stamps, atlas, id);
    let svg = `<defs><image id="${id}-atlas" width="${atlas.width}" height="${atlas.height}" href="${atlas.url}"/>`;
    for (let i = 0; i < 16; i++) {
      const [x, y, w, h] = atlasRect(atlas, i);
      svg += `<symbol id="${id}-${i}" viewBox="${x} ${y} ${w} ${h}" preserveAspectRatio="none" overflow="hidden"><use href="#${id}-atlas"/></symbol>`;
    }
    svg += "</defs>";
    if (ink) svg += `<defs><filter id="${id}-gray" color-interpolation-filters="sRGB"><feColorMatrix type="saturate" values="0"/></filter></defs>`;
    svg += `<g class="fantasy-brushes"${ink ? ` filter="url(#${id}-gray)"` : ""}>`;
    const f = (n) => n.toFixed(2);
    for (const s of stamps) svg += `<use href="#${id}-${s.brush}" x="${f(-s.width / 2)}" y="${f(-s.height * 0.93)}" width="${f(s.width)}" height="${f(s.height)}" opacity="${s.opacity}" transform="translate(${f(s.x)} ${f(s.y)}) scale(${s.flip ? -1 : 1} 1)"/>`;
    return svg + "</g>";
  }
  function paintingCanvas(ctx, stamps, image, atlas, camera, ink = false) {
    if ("kind" in atlas) {
      familyPaintingCanvas(ctx, stamps, image, atlas, camera);
      return;
    }
    ctx.save();
    if (ink) ctx.filter = "grayscale(1)";
    for (const s of stamps) {
      const [sx, sy, sw, sh] = atlasRect(atlas, s.brush), w = s.width * camera.scale, h = s.height * camera.scale;
      const x = (s.x - camera.x) * camera.scale, y = (s.y - camera.y) * camera.scale;
      ctx.globalAlpha = s.opacity;
      if (s.flip) {
        ctx.save();
        ctx.translate(x, y);
        ctx.scale(-1, 1);
        ctx.drawImage(image, sx, sy, sw, sh, -w / 2, -h * 0.93, w, h);
        ctx.restore();
      } else ctx.drawImage(image, sx, sy, sw, sh, x - w / 2, y - h * 0.93, w, h);
    }
    ctx.restore();
  }
  function familyPaintingSvg(stamps, library, id) {
    const frames = /* @__PURE__ */ new Map(), families = /* @__PURE__ */ new Map();
    for (const stamp of stamps) {
      const f2 = brushFrame(library, stamp);
      frames.set(f2.token, f2);
      families.set(f2.family, f2.atlas);
    }
    let svg = "<defs>";
    for (const [family, atlas] of families) svg += `<image id="${id}-sheet-${family}" width="${atlas.width}" height="${atlas.height}" href="${atlas.url}"/>`;
    for (const frame of frames.values()) {
      const [x, y, w, h] = frame.rect;
      svg += `<symbol id="${id}-${frame.token}" viewBox="${x} ${y} ${w} ${h}" overflow="hidden" preserveAspectRatio="none"><use href="#${id}-sheet-${frame.family}"/></symbol>`;
    }
    svg += '</defs><g class="fantasy-brushes">';
    const f = (n) => n.toFixed(2);
    for (const s of stamps) {
      const frame = brushFrame(library, s), h = s.width * frame.aspect;
      svg += `<use class="${s.brush >= 4 && s.brush <= 8 ? "fantasy-relief" : "fantasy-landuse"}" data-brush="${frame.family}" data-variant="${frame.index % 8}" href="#${id}-${frame.token}" x="${f(-s.width / 2)}" y="${f(-h * 0.95)}" width="${f(s.width)}" height="${f(h)}" opacity="${s.opacity}" transform="translate(${f(s.x)} ${f(s.y)}) scale(${s.flip ? -1 : 1} 1)"/>`;
    }
    return svg + "</g>";
  }
  function familyPaintingCanvas(ctx, stamps, images, library, camera) {
    ctx.save();
    for (const s of stamps) {
      const frame = brushFrame(library, s), image = images.get(frame.family);
      if (!image) continue;
      const [sx, sy, sw, sh] = frame.rect, w = s.width * camera.scale, h = w * frame.aspect;
      const x = (s.x - camera.x) * camera.scale, y = (s.y - camera.y) * camera.scale;
      ctx.globalAlpha = s.opacity;
      if (s.flip) {
        ctx.save();
        ctx.translate(x, y);
        ctx.scale(-1, 1);
        ctx.drawImage(image, sx, sy, sw, sh, -w / 2, -h * 0.95, w, h);
        ctx.restore();
      } else ctx.drawImage(image, sx, sy, sw, sh, x - w / 2, y - h * 0.95, w, h);
    }
    ctx.restore();
  }
  return __toCommonJS(fantasyPainting_exports);
})();
