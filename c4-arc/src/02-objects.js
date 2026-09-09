/* Port of engine/objects.py -- segmentation and object-level features.
 *
 * ARC tasks are about things rather than pixels, so the segmentation
 * vocabulary bounds what any downstream solver can express. Several
 * segmentations are exposed and the portfolio tries each.
 */

function Obj(cells, grid, color) {
  this.cells = cells;                       /* Set<encoded cell> */
  var bb = G.bboxOf(cells);
  this.r0 = bb[0]; this.c0 = bb[1]; this.r1 = bb[2]; this.c1 = bb[3];
  this.grid_shape = [grid.length, grid[0].length];
  if (color === null || color === undefined) {
    var cc = new Int32Array(G.NCOLORS), it = cells.values(), s = it.next();
    while (!s.done) { cc[grid[s.value >> 6][s.value & 63]]++; s = it.next(); }
    color = G.modalOf(cc);
  }
  this.color = color;
  this._patch = null;
  this._mask = null;
  this._normKey = null;
  this._holes = {};
  this._interior = null;
  this._build(grid);
}

Obj.prototype._build = function (grid) {
  var h = this.r1 - this.r0 + 1, w = this.c1 - this.c0 + 1;
  var patch = [], mask = [], r, c, row, mrow;
  for (r = 0; r < h; r++) {
    row = new Array(w); mrow = new Array(w);
    for (c = 0; c < w; c++) { row[c] = null; mrow[c] = 0; }
    patch.push(row); mask.push(mrow);
  }
  var it = this.cells.values(), s = it.next(), p, rr, cc;
  while (!s.done) {
    p = s.value; rr = (p >> 6) - this.r0; cc = (p & 63) - this.c0;
    patch[rr][cc] = grid[p >> 6][p & 63];
    mask[rr][cc] = 1;
    s = it.next();
  }
  this._patch = patch;
  this._mask = mask;
};

Obj.prototype.height = function () { return this.r1 - this.r0 + 1; };
Obj.prototype.width = function () { return this.c1 - this.c0 + 1; };
Obj.prototype.size = function () { return this.cells.size; };
Obj.prototype.bbox_area = function () { return this.height() * this.width(); };
Obj.prototype.patch = function () { return this._patch; };
Obj.prototype.mask = function () { return this._mask; };

Obj.prototype.filled = function (bg) {
  if (bg === undefined) bg = 0;
  var out = [], r, c, row, nr;
  for (r = 0; r < this._patch.length; r++) {
    row = this._patch[r]; nr = new Array(row.length);
    for (c = 0; c < row.length; c++) nr[c] = row[c] === null ? bg : row[c];
    out.push(nr);
  }
  return out;
};

Obj.prototype.is_rect = function () { return this.size() === this.bbox_area(); };
Obj.prototype.is_square = function () { return this.height() === this.width(); };

Obj.prototype.colors = function () {
  var m = 0, r, c, row;
  for (r = 0; r < this._patch.length; r++) {
    row = this._patch[r];
    for (c = 0; c < row.length; c++) if (row[c] !== null) m |= 1 << row[c];
  }
  return m;
};

Obj.prototype.touches_border = function () {
  return this.r0 === 0 || this.c0 === 0 ||
         this.r1 === this.grid_shape[0] - 1 || this.c1 === this.grid_shape[1] - 1;
};

/* Shape identity modulo translation (not colour). */
Obj.prototype.norm_key = function () {
  if (this._normKey === null) {
    var parts = [], r;
    for (r = 0; r < this._mask.length; r++) parts.push(this._mask[r].join(""));
    this._normKey = parts.join("|");
  }
  return this._normKey;
};

Obj.prototype.holes_count = function (diag) {
  diag = !!diag;
  var k = diag ? "d" : "o";
  if (this._holes[k] !== undefined) return this._holes[k];
  var n = this._holesCount(diag);
  this._holes[k] = n;
  return n;
};

Obj.prototype._holesCount = function (diag) {
  var h = this.height(), w = this.width(), m = this._mask;
  var seen = [], r, c, i, row;
  for (r = 0; r < h; r++) { row = new Array(w); for (c = 0; c < w; c++) row[c] = false; seen.push(row); }
  var stack = [];
  for (r = 0; r < h; r++) for (i = 0; i < 2; i++) {
    c = i ? w - 1 : 0;
    if (!m[r][c] && !seen[r][c]) { seen[r][c] = true; stack.push([r, c]); }
  }
  for (c = 0; c < w; c++) for (i = 0; i < 2; i++) {
    r = i ? h - 1 : 0;
    if (!m[r][c] && !seen[r][c]) { seen[r][c] = true; stack.push([r, c]); }
  }
  var nb = diag ? G.N8 : G.N4, cur, nr, nc, d;
  while (stack.length) {
    cur = stack.pop();
    for (d = 0; d < nb.length; d++) {
      nr = cur[0] + nb[d][0]; nc = cur[1] + nb[d][1];
      if (nr >= 0 && nr < h && nc >= 0 && nc < w && !seen[nr][nc] && !m[nr][nc]) {
        seen[nr][nc] = true; stack.push([nr, nc]);
      }
    }
  }
  var cnt = 0, vis2 = [];
  for (r = 0; r < h; r++) { row = new Array(w); for (c = 0; c < w; c++) row[c] = false; vis2.push(row); }
  for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
    if (m[r][c] || seen[r][c] || vis2[r][c]) continue;
    cnt++;
    var st = [[r, c]];
    vis2[r][c] = true;
    while (st.length) {
      cur = st.pop();
      for (d = 0; d < nb.length; d++) {
        nr = cur[0] + nb[d][0]; nc = cur[1] + nb[d][1];
        if (nr >= 0 && nr < h && nc >= 0 && nc < w && !m[nr][nc] && !seen[nr][nc] && !vis2[nr][nc]) {
          vis2[nr][nc] = true; st.push([nr, nc]);
        }
      }
    }
  }
  return cnt;
};

/* ------------------------------------------------------------ segmentations */

function segConnected(grid, bg, diag, sameColor) {
  var regs = G.floodRegions(grid, bg, diag, sameColor), out = [], i;
  for (i = 0; i < regs.length; i++) out.push(new Obj(regs[i][1], grid, regs[i][0]));
  return out;
}

function segByColor(grid, bg) {
  var h = grid.length, w = grid[0].length, buckets = {}, r, c, v, row;
  for (r = 0; r < h; r++) {
    row = grid[r];
    for (c = 0; c < w; c++) {
      v = row[c];
      if (v !== bg) { if (!buckets[v]) buckets[v] = []; buckets[v].push(r * 64 + c); }
    }
  }
  var keys = Object.keys(buckets).map(Number).sort(function (a, b) { return a - b; });
  var out = [], i;
  for (i = 0; i < keys.length; i++) out.push(new Obj(new Set(buckets[keys[i]]), grid, keys[i]));
  return out;
}

function segCells(grid, bg) {
  var h = grid.length, w = grid[0].length, out = [], r, c;
  for (r = 0; r < h; r++) for (c = 0; c < w; c++)
    if (grid[r][c] !== bg) out.push(new Obj(new Set([r * 64 + c]), grid, grid[r][c]));
  return out;
}

/* Components where cells within Chebyshev distance ``gap`` are linked: real
   ARC objects are often scattered, and strict adjacency splits a dashed line
   into singletons. */
function segConnectedGap(grid, bg, gap, sameColor) {
  if (sameColor === undefined) sameColor = true;
  var h = grid.length, w = grid[0].length, pts = [], r, c;
  for (r = 0; r < h; r++) for (c = 0; c < w; c++) if (grid[r][c] !== bg) pts.push(r * 64 + c);
  if (pts.length > 400) return [];
  var idx = new Map(), i;
  for (i = 0; i < pts.length; i++) idx.set(pts[i], i);
  var parent = new Int32Array(pts.length);
  for (i = 0; i < pts.length; i++) parent[i] = i;
  function find(x) { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
  var dr, dc, j, a, b;
  for (i = 0; i < pts.length; i++) {
    r = pts[i] >> 6; c = pts[i] & 63;
    for (dr = -gap; dr <= gap; dr++) for (dc = -gap; dc <= gap; dc++) {
      if (dr === 0 && dc === 0) continue;
      if (r + dr < 0 || r + dr >= h || c + dc < 0 || c + dc >= w) continue;
      j = idx.get((r + dr) * 64 + (c + dc));
      if (j === undefined || j < i) continue;
      if (sameColor && grid[r + dr][c + dc] !== grid[r][c]) continue;
      a = find(i); b = find(j);
      if (a !== b) parent[b] = a;
    }
  }
  var groups = new Map(), root;
  for (i = 0; i < pts.length; i++) {
    root = find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(pts[i]);
  }
  var out = [];
  groups.forEach(function (cs) { out.push(new Obj(new Set(cs), grid)); });
  return out;
}

/* Every row (or column) as one object, background included: a band of the grid
   is a thing a rule can talk about, and no connectivity segmentation names it. */
function segLines(grid, bg, axis) {
  var h = grid.length, w = grid[0].length, out = [], r, c, cells;
  if (axis === "row") {
    for (r = 0; r < h; r++) {
      cells = [];
      for (c = 0; c < w; c++) cells.push(r * 64 + c);
      out.push(new Obj(new Set(cells), grid));
    }
  } else {
    for (c = 0; c < w; c++) {
      cells = [];
      for (r = 0; r < h; r++) cells.push(r * 64 + c);
      out.push(new Obj(new Set(cells), grid));
    }
  }
  return out;
}

var SEGMENTATIONS = [
  ["rows", function (g, bg) { return segLines(g, bg, "row"); }],
  ["cols", function (g, bg) { return segLines(g, bg, "col"); }],
  ["g2", function (g, bg) { return segConnectedGap(g, bg, 2, true); }],
  ["g2m", function (g, bg) { return segConnectedGap(g, bg, 2, false); }],
  ["g3", function (g, bg) { return segConnectedGap(g, bg, 3, true); }],
  ["c4", function (g, bg) { return segConnected(g, bg, false, true); }],
  ["c8", function (g, bg) { return segConnected(g, bg, true, true); }],
  ["m4", function (g, bg) { return segConnected(g, bg, false, false); }],
  ["m8", function (g, bg) { return segConnected(g, bg, true, false); }],
  ["color", segByColor],
  ["cells", segCells]
];
var SEG_NAMES = SEGMENTATIONS.map(function (s) { return s[0]; });

var _SEG_CACHE = new Map();
var _SEG_CAP = 512;

/* Hundreds of hypotheses ask for the same segmentation of the same grid;
   recomputing connected components each time dominated the runtime. */
function segment(grid, mode, bg) {
  if (bg === null || bg === undefined) bg = G.background(grid);
  var key = G.gkey(grid) + "#" + mode + "#" + bg;
  var hit = _SEG_CACHE.get(key);
  if (hit !== undefined) return hit;
  var i, res = null;
  for (i = 0; i < SEGMENTATIONS.length; i++)
    if (SEGMENTATIONS[i][0] === mode) { res = SEGMENTATIONS[i][1](grid, bg); break; }
  if (res === null) throw new Error("unknown segmentation " + mode);
  if (_SEG_CACHE.size > _SEG_CAP) _SEG_CACHE.clear();
  _SEG_CACHE.set(key, res);
  return res;
}

/* ------------------------------------------------------- object rankings */

var OBJ_RANKERS = {
  size: function (o) { return o.size(); },
  bbox_area: function (o) { return o.bbox_area(); },
  height: function (o) { return o.height(); },
  width: function (o) { return o.width(); },
  holes: function (o) { return o.holes_count(); },
  ncolors: function (o) { return G.csSize(o.colors()); },
  top: function (o) { return -o.r0; },
  bottom: function (o) { return o.r1; },
  left: function (o) { return -o.c0; },
  right: function (o) { return o.c1; },
  density: function (o) { return o.size() / o.bbox_area(); }
};
var OBJ_RANKER_NAMES = ["size", "bbox_area", "height", "width", "holes",
                        "ncolors", "top", "bottom", "left", "right", "density"];

function selectExtreme(objs, ranker, wantMax, strict) {
  if (strict === undefined) strict = true;
  if (!objs.length) return null;
  var f = OBJ_RANKERS[ranker], vals = [], i, tgt;
  for (i = 0; i < objs.length; i++) vals.push(f(objs[i]));
  tgt = vals[0];
  for (i = 1; i < vals.length; i++) {
    if (wantMax ? vals[i] > tgt : vals[i] < tgt) tgt = vals[i];
  }
  var hits = [];
  for (i = 0; i < objs.length; i++) if (vals[i] === tgt) hits.push(objs[i]);
  if (strict && hits.length !== 1) return null;
  return hits[0];
}

function countBy(objs, keyfn) {
  var m = new Map(), i, k;
  for (i = 0; i < objs.length; i++) {
    k = keyfn(objs[i]);
    m.set(k, (m.get(k) || 0) + 1);
  }
  return m;
}

function selectUniqueShape(objs) {
  var cnt = countBy(objs, function (o) { return o.norm_key(); }), hits = [], i;
  for (i = 0; i < objs.length; i++) if (cnt.get(objs[i].norm_key()) === 1) hits.push(objs[i]);
  return hits.length === 1 ? hits[0] : null;
}

function selectMajorityShape(objs) {
  var cnt = countBy(objs, function (o) { return o.norm_key(); });
  if (!cnt.size) return null;
  var bestK = null, bestN = -1, order = [];
  cnt.forEach(function (n, k) { order.push([k, n]); });
  var i;
  for (i = 0; i < order.length; i++) if (order[i][1] > bestN) { bestN = order[i][1]; bestK = order[i][0]; }
  if (bestN < 2) return null;
  for (i = 0; i < objs.length; i++) if (objs[i].norm_key() === bestK) return objs[i];
  return null;
}

function selectUniqueColor(objs) {
  var cnt = countBy(objs, function (o) { return o.color; }), hits = [], i;
  for (i = 0; i < objs.length; i++) if (cnt.get(objs[i].color) === 1) hits.push(objs[i]);
  return hits.length === 1 ? hits[0] : null;
}

function selectSymmetric(objs, want) {
  var hits = [], i;
  for (i = 0; i < objs.length; i++)
    if ((G.symmetries(objs[i].filled(0)).length > 0) === want) hits.push(objs[i]);
  return hits.length === 1 ? hits[0] : null;
}

var SELECTORS = [];
(function () {
  var i, r;
  for (i = 0; i < OBJ_RANKER_NAMES.length; i++) {
    r = OBJ_RANKER_NAMES[i];
    SELECTORS.push(["max_" + r, (function (rr) { return function (os) { return selectExtreme(os, rr, true); }; })(r)]);
    SELECTORS.push(["min_" + r, (function (rr) { return function (os) { return selectExtreme(os, rr, false); }; })(r)]);
  }
  SELECTORS.push(["unique_shape", selectUniqueShape]);
  SELECTORS.push(["majority_shape", selectMajorityShape]);
  SELECTORS.push(["unique_color", selectUniqueColor]);
  SELECTORS.push(["symmetric", function (os) { return selectSymmetric(os, true); }]);
  SELECTORS.push(["asymmetric", function (os) { return selectSymmetric(os, false); }]);
})();

var _FEAT_CACHE = new Map();

function featuresOf(grid, mode, bg) {
  var key = G.gkey(grid) + "#" + mode + "#" + bg;
  var hit = _FEAT_CACHE.get(key);
  if (hit !== undefined) return hit;
  var objs = segment(grid, mode, bg);
  var shared = sharedStats(objs, grid), feats = [], i;
  for (i = 0; i < objs.length; i++) feats.push(objectFeatures(objs[i], objs, grid, shared, i));
  if (_FEAT_CACHE.size > 256) _FEAT_CACHE.clear();
  var res = [objs, feats];
  _FEAT_CACHE.set(key, res);
  return res;
}

function sharedStats(objs, grid) {
  var sizes = [], i;
  for (i = 0; i < objs.length; i++) sizes.push(objs[i].size());
  sizes.sort(function (a, b) { return b - a; });
  return {
    sizes: sizes,
    shape_cnt: countBy(objs, function (o) { return o.norm_key(); }),
    color_cnt: countBy(objs, function (o) { return o.color; }),
    dims: [grid.length, grid[0].length],
    containment: containment(objs)
  };
}

/* For each object: what encloses it, and how much it encloses. "Colour each
   shape like the box it sits in" is a rule whose evidence is entirely
   relational. */
function containment(objs) {
  var n = objs.length, inner = [], i, j, o, p, best, cnt;
  for (i = 0; i < n; i++) inner.push([-1, 0]);
  if (n > 60) return inner;
  for (i = 0; i < n; i++) {
    o = objs[i]; best = -1; cnt = 0;
    for (j = 0; j < n; j++) {
      if (i === j) continue;
      p = objs[j];
      if (p.r0 <= o.r0 && p.c0 <= o.c0 && p.r1 >= o.r1 && p.c1 >= o.c1 &&
          p.bbox_area() > o.bbox_area()) {
        if (best < 0 || p.bbox_area() < objs[best].bbox_area()) best = j;
      }
      if (o.r0 <= p.r0 && o.c0 <= p.c0 && o.r1 >= p.r1 && o.c1 >= p.c1 &&
          o.bbox_area() > p.bbox_area()) cnt++;
    }
    inner[i] = [best >= 0 ? objs[best].color : -1, cnt];
  }
  return inner;
}

function objectFeatures(o, objs, grid, shared, index) {
  var n = objs.length;
  if (!shared) shared = sharedStats(objs, grid);
  var sizes = shared.sizes, shapeCnt = shared.shape_cnt, colorCnt = shared.color_cnt;
  var ghh = shared.dims[0], gww = shared.dims[1];
  if (index === undefined || index === null) index = objs.indexOf(o);
  var cont = -1, ncont = 0;
  if (index >= 0 && shared.containment[index]) {
    cont = shared.containment[index][0];
    ncont = shared.containment[index][1];
  }
  return {
    container: cont,
    n_contains: ncont,
    color: o.color,
    size: o.size(),
    h: o.height(),
    w: o.width(),
    bbox: o.bbox_area(),
    square: o.is_square() ? 1 : 0,
    rect: o.is_rect() ? 1 : 0,
    holes: o.holes_count(),
    ncolors: G.csSize(o.colors()),
    border: o.touches_border() ? 1 : 0,
    size_rank: sizes.indexOf(o.size()),
    is_largest: o.size() === sizes[0] ? 1 : 0,
    is_smallest: o.size() === sizes[sizes.length - 1] ? 1 : 0,
    shape_freq: shapeCnt.get(o.norm_key()),
    shape_unique: shapeCnt.get(o.norm_key()) === 1 ? 1 : 0,
    color_freq: colorCnt.get(o.color),
    color_unique: colorCnt.get(o.color) === 1 ? 1 : 0,
    n_objects: n,
    r0: o.r0, c0: o.c0,
    row_band: o.r0 < ghh / 3 ? 0 : (o.r0 < 2 * ghh / 3 ? 1 : 2),
    col_band: o.c0 < gww / 3 ? 0 : (o.c0 < 2 * gww / 3 ? 1 : 2)
  };
}

function gridFromObjects(objs, h, w, bg) {
  var out = G.constGrid(h, w, bg), i, o, p, r, c, v, rr, cc;
  for (i = 0; i < objs.length; i++) {
    o = objs[i]; p = o.patch();
    for (r = 0; r < o.height(); r++) for (c = 0; c < o.width(); c++) {
      v = p[r][c];
      if (v !== null) {
        rr = o.r0 + r; cc = o.c0 + c;
        if (rr >= 0 && rr < h && cc >= 0 && cc < w) out[rr][cc] = v;
      }
    }
  }
  return out;
}

function histogramNoBg(g, bg) {
  var h = new Int32Array(G.NCOLORS), src = G.histogram(g), i;
  for (i = 0; i < G.NCOLORS; i++) h[i] = src[i];
  h[bg] = 0;
  return h;
}

function cropObj(grid, o) { return G.subgrid(grid, o.r0, o.c0, o.r1, o.c1); }

var O = {
  Obj: Obj, segConnected: segConnected, segByColor: segByColor, segCells: segCells,
  segConnectedGap: segConnectedGap, segLines: segLines,
  SEGMENTATIONS: SEGMENTATIONS, SEG_NAMES: SEG_NAMES, segment: segment,
  OBJ_RANKERS: OBJ_RANKERS, OBJ_RANKER_NAMES: OBJ_RANKER_NAMES,
  selectExtreme: selectExtreme, countBy: countBy,
  selectUniqueShape: selectUniqueShape, selectMajorityShape: selectMajorityShape,
  selectUniqueColor: selectUniqueColor, selectSymmetric: selectSymmetric,
  SELECTORS: SELECTORS, featuresOf: featuresOf, sharedStats: sharedStats,
  containment: containment, objectFeatures: objectFeatures,
  gridFromObjects: gridFromObjects, histogramNoBg: histogramNoBg,
  objectPatchGrid: function (o, bg) { return o.filled(bg === undefined ? 0 : bg); },
  cropObj: cropObj
};
