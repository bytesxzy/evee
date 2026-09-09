/* Port of engine/grid.py -- core grid algebra.
 *
 * A Grid is an Array of Arrays of small ints (0-9). Python's tuples gave the
 * engine hashability and structural equality for free; here every place that
 * needed either uses gkey(), a cached string form, so grids stay plain arrays
 * and cell access stays fast.
 *
 * Cell coordinates are packed as r * 64 + c wherever Python used a (r, c)
 * tuple inside a set: grids never exceed 60x60, so the packing is total, and a
 * Set of ints replaces a frozenset of tuples with the same semantics.
 */

var NCOLORS = 10;

function enc(r, c) { return r * 64 + c; }
function decR(p) { return p >> 6; }
function decC(p) { return p & 63; }

/* Structural identity. Python compared tuples; every such comparison here goes
   through this cached key, so a grid is hashed and compared exactly once. */
function gkey(g) {
  var k = g.__k;
  if (k === undefined) {
    var parts = [], i;
    for (i = 0; i < g.length; i++) parts.push(g[i].join(","));
    k = parts.join(";");
    g.__k = k;
  }
  return k;
}

function gEq(a, b) {
  if (a === b) return true;
  if (a === null || b === null || a === undefined || b === undefined) return false;
  return gkey(a) === gkey(b);
}

/* Colour sets are 10-bit masks: the palette algebra below is set union,
   difference and subset, all of which are one machine instruction this way. */
function csAdd(m, v) { return m | (1 << v); }
function csHas(m, v) { return (m & (1 << v)) !== 0; }
function csUnion(a, b) { return a | b; }
function csDiff(a, b) { return a & ~b; }
function csSubset(a, b) { return (a & ~b) === 0; }
function csSize(m) { var n = 0; while (m) { n += m & 1; m >>= 1; } return n; }
function csList(m) { var o = [], v; for (v = 0; v < NCOLORS; v++) if (m & (1 << v)) o.push(v); return o; }
function csFrom(list) { var m = 0, i; for (i = 0; i < list.length; i++) m |= 1 << list[i]; return m; }

function dims(g) { return [g.length, g.length ? g[0].length : 0]; }
function gh(g) { return g.length; }
function gw(g) { return g.length ? g[0].length : 0; }
function area(g) { return g.length * (g.length ? g[0].length : 0); }

function valid(g) {
  if (!Array.isArray(g) || !g.length || !Array.isArray(g[0])) return false;
  var w = g[0].length, i;
  if (w === 0 || w > 60 || g.length > 60) return false;
  for (i = 0; i < g.length; i++)
    if (!Array.isArray(g[i]) || g[i].length !== w) return false;
  return true;
}

function isGrid(g) {
  if (!valid(g)) return false;
  var r, c, row, v;
  for (r = 0; r < g.length; r++) {
    row = g[r];
    for (c = 0; c < row.length; c++) {
      v = row[c];
      if (typeof v !== "number" || (v | 0) !== v || v < 0 || v >= NCOLORS) return false;
    }
  }
  return true;
}

function asGrid(value) {
  if (!Array.isArray(value)) throw new Error("grid must be an array of rows");
  var g = [], i;
  for (i = 0; i < value.length; i++) {
    if (!Array.isArray(value[i])) throw new Error("grid rows must be arrays");
    g.push(value[i].slice());
  }
  if (!isGrid(g))
    throw new Error("grid must be rectangular, 1..60 by 1..60, colours 0..9");
  return g;
}

function constGrid(h, w, c) {
  var out = [], r, row, i;
  for (r = 0; r < h; r++) { row = new Array(w); for (i = 0; i < w; i++) row[i] = c; out.push(row); }
  return out;
}

function copyGrid(g) {
  var out = [], r;
  for (r = 0; r < g.length; r++) out.push(g[r].slice());
  return out;
}

/* ---------------------------------------------------------- dihedral group */

function transpose(g) {
  var h = g.length, w = g[0].length, out = [], r, c, row;
  for (c = 0; c < w; c++) { row = new Array(h); for (r = 0; r < h; r++) row[r] = g[r][c]; out.push(row); }
  return out;
}

function flipH(g) { var out = [], r; for (r = 0; r < g.length; r++) out.push(g[r].slice().reverse()); return out; }
function flipV(g) { var out = [], r; for (r = g.length - 1; r >= 0; r--) out.push(g[r].slice()); return out; }
function rot90(g) {
  var h = g.length, w = g[0].length, out = [], r, c, row;
  for (c = 0; c < w; c++) { row = new Array(h); for (r = 0; r < h; r++) row[r] = g[h - 1 - r][c]; out.push(row); }
  return out;
}
function rot180(g) { return flipH(flipV(g)); }
function rot270(g) { var t = transpose(g); return t.reverse(); }
function antiTranspose(g) { var t = transpose(flipH(g)); return t.reverse(); }

var DIHEDRAL = [
  ["id", function (g) { return g; }],
  ["rot90", rot90],
  ["rot180", rot180],
  ["rot270", rot270],
  ["flip_h", flipH],
  ["flip_v", flipV],
  ["transpose", transpose],
  ["anti_transpose", antiTranspose]
];
var DIHEDRAL_MAP = {};
(function () { var i; for (i = 0; i < DIHEDRAL.length; i++) DIHEDRAL_MAP[DIHEDRAL[i][0]] = DIHEDRAL[i][1]; })();

function orbit(g) {
  var out = {}, i;
  for (i = 0; i < DIHEDRAL.length; i++) out[DIHEDRAL[i][0]] = DIHEDRAL[i][1](g);
  return out;
}

/* -------------------------------------------------------- colour statistics */

function histogram(g) {
  var h = new Int32Array(NCOLORS), r, c, row;
  for (r = 0; r < g.length; r++) { row = g[r]; for (c = 0; c < row.length; c++) h[row[c]]++; }
  return h;
}

function palette(g) {
  var m = 0, r, c, row;
  for (r = 0; r < g.length; r++) { row = g[r]; for (c = 0; c < row.length; c++) m |= 1 << row[c]; }
  return m;
}

/* max(h.items(), key=(count, -colour)): highest count, lowest colour on ties. */
function modalOf(h) {
  var best = -1, bn = -1, v;
  for (v = 0; v < NCOLORS; v++) if (h[v] > bn) { bn = h[v]; best = v; }
  return best;
}

function mostCommonColor(g) { return modalOf(histogram(g)); }

function leastCommonColor(g) {
  var h = histogram(g), best = -1, bn = Infinity, v;
  for (v = 0; v < NCOLORS; v++) if (h[v] > 0 && h[v] < bn) { bn = h[v]; best = v; }
  if (best < 0) { for (v = 0; v < NCOLORS; v++) if (h[v] < bn) { bn = h[v]; best = v; } }
  return best;
}

/* Python iterates every colour key present in the Counter, including zero
   counts only when they were inserted; a histogram over the grid only ever
   holds colours that occur, so min() there is over present colours. */
function background(g) {
  var h = histogram(g), n = area(g);
  if (h[0] * 4 >= n) return 0;
  var mc = modalOf(h);
  if (h[0] && h[0] >= 0.25 * h[mc]) return 0;
  return mc;
}

function bgOr(g, bg) { return (bg === null || bg === undefined) ? background(g) : bg; }

function countColor(g, c) {
  var n = 0, r, row, i;
  for (r = 0; r < g.length; r++) { row = g[r]; for (i = 0; i < row.length; i++) if (row[i] === c) n++; }
  return n;
}

function replaceColor(g, a, b) {
  var out = [], r, c, row, nr;
  for (r = 0; r < g.length; r++) {
    row = g[r]; nr = new Array(row.length);
    for (c = 0; c < row.length; c++) nr[c] = row[c] === a ? b : row[c];
    out.push(nr);
  }
  return out;
}

/* ``m`` is a plain object or Map from colour to colour; absent keys pass. */
function applyCmap(g, m) {
  var get = (m instanceof Map) ? function (v) { return m.has(v) ? m.get(v) : v; }
                               : function (v) { return m[v] === undefined ? v : m[v]; };
  var out = [], r, c, row, nr;
  for (r = 0; r < g.length; r++) {
    row = g[r]; nr = new Array(row.length);
    for (c = 0; c < row.length; c++) nr[c] = get(row[c]);
    out.push(nr);
  }
  return out;
}

/* ------------------------------------------------------ cropping / subgrids */

function bboxOf(cells) {
  var r0 = Infinity, r1 = -Infinity, c0 = Infinity, c1 = -Infinity, r, c, p;
  var it = (cells instanceof Set) ? cells.values() : cells[Symbol.iterator]();
  var step = it.next();
  while (!step.done) {
    p = step.value; r = p >> 6; c = p & 63;
    if (r < r0) r0 = r;
    if (r > r1) r1 = r;
    if (c < c0) c0 = c;
    if (c > c1) c1 = c;
    step = it.next();
  }
  return [r0, c0, r1, c1];
}

function subgrid(g, r0, c0, r1, c1) {
  var h = g.length, w = g.length ? g[0].length : 0, out = [], r;
  if (r0 < 0 || c0 < 0 || r1 >= h || c1 >= w || r1 < r0 || c1 < c0) return null;
  for (r = r0; r <= r1; r++) out.push(g[r].slice(c0, c1 + 1));
  return out;
}

function cropToContent(g, bg) {
  if (bg === null || bg === undefined) bg = background(g);
  var r0 = Infinity, r1 = -Infinity, c0 = Infinity, c1 = -Infinity, r, c, row, any = false;
  for (r = 0; r < g.length; r++) {
    row = g[r];
    for (c = 0; c < row.length; c++) if (row[c] !== bg) {
      any = true;
      if (r < r0) r0 = r;
      if (r > r1) r1 = r;
      if (c < c0) c0 = c;
      if (c > c1) c1 = c;
    }
  }
  if (!any) return null;
  return subgrid(g, r0, c0, r1, c1);
}

function trimBorder(g, n) {
  if (n === undefined) n = 1;
  var h = g.length, w = g[0].length;
  if (h <= 2 * n || w <= 2 * n) return null;
  return subgrid(g, n, n, h - 1 - n, w - 1 - n);
}

function pad(g, n, c) {
  var w = g[0].length, out = [], r, i, row, side = [];
  for (i = 0; i < n; i++) side.push(c);
  for (i = 0; i < n; i++) out.push(constGrid(1, w + 2 * n, c)[0]);
  for (r = 0; r < g.length; r++) { row = side.concat(g[r], side); out.push(row); }
  for (i = 0; i < n; i++) out.push(constGrid(1, w + 2 * n, c)[0]);
  return out;
}

function half(g, which) {
  var h = g.length, w = g[0].length, out = [], r;
  if (which === "top") return h >= 2 ? copyGrid(g.slice(0, Math.floor(h / 2))) : null;
  if (which === "bottom") return h >= 2 ? copyGrid(g.slice(Math.floor((h + 1) / 2))) : null;
  if (which === "left") {
    if (w < 2) return null;
    for (r = 0; r < h; r++) out.push(g[r].slice(0, Math.floor(w / 2)));
    return out;
  }
  if (which === "right") {
    if (w < 2) return null;
    for (r = 0; r < h; r++) out.push(g[r].slice(Math.floor((w + 1) / 2)));
    return out;
  }
  return null;
}

function quadrant(g, i) {
  var h = g.length, w = g[0].length, hh = Math.floor(h / 2), hw = Math.floor(w / 2), out = [], r;
  if (hh === 0 || hw === 0) return null;
  if (i === 0) { for (r = 0; r < hh; r++) out.push(g[r].slice(0, hw)); return out; }
  if (i === 1) { for (r = 0; r < hh; r++) out.push(g[r].slice(w - hw)); return out; }
  if (i === 2) { for (r = h - hh; r < h; r++) out.push(g[r].slice(0, hw)); return out; }
  for (r = h - hh; r < h; r++) out.push(g[r].slice(w - hw));
  return out;
}

/* ------------------------------------------------- scaling / tiling / joins */

function upscale(g, ky, kx) {
  if (ky < 1 || kx < 1 || g.length * ky > 60 || g[0].length * kx > 60) return null;
  var out = [], r, c, i, row, nr;
  for (r = 0; r < g.length; r++) {
    row = g[r]; nr = [];
    for (c = 0; c < row.length; c++) for (i = 0; i < kx; i++) nr.push(row[c]);
    for (i = 0; i < ky; i++) out.push(nr.slice());
  }
  return out;
}

function downscale(g, ky, kx) {
  var h = g.length, w = g[0].length;
  if (ky < 1 || kx < 1 || h % ky || w % kx) return null;
  var out = [], r, c, rr, cc, v, row;
  for (r = 0; r < h; r += ky) {
    row = [];
    for (c = 0; c < w; c += kx) {
      v = g[r][c];
      for (rr = r; rr < r + ky; rr++) for (cc = c; cc < c + kx; cc++)
        if (g[rr][cc] !== v) return null;
      row.push(v);
    }
    out.push(row);
  }
  return out;
}

function blockReduceMode(g, ky, kx) {
  var h = g.length, w = g[0].length;
  if (ky < 1 || kx < 1 || h % ky || w % kx) return null;
  var out = [], r, c, rr, cc, row, cnt;
  for (r = 0; r < h; r += ky) {
    row = [];
    for (c = 0; c < w; c += kx) {
      cnt = new Int32Array(NCOLORS);
      for (rr = r; rr < r + ky; rr++) for (cc = c; cc < c + kx; cc++) cnt[g[rr][cc]]++;
      row.push(modalOf(cnt));
    }
    out.push(row);
  }
  return out;
}

function blockReduceNonbg(g, ky, kx, bg) {
  var h = g.length, w = g[0].length;
  if (ky < 1 || kx < 1 || h % ky || w % kx) return null;
  var out = [], r, c, rr, cc, row, m, v;
  for (r = 0; r < h; r += ky) {
    row = [];
    for (c = 0; c < w; c += kx) {
      m = 0;
      for (rr = r; rr < r + ky; rr++) for (cc = c; cc < c + kx; cc++) {
        v = g[rr][cc];
        if (v !== bg) m |= 1 << v;
      }
      if (csSize(m) > 1) return null;
      row.push(m ? csList(m)[0] : bg);
    }
    out.push(row);
  }
  return out;
}

function tile(g, ky, kx) {
  var h = g.length, w = g[0].length;
  if (ky < 1 || kx < 1 || h * ky > 60 || w * kx > 60) return null;
  var body = [], out = [], r, i, nr;
  for (r = 0; r < h; r++) { nr = []; for (i = 0; i < kx; i++) nr = nr.concat(g[r]); body.push(nr); }
  for (i = 0; i < ky; i++) for (r = 0; r < h; r++) out.push(body[r].slice());
  return out;
}

function hconcat(a, b) {
  if (!a || !b || a.length !== b.length) return null;
  if (a[0].length + b[0].length > 60) return null;
  var out = [], r;
  for (r = 0; r < a.length; r++) out.push(a[r].concat(b[r]));
  return out;
}

function vconcat(a, b) {
  if (!a || !b || a[0].length !== b[0].length) return null;
  if (a.length + b.length > 60) return null;
  return copyGrid(a).concat(copyGrid(b));
}

function paste(base, patch, r0, c0) {
  var h = base.length, w = base[0].length, ph = patch.length, pw = patch[0].length;
  var out = copyGrid(base), r, c, rr, cc, prow, orow;
  for (r = 0; r < ph; r++) {
    rr = r0 + r;
    if (rr >= 0 && rr < h) {
      prow = patch[r]; orow = out[rr];
      for (c = 0; c < pw; c++) { cc = c0 + c; if (cc >= 0 && cc < w) orow[cc] = prow[c]; }
    }
  }
  return out;
}

function pasteMasked(base, patch, r0, c0, transparent) {
  var h = base.length, w = base[0].length, ph = patch.length, pw = patch[0].length;
  var out = copyGrid(base), r, c, rr, cc, prow, orow, v;
  for (r = 0; r < ph; r++) {
    rr = r0 + r;
    if (rr >= 0 && rr < h) {
      prow = patch[r]; orow = out[rr];
      for (c = 0; c < pw; c++) {
        v = prow[c]; cc = c0 + c;
        if (v !== transparent && cc >= 0 && cc < w) orow[cc] = v;
      }
    }
  }
  return out;
}

/* ------------------------------------------------------ structural analysis */

function uniformRows(g) {
  var out = [], r, c, row, ok;
  for (r = 0; r < g.length; r++) {
    row = g[r]; ok = true;
    for (c = 1; c < row.length; c++) if (row[c] !== row[0]) { ok = false; break; }
    if (ok) out.push(r);
  }
  return out;
}

function uniformCols(g) { return uniformRows(transpose(g)); }

function rowEq(a, b) {
  if (a.length !== b.length) return false;
  var i;
  for (i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function dedupRows(g) {
  var out = [g[0].slice()], r;
  for (r = 1; r < g.length; r++) if (!rowEq(g[r], out[out.length - 1])) out.push(g[r].slice());
  return out;
}

function dedupCols(g) { return transpose(dedupRows(transpose(g))); }
function dedup(g) { return dedupCols(dedupRows(g)); }

function rowPeriod(g) {
  var h = g.length, p, r, ok;
  for (p = 1; p <= h; p++) {
    ok = true;
    for (r = 0; r < h - p; r++) if (!rowEq(g[r], g[r + p])) { ok = false; break; }
    if (ok) return p;
  }
  return h;
}

function colPeriod(g) { return rowPeriod(transpose(g)); }

function symmetries(g, ignore) {
  var out = [], i, name, t, r, c, row, tr, ok;
  var useIgnore = (ignore !== null && ignore !== undefined);
  for (i = 0; i < DIHEDRAL.length; i++) {
    name = DIHEDRAL[i][0];
    if (name === "id") continue;
    t = DIHEDRAL[i][1](g);
    if (t.length !== g.length || t[0].length !== g[0].length) continue;
    if (!useIgnore) {
      if (gEq(t, g)) out.push(name);
    } else {
      ok = true;
      for (r = 0; r < g.length && ok; r++) {
        row = g[r]; tr = t[r];
        for (c = 0; c < row.length; c++)
          if (row[c] !== ignore && tr[c] !== ignore && row[c] !== tr[c]) { ok = false; break; }
      }
      if (ok) out.push(name);
    }
  }
  return out;
}

/* ---------------------------------------------------------- connectivity */

var N4 = [[-1, 0], [1, 0], [0, -1], [0, 1]];
var N8 = N4.concat([[-1, -1], [-1, 1], [1, -1], [1, 1]]);

/* Returns [[colourOrNull, Set<encodedCell>], ...] */
function floodRegions(g, bg, diag, sameColor) {
  if (sameColor === undefined) sameColor = true;
  var h = g.length, w = g[0].length, nb = diag ? N8 : N4;
  var seen = new Uint8Array(h * w), out = [], r, c, i, col, q, qi, cells, multi, cr, cc, nr, ncc, v, d;
  for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
    if (seen[r * w + c] || g[r][c] === bg) continue;
    col = g[r][c];
    q = [r * 64 + c]; qi = 0;
    seen[r * w + c] = 1;
    cells = []; multi = false;
    while (qi < q.length) {
      var p = q[qi++]; cr = p >> 6; cc = p & 63;
      cells.push(p);
      for (d = 0; d < nb.length; d++) {
        nr = cr + nb[d][0]; ncc = cc + nb[d][1];
        if (nr < 0 || nr >= h || ncc < 0 || ncc >= w || seen[nr * w + ncc]) continue;
        v = g[nr][ncc];
        if (v === bg) continue;
        if (sameColor && v !== col) continue;
        if (v !== col) multi = true;
        seen[nr * w + ncc] = 1;
        q.push(nr * 64 + ncc);
      }
    }
    out.push([(multi || !sameColor) ? null : col, new Set(cells)]);
  }
  return out;
}

function holes(g, bg, diag) {
  var h = g.length, w = g[0].length, nb = diag ? N8 : N4;
  var seen = new Uint8Array(h * w), q = [], qi = 0, r, c, d, cr, cc, nr, ncc, out = [];
  for (r = 0; r < h; r++) {
    for (var k = 0; k < 2; k++) {
      c = k ? w - 1 : 0;
      if (g[r][c] === bg && !seen[r * w + c]) { seen[r * w + c] = 1; q.push(r * 64 + c); }
    }
  }
  for (c = 0; c < w; c++) {
    for (var k2 = 0; k2 < 2; k2++) {
      r = k2 ? h - 1 : 0;
      if (g[r][c] === bg && !seen[r * w + c]) { seen[r * w + c] = 1; q.push(r * 64 + c); }
    }
  }
  while (qi < q.length) {
    var p = q[qi++]; cr = p >> 6; cc = p & 63;
    for (d = 0; d < nb.length; d++) {
      nr = cr + nb[d][0]; ncc = cc + nb[d][1];
      if (nr >= 0 && nr < h && ncc >= 0 && ncc < w && !seen[nr * w + ncc] && g[nr][ncc] === bg) {
        seen[nr * w + ncc] = 1; q.push(nr * 64 + ncc);
      }
    }
  }
  for (r = 0; r < h; r++) for (c = 0; c < w; c++)
    if (g[r][c] === bg && !seen[r * w + c]) out.push(r * 64 + c);
  return out;
}

function fillHoles(g, color, bg, diag) {
  if (bg === null || bg === undefined) bg = background(g);
  var hs = holes(g, bg, diag);
  if (!hs.length) return g;
  var out = copyGrid(g), i;
  for (i = 0; i < hs.length; i++) out[hs[i] >> 6][hs[i] & 63] = color;
  return out;
}

/* ------------------------------------------------------------ masks / logic */

function cellwiseOp(a, b, fn) {
  if (!a || !b || a.length !== b.length || a[0].length !== b[0].length) return null;
  var out = [], r, c, row;
  for (r = 0; r < a.length; r++) {
    row = new Array(a[r].length);
    for (c = 0; c < a[r].length; c++) row[c] = fn(a[r][c], b[r][c]);
    out.push(row);
  }
  return out;
}

function gravity(g, bg, direction) {
  bg = bgOr(g, bg);
  var h = g.length, w = g[0].length, r, c, vals, out = [], row;
  if (direction === "down" || direction === "up") {
    var cols = [];
    for (c = 0; c < w; c++) {
      vals = [];
      for (r = 0; r < h; r++) if (g[r][c] !== bg) vals.push(g[r][c]);
      var padn = [];
      for (r = 0; r < h - vals.length; r++) padn.push(bg);
      cols.push(direction === "down" ? padn.concat(vals) : vals.concat(padn));
    }
    for (r = 0; r < h; r++) { row = new Array(w); for (c = 0; c < w; c++) row[c] = cols[c][r]; out.push(row); }
    return out;
  }
  for (r = 0; r < h; r++) {
    vals = [];
    for (c = 0; c < w; c++) if (g[r][c] !== bg) vals.push(g[r][c]);
    var pd = [];
    for (c = 0; c < w - vals.length; c++) pd.push(bg);
    out.push(direction === "right" ? pd.concat(vals) : vals.concat(pd));
  }
  return out;
}

function translate(g, dr, dc, fill) {
  fill = bgOr(g, fill);
  var h = g.length, w = g[0].length, out = constGrid(h, w, fill), r, c, nr, nc;
  for (r = 0; r < h; r++) {
    nr = r + dr;
    if (nr >= 0 && nr < h) for (c = 0; c < w; c++) {
      nc = c + dc;
      if (nc >= 0 && nc < w) out[nr][nc] = g[r][c];
    }
  }
  return out;
}

function wrapTranslate(g, dr, dc) {
  var h = g.length, w = g[0].length, out = [], r, c, row;
  for (r = 0; r < h; r++) {
    row = new Array(w);
    for (c = 0; c < w; c++) row[c] = g[((r - dr) % h + h) % h][((c - dc) % w + w) % w];
    out.push(row);
  }
  return out;
}

var G = {
  NCOLORS: NCOLORS, enc: enc, decR: decR, decC: decC, gkey: gkey, gEq: gEq,
  csAdd: csAdd, csHas: csHas, csUnion: csUnion, csDiff: csDiff, csSubset: csSubset,
  csSize: csSize, csList: csList, csFrom: csFrom,
  dims: dims, gh: gh, gw: gw, area: area, valid: valid, isGrid: isGrid, asGrid: asGrid,
  constGrid: constGrid, copyGrid: copyGrid,
  transpose: transpose, flipH: flipH, flipV: flipV, rot90: rot90, rot180: rot180,
  rot270: rot270, antiTranspose: antiTranspose, DIHEDRAL: DIHEDRAL,
  DIHEDRAL_MAP: DIHEDRAL_MAP, orbit: orbit,
  histogram: histogram, palette: palette, modalOf: modalOf,
  mostCommonColor: mostCommonColor, leastCommonColor: leastCommonColor,
  background: background, bgOr: bgOr, countColor: countColor,
  replaceColor: replaceColor, applyCmap: applyCmap,
  bboxOf: bboxOf, subgrid: subgrid, cropToContent: cropToContent,
  trimBorder: trimBorder, pad: pad, half: half, quadrant: quadrant,
  upscale: upscale, downscale: downscale, blockReduceMode: blockReduceMode,
  blockReduceNonbg: blockReduceNonbg, tile: tile, hconcat: hconcat,
  vconcat: vconcat, paste: paste, pasteMasked: pasteMasked,
  uniformRows: uniformRows, uniformCols: uniformCols, rowEq: rowEq,
  dedupRows: dedupRows, dedupCols: dedupCols, dedup: dedup,
  rowPeriod: rowPeriod, colPeriod: colPeriod, symmetries: symmetries,
  N4: N4, N8: N8, floodRegions: floodRegions, holes: holes, fillHoles: fillHoles,
  cellwise: cellwiseOp, gravity: gravity, translate: translate,
  wrapTranslate: wrapTranslate
};
