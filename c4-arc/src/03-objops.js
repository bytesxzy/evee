/* Port of engine/objops.py -- object-level operators for the enumerator.
 *
 * The synthesiser composes grid-to-grid functions and its library was written
 * at the level of pixels and whole grids, so nothing in it could say *which
 * thing* to act on. Each operator here is a selector crossed with an action,
 * still an ordinary grid-to-grid function, so it drops straight into the
 * existing search and therefore into every family built on that search.
 */

var OBJOPS_ENABLED = true;

var _OO_SEG_CACHE = new Map();
var _OO_SEG_KEYS = [];

function ooSegment(g, seg, bg) {
  var key = G.gkey(g) + "#" + seg + "#" + bg;
  var hit = _OO_SEG_CACHE.get(key);
  if (hit !== undefined) return hit;
  var objs = O.segment(g, seg, bg);
  _OO_SEG_CACHE.set(key, objs);
  _OO_SEG_KEYS.push(key);
  if (_OO_SEG_KEYS.length > 400) {
    var i;
    for (i = 0; i < 200; i++) _OO_SEG_CACHE.delete(_OO_SEG_KEYS[i]);
    _OO_SEG_KEYS.splice(0, 200);
  }
  return objs;
}

/* ------------------------------- selectors: scene -> the objects acted on */

function _ooSym(o) {
  var p = o.filled(0);
  return G.gEq(p, G.flipH(p)) || G.gEq(p, G.flipV(p));
}

function _ooSelectors() {
  function pickBy(objs, valfn, cmp) {
    var i, m = valfn(objs[0]);
    for (i = 1; i < objs.length; i++) m = cmp(valfn(objs[i]), m) ? valfn(objs[i]) : m;
    var out = [];
    for (i = 0; i < objs.length; i++) if (valfn(objs[i]) === m) out.push(objs[i]);
    return out;
  }
  var gt = function (a, b) { return a > b; }, lt = function (a, b) { return a < b; };
  function counted(objs, keyfn) { return O.countBy(objs, keyfn); }

  return [
    ["all", 0.0, function (objs) { return objs.slice(); }],
    ["big", 0.4, function (objs) { return pickBy(objs, function (o) { return o.size(); }, gt); }],
    ["small", 0.4, function (objs) { return pickBy(objs, function (o) { return o.size(); }, lt); }],
    ["wide", 0.6, function (objs) { return pickBy(objs, function (o) { return o.bbox_area(); }, gt); }],
    ["ucol", 0.5, function (objs) {
      var c = counted(objs, function (o) { return o.color; }), out = [], i;
      for (i = 0; i < objs.length; i++) if (c.get(objs[i].color) === 1) out.push(objs[i]);
      return out;
    }],
    ["mcol", 0.7, function (objs) {
      var c = counted(objs, function (o) { return o.color; }), m = 0, out = [], i;
      c.forEach(function (v) { if (v > m) m = v; });
      for (i = 0; i < objs.length; i++) if (c.get(objs[i].color) === m) out.push(objs[i]);
      return out;
    }],
    ["ushp", 0.5, function (objs) {
      var c = counted(objs, function (o) { return o.norm_key(); }), out = [], i;
      for (i = 0; i < objs.length; i++) if (c.get(objs[i].norm_key()) === 1) out.push(objs[i]);
      return out;
    }],
    ["mshp", 0.6, function (objs) {
      var c = counted(objs, function (o) { return o.norm_key(); }), m = 0, out = [], i;
      c.forEach(function (v) { if (v > m) m = v; });
      for (i = 0; i < objs.length; i++) if (c.get(objs[i].norm_key()) === m) out.push(objs[i]);
      return out;
    }],
    ["usz", 0.7, function (objs) {
      var c = counted(objs, function (o) { return o.size(); }), out = [], i;
      for (i = 0; i < objs.length; i++) if (c.get(objs[i].size()) === 1) out.push(objs[i]);
      return out;
    }],
    ["hole", 0.5, function (objs) { return objs.filter(function (o) { return o.holes_count() > 0; }); }],
    ["nohole", 0.7, function (objs) { return objs.filter(function (o) { return o.holes_count() === 0; }); }],
    ["edge", 0.5, function (objs) { return objs.filter(function (o) { return o.touches_border(); }); }],
    ["in", 0.5, function (objs) { return objs.filter(function (o) { return !o.touches_border(); }); }],
    ["sq", 0.6, function (objs) { return objs.filter(function (o) { return o.is_square(); }); }],
    ["rc", 0.7, function (objs) { return objs.filter(function (o) { return o.is_rect(); }); }],
    ["sym", 0.7, function (objs) { return objs.filter(_ooSym); }],
    ["multi", 0.7, function (objs) { return objs.filter(function (o) { return G.csSize(o.colors()) > 1; }); }]
  ];
}

var _OO_DIRS = { u: [-1, 0], d: [1, 0], l: [0, -1], r: [0, 1] };

/* ------------------------- actions: (grid, chosen, all, bg) -> grid */

function _actDelete(g, sel, objs, bg) {
  var out = G.copyGrid(g), i, it, s;
  for (i = 0; i < sel.length; i++) {
    it = sel[i].cells.values(); s = it.next();
    while (!s.done) { out[s.value >> 6][s.value & 63] = bg; s = it.next(); }
  }
  return out;
}

function _actKeep(g, sel, objs, bg) {
  var keep = new Set(), i, it, s;
  for (i = 0; i < sel.length; i++) {
    it = sel[i].cells.values(); s = it.next();
    while (!s.done) { keep.add(s.value); s = it.next(); }
  }
  var h = g.length, w = g[0].length, out = [], r, c, row;
  for (r = 0; r < h; r++) {
    row = new Array(w);
    for (c = 0; c < w; c++) row[c] = keep.has(r * 64 + c) ? g[r][c] : bg;
    out.push(row);
  }
  return out;
}

function _actCrop(g, sel, objs, bg) {
  if (!sel.length) return null;
  var r0 = Infinity, c0 = Infinity, r1 = -Infinity, c1 = -Infinity, i;
  for (i = 0; i < sel.length; i++) {
    if (sel[i].r0 < r0) r0 = sel[i].r0;
    if (sel[i].c0 < c0) c0 = sel[i].c0;
    if (sel[i].r1 > r1) r1 = sel[i].r1;
    if (sel[i].c1 > c1) c1 = sel[i].c1;
  }
  return G.subgrid(g, r0, c0, r1, c1);
}

function _actCropmask(g, sel, objs, bg) {
  if (sel.length !== 1) return null;
  return sel[0].filled(bg);
}

function _actRecolor(color) {
  return function (g, sel, objs, bg) {
    var out = G.copyGrid(g), i, it, s;
    for (i = 0; i < sel.length; i++) {
      it = sel[i].cells.values(); s = it.next();
      while (!s.done) { out[s.value >> 6][s.value & 63] = color; s = it.next(); }
    }
    return out;
  };
}

function _actFill(color) {
  return function (g, sel, objs, bg) {
    var out = G.copyGrid(g), i, cells, j, touched = false;
    for (i = 0; i < sel.length; i++) {
      cells = _ooInterior(sel[i]);
      for (j = 0; j < cells.length; j++) { out[cells[j] >> 6][cells[j] & 63] = color; touched = true; }
    }
    return touched ? out : null;
  };
}

function _actBox(color) {
  return function (g, sel, objs, bg) {
    var out = G.copyGrid(g), i, r, c, o;
    for (i = 0; i < sel.length; i++) {
      o = sel[i];
      for (r = o.r0; r <= o.r1; r++) for (c = o.c0; c <= o.c1; c++) out[r][c] = color;
    }
    return out;
  };
}

function _actOutline(color) {
  return function (g, sel, objs, bg) {
    var h = g.length, w = g[0].length, out = G.copyGrid(g), touched = false;
    var i, it, s, r, c, dr, dc, rr, cc, o;
    for (i = 0; i < sel.length; i++) {
      o = sel[i]; it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        for (dr = -1; dr <= 1; dr++) for (dc = -1; dc <= 1; dc++) {
          rr = r + dr; cc = c + dc;
          if (rr >= 0 && rr < h && cc >= 0 && cc < w &&
              !o.cells.has(rr * 64 + cc) && g[rr][cc] === bg) {
            out[rr][cc] = color; touched = true;
          }
        }
        s = it.next();
      }
    }
    return touched ? out : null;
  };
}

function _toEdge(o, h, w, dr, dc) {
  if (dr < 0) return o.r0;
  if (dr > 0) return h - 1 - o.r1;
  if (dc < 0) return o.c0;
  return w - 1 - o.c1;
}

function _untilBlocked(o, h, w, dr, dc, blocked) {
  var limit = _toEdge(o, h, w, dr, dc), k, it, s, r, c;
  for (k = 1; k <= limit; k++) {
    it = o.cells.values(); s = it.next();
    while (!s.done) {
      r = (s.value >> 6) + dr * k; c = (s.value & 63) + dc * k;
      if (blocked.has(r * 64 + c)) return k - 1;
      s = it.next();
    }
  }
  return limit;
}

function _actMove(d, mode) {
  var dr = _OO_DIRS[d][0], dc = _OO_DIRS[d][1];
  return function (g, sel, objs, bg) {
    var h = g.length, w = g[0].length, blocked = new Set(), i, it, s, o, k;
    for (i = 0; i < objs.length; i++) {
      if (sel.indexOf(objs[i]) >= 0) continue;
      it = objs[i].cells.values(); s = it.next();
      while (!s.done) { blocked.add(s.value); s = it.next(); }
    }
    var out = G.copyGrid(g), draws = [], r, c;
    for (i = 0; i < sel.length; i++) {
      o = sel[i];
      k = (mode === "edge") ? _toEdge(o, h, w, dr, dc) : _untilBlocked(o, h, w, dr, dc, blocked);
      if (k <= 0) continue;
      it = o.cells.values(); s = it.next();
      while (!s.done) { out[s.value >> 6][s.value & 63] = bg; s = it.next(); }
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        draws.push([r + dr * k, c + dc * k, g[r][c]]);
        s = it.next();
      }
    }
    if (!draws.length) return null;
    for (i = 0; i < draws.length; i++) {
      r = draws[i][0]; c = draws[i][1];
      if (!(r >= 0 && r < h && c >= 0 && c < w)) return null;
      out[r][c] = draws[i][2];
    }
    return out;
  };
}

function _ooInterior(o) {
  if (o._interior !== null && o._interior !== undefined) return o._interior;
  var hh = o.r1 - o.r0 + 1, ww = o.c1 - o.c0 + 1, inside = [], r, c, row;
  for (r = 0; r < hh; r++) { row = new Array(ww); for (c = 0; c < ww; c++) row[c] = true; inside.push(row); }
  var stack = [];
  for (r = 0; r < hh; r++) { stack.push([r, 0]); stack.push([r, ww - 1]); }
  for (c = 0; c < ww; c++) { stack.push([0, c]); stack.push([hh - 1, c]); }
  var cur;
  while (stack.length) {
    cur = stack.pop(); r = cur[0]; c = cur[1];
    if (!(r >= 0 && r < hh && c >= 0 && c < ww) || !inside[r][c]) continue;
    if (o.cells.has((r + o.r0) * 64 + (c + o.c0))) continue;
    inside[r][c] = false;
    stack.push([r + 1, c]); stack.push([r - 1, c]); stack.push([r, c + 1]); stack.push([r, c - 1]);
  }
  var out = [];
  for (r = 0; r < hh; r++) for (c = 0; c < ww; c++)
    if (inside[r][c] && !o.cells.has((r + o.r0) * 64 + (c + o.c0)))
      out.push((r + o.r0) * 64 + (c + o.c0));
  o._interior = out;
  return out;
}

/* Paint colours come from the task, never from the full palette. */
function _ooActions(ctx) {
  var news = G.csList(ctx.new_colors()).slice(0, 2);
  var counts = new Int32Array(G.NCOLORS), order = [], outs = ctx.outputs(), i, r, c, row, v;
  for (i = 0; i < outs.length; i++)
    for (r = 0; r < outs[i].length; r++) {
      row = outs[i][r];
      for (c = 0; c < row.length; c++) {
        v = row[c];
        if (!counts[v]) order.push(v);
        counts[v]++;
      }
    }
  order.sort(function (a, b) { return counts[b] - counts[a]; });
  var hot = order.slice(0, 4), paints = [];
  for (i = 0; i < news.length; i++) if (paints.indexOf(news[i]) < 0) paints.push(news[i]);
  for (i = 0; i < hot.length; i++) if (paints.indexOf(hot[i]) < 0) paints.push(hot[i]);
  paints = paints.slice(0, 4);
  var acts = [["del", 0.6, _actDelete], ["only", 0.6, _actKeep],
              ["crop", 0.8, _actCrop], ["cropm", 1.0, _actCropmask]];
  for (i = 0; i < Math.min(3, paints.length); i++)
    acts.push(["rec" + paints[i], 0.9, _actRecolor(paints[i])]);
  for (i = 0; i < Math.min(2, paints.length); i++)
    acts.push(["fil" + paints[i], 1.0, _actFill(paints[i])]);
  for (i = 0; i < Math.min(1, paints.length); i++) {
    acts.push(["box" + paints[i], 1.1, _actBox(paints[i])]);
    acts.push(["out" + paints[i], 1.1, _actOutline(paints[i])]);
  }
  var dirs = "udlr";
  for (i = 0; i < 4; i++) {
    acts.push(["slide" + dirs[i], 1.0, _actMove(dirs[i], "block")]);
    acts.push(["snap" + dirs[i], 1.1, _actMove(dirs[i], "edge")]);
  }
  return acts;
}

function _ooMk(seg, bg, selFn, actFn, cap) {
  return function (g) {
    var b = G.bgOr(g, bg);
    var objs = ooSegment(g, seg, b);
    if (!objs.length || objs.length > cap) return null;
    var sel = selFn(objs);
    if (!sel.length) return null;
    var out = actFn(g, sel, objs, b);
    if (out === null || out === undefined || !G.valid(out)) return null;
    return G.gEq(out, g) ? null : out;
  };
}

/* (name, cost, fn) triples: one per selector x action x segmentation. */
function objectOps(ctx, cap, segs) {
  if (!OBJOPS_ENABLED) return [];
  if (cap === undefined) cap = 28;
  if (segs === undefined) segs = ["c8"];
  var sels = _ooSelectors(), acts = _ooActions(ctx);
  var bg = ctx.bg_varies() ? null : ctx.bg();
  var ops = [], si, i, j;
  for (si = 0; si < segs.length; si++)
    for (i = 0; i < sels.length; i++)
      for (j = 0; j < acts.length; j++)
        ops.push([segs[si] + "." + sels[i][0] + "." + acts[j][0],
                  1.6 + sels[i][1] + acts[j][1],
                  _ooMk(segs[si], bg, sels[i][2], acts[j][2], cap)]);
  return ops;
}

var OPS = {
  get ENABLED() { return OBJOPS_ENABLED; },
  set ENABLED(v) { OBJOPS_ENABLED = v; },
  objectOps: objectOps, interior: _ooInterior, segment: ooSegment
};
