/* Port of engine/solvers/motion.py -- constraint-fitted and relational
 * per-object translations.
 *
 * "Everything red moves two down" is a rule no pixel rule and no in-place
 * object edit can express: the object survives, unchanged, somewhere else.
 * Displacements are recovered by matching each input object to the output
 * object with the same patch; identical objects are matched jointly through
 * shared displacement constraints rather than greedily to the nearest copy.
 */

(function () {
  var _h = mkHyp("objects");
  var _SEGS = ["c8", "m8", "c4"];
  var _KEYS = ["color", "size", "shape", "dims", "all"];

  function _patchKey(o) {
    var parts = [], r, c, row, vals;
    for (r = 0; r < o._patch.length; r++) {
      row = o._patch[r]; vals = [];
      for (c = 0; c < row.length; c++) vals.push(row[c] === null ? "n" : row[c]);
      parts.push(vals.join(","));
    }
    return parts.join("|");
  }

  function _keyOf(o, kind) {
    if (kind === "color") return "c" + o.color;
    if (kind === "size") return "s" + o.size();
    if (kind === "shape") return "m" + o.norm_key();
    if (kind === "dims") return "d" + o.height() + "x" + o.width();
    return "0";
  }

  function _apply(g, seg, kind, table, bg) {
    bg = G.bgOr(g, bg);
    var objs = O.segment(g, seg, bg);
    if (!objs.length || objs.length > 30) return null;
    var h = g.length, w = g[0].length, out = G.constGrid(h, w, bg), i, o, d, it, s, r, c, nr, nc;
    for (i = 0; i < objs.length; i++) {
      o = objs[i];
      d = table.get(_keyOf(o, kind));
      if (d === undefined) return null;
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        nr = r + d[0]; nc = c + d[1];
        if (!(nr >= 0 && nr < h && nc >= 0 && nc < w)) return null;
        if (out[nr][nc] !== bg) return null;
        out[nr][nc] = g[r][c];
        s = it.next();
      }
    }
    return out;
  }

  function _fit(ctx, seg, kind, bg) {
    var domains = new Map(), n = 0, changed = false, t, a, b, abg, ins, outs, i, j, byPatch, o, possible, k;
    for (t = 0; t < ctx.train.length; t++) {
      a = ctx.train[t][0]; b = ctx.train[t][1];
      if (a.length !== b.length || a[0].length !== b[0].length) return null;
      abg = G.bgOr(a, bg);
      ins = O.segment(a, seg, abg);
      outs = O.segment(b, seg, abg);
      if (!ins.length || ins.length > 30 || ins.length !== outs.length) return null;
      if (!G.gEq(a, b)) changed = true;
      byPatch = new Map();
      for (i = 0; i < outs.length; i++) {
        k = _patchKey(outs[i]);
        if (!byPatch.has(k)) byPatch.set(k, []);
        byPatch.get(k).push(outs[i]);
      }
      for (i = 0; i < ins.length; i++) {
        o = ins[i];
        /* Every occurrence of a property must admit the SAME displacement. */
        possible = new Map();
        var matches = byPatch.get(_patchKey(o)) || [];
        for (j = 0; j < matches.length; j++)
          possible.set((matches[j].r0 - o.r0) + "," + (matches[j].c0 - o.c0),
                       [matches[j].r0 - o.r0, matches[j].c0 - o.c0]);
        k = _keyOf(o, kind);
        if (domains.has(k)) {
          var cur = domains.get(k), inter = new Map();
          cur.forEach(function (v, kk) { if (possible.has(kk)) inter.set(kk, v); });
          domains.set(k, inter);
        } else domains.set(k, possible);
        if (!domains.get(k).size) return null;
        n += 1;
      }
    }
    if (!changed || !domains.size || domains.size * 2 > n) return null;
    var keys = [], choices = [];
    domains.forEach(function (v, kk) {
      keys.push(kk);
      var arr = [];
      v.forEach(function (d) { arr.push(d); });
      arr.sort(function (x, y) {
        var ax = Math.abs(x[0]) + Math.abs(x[1]), ay = Math.abs(y[0]) + Math.abs(y[1]);
        return (ax - ay) || (x[0] - y[0]) || (x[1] - y[1]);
      });
      choices.push(arr);
    });
    /* Bounded ambiguity search: repeated shapes must not explode the work. */
    var idx = new Array(keys.length), total = 1;
    for (i = 0; i < keys.length; i++) { idx[i] = 0; total *= choices[i].length; }
    var count = 0;
    while (count < 128 && count < total) {
      if (ctx.timed_out()) break;
      var table = new Map();
      for (i = 0; i < keys.length; i++) table.set(keys[i], choices[i][idx[i]]);
      var ok = true;
      for (t = 0; t < ctx.train.length; t++) {
        var p = _apply(ctx.train[t][0], seg, kind, table, bg);
        if (p === null || !G.gEq(p, ctx.train[t][1])) { ok = false; break; }
      }
      if (ok) return table;
      count++;
      for (i = keys.length - 1; i >= 0; i--) {
        idx[i]++;
        if (idx[i] < choices[i].length) break;
        idx[i] = 0;
      }
    }
    return null;
  }

  /* An anchor must be uniquely identifiable; traversal order is no evidence. */
  function _anchor(objs, selector) {
    var candidates = [], i, m;
    if (selector === "largest") {
      m = objs[0].size();
      for (i = 1; i < objs.length; i++) if (objs[i].size() > m) m = objs[i].size();
      for (i = 0; i < objs.length; i++) if (objs[i].size() === m) candidates.push(objs[i]);
    } else if (selector === "smallest") {
      m = objs[0].size();
      for (i = 1; i < objs.length; i++) if (objs[i].size() < m) m = objs[i].size();
      for (i = 0; i < objs.length; i++) if (objs[i].size() === m) candidates.push(objs[i]);
    } else {
      var col = parseInt(selector.split("#")[1], 10);
      for (i = 0; i < objs.length; i++) if (objs[i].color === col) candidates.push(objs[i]);
    }
    return candidates.length === 1 ? candidates[0] : null;
  }

  function _offset(start, size, anchorStart, anchorSize, mode) {
    var target;
    if (mode === "keep") return 0;
    if (mode === "near") target = anchorStart;
    else if (mode === "far") target = anchorStart + anchorSize - size;
    else if (mode === "center") {
      /* A half-cell centre gives two placements; do not break that tie. */
      if ((anchorSize - size) % 2) return null;
      target = anchorStart + (anchorSize - size) / 2;
    } else if (mode === "before") target = anchorStart - size;
    else if (mode === "after") target = anchorStart + anchorSize;
    else throw new Error("unknown alignment mode");
    return target - start;
  }

  function _applyRelative(g, seg, bg, selector, rowMode, colMode) {
    bg = G.bgOr(g, bg);
    var objs = O.segment(g, seg, bg);
    if (!(objs.length >= 2 && objs.length <= 30)) return null;
    var anchor = _anchor(objs, selector);
    if (anchor === null) return null;
    var h = g.length, w = g[0].length, out = G.constGrid(h, w, bg), occupied = new Set();
    var i, o, dr, dc, it, s, r, c, nr, nc;
    for (i = 0; i < objs.length; i++) {
      o = objs[i];
      if (o === anchor) { dr = 0; dc = 0; }
      else {
        dr = _offset(o.r0, o.height(), anchor.r0, anchor.height(), rowMode);
        dc = _offset(o.c0, o.width(), anchor.c0, anchor.width(), colMode);
      }
      if (dr === null || dc === null) return null;
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        nr = r + dr; nc = c + dc;
        if (!(nr >= 0 && nr < h && nc >= 0 && nc < w) || occupied.has(nr * 64 + nc)) return null;
        occupied.add(nr * 64 + nc);
        out[nr][nc] = g[r][c];
        s = it.next();
      }
    }
    return out;
  }

  /* Small reusable geometry grammar; accept only complete demonstrated rules. */
  function _relativeRules(ctx, bg) {
    var same = true, i, j, t;
    for (t = 0; t < ctx.train.length; t++) if (!G.gEq(ctx.train[t][0], ctx.train[t][1])) { same = false; break; }
    if (same) return [];
    var selectors = ["largest", "smallest"], pal = G.csList(ctx.in_palette());
    for (i = 0; i < pal.length; i++) if (pal[i] !== bg) selectors.push("color#" + pal[i]);
    var modes = ["near", "far", "center", "before", "after"], alignments = [];
    for (i = 0; i < modes.length; i++) alignments.push([modes[i], "keep"]);
    for (i = 0; i < modes.length; i++) alignments.push(["keep", modes[i]]);
    for (i = 0; i < modes.length; i++) for (j = 0; j < modes.length; j++) {
      var ra = (modes[i] === "before" || modes[i] === "after");
      var ca = (modes[j] === "before" || modes[j] === "after");
      if (ra !== ca) alignments.push([modes[i], modes[j]]);
    }
    var res = [], s, sel, k, scenes, ok, ins = ctx.inputs();
    for (s = 0; s < _SEGS.length; s++)
      for (sel = 0; sel < selectors.length; sel++) {
        scenes = []; ok = true;
        for (i = 0; i < ins.length; i++) scenes.push(O.segment(ins[i], _SEGS[s], G.bgOr(ins[i], bg)));
        for (i = 0; i < scenes.length; i++)
          if (!(scenes[i].length >= 2 && scenes[i].length <= 30) ||
              _anchor(scenes[i], selectors[sel]) === null) { ok = false; break; }
        if (!ok) continue;
        for (k = 0; k < alignments.length; k++) {
          if (ctx.timed_out() || res.length >= 12) return res;
          var rm = alignments[k][0], cm = alignments[k][1];
          var hyp = _h("align_" + _SEGS[s] + "_" + selectors[sel] + "_" + rm + "_" + cm + "_bg" + bg,
                       (function (sg, se, r2, c2) {
                         return function (g) { return _applyRelative(g, sg, bg, se, r2, c2); };
                       })(_SEGS[s], selectors[sel], rm, cm),
                       5.0 + (selectors[sel].indexOf("color#") === 0 ? 0.3 : 0.0) +
                       (rm !== "keep" && cm !== "keep" ? 0.4 : 0.0));
          if (hyp.fits(ctx.train)) res.push(hyp);
        }
      }
    return res;
  }

  function generate(ctx) {
    if (!ctx.same_shape()) return [];
    var res = [], bgs = ctx.bg_varies() ? [ctx.bg(), null] : [ctx.bg()], bi, s, k, t;
    for (bi = 0; bi < bgs.length; bi++) {
      for (s = 0; s < _SEGS.length; s++) {
        if (ctx.timed_out()) break;
        for (k = 0; k < _KEYS.length; k++) {
          try { t = _fit(ctx, _SEGS[s], _KEYS[k], bgs[bi]); } catch (e) { t = null; }
          if (t === null) continue;
          res.push(_h("move_" + _SEGS[s] + "_by_" + _KEYS[k],
                      (function (sg, kd, tb, bb) { return function (g) { return _apply(g, sg, kd, tb, bb); }; })(_SEGS[s], _KEYS[k], t, bgs[bi]),
                      4.0 + 0.1 * t.size));
        }
      }
      res = res.concat(_relativeRules(ctx, bgs[bi]));
    }
    return res;
  }

  defSolver("motion", "objects", generate);
})();
