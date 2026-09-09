/* Port of engine/solvers/objproc.py -- per-object process induction over a
 * compositional action algebra.
 *
 * objects_map learns "which colour does each object become"; that is one
 * action family out of many. Here each object proposes actions from a wider
 * algebra -- recolour, erase, bounding-box paints, halo, hole fill, translate,
 * slide-until-blocked, snap-to-edge, in-place dihedral, ray casting, stencils --
 * keeping only those whose local effect matches the demonstrated output, and
 * then a low-capacity decision function from features to action must be
 * consistent with every object of every training pair simultaneously.
 */

var OBJPROC = {};

(function () {
  var _h = mkHyp("objects");
  var _SEGS = ["c4", "c8", "m4", "m8", "color", "g2m", "rows", "cols"];
  var _FEATURE_KEYS = ["color", "size", "h", "w", "bbox", "square", "rect", "holes",
    "ncolors", "border", "size_rank", "is_largest", "is_smallest",
    "shape_freq", "shape_unique", "color_freq", "color_unique",
    "row_band", "col_band", "container", "n_contains"];
  var _PAIR_KEYS = ["color", "size_rank", "shape_unique", "is_largest", "holes",
    "border", "square", "color_unique"];
  var _SHAPE_KEY = "__shape__";
  var _DIRS = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };

  /* structural cost of each action family; cheap actions are preferred when
     several explain the same evidence */
  var _ACTION_COST = {
    keep: 0.0, del: 0.6, solid: 0.8, bbox: 1.2, bboxout: 1.4,
    bboxin: 1.4, halo: 1.4, fill: 1.2, move: 1.6, slide: 1.3,
    edge: 1.3, geom: 1.5, ray: 1.6, rays4: 1.6, cross: 1.6,
    patch: 3.2, toward: 1.4, step: 1.5, into: 1.6,
    copy: 1.7, reflect: 1.5
  };

  function actKey(a) {
    var parts = [], i, v;
    for (i = 0; i < a.length; i++) {
      v = a[i];
      if (Array.isArray(v)) {
        if (v.length && v[0] === "rel") parts.push("rel." + v[1]);
        else parts.push("[" + G.gkey(v) + "]");
      } else parts.push(String(v));
    }
    return parts.join(":");
  }

  /* Chebyshev distance between two cell sets, cheap bbox bound first. */
  function _dist(a, b) {
    var dr = Math.max(a.r0 - b.r1, b.r0 - a.r1, 0);
    var dc = Math.max(a.c0 - b.c1, b.c0 - a.c1, 0);
    var lo = Math.max(dr, dc);
    if (lo > 0 && (a.cells.size > 60 || b.cells.size > 60)) return lo;
    var best = null, it1 = a.cells.values(), s1 = it1.next(), it2, s2, d, r1, c1;
    while (!s1.done) {
      r1 = s1.value >> 6; c1 = s1.value & 63;
      it2 = b.cells.values(); s2 = it2.next();
      while (!s2.done) {
        d = Math.max(Math.abs(r1 - (s2.value >> 6)), Math.abs(c1 - (s2.value & 63)));
        if (best === null || d < best) {
          best = d;
          if (best <= lo) return best;
        }
        s2 = it2.next();
      }
      s1 = it1.next();
    }
    return best === null ? 1000000 : best;
  }

  /* Turn a relational colour reference into a concrete colour: "paint the
     shape the colour of the lone marker" cannot be written with a literal. */
  function _resolve(color, scene, o) {
    if (!Array.isArray(color)) return color;
    var kind = color[1], objs = (scene && scene.objs) || [], i, pick, counts, cands;
    if (!objs.length) return null;
    if (kind === "big") {
      pick = objs[0];
      for (i = 1; i < objs.length; i++)
        if (objs[i].size() > pick.size() ||
            (objs[i].size() === pick.size() && (-objs[i].r0 > -pick.r0 ||
             (-objs[i].r0 === -pick.r0 && -objs[i].c0 > -pick.c0)))) pick = objs[i];
    } else if (kind === "small") {
      pick = objs[0];
      for (i = 1; i < objs.length; i++)
        if (objs[i].size() < pick.size() ||
            (objs[i].size() === pick.size() && (objs[i].r0 < pick.r0 ||
             (objs[i].r0 === pick.r0 && objs[i].c0 < pick.c0)))) pick = objs[i];
    } else if (kind === "ucol" || kind === "ushp" || kind === "usz") {
      var f = kind === "ucol" ? function (x) { return x.color; }
            : kind === "ushp" ? function (x) { return x.norm_key(); }
            : function (x) { return x.size(); };
      counts = O.countBy(objs, f);
      cands = [];
      for (i = 0; i < objs.length; i++) if (counts.get(f(objs[i])) === 1) cands.push(objs[i]);
      if (cands.length !== 1) return null;
      pick = cands[0];
    } else if (kind === "near" || kind === "far" || kind === "neardiff" || kind === "nearbig") {
      if (!o || objs.length < 2) return null;
      var others = [];
      for (i = 0; i < objs.length; i++) if (objs[i] !== o) others.push(objs[i]);
      if (kind === "neardiff") others = others.filter(function (x) { return x.color !== o.color; });
      else if (kind === "nearbig") others = others.filter(function (x) { return x.size() > o.size(); });
      if (!others.length || others.length > 40) return null;
      var ds = [], want = null;
      for (i = 0; i < others.length; i++) ds.push([_dist(o, others[i]), others[i]]);
      for (i = 0; i < ds.length; i++)
        if (want === null || (kind === "far" ? ds[i][0] > want : ds[i][0] < want)) want = ds[i][0];
      var hits = 0;
      for (i = 0; i < ds.length; i++) if (ds[i][0] === want) hits |= 1 << ds[i][1].color;
      if (G.csSize(hits) !== 1) return null;   /* ambiguous anchor is not a rule */
      return G.csList(hits)[0];
    } else return null;
    return pick.color;
  }

  var _REL_COLORS = [["rel", "big"], ["rel", "small"], ["rel", "ucol"],
    ["rel", "ushp"], ["rel", "usz"], ["rel", "near"], ["rel", "far"],
    ["rel", "neardiff"], ["rel", "nearbig"]];

  /* Cells of the object's bounding box enclosed by it (its holes). */
  function _interior(o, g, bg) {
    if (o._procInterior !== undefined && o._procInterior !== null) return o._procInterior;
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
    o._procInterior = out;
    return out;
  }

  function _pickAnchor(objs, o, kind) {
    var others = [], i;
    for (i = 0; i < objs.length; i++) if (objs[i] !== o) others.push(objs[i]);
    if (!others.length) return null;
    var best, cands, ds;
    if (kind === "big") {
      best = others[0];
      for (i = 1; i < others.length; i++)
        if (others[i].size() > best.size() ||
            (others[i].size() === best.size() && (-others[i].r0 > -best.r0 ||
             (-others[i].r0 === -best.r0 && -others[i].c0 > -best.c0)))) best = others[i];
      return best;
    }
    if (kind === "small") {
      best = others[0];
      for (i = 1; i < others.length; i++)
        if (others[i].size() < best.size() ||
            (others[i].size() === best.size() && (others[i].r0 < best.r0 ||
             (others[i].r0 === best.r0 && others[i].c0 < best.c0)))) best = others[i];
      return best;
    }
    if (kind === "diff") {
      cands = others.filter(function (x) { return x.color !== o.color; });
      if (!cands.length) return null;
      ds = cands.map(function (x) { return [_dist(o, x), x.r0, x.c0, x]; });
      ds.sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]); });
      if (ds.length > 1 && ds[0][0] === ds[1][0] && ds[0][3].color !== ds[1][3].color) return null;
      return ds[0][3];
    }
    if (kind === "bigger") {
      cands = others.filter(function (x) { return x.size() > o.size(); });
      if (!cands.length) return null;
      ds = cands.map(function (x) { return [_dist(o, x), x.r0, x.c0, x]; });
      ds.sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]); });
      return ds[0][3];
    }
    return null;
  }

  /* Displacement that carries ``o`` to its anchor: slide until you touch it,
     take one step, or land inside it. */
  function _anchorOffset(g, o, act, bg, objs) {
    var kind = act[0], anchor = _pickAnchor(objs, o, act[1]);
    if (anchor === null) return null;
    if (kind === "into")
      return [Math.floor((anchor.r0 + anchor.r1 - o.r0 - o.r1) / 2),
              Math.floor((anchor.c0 + anchor.c1 - o.c0 - o.c1) / 2)];
    var overlapCols = o.c0 <= anchor.c1 && anchor.c0 <= o.c1;
    var overlapRows = o.r0 <= anchor.r1 && anchor.r0 <= o.r1;
    var sr = (anchor.r0 + anchor.r1) - (o.r0 + o.r1);
    var sc = (anchor.c0 + anchor.c1) - (o.c0 + o.c1);
    var dr, dc;
    if (overlapCols && !overlapRows) { dr = sr > 0 ? 1 : -1; dc = 0; }
    else if (overlapRows && !overlapCols) { dr = 0; dc = sc > 0 ? 1 : -1; }
    else if (!overlapRows && !overlapCols) { dr = sr > 0 ? 1 : -1; dc = sc > 0 ? 1 : -1; }
    else return null;
    if (kind === "step") return [dr, dc];
    var h = g.length, w = g[0].length, blocked = new Set(), i, it, s;
    for (i = 0; i < objs.length; i++) {
      if (objs[i] === o) continue;
      it = objs[i].cells.values(); s = it.next();
      while (!s.done) { blocked.add(s.value); s = it.next(); }
    }
    var k = 0, n, r, c, rr, cc, bad;
    for (;;) {
      n = k + 1; bad = false;
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        rr = r + dr * n; cc = c + dc * n;
        if (!(rr >= 0 && rr < h && cc >= 0 && cc < w) || blocked.has(rr * 64 + cc)) { bad = true; break; }
        s = it.next();
      }
      if (bad) return [dr * k, dc * k];
      k = n;
      if (k > h + w) return [0, 0];
    }
  }

  function _slideOffset(g, o, d, bg) {
    var dr = _DIRS[d][0], dc = _DIRS[d][1], h = g.length, w = g[0].length;
    var k = 0, n, it, s, r, c, rr, cc, bad;
    for (;;) {
      n = k + 1; bad = false;
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        rr = r + dr * n; cc = c + dc * n;
        if (!(rr >= 0 && rr < h && cc >= 0 && cc < w)) { bad = true; break; }
        if (!o.cells.has(rr * 64 + cc) && g[rr][cc] !== bg) { bad = true; break; }
        s = it.next();
      }
      if (bad) return [dr * k, dc * k];
      k = n;
      if (k > h + w) return [0, 0];
    }
  }

  function _edgeOffset(g, o, d) {
    var h = g.length, w = g[0].length;
    if (d === "up") return [-o.r0, 0];
    if (d === "down") return [h - 1 - o.r1, 0];
    if (d === "left") return [0, -o.c0];
    return [0, w - 1 - o.c1];
  }

  var _GEOMS = { rot180: G.rot180, flip_h: G.flipH, flip_v: G.flipV,
                 rot90: G.rot90, rot270: G.rot270, transpose: G.transpose };

  function _reflectWrites(g, o, axis, bg, h, w) {
    var out = new Map(), it = o.cells.values(), s = it.next(), r, c, rr, cc;
    while (!s.done) {
      r = s.value >> 6; c = s.value & 63;
      if (axis === "h") { rr = r; cc = w - 1 - c; }
      else if (axis === "v") { rr = h - 1 - r; cc = c; }
      else if (axis === "both") { rr = h - 1 - r; cc = w - 1 - c; }
      else if (axis === "d" && h === w) { rr = c; cc = r; }
      else return null;
      if (!o.cells.has(rr * 64 + cc)) out.set(rr * 64 + cc, g[r][c]);
      s = it.next();
    }
    return out.size ? out : null;
  }

  function _geomWrites(g, o, name, bg) {
    var fn = _GEOMS[name];
    if (!fn) return null;
    var hh = o.height(), ww = o.width();
    if ((name === "rot90" || name === "rot270" || name === "transpose") && hh !== ww) return null;
    var patch = o.patch(), filled = o.filled(bg), mask = o.mask();
    var tp = fn(filled), tm = fn(mask);
    if (tp.length !== hh || tp[0].length !== ww) return null;
    var out = new Map(), it = o.cells.values(), s = it.next(), r, c;
    while (!s.done) { out.set(s.value, bg); s = it.next(); }
    for (r = 0; r < hh; r++) for (c = 0; c < ww; c++)
      if (tm[r][c]) out.set((o.r0 + r) * 64 + (o.c0 + c), tp[r][c]);
    return out;
  }

  function _rayWrites(g, o, act, bg, h, w) {
    var kind = act[0], dirs, c;
    if (kind === "ray") { dirs = [act[1]]; c = act[2]; }
    else { dirs = ["up", "down", "left", "right"]; c = act[1]; }
    var out = new Map(), d, dr, dc, it, s, r, cc, rr, c2;
    for (d = 0; d < dirs.length; d++) {
      dr = _DIRS[dirs[d]][0]; dc = _DIRS[dirs[d]][1];
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; cc = s.value & 63;
        rr = r + dr; c2 = cc + dc;
        while (rr >= 0 && rr < h && c2 >= 0 && c2 < w) {
          if (o.cells.has(rr * 64 + c2)) break;
          if (g[rr][c2] !== bg) break;
          out.set(rr * 64 + c2, c);
          rr += dr; c2 += dc;
        }
        s = it.next();
      }
    }
    return out.size ? out : null;
  }

  /* Cell writes an action performs, or null when it does not apply. */
  function _writes(g, o, act, bg, scene) {
    var kind = act[0];
    if (act.length > 1 && Array.isArray(act[1]) && act[1][0] === "rel") {
      var cres = _resolve(act[1], scene || {}, o);
      if (cres === null) return null;
      act = [kind, cres].concat(act.slice(2));
    }
    var h = g.length, w = g[0].length, out, it, s, r, c, rr, c2, dr, dc, off, i, j, v;
    if (kind === "keep") return new Map();
    if (kind === "del") {
      out = new Map();
      it = o.cells.values(); s = it.next();
      while (!s.done) { out.set(s.value, bg); s = it.next(); }
      return out;
    }
    if (kind === "solid") {
      out = new Map();
      it = o.cells.values(); s = it.next();
      while (!s.done) { out.set(s.value, act[1]); s = it.next(); }
      return out;
    }
    if (kind === "bbox") {
      out = new Map();
      for (r = o.r0; r <= o.r1; r++) for (c = o.c0; c <= o.c1; c++) out.set(r * 64 + c, act[1]);
      return out;
    }
    if (kind === "bboxout") {
      out = new Map();
      for (r = o.r0; r <= o.r1; r++) for (c = o.c0; c <= o.c1; c++)
        if (r === o.r0 || r === o.r1 || c === o.c0 || c === o.c1) out.set(r * 64 + c, act[1]);
      return out;
    }
    if (kind === "bboxin") {
      out = new Map();
      for (r = o.r0; r <= o.r1; r++) for (c = o.c0; c <= o.c1; c++)
        if (!o.cells.has(r * 64 + c)) out.set(r * 64 + c, act[1]);
      return out;
    }
    if (kind === "halo") {
      out = new Map();
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        for (dr = -1; dr <= 1; dr++) for (dc = -1; dc <= 1; dc++) {
          rr = r + dr; c2 = c + dc;
          if (rr >= 0 && rr < h && c2 >= 0 && c2 < w && !o.cells.has(rr * 64 + c2) && g[rr][c2] === bg)
            out.set(rr * 64 + c2, act[1]);
        }
        s = it.next();
      }
      return out.size ? out : null;
    }
    if (kind === "fill") {
      out = new Map();
      var cells = _interior(o, g, bg);
      for (i = 0; i < cells.length; i++) out.set(cells[i], act[1]);
      return out.size ? out : null;
    }
    if (kind === "move" || kind === "slide" || kind === "edge") {
      if (kind === "move") { dr = act[1]; dc = act[2]; }
      else if (kind === "slide") { off = _slideOffset(g, o, act[1], bg); dr = off[0]; dc = off[1]; }
      else { off = _edgeOffset(g, o, act[1]); dr = off[0]; dc = off[1]; }
      if (dr === 0 && dc === 0) return null;
      out = new Map();
      it = o.cells.values(); s = it.next();
      while (!s.done) { out.set(s.value, bg); s = it.next(); }
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        rr = r + dr; c2 = c + dc;
        if (!(rr >= 0 && rr < h && c2 >= 0 && c2 < w)) return null;
        out.set(rr * 64 + c2, g[r][c]);
        s = it.next();
      }
      return out;
    }
    if (kind === "toward" || kind === "step" || kind === "into") {
      off = _anchorOffset(g, o, act, bg, (scene && scene.objs) || []);
      if (off === null) return null;
      dr = off[0]; dc = off[1];
      if (dr === 0 && dc === 0) return null;
      out = new Map();
      it = o.cells.values(); s = it.next();
      while (!s.done) { out.set(s.value, bg); s = it.next(); }
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        rr = r + dr; c2 = c + dc;
        if (!(rr >= 0 && rr < h && c2 >= 0 && c2 < w)) return null;
        out.set(rr * 64 + c2, g[r][c]);
        s = it.next();
      }
      return out;
    }
    if (kind === "copy") {
      /* the object stays and a duplicate appears elsewhere */
      dr = act[1]; dc = act[2];
      out = new Map();
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        rr = r + dr; c2 = c + dc;
        if (!(rr >= 0 && rr < h && c2 >= 0 && c2 < w)) return null;
        out.set(rr * 64 + c2, g[r][c]);
        s = it.next();
      }
      return out;
    }
    if (kind === "reflect") return _reflectWrites(g, o, act[1], bg, h, w);
    if (kind === "geom") return _geomWrites(g, o, act[1], bg);
    if (kind === "ray" || kind === "rays4" || kind === "cross") return _rayWrites(g, o, act, bg, h, w);
    if (kind === "patch") {
      var pad = act[1], rows = act[2], r0 = o.r0 - pad, c0 = o.c0 - pad;
      out = new Map();
      for (i = 0; i < rows.length; i++) for (j = 0; j < rows[i].length; j++) {
        v = rows[i][j];
        if (v === null || v === undefined) continue;
        r = r0 + i; c = c0 + j;
        if (!(r >= 0 && r < h && c >= 0 && c < w)) return null;
        out.set(r * 64 + c, v);
      }
      return out;
    }
    return null;
  }

  function _candidateActions(colors, geoms) {
    if (geoms === undefined) geoms = true;
    var acts = [["keep"], ["del"]], i, d, dirs = ["up", "down", "left", "right"];
    for (i = 0; i < colors.length; i++) {
      acts.push(["solid", colors[i]]);
      acts.push(["bbox", colors[i]]);
      acts.push(["fill", colors[i]]);
      acts.push(["halo", colors[i]]);
      acts.push(["bboxout", colors[i]]);
      acts.push(["bboxin", colors[i]]);
    }
    for (d = 0; d < dirs.length; d++) { acts.push(["slide", dirs[d]]); acts.push(["edge", dirs[d]]); }
    if (geoms) {
      var gs = ["rot180", "flip_h", "flip_v", "rot90", "rot270", "transpose"];
      for (i = 0; i < gs.length; i++) acts.push(["geom", gs[i]]);
    }
    for (i = 0; i < colors.length; i++) {
      acts.push(["rays4", colors[i]]);
      for (d = 0; d < dirs.length; d++) acts.push(["ray", dirs[d], colors[i]]);
    }
    return acts;
  }

  /* Displacements at which the object's exact patch reappears in the output. */
  function _matchOffsets(g, out, o, bg, limit, raw) {
    if (limit === undefined) limit = 4;
    var h = g.length, w = g[0].length, found = [], hh = o.height(), ww = o.width();
    var cells = [], it = o.cells.values(), s = it.next(), r, c;
    while (!s.done) {
      r = s.value >> 6; c = s.value & 63;
      cells.push([r - o.r0, c - o.c0, g[r][c]]);
      s = it.next();
    }
    var r0, c0, dr, dc, ok, i;
    for (r0 = 0; r0 <= h - hh; r0++) {
      for (c0 = 0; c0 <= w - ww; c0++) {
        dr = r0 - o.r0; dc = c0 - o.c0;
        if (dr === 0 && dc === 0) continue;
        ok = true;
        for (i = 0; i < cells.length; i++)
          if (out[r0 + cells[i][0]][c0 + cells[i][1]] !== cells[i][2]) { ok = false; break; }
        if (ok) {
          found.push([Math.abs(dr) + Math.abs(dc), dr, dc]);
          if (found.length > 24) break;
        }
      }
      if (found.length > 24) break;
    }
    found.sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]); });
    var top = found.slice(0, limit);
    if (raw) return top;
    return top.map(function (t) { return ["move", t[1], t[2]]; });
  }

  /* Read the output's content around the object as a reusable stencil: the
     dictionary end of the algebra, powerful enough to be dangerous. */
  function _stencils(g, out, o, pads) {
    if (pads === undefined) pads = [0, 1];
    var h = g.length, w = g[0].length, acts = [], p, r0, c0, r1, c1, r, c, rows, row;
    for (p = 0; p < pads.length; p++) {
      r0 = o.r0 - pads[p]; c0 = o.c0 - pads[p];
      r1 = o.r1 + pads[p]; c1 = o.c1 + pads[p];
      if (r0 < 0 || c0 < 0 || r1 >= h || c1 >= w) continue;
      if ((r1 - r0 + 1) * (c1 - c0 + 1) > 90) continue;
      rows = [];
      for (r = r0; r <= r1; r++) {
        row = new Array(c1 - c0 + 1);
        for (c = c0; c <= c1; c++) row[c - c0] = out[r][c];
        rows.push(row);
      }
      acts.push(["patch", pads[p], rows]);
    }
    return acts;
  }

  function _consistent(g, out, o, writes) {
    if (writes === null || writes === undefined) return false;
    var bad = false;
    writes.forEach(function (v, rc) { if (out[rc >> 6][rc & 63] !== v) bad = true; });
    if (bad) return false;
    var it = o.cells.values(), s = it.next(), r, c;
    while (!s.done) {
      r = s.value >> 6; c = s.value & 63;
      if (!writes.has(s.value) && out[r][c] !== g[r][c]) return false;
      s = it.next();
    }
    return true;
  }

  /* Two passes: every erasure first, then every draw. */
  function _applyLabelling(g, objs, actions, bg) {
    var out = G.copyGrid(g), draws = [], scene = { objs: objs }, i, wr;
    for (i = 0; i < objs.length; i++) {
      wr = _writes(g, objs[i], actions[i], bg, scene);
      if (wr === null || wr === undefined) return null;
      wr.forEach(function (v, rc) {
        if (v === bg) out[rc >> 6][rc & 63] = v;
        else draws.push([rc, v]);
      });
    }
    for (i = 0; i < draws.length; i++) out[draws[i][0] >> 6][draws[i][0] & 63] = draws[i][1];
    return out;
  }

  function _keyOf(o, objs, g, keys, shared) {
    if (!keys.length) return "";
    var feats = null, vals = [], i;
    for (i = 0; i < keys.length; i++) {
      if (keys[i] === _SHAPE_KEY) vals.push(o.norm_key());
      else {
        if (feats === null) feats = O.objectFeatures(o, objs, g, shared, objs.indexOf(o));
        vals.push(feats[keys[i]]);
      }
    }
    return vals.join("|");
  }

  function _procRule(seg, bg, keys, table, def, maxobj) {
    if (maxobj === undefined) maxobj = 80;
    return function (g) {
      var b = G.bgOr(g, bg), objs = O.segment(g, seg, b);
      if (!objs.length || objs.length > maxobj) return null;
      var shared = O.sharedStats(objs, g), acts = [], i, k, act;
      for (i = 0; i < objs.length; i++) {
        k = _keyOf(objs[i], objs, g, keys, shared);
        act = table.has(k) ? table.get(k) : def;
        if (act === null || act === undefined) return null;
        acts.push(act);
      }
      return _applyLabelling(g, objs, acts, b);
    };
  }

  function _bgCandidates(ctx) {
    var cands = [ctx.bg()], i, c;
    if (ctx.bg_varies()) cands.push(null);
    var counts = new Int32Array(G.NCOLORS), ins = ctx.inputs(), hist;
    for (i = 0; i < ins.length; i++) {
      hist = G.histogram(ins[i]);
      for (c = 0; c < G.NCOLORS; c++) counts[c] += hist[c];
    }
    var order = [];
    for (c = 0; c < G.NCOLORS; c++) if (counts[c] > 0) order.push(c);
    order.sort(function (a, b) { return counts[b] - counts[a]; });
    for (i = 0; i < Math.min(2, order.length); i++)
      if (cands.indexOf(order[i]) < 0) cands.push(order[i]);
    return cands.slice(0, 3);
  }

  function _colors(ctx) {
    var m = G.csAdd(ctx.out_palette(), ctx.bg()), t, a, b, r, c;
    for (t = 0; t < ctx.train.length; t++) {
      a = ctx.train[t][0]; b = ctx.train[t][1];
      for (r = 0; r < Math.min(a.length, b.length); r++)
        for (c = 0; c < Math.min(a[r].length, b[r].length); c++)
          if (a[r][c] !== b[r][c]) m |= 1 << b[r][c];
    }
    return G.csList(m).slice(0, 9);
  }

  /* When no demonstration ever removes anything, an action that removes
     something is a wrong explanation that happens to look cheap locally. */
  function _penalty(ctx) {
    var pen = {}, same = true, t, i, ha, hb;
    for (t = 0; t < ctx.train.length; t++) {
      ha = G.histogram(ctx.train[t][0]); hb = G.histogram(ctx.train[t][1]);
      for (i = 0; i < G.NCOLORS; i++) if (ha[i] !== hb[i]) { same = false; break; }
      if (!same) break;
    }
    if (same) { pen.del = 2.5; pen.solid = 1.5; pen.bbox = 1.5; }
    return pen;
  }

  function _costOf(act) {
    var base = _ACTION_COST[act[0]] === undefined ? 2.0 : _ACTION_COST[act[0]];
    if (act.length > 1 && Array.isArray(act[1]) && act[1][0] === "rel") base -= 0.25;
    return base;
  }

  function _rank(okMap, penalty) {
    var arr = [];
    okMap.forEach(function (act, k) { arr.push([act, k]); });
    arr.sort(function (x, y) {
      var cx = _costOf(x[0]) + (penalty ? (penalty[x[0][0]] || 0.0) : 0.0);
      var cy = _costOf(y[0]) + (penalty ? (penalty[y[0][0]] || 0.0) : 0.0);
      return (cx - cy) || (x[0].length - y[0].length) || cmpStr(x[1], y[1]);
    });
    return arr.map(function (x) { return x[0]; });
  }

  /* Per-object survivor sets, or null when this reading does not apply. */
  function _observe(ctx, seg, bg, colors, baseActs) {
    var rows = [], t, a, b, abg, objs, shared, per, i, o, scene, ok, act, j, k, offs, sten;
    for (t = 0; t < ctx.train.length; t++) {
      a = ctx.train[t][0]; b = ctx.train[t][1];
      if (a.length !== b.length || a[0].length !== b[0].length) return null;
      abg = G.bgOr(a, bg);
      objs = O.segment(a, seg, abg);
      if (!objs.length || objs.length > 45) return null;
      shared = O.sharedStats(objs, a);
      per = [];
      for (i = 0; i < objs.length; i++) {
        o = objs[i];
        scene = { objs: objs };
        ok = new Map();
        for (j = 0; j < baseActs.length; j++)
          if (_consistent(a, b, o, _writes(a, o, baseActs[j], abg, scene))) ok.set(actKey(baseActs[j]), baseActs[j]);
        var kinds = ["solid", "fill", "bbox"];
        for (j = 0; j < _REL_COLORS.length; j++) for (k = 0; k < kinds.length; k++) {
          act = [kinds[k], _REL_COLORS[j]];
          if (_consistent(a, b, o, _writes(a, o, act, abg, scene))) ok.set(actKey(act), act);
        }
        var tks = ["toward", "step", "into"], anchors = ["big", "small", "diff", "bigger"];
        for (j = 0; j < tks.length; j++) for (k = 0; k < anchors.length; k++) {
          act = [tks[j], anchors[k]];
          if (_consistent(a, b, o, _writes(a, o, act, abg, scene))) ok.set(actKey(act), act);
        }
        var axes = ["h", "v", "both", "d"];
        for (j = 0; j < axes.length; j++) {
          act = ["reflect", axes[j]];
          if (_consistent(a, b, o, _writes(a, o, act, abg, scene))) ok.set(actKey(act), act);
        }
        offs = _matchOffsets(a, b, o, abg, 6, true);
        for (j = 0; j < offs.length; j++) {
          act = ["copy", offs[j][1], offs[j][2]];
          if (_consistent(a, b, o, _writes(a, o, act, abg, scene))) ok.set(actKey(act), act);
        }
        offs = _matchOffsets(a, b, o, abg);
        for (j = 0; j < offs.length; j++) {
          act = offs[j];
          if (!ok.has(actKey(act)) && _consistent(a, b, o, _writes(a, o, act, abg, scene)))
            ok.set(actKey(act), act);
        }
        sten = _stencils(a, b, o);
        for (j = 0; j < sten.length; j++) ok.set(actKey(sten[j]), sten[j]);
        if (!ok.size) return null;
        per.push([o, ok]);
      }
      rows.push([a, b, objs, shared, per]);
    }
    return rows;
  }

  /* Taking the cheapest survivor in every group is not enough: local
     consistency cannot tell "deleted" from "moved away", and deletion is the
     cheaper reading. Try the cheapest assembly, then single substitutions. */
  function _tables(ranked, order, limit) {
    if (limit === undefined) limit = 14;
    var base = new Map(), i, j, out = [], t;
    for (i = 0; i < order.length; i++) base.set(order[i], ranked.get(order[i])[0]);
    out.push(base);
    for (i = 0; i < order.length; i++) {
      var alts = ranked.get(order[i]);
      for (j = 1; j < alts.length; j++) {
        t = new Map(base);
        t.set(order[i], alts[j]);
        out.push(t);
        if (out.length >= limit) return out;
      }
    }
    var allMulti = order.length > 0;
    for (i = 0; i < order.length; i++) if (ranked.get(order[i]).length <= 1) allMulti = false;
    if (allMulti) {
      t = new Map();
      for (i = 0; i < order.length; i++) t.set(order[i], ranked.get(order[i])[1]);
      out.push(t);
    }
    return out.slice(0, limit);
  }

  /* Per-group action shortlists consistent with every object, or null. */
  function _induce(rows, keys, penalty) {
    var groups = new Map(), order = [], sizes = new Map(), n = 0, i, j, k, cur, ok;
    for (i = 0; i < rows.length; i++) {
      var a = rows[i][0], objs = rows[i][2], shared = rows[i][3], per = rows[i][4];
      for (j = 0; j < per.length; j++) {
        n += 1;
        k = _keyOf(per[j][0], objs, a, keys, shared);
        sizes.set(k, (sizes.get(k) || 0) + 1);
        cur = groups.get(k);
        ok = per[j][1];
        if (cur === undefined) { groups.set(k, new Map(ok)); order.push(k); }
        else {
          var inter = new Map();
          cur.forEach(function (act, ak) { if (ok.has(ak)) inter.set(ak, act); });
          groups.set(k, inter);
        }
        if (!groups.get(k).size) return null;
      }
    }
    if (!groups.size) return null;
    if (keys.length && groups.size > Math.max(2, Math.floor(0.7 * n))) return null;
    var ranked = new Map();
    for (i = 0; i < order.length; i++) ranked.set(order[i], _rank(groups.get(order[i]), penalty).slice(0, 3));
    /* a stencil seen once is a copy of the answer, not a rule */
    for (i = 0; i < order.length; i++)
      if (ranked.get(order[i])[0][0] === "patch" && sizes.get(order[i]) < 2) return null;
    return [ranked, order, n];
  }

  function _verify(rule, ctx) {
    var i, p;
    for (i = 0; i < ctx.train.length; i++) {
      p = rule(ctx.train[i][0]);
      if (p === null || !G.gEq(p, ctx.train[i][1])) return false;
    }
    return true;
  }

  function _modal(table) {
    var cnt = new Map(), order = [], best = null, bestN = -1;
    table.forEach(function (act) {
      var ak = actKey(act);
      if (!cnt.has(ak)) { cnt.set(ak, { n: 0, act: act }); order.push(ak); }
      cnt.get(ak).n += 1;
    });
    var i;
    for (i = 0; i < order.length; i++) if (cnt.get(order[i]).n > bestN) { bestN = cnt.get(order[i]).n; best = cnt.get(order[i]).act; }
    return best;
  }

  function _sigOf(rule, ctx) {
    var parts = [], i, p;
    for (i = 0; i < ctx.test_inputs.length; i++) {
      p = rule(ctx.test_inputs[i]);
      parts.push((p === null || p === undefined) ? "*" : G.gkey(p));
    }
    return parts.join("~");
  }

  function _distinctActions(table) {
    var s = new Set();
    table.forEach(function (act) { s.add(actKey(act)); });
    return s.size;
  }

  function _pairs(ctx, colors, base, seen, room) {
    var out = [], bgs = _bgCandidates(ctx).slice(0, 2), segs = _SEGS.slice(0, 4), bi, si, rows, i, j;
    for (bi = 0; bi < bgs.length; bi++) {
      for (si = 0; si < segs.length; si++) {
        if (ctx.timed_out() || out.length >= room) return out;
        try { rows = _observe(ctx, segs[si], bgs[bi], colors, base); } catch (e) { rows = null; }
        if (!rows) continue;
        for (i = 0; i < _PAIR_KEYS.length; i++) for (j = i + 1; j < _PAIR_KEYS.length; j++) {
          if (ctx.timed_out() || out.length >= room) return out;
          var got = _induce(rows, [_PAIR_KEYS[i], _PAIR_KEYS[j]], _penalty(ctx));
          if (got === null) continue;
          var ranked = got[0], order = got[1], table = new Map(), k;
          for (k = 0; k < order.length; k++) table.set(order[k], ranked.get(order[k])[0]);
          if (_distinctActions(table) < 2) continue;
          var defs = [null, _modal(table)], d;
          for (d = 0; d < 2; d++) {
            var rule = _procRule(segs[si], bgs[bi], [_PAIR_KEYS[i], _PAIR_KEYS[j]], table, defs[d]);
            var sig;
            try {
              if (!_verify(rule, ctx)) continue;
              sig = _sigOf(rule, ctx);
            } catch (e) { continue; }
            if (seen.has(sig)) continue;
            seen.add(sig);
            var cost = 3.4 + 0.3 * table.size +
                       ((segs[si] === "rows" || segs[si] === "cols") ? 1.6 : 0.0) +
                       (defs[d] !== null ? 0.4 : 0.0);
            out.push(_h("proc2[" + segs[si] + "/" + (bgs[bi] === null ? "auto" : bgs[bi]) + "/" +
                        _PAIR_KEYS[i] + "+" + _PAIR_KEYS[j] + "|" + table.size + "]", rule, cost));
            break;
          }
        }
      }
    }
    return out;
  }

  function generate(ctx) {
    if (!ctx.same_shape()) return [];
    var colors = _colors(ctx), base = _candidateActions(colors), penalty = _penalty(ctx);
    var out = [], seen = new Set(), bgs = _bgCandidates(ctx), bi, si, rows, ki;
    var keysets = [[]], i;
    for (i = 0; i < _FEATURE_KEYS.length; i++) keysets.push([_FEATURE_KEYS[i]]);
    keysets.push([_SHAPE_KEY]);
    for (bi = 0; bi < bgs.length; bi++) {
      if (ctx.timed_out()) break;
      for (si = 0; si < _SEGS.length; si++) {
        if (ctx.timed_out()) break;
        try { rows = _observe(ctx, _SEGS[si], bgs[bi], colors, base); } catch (e) { rows = null; }
        if (!rows) continue;
        var foundSingle = false;
        for (ki = 0; ki < keysets.length; ki++) {
          if (ctx.timed_out()) break;
          var got = _induce(rows, keysets[ki], penalty);
          if (got === null) continue;
          var ranked = got[0], order = got[1], hit = false;
          var tables = _tables(ranked, order), ti;
          for (ti = 0; ti < tables.length; ti++) {
            if (ctx.timed_out() || hit) break;
            var table = tables[ti];
            if (keysets[ki].length && _distinctActions(table) < 2) continue;
            var defs = keysets[ki].length ? [null, _modal(table)] : [table.get("")], d;
            for (d = 0; d < defs.length; d++) {
              var rule = _procRule(_SEGS[si], bgs[bi], keysets[ki], table, defs[d]);
              var sig;
              try {
                if (!_verify(rule, ctx)) continue;
                sig = _sigOf(rule, ctx);
              } catch (e) { continue; }
              if (seen.has(sig)) { hit = true; break; }
              seen.add(sig);
              var sum = 0;
              table.forEach(function (act) { sum += _costOf(act); });
              var cost = 2.2 + 0.7 * keysets[ki].length + 0.28 * table.size +
                         ((_SEGS[si] === "rows" || _SEGS[si] === "cols") ? 1.6 : 0.0) +
                         sum / Math.max(1, table.size) +
                         (defs[d] !== null && defs[d] !== undefined ? 0.4 : 0.0);
              var name = "proc[" + _SEGS[si] + "/" + (bgs[bi] === null ? "auto" : bgs[bi]) + "/" +
                         (keysets[ki].length ? keysets[ki].join("+") : "all") + "|" + table.size + "]";
              out.push(_h(name, rule, cost));
              if (!keysets[ki].length) foundSingle = true;
              hit = true;
              break;
            }
          }
          if (out.length >= 40) break;
        }
        if (foundSingle && out.length >= 12) break;
      }
      if (out.length >= 40) break;
    }
    if (out.length < 40 && !ctx.timed_out())
      out = out.concat(_pairs(ctx, colors, base, seen, 40 - out.length));
    return out;
  }

  OBJPROC.dist = _dist;
  OBJPROC.interior = _interior;
  OBJPROC.consistent = _consistent;
  OBJPROC.actKey = actKey;

  defSolver("objproc", "objects", generate);
})();
