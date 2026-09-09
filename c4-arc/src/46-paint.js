/* Port of engine/solvers/paint.py -- drawing rules that need geometry rather
 * than a lookup.
 *
 * Extend every straight segment to the edge; complete a rectangle from its
 * corner markers; reflect one side of a separator onto the other; grow every
 * region to a fixed point; make each object symmetric in its own box. No table
 * over local contexts can express "continue this line until it hits
 * something", because the answer at a cell depends on structure arbitrarily
 * far away.
 */

(function () {
  var _h = mkHyp("sequence");
  var DIR_NAMES = ["up", "down", "left", "right"];
  var DIRS = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };
  var DIAG_NAMES = ["ul", "ur", "dl", "dr"];
  var DIAG = { ul: [-1, -1], ur: [-1, 1], dl: [1, -1], dr: [1, 1] };

  function _extendLines(g, bg, both, stop) {
    bg = G.bgOr(g, bg);
    var h = g.length, w = g[0].length, out = G.copyGrid(g), drawn = false;
    var objs = O.segment(g, "c8", bg), i, o, dirs, d, r, c, nr, nc;
    for (i = 0; i < objs.length; i++) {
      o = objs[i];
      if (o.size() < 2) continue;
      if (o.height() === 1) dirs = [[0, 1], [0, -1]];
      else if (o.width() === 1) dirs = [[1, 0], [-1, 0]];
      else continue;
      if (!o.is_rect()) continue;
      for (d = 0; d < dirs.length; d++) {
        r = dirs[d][0] > 0 ? o.r1 : o.r0;
        c = dirs[d][1] > 0 ? o.c1 : o.c0;
        nr = r + dirs[d][0]; nc = c + dirs[d][1];
        while (nr >= 0 && nr < h && nc >= 0 && nc < w) {
          if (g[nr][nc] !== bg) {
            if (stop) break;
          } else { out[nr][nc] = o.color; drawn = true; }
          nr += dirs[d][0]; nc += dirs[d][1];
        }
        if (!both) break;
      }
    }
    return drawn ? out : null;
  }

  function _rectFromCorners(g, bg, fill, outlineOnly) {
    bg = G.bgOr(g, bg);
    var pts = new Map(), r, c, v;
    for (r = 0; r < g.length; r++) for (c = 0; c < g[r].length; c++) {
      v = g[r][c];
      if (v !== bg) {
        if (!pts.has(v)) pts.set(v, []);
        pts.get(v).push(r * 64 + c);
      }
    }
    var out = G.copyGrid(g), drawn = false;
    pts.forEach(function (ps, val) {
      if (ps.length !== 2 && ps.length !== 4) return;
      var bb = G.bboxOf(ps), r0 = bb[0], c0 = bb[1], r1 = bb[2], c1 = bb[3];
      if (r1 - r0 < 1 || c1 - c0 < 1) return;
      var col = fill === null ? val : fill, rr, cc, edge;
      for (rr = r0; rr <= r1; rr++) for (cc = c0; cc <= c1; cc++) {
        edge = (rr === r0 || rr === r1 || cc === c0 || cc === c1);
        if (outlineOnly && !edge) continue;
        if (g[rr][cc] === bg) { out[rr][cc] = col; drawn = true; }
      }
    });
    return drawn ? out : null;
  }

  function _axisLines(g) {
    var h = g.length, w = g[0].length, rows = [], cols = [], r, c, uni;
    for (r = 0; r < h; r++) {
      uni = true;
      for (c = 1; c < w; c++) if (g[r][c] !== g[r][0]) { uni = false; break; }
      if (uni) rows.push(r);
    }
    var t = G.transpose(g);
    for (c = 0; c < w; c++) {
      uni = true;
      for (r = 1; r < t[c].length; r++) if (t[c][r] !== t[c][0]) { uni = false; break; }
      if (uni) cols.push(c);
    }
    return [rows, cols];
  }

  function _reflectAcross(g, bg, useRow, overwrite) {
    bg = G.bgOr(g, bg);
    var al = _axisLines(g), idx = useRow ? al[0] : al[1];
    if (idx.length !== 1) return null;
    var a = idx[0], h = g.length, w = g[0].length, out = G.copyGrid(g), drawn = false;
    var r, c, v, nr, nc;
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
      v = g[r][c];
      if (v === bg) continue;
      if (useRow) { nr = 2 * a - r; nc = c; } else { nr = r; nc = 2 * a - c; }
      if (nr >= 0 && nr < h && nc >= 0 && nc < w && !(r === nr && c === nc)) {
        if (overwrite || g[nr][nc] === bg) {
          if (out[nr][nc] !== v) { out[nr][nc] = v; drawn = true; }
        }
      }
    }
    return drawn ? out : null;
  }

  /* Each satellite aligned with the anchor grows a line towards it; ones that
     are not aligned are deliberately left untouched. */
  function _connectToAnchor(g, bg, seg, anchorMode) {
    bg = G.bgOr(g, bg);
    var objs = O.segment(g, seg, bg);
    if (objs.length < 2 || objs.length > 40) return null;
    var anchor = objs[0], i, o, r, c;
    for (i = 1; i < objs.length; i++) {
      if (anchorMode === "largest") {
        if (objs[i].size() > anchor.size() ||
            (objs[i].size() === anchor.size() && objs[i].bbox_area() > anchor.bbox_area())) anchor = objs[i];
      } else {
        if (objs[i].bbox_area() > anchor.bbox_area() ||
            (objs[i].bbox_area() === anchor.bbox_area() && objs[i].size() > anchor.size())) anchor = objs[i];
      }
    }
    var out = G.copyGrid(g), drawn = false, v, lo, hi, span, ok, k;
    for (i = 0; i < objs.length; i++) {
      o = objs[i];
      if (o === anchor) continue;
      v = o.color;
      lo = Math.max(o.r0, anchor.r0); hi = Math.min(o.r1, anchor.r1);
      if (lo <= hi) {
        span = [];
        if (o.c0 > anchor.c1) { for (k = anchor.c1 + 1; k < o.c0; k++) span.push(k); }
        else if (o.c1 < anchor.c0) { for (k = o.c1 + 1; k < anchor.c0; k++) span.push(k); }
        for (r = lo; r <= hi; r++) {
          ok = true;
          for (k = 0; k < span.length; k++) if (g[r][span[k]] !== bg) { ok = false; break; }
          if (ok) for (k = 0; k < span.length; k++) { out[r][span[k]] = v; drawn = true; }
        }
      }
      lo = Math.max(o.c0, anchor.c0); hi = Math.min(o.c1, anchor.c1);
      if (lo <= hi) {
        span = [];
        if (o.r0 > anchor.r1) { for (k = anchor.r1 + 1; k < o.r0; k++) span.push(k); }
        else if (o.r1 < anchor.r0) { for (k = o.r1 + 1; k < anchor.r0; k++) span.push(k); }
        for (c = lo; c <= hi; c++) {
          ok = true;
          for (k = 0; k < span.length; k++) if (g[span[k]][c] !== bg) { ok = false; break; }
          if (ok) for (k = 0; k < span.length; k++) { out[span[k]][c] = v; drawn = true; }
        }
      }
    }
    return drawn ? out : null;
  }

  /* Slide every satellite straight at the anchor until it touches: the
     direction is per object, read off the geometry, and it arrives intact. */
  function _moveToAnchor(g, bg, seg, diagTouch, keepAnchor) {
    bg = G.bgOr(g, bg);
    var objs = O.segment(g, seg, bg);
    if (objs.length < 2 || objs.length > 40) return null;
    var anchor = objs[0], i;
    for (i = 1; i < objs.length; i++)
      if (objs[i].size() > anchor.size() ||
          (objs[i].size() === anchor.size() && objs[i].bbox_area() > anchor.bbox_area())) anchor = objs[i];
    if (anchor.size() < 2) return null;
    var h = g.length, w = g[0].length, occupied = new Set(), out = G.constGrid(h, w, bg);
    var it = anchor.cells.values(), s = it.next(), r, c;
    while (!s.done) {
      r = s.value >> 6; c = s.value & 63;
      occupied.add(s.value);
      out[r][c] = g[r][c];
      s = it.next();
    }
    var nb = diagTouch ? G.N8 : G.N4, moved = false;
    var order = objs.slice();
    order.sort(function (a, b) {
      var da = Math.min(Math.abs(a.r0 - anchor.r0) + Math.abs(a.c0 - anchor.c0), 1000000);
      var db = Math.min(Math.abs(b.r0 - anchor.r0) + Math.abs(b.c0 - anchor.c0), 1000000);
      return db - da;
    });
    var o, dr, dc, best, step, cells, j, d, ok, touching;
    for (i = 0; i < order.length; i++) {
      o = order[i];
      if (o === anchor) continue;
      if (o.r1 < anchor.r0) dr = 1; else if (o.r0 > anchor.r1) dr = -1; else dr = 0;
      if (o.c1 < anchor.c0) dc = 1; else if (o.c0 > anchor.c1) dc = -1; else dc = 0;
      if (dr === 0 && dc === 0) return null;
      best = 0; step = 0;
      while (step < 60) {
        cells = [];
        it = o.cells.values(); s = it.next();
        while (!s.done) { cells.push([(s.value >> 6) + dr * step, (s.value & 63) + dc * step]); s = it.next(); }
        ok = true;
        for (j = 0; j < cells.length; j++)
          if (!(cells[j][0] >= 0 && cells[j][0] < h && cells[j][1] >= 0 && cells[j][1] < w)) { ok = false; break; }
        if (!ok) break;
        ok = true;
        for (j = 0; j < cells.length; j++) if (occupied.has(cells[j][0] * 64 + cells[j][1])) { ok = false; break; }
        if (!ok) break;
        touching = false;
        for (j = 0; j < cells.length && !touching; j++)
          for (d = 0; d < nb.length; d++)
            if (occupied.has((cells[j][0] + nb[d][0]) * 64 + (cells[j][1] + nb[d][1]))) { touching = true; break; }
        best = step;
        if (touching) break;
        step += 1;
      }
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        out[r + dr * best][c + dc * best] = g[r][c];
        occupied.add((r + dr * best) * 64 + (c + dc * best));
        s = it.next();
      }
      if (best) moved = true;
    }
    return moved ? out : null;
  }

  function _rayStriped(g, bg, dr, dc, alt, period) {
    bg = G.bgOr(g, bg);
    var h = g.length, w = g[0].length, src = [], r, c;
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) if (g[r][c] !== bg) src.push([r, c, g[r][c]]);
    if (!src.length || src.length > 60) return null;
    var out = G.copyGrid(g), drawn = false, i, k, nr, nc;
    for (i = 0; i < src.length; i++) {
      k = 1; nr = src[i][0] + dr; nc = src[i][1] + dc;
      while (nr >= 0 && nr < h && nc >= 0 && nc < w) {
        if (g[nr][nc] === bg) { out[nr][nc] = (k % period) ? alt : src[i][2]; drawn = true; }
        nr += dr; nc += dc; k += 1;
      }
    }
    return drawn ? out : null;
  }

  /* Two copies of a shape define a step; continue it to the edges. */
  function _repeatTranslation(g, bg, seg, both) {
    bg = G.bgOr(g, bg);
    var objs = O.segment(g, seg, bg);
    if (objs.length < 2 || objs.length > 40) return null;
    var h = g.length, w = g[0].length, groups = new Map(), i, k, o, parts, r, c, row, vals;
    for (i = 0; i < objs.length; i++) {
      o = objs[i];
      parts = [];
      for (r = 0; r < o._patch.length; r++) {
        row = o._patch[r]; vals = [];
        for (c = 0; c < row.length; c++) vals.push(row[c] === null ? "n" : row[c]);
        parts.push(vals.join(","));
      }
      k = parts.join("|");
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(o);
    }
    var out = G.copyGrid(g), drawn = false;
    groups.forEach(function (members) {
      if (members.length < 2) return;
      members.sort(function (a, b) { return (a.r0 - b.r0) || (a.c0 - b.c0); });
      var deltas = new Set(), j;
      for (j = 1; j < members.length; j++)
        deltas.add((members[j].r0 - members[j - 1].r0) + "," + (members[j].c0 - members[j - 1].c0));
      if (deltas.size !== 1) return;
      var d = deltas.values().next().value.split(",");
      var dr = parseInt(d[0], 10), dc = parseInt(d[1], 10);
      if (dr === 0 && dc === 0) return;
      var proto = members[0], last = members[members.length - 1];
      var steps = [[last.r0 + dr, last.c0 + dc, dr, dc]];
      if (both) steps.push([proto.r0 - dr, proto.c0 - dc, -dr, -dc]);
      var si, r0, c0, sr, sc, it, s, cells, ok;
      for (si = 0; si < steps.length; si++) {
        r0 = steps[si][0]; c0 = steps[si][1]; sr = steps[si][2]; sc = steps[si][3];
        for (;;) {
          cells = [];
          it = proto.cells.values(); s = it.next();
          while (!s.done) {
            cells.push([r0 + ((s.value >> 6) - proto.r0), c0 + ((s.value & 63) - proto.c0), s.value]);
            s = it.next();
          }
          ok = true;
          for (j = 0; j < cells.length; j++)
            if (!(cells[j][0] >= 0 && cells[j][0] < h && cells[j][1] >= 0 && cells[j][1] < w)) { ok = false; break; }
          if (!ok) break;
          for (j = 0; j < cells.length; j++) {
            out[cells[j][0]][cells[j][1]] = g[cells[j][2] >> 6][cells[j][2] & 63];
            drawn = true;
          }
          r0 += sr; c0 += sc;
        }
      }
    });
    return drawn ? out : null;
  }

  function _stampAtMarkers(g, bg, seg, recolor, keepMarker) {
    bg = G.bgOr(g, bg);
    var objs = O.segment(g, seg, bg);
    if (objs.length < 2 || objs.length > 40) return null;
    var big = objs[0], i;
    for (i = 1; i < objs.length; i++) if (objs[i].size() > big.size()) big = objs[i];
    if (big.size() < 3) return null;
    var markers = objs.filter(function (o) { return o.size() === 1; });
    if (!markers.length) return null;
    var h = g.length, w = g[0].length;
    var cr = Math.floor((big.r0 + big.r1) / 2), cc = Math.floor((big.c0 + big.c1) / 2);
    var out = G.copyGrid(g), drawn = false, m, mr, mc, it, s, r, c, nr, nc, v;
    for (i = 0; i < markers.length; i++) {
      m = markers[i];
      var first = m.cells.values().next().value;
      mr = first >> 6; mc = first & 63;
      it = big.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        nr = mr + (r - cr); nc = mc + (c - cc);
        if (nr >= 0 && nr < h && nc >= 0 && nc < w) {
          v = recolor ? m.color : g[r][c];
          if (out[nr][nc] !== v) { out[nr][nc] = v; drawn = true; }
        }
        s = it.next();
      }
      if (keepMarker) out[mr][mc] = m.color;
    }
    return drawn ? out : null;
  }

  /* A ray that reflects off the grid edges: the direction at a cell depends on
     the whole path taken to reach it. */
  function _bounce(g, bg, startColor, diag, maxSteps) {
    if (maxSteps === undefined) maxSteps = 200;
    bg = G.bgOr(g, bg);
    var h = g.length, w = g[0].length, src = [], r, c;
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) if (g[r][c] === startColor) src.push([r, c]);
    if (src.length !== 1) return null;
    var dirs = diag ? [[1, 1], [1, -1], [-1, 1], [-1, -1]] : [[1, 0], [0, 1]];
    var out = G.copyGrid(g), drawn = false, d, cr, cc, vr, vc, i, nr, nc;
    for (d = 0; d < dirs.length; d++) {
      cr = src[0][0]; cc = src[0][1]; vr = dirs[d][0]; vc = dirs[d][1];
      for (i = 0; i < maxSteps; i++) {
        nr = cr + vr; nc = cc + vc;
        if (!(nr >= 0 && nr < h)) { vr = -vr; nr = cr + vr; }
        if (!(nc >= 0 && nc < w)) { vc = -vc; nc = cc + vc; }
        if (!(nr >= 0 && nr < h && nc >= 0 && nc < w)) break;
        if (g[nr][nc] === bg) { out[nr][nc] = startColor; drawn = true; }
        cr = nr; cc = nc;
      }
    }
    return drawn ? out : null;
  }

  function _bars(g, bg, vertical, orderDesc, outShape) {
    bg = G.bgOr(g, bg);
    var hist = G.histogram(g), items = [], k;
    for (k = 0; k < G.NCOLORS; k++) if (k !== bg && hist[k] > 0) items.push([k, hist[k]]);
    if (!items.length || items.length > 10) return null;
    items.sort(function (a, b) { return (b[1] - a[1]) || (a[0] - b[0]); });
    if (!orderDesc) items.reverse();
    var n = items.length, m = 0, i, j;
    for (i = 0; i < n; i++) if (items[i][1] > m) m = items[i][1];
    if (m > 30 || n > 30) return null;
    var h = vertical ? m : n, w = vertical ? n : m;
    if (outShape !== null && outShape !== undefined && (outShape[0] !== h || outShape[1] !== w)) return null;
    var out = G.constGrid(h, w, bg);
    for (i = 0; i < n; i++) for (j = 0; j < items[i][1]; j++) {
      if (vertical) out[h - 1 - j][i] = items[i][0];
      else out[i][j] = items[i][0];
    }
    return out;
  }

  function _moveToMarkers(g, bg, marker, mode) {
    bg = G.bgOr(g, bg);
    var h = g.length, w = g[0].length, marks = [], body = [], r, c, v;
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
      v = g[r][c];
      if (v === marker) marks.push(r * 64 + c);
      else if (v !== bg) body.push(r * 64 + c);
    }
    if (marks.length < 2 || marks.length > 8) return null;
    if (!body.length) return null;
    var mb = G.bboxOf(marks), bb = G.bboxOf(body), dr, dc, i;
    if (mode === "center") {
      dr = Math.floor(((mb[0] + mb[2]) - (bb[0] + bb[2])) / 2);
      dc = Math.floor(((mb[1] + mb[3]) - (bb[1] + bb[3])) / 2);
    } else if (mode === "inside") { dr = (mb[0] + 1) - bb[0]; dc = (mb[1] + 1) - bb[1]; }
    else { dr = mb[0] - bb[0]; dc = mb[1] - bb[1]; }
    if (dr === 0 && dc === 0) return null;
    var out = G.constGrid(h, w, bg), nr, nc;
    for (i = 0; i < marks.length; i++) out[marks[i] >> 6][marks[i] & 63] = marker;
    for (i = 0; i < body.length; i++) {
      r = body[i] >> 6; c = body[i] & 63;
      nr = r + dr; nc = c + dc;
      if (!(nr >= 0 && nr < h && nc >= 0 && nc < w)) return null;
      out[nr][nc] = g[r][c];
    }
    return out;
  }

  /* Grow every coloured region simultaneously until the grid is full. */
  function _voronoi(g, bg, diag, tieBg) {
    bg = G.bgOr(g, bg);
    var h = g.length, w = g[0].length, dist = [], col = [], tie = [], r, c, row, crow, trow;
    var q = [], qi = 0;
    for (r = 0; r < h; r++) {
      row = new Array(w); crow = new Array(w); trow = new Array(w);
      for (c = 0; c < w; c++) { row[c] = -1; crow[c] = bg; trow[c] = false; }
      dist.push(row); col.push(crow); tie.push(trow);
    }
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) if (g[r][c] !== bg) {
      dist[r][c] = 0; col[r][c] = g[r][c]; q.push(r * 64 + c);
    }
    if (!q.length || q.length === h * w) return null;
    var nb = diag ? G.N8 : G.N4, cr, cc, d, nr, nc;
    while (qi < q.length) {
      cr = q[qi] >> 6; cc = q[qi] & 63; qi++;
      for (d = 0; d < nb.length; d++) {
        nr = cr + nb[d][0]; nc = cc + nb[d][1];
        if (!(nr >= 0 && nr < h && nc >= 0 && nc < w)) continue;
        if (dist[nr][nc] === -1) {
          dist[nr][nc] = dist[cr][cc] + 1;
          col[nr][nc] = col[cr][cc];
          q.push(nr * 64 + nc);
        } else if (dist[nr][nc] === dist[cr][cc] + 1 && col[nr][nc] !== col[cr][cc]) {
          tie[nr][nc] = true;
          if (!tieBg && col[cr][cc] < col[nr][nc]) col[nr][nc] = col[cr][cc];
        }
      }
    }
    var out = [];
    for (r = 0; r < h; r++) {
      row = new Array(w);
      for (c = 0; c < w; c++) {
        if (g[r][c] !== bg) row[c] = g[r][c];
        else if (tie[r][c] && tieBg) row[c] = bg;
        else row[c] = col[r][c];
      }
      out.push(row);
    }
    return out;
  }

  function _fillEnclosed(g, bg, diag, color, onlyRect) {
    bg = G.bgOr(g, bg);
    var h = g.length, w = g[0].length, seen = [], r, c, row, q = [], qi = 0, i;
    for (r = 0; r < h; r++) { row = new Array(w); for (c = 0; c < w; c++) row[c] = false; seen.push(row); }
    for (r = 0; r < h; r++) for (i = 0; i < 2; i++) {
      c = i ? w - 1 : 0;
      if (g[r][c] === bg && !seen[r][c]) { seen[r][c] = true; q.push(r * 64 + c); }
    }
    for (c = 0; c < w; c++) for (i = 0; i < 2; i++) {
      r = i ? h - 1 : 0;
      if (g[r][c] === bg && !seen[r][c]) { seen[r][c] = true; q.push(r * 64 + c); }
    }
    var nb = diag ? G.N8 : G.N4, cr, cc, d, nr, nc;
    while (qi < q.length) {
      cr = q[qi] >> 6; cc = q[qi] & 63; qi++;
      for (d = 0; d < nb.length; d++) {
        nr = cr + nb[d][0]; nc = cc + nb[d][1];
        if (nr >= 0 && nr < h && nc >= 0 && nc < w && !seen[nr][nc] && g[nr][nc] === bg) {
          seen[nr][nc] = true; q.push(nr * 64 + nc);
        }
      }
    }
    var out = G.copyGrid(g), drawn = false, vis = [];
    for (r = 0; r < h; r++) { row = new Array(w); for (c = 0; c < w; c++) row[c] = false; vis.push(row); }
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
      if (g[r][c] !== bg || seen[r][c] || vis[r][c]) continue;
      var comp = [], border = new Map(), border_order = [], st = [[r, c]], cur;
      vis[r][c] = true;
      while (st.length) {
        cur = st.pop();
        comp.push(cur[0] * 64 + cur[1]);
        for (d = 0; d < G.N4.length; d++) {
          nr = cur[0] + G.N4[d][0]; nc = cur[1] + G.N4[d][1];
          if (!(nr >= 0 && nr < h && nc >= 0 && nc < w)) continue;
          if (g[nr][nc] === bg) {
            if (!vis[nr][nc] && !seen[nr][nc]) { vis[nr][nc] = true; st.push([nr, nc]); }
          } else {
            if (!border.has(g[nr][nc])) { border.set(g[nr][nc], 0); border_order.push(g[nr][nc]); }
            border.set(g[nr][nc], border.get(g[nr][nc]) + 1);
          }
        }
      }
      if (!border.size) continue;
      if (onlyRect) {
        var bb = G.bboxOf(comp);
        if (comp.length !== (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)) continue;
      }
      var colv;
      if (color !== null && color !== undefined) colv = color;
      else {
        colv = border_order[0];
        for (i = 1; i < border_order.length; i++)
          if (border.get(border_order[i]) > border.get(colv)) colv = border_order[i];
      }
      for (i = 0; i < comp.length; i++) { out[comp[i] >> 6][comp[i] & 63] = colv; drawn = true; }
    }
    return drawn ? out : null;
  }

  function _symmetrise(g, bg, seg, mode) {
    bg = G.bgOr(g, bg);
    var objs = O.segment(g, seg, bg);
    if (!objs.length || objs.length > 60) return null;
    var out = G.copyGrid(g), drawn = false, i, o, p, variants, vi, r, c, v;
    for (i = 0; i < objs.length; i++) {
      o = objs[i];
      p = o.filled(bg);
      variants = [];
      if (mode === "h" || mode === "both") variants.push(G.flipH(p));
      if (mode === "v" || mode === "both") variants.push(G.flipV(p));
      if (mode === "both") variants.push(G.rot180(p));
      for (vi = 0; vi < variants.length; vi++)
        for (r = 0; r < o.height(); r++) for (c = 0; c < o.width(); c++) {
          v = variants[vi][r][c];
          if (v !== bg && out[o.r0 + r][o.c0 + c] === bg) {
            out[o.r0 + r][o.c0 + c] = v; drawn = true;
          }
        }
    }
    return drawn ? out : null;
  }

  var _RAY_MODES = ["none", "h", "v", "both"];

  /* Each colour emits rays in the directions its own colour dictates; the draw
     order is by direction, so where a row line crosses a column line, which
     shows through is a property of the two directions. */
  function _colorRays(g, bg, assign, modeOrder) {
    bg = G.bgOr(g, bg);
    var h = g.length, w = g[0].length, out = G.copyGrid(g), src = [], r, c;
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) if (g[r][c] !== bg) src.push([r, c, g[r][c]]);
    if (!src.length || src.length > 120) return null;
    var drawn = false, wi, i, mode, cc, rr;
    for (wi = 0; wi < modeOrder.length; wi++) {
      for (i = 0; i < src.length; i++) {
        mode = assign.has(src[i][2]) ? assign.get(src[i][2]) : "none";
        if (mode !== modeOrder[wi]) continue;
        r = src[i][0]; c = src[i][1];
        if (mode === "h" || mode === "both")
          for (cc = 0; cc < w; cc++) if (g[r][cc] === bg) { out[r][cc] = src[i][2]; drawn = true; }
        if (mode === "v" || mode === "both")
          for (rr = 0; rr < h; rr++) if (g[rr][c] === bg) { out[rr][c] = src[i][2]; drawn = true; }
      }
    }
    return drawn ? out : null;
  }

  function _rayAssignments(colors) {
    if (!colors.length || colors.length > 3) return [];
    var out = [new Map()], i, j, k, nxt;
    for (i = 0; i < colors.length; i++) {
      nxt = [];
      for (j = 0; j < out.length; j++)
        for (k = 0; k < _RAY_MODES.length; k++) {
          var d = new Map(out[j]);
          d.set(colors[i], _RAY_MODES[k]);
          nxt.push(d);
        }
      out = nxt;
      if (out.length > 300) return out;
    }
    return out.filter(function (a) {
      var any = false;
      a.forEach(function (v) { if (v !== "none") any = true; });
      return any;
    });
  }

  function _rules(ctx, bg) {
    var res = [], i, j, k, bothList = [true, false];
    for (i = 0; i < 2; i++) for (j = 0; j < 2; j++)
      res.push(_h("extend_lines" + (bothList[i] ? "True" : "False") + (bothList[j] ? "True" : "False"),
                  (function (b2, s2) { return function (g) { return _extendLines(g, bg, b2, s2); }; })(bothList[i], bothList[j]),
                  4.5));
    var outPal = G.csList(ctx.out_palette());
    for (i = 0; i < 2; i++) {
      (function (ol) {
        res.push(_h("rect_corners" + (ol ? "True" : "False"),
                    function (g) { return _rectFromCorners(g, bg, null, ol); }, 4.8));
        var m;
        for (m = 0; m < outPal.length; m++)
          res.push(_h("rect_corners" + (ol ? "True" : "False") + "#" + outPal[m],
                      (function (c) { return function (g) { return _rectFromCorners(g, bg, c, ol); }; })(outPal[m]),
                      5.5));
      })(bothList[i]);
    }
    for (i = 0; i < 2; i++) for (j = 0; j < 2; j++)
      res.push(_h("reflect_" + (bothList[i] ? "row" : "col") + (bothList[j] ? "True" : "False"),
                  (function (u2, o2) { return function (g) { return _reflectAcross(g, bg, u2, o2); }; })(bothList[i], bothList[j]),
                  4.5));

    var cols = G.csList(G.csDiff(ctx.in_palette(), bg === null ? 0 : (1 << bg)));
    var all = ctx.all_inputs(), busiest = 0, r, c, n;
    for (i = 0; i < all.length; i++) {
      n = 0;
      var ab = G.bgOr(all[i], bg);
      for (r = 0; r < all[i].length; r++) for (c = 0; c < all[i][r].length; c++) if (all[i][r][c] !== ab) n++;
      if (n > busiest) busiest = n;
    }
    var orders = [["v", "both", "h"], ["h", "both", "v"]];
    var assigns = busiest <= 40 ? _rayAssignments(cols) : [];
    for (i = 0; i < assigns.length; i++) {
      var entries = [];
      assigns[i].forEach(function (v, kk) { entries.push([kk, v]); });
      entries.sort(function (a, b) { return a[0] - b[0]; });
      var nm = entries.map(function (e) { return e[0] + e[1].charAt(0); }).join("");
      var live = 0;
      assigns[i].forEach(function (v) { if (v !== "none") live++; });
      for (j = 0; j < orders.length; j++)
        res.push(_h("rays_" + nm + j,
                    (function (a2, o2) { return function (g) { return _colorRays(g, bg, a2, o2); }; })(assigns[i], orders[j]),
                    5.5 + 0.2 * live));
    }
    for (i = 0; i < cols.length; i++) for (j = 0; j < 2; j++)
      res.push(_h("bounce#" + cols[i] + (bothList[j] ? "True" : "False"),
                  (function (c2, d2) { return function (g) { return _bounce(g, bg, c2, d2); }; })(cols[i], bothList[j]),
                  5.6));
    var modes = ["center", "corner", "inside"];
    for (i = 0; i < cols.length; i++) for (j = 0; j < modes.length; j++)
      res.push(_h("moveto#" + cols[i] + "_" + modes[j],
                  (function (m2, md) { return function (g) { return _moveToMarkers(g, bg, m2, md); }; })(cols[i], modes[j]),
                  5.4));
    var diags = [false, true], tieList = [true, false];
    for (i = 0; i < 2; i++) {
      for (j = 0; j < 2; j++)
        res.push(_h("voronoi" + (diags[i] ? "True" : "False") + (tieList[j] ? "True" : "False"),
                    (function (d2, t2) { return function (g) { return _voronoi(g, bg, d2, t2); }; })(diags[i], tieList[j]),
                    4.4));
      res.push(_h("fill_enclosed" + (diags[i] ? "True" : "False"),
                  (function (d2) { return function (g) { return _fillEnclosed(g, bg, d2, null, false); }; })(diags[i]),
                  4.2));
      for (k = 0; k < outPal.length; k++) for (j = 0; j < 2; j++)
        res.push(_h("fill_enclosed" + (diags[i] ? "True" : "False") + "#" + outPal[k] + (diags[j] ? "r" : ""),
                    (function (d2, c2, r2) { return function (g) { return _fillEnclosed(g, bg, d2, c2, r2); }; })(diags[i], outPal[k], diags[j]),
                    4.6));
    }
    var segs3 = ["c8", "m8", "c4"];
    for (i = 0; i < segs3.length; i++) for (j = 0; j < 2; j++)
      res.push(_h("moveanchor_" + segs3[i] + (bothList[j] ? "True" : "False"),
                  (function (s2, d2) { return function (g) { return _moveToAnchor(g, bg, s2, d2, true); }; })(segs3[i], bothList[j]),
                  4.8));
    var singles = [];
    for (i = 0; i < DIR_NAMES.length; i++) singles.push([DIR_NAMES[i], DIRS[DIR_NAMES[i]]]);
    for (i = 0; i < DIAG_NAMES.length; i++) singles.push([DIAG_NAMES[i], DIAG[DIAG_NAMES[i]]]);
    for (i = 0; i < singles.length; i++) for (k = 0; k < outPal.length; k++)
      res.push(_h("stripe_" + singles[i][0] + "#" + outPal[k],
                  (function (v2, a2) { return function (g) { return _rayStriped(g, bg, v2[0], v2[1], a2, 2); }; })(singles[i][1], outPal[k]),
                  5.6));
    var segs2 = ["c8", "m8"];
    for (i = 0; i < segs2.length; i++) {
      for (j = 0; j < 2; j++)
        res.push(_h("repeat_" + segs2[i] + (bothList[j] ? "True" : "False"),
                    (function (s2, b2) { return function (g) { return _repeatTranslation(g, bg, s2, b2); }; })(segs2[i], bothList[j]),
                    5.0));
      var recs = [false, true], keeps = [true, false], ri, ki;
      for (ri = 0; ri < 2; ri++) for (ki = 0; ki < 2; ki++)
        res.push(_h("stamp_" + segs2[i] + (recs[ri] ? "True" : "False") + (keeps[ki] ? "True" : "False"),
                    (function (s2, r2, k2) { return function (g) { return _stampAtMarkers(g, bg, s2, r2, k2); }; })(segs2[i], recs[ri], keeps[ki]),
                    5.2));
    }
    var ams = ["largest", "bbox"];
    for (i = 0; i < segs3.length; i++) for (j = 0; j < ams.length; j++)
      res.push(_h("anchor_" + segs3[i] + "_" + ams[j],
                  (function (s2, a2) { return function (g) { return _connectToAnchor(g, bg, s2, a2); }; })(segs3[i], ams[j]),
                  4.6));
    var syms = ["h", "v", "both"];
    for (i = 0; i < segs2.length; i++) for (j = 0; j < syms.length; j++)
      res.push(_h("symmetrise_" + segs2[i] + "_" + syms[j],
                  (function (s2, m2) { return function (g) { return _symmetrise(g, bg, s2, m2); }; })(segs2[i], syms[j]),
                  5.0));
    return res;
  }

  function generate(ctx) {
    var res = [], bgs = ctx.bg_varies() ? [ctx.bg(), null] : [ctx.bg()], i, j, k;
    var flags = [true, false];
    /* bar charts change the grid shape, so they sit outside the same-shape gate */
    for (i = 0; i < bgs.length; i++) for (j = 0; j < 2; j++) for (k = 0; k < 2; k++)
      res.push(_h("bars" + (flags[j] ? "True" : "False") + (flags[k] ? "True" : "False"),
                  (function (v2, d2, b2) { return function (g) { return _bars(g, b2, v2, d2, null); }; })(flags[j], flags[k], bgs[i]),
                  5.5));
    if (!ctx.same_shape()) return res;
    for (i = 0; i < bgs.length; i++) res = res.concat(_rules(ctx, bgs[i]));
    return res;
  }

  defSolver("paint", "sequence", generate);
})();
