/* Port of engine/solvers/panelabs.py -- solve the grid of panels as a task in
 * its own right.
 *
 * When a grid is a lattice of equal panels the rule almost never operates on
 * pixels. Abstract each panel to a single cell, hand the resulting small task
 * back to the portfolio, and render the answer by stamping the learned patch
 * for each predicted panel code. The abstraction is lossy by design; the
 * render table, learned from the training outputs, puts the detail back.
 */

var PANELABS = {};

(function () {
  /* One colour standing for a whole panel: its dominant non-background. */
  function _code(panel, bg) {
    var hist = G.histogram(panel), best = -1, bestN = -1, k;
    for (k = 0; k < G.NCOLORS; k++) if (k !== bg && hist[k] > 0 && hist[k] > bestN) { bestN = hist[k]; best = k; }
    return best < 0 ? bg : best;
  }

  /* Panel matrix -> small grid. ``role`` collapses every non-empty panel to 1,
     which is what tasks about *which* panels are occupied need; the raw form
     keeps the panel's colour. Neither subsumes the other. */
  function _abstract(mat, bg, role) {
    var out = [], i, j, r, k;
    for (i = 0; i < mat.length; i++) {
      r = [];
      for (j = 0; j < mat[i].length; j++) {
        if (mat[i][j] === null || mat[i][j] === undefined) return null;
        k = _code(mat[i][j], bg);
        r.push(role ? (k === bg ? 0 : 1) : k);
      }
      out.push(r);
    }
    return out;
  }

  function _fillColor(g, bg, mode) {
    if (mode === "sep") return PART.sepColorOf(g);
    if (mode === "motif") {
      var hist = G.histogram(g), sep = PART.sepColorOf(g), best = -1, bestN = Infinity, k;
      for (k = 0; k < G.NCOLORS; k++) {
        if (k === bg || k === sep || hist[k] === 0) continue;
        if (hist[k] < bestN) { bestN = hist[k]; best = k; }
      }
      return best < 0 ? null : best;
    }
    return mode;
  }

  /* code -> panel patch, learned from the training outputs. */
  function _renderTable(ctx, dec, bg) {
    var table = new Map(), shape = null, t, mat, i, j, p, k, prev;
    for (t = 0; t < ctx.train.length; t++) {
      mat = dec(ctx.train[t][1]);
      if (!mat) return null;
      for (i = 0; i < mat.length; i++) for (j = 0; j < mat[i].length; j++) {
        p = mat[i][j];
        if (p === null || p === undefined) return null;
        if (shape === null) shape = [p.length, p[0].length];
        else if (p.length !== shape[0] || p[0].length !== shape[1]) return null;
        k = _code(p, bg);
        prev = table.get(k);
        if (prev === undefined) table.set(k, p);
        else if (!G.gEq(prev, p)) return null;
      }
    }
    return table.size ? table : null;
  }

  /* A single stencil shared by every non-empty output panel: a literal
     code -> patch table cannot survive a task that draws the same motif in a
     different colour in each grid. */
  function _renderShape(ctx, dec, bg) {
    var mask = null, shape = null, t, mat, i, j, p, cols, m, r, c, row;
    for (t = 0; t < ctx.train.length; t++) {
      mat = dec(ctx.train[t][1]);
      if (!mat) return null;
      for (i = 0; i < mat.length; i++) for (j = 0; j < mat[i].length; j++) {
        p = mat[i][j];
        if (p === null || p === undefined) return null;
        if (shape === null) shape = [p.length, p[0].length];
        else if (p.length !== shape[0] || p[0].length !== shape[1]) return null;
        cols = G.csDiff(G.palette(p), 1 << bg);
        if (!cols) continue;
        if (G.csSize(cols) > 1) return null;
        m = [];
        for (r = 0; r < p.length; r++) {
          row = new Array(p[r].length);
          for (c = 0; c < p[r].length; c++) row[c] = p[r][c] !== bg ? 1 : 0;
          m.push(row);
        }
        if (mask === null) mask = m;
        else if (!G.gEq(mask, m)) return null;
      }
    }
    return mask;
  }

  /* Top-left corner of each panel, recovered by matching the decomposition. */
  function _positions(g, decName, dec) {
    var mat = dec(g);
    if (!mat) return null;
    var h = g.length, w = g[0].length;
    var ph = mat[0][0] ? mat[0][0].length : 0, pw = mat[0][0] ? mat[0][0][0].length : 0;
    if (ph === 0 || pw === 0) return null;
    var nr = mat.length, nc = mat[0].length, rows = [], cols = [], i, j, r, c, win, k;
    r = 0;
    for (i = 0; i < nr; i++) {
      while (r + ph <= h) {
        win = [];
        for (k = r; k < r + ph; k++) win.push(g[k].slice(0, pw));
        if (G.gEq(win, mat[i][0])) break;
        r += 1;
      }
      if (r + ph > h) return null;
      rows.push(r);
      r += ph;
    }
    c = 0;
    for (j = 0; j < nc; j++) {
      while (c + pw <= w) {
        win = [];
        for (k = rows[0]; k < rows[0] + ph; k++) win.push(g[k].slice(c, c + pw));
        if (G.gEq(win, mat[0][j])) break;
        c += 1;
      }
      if (c + pw > w) return null;
      cols.push(c);
      c += pw;
    }
    return [rows, cols, ph, pw];
  }

  function _panelRule(dec, bg, hyp, table, mask, role, cmode) {
    function patchOf(code, ph, pw, fill) {
      var r, c, row, out;
      if (role) {
        if (mask === null || mask.length !== ph || mask[0].length !== pw) return null;
        if (!code) return G.constGrid(ph, pw, bg);
        if (fill === null || fill === undefined) return null;
        out = [];
        for (r = 0; r < mask.length; r++) {
          row = new Array(mask[r].length);
          for (c = 0; c < mask[r].length; c++) row[c] = mask[r][c] ? fill : bg;
          out.push(row);
        }
        return out;
      }
      if (mask !== null && mask !== undefined) {
        if (mask.length !== ph || mask[0].length !== pw) return null;
        if (code === bg) return G.constGrid(ph, pw, bg);
        out = [];
        for (r = 0; r < mask.length; r++) {
          row = new Array(mask[r].length);
          for (c = 0; c < mask[r].length; c++) row[c] = mask[r][c] ? code : bg;
          out.push(row);
        }
        return out;
      }
      var v = table.get(code);
      return v === undefined ? null : v;
    }
    return function (g) {
      var mat = dec(g);
      if (!mat) return null;
      var small = _abstract(mat, bg, role);
      if (small === null) return null;
      var pred = hyp.fn(small);
      if (pred === null || pred === undefined || !G.valid(pred)) return null;
      if (pred.length !== mat.length || pred[0].length !== mat[0].length) return null;
      var pos = _positions(g, null, dec);
      if (pos === null) return null;
      var rows = pos[0], cols = pos[1], ph = pos[2], pw = pos[3];
      var fill = role ? _fillColor(g, bg, cmode) : null;
      if (role && (fill === null || fill === undefined)) return null;
      var out = G.copyGrid(g), i, j, patch, rr, cc;
      for (i = 0; i < rows.length; i++) for (j = 0; j < cols.length; j++) {
        patch = patchOf(pred[i][j], ph, pw, fill);
        if (patch === null) return null;
        if (patch.length !== ph || patch[0].length !== pw) return null;
        for (rr = 0; rr < ph; rr++)
          for (cc = 0; cc < pw; cc++) out[rows[i] + rr][cols[j] + cc] = patch[rr][cc];
      }
      return out;
    };
  }

  function _modules() {
    var names = ["geometry", "colormap", "cellwise", "symmetry", "tiling",
                 "partition", "objects_map", "sequence"], out = [], i, m;
    for (i = 0; i < names.length; i++) { m = moduleByName(names[i]); if (m) out.push(m); }
    return out;
  }

  function _one(ctx, dname, dcost, dec, bg, role, deadline) {
    var res = [], pairs = [], t, ma, mb, sa, sb, tins = [], table = null, mask = null, i, j;
    try {
      for (t = 0; t < ctx.train.length; t++) {
        ma = dec(ctx.train[t][0]); mb = dec(ctx.train[t][1]);
        if (!ma || !mb || ma.length !== mb.length || ma[0].length !== mb[0].length) return res;
        sa = _abstract(ma, bg, role); sb = _abstract(mb, bg, role);
        if (sa === null || sb === null || G.gEq(sa, sb)) return res;
        pairs.push([sa, sb]);
      }
      if (!pairs.length) return res;
      table = role ? null : _renderTable(ctx, dec, bg);
      mask = _renderShape(ctx, dec, bg);
      if (table === null && mask === null) return res;
      for (t = 0; t < ctx.test_inputs.length; t++) {
        var m2 = dec(ctx.test_inputs[t]);
        if (!m2) return res;
        var s2 = _abstract(m2, bg, role);
        if (s2 === null) return res;
        tins.push(s2);
      }
    } catch (e) { return res; }

    var sub = new Ctx(pairs, tins);
    sub.deadline = Math.min(deadline, nowMs() + 1200);
    var variants = [];
    if (role) {
      if (mask !== null) {
        var cms = ["sep", "motif"].concat(G.csList(ctx.out_palette()).slice(0, 4));
        for (i = 0; i < cms.length; i++)
          variants.push(["role:" + cms[i], null, mask, true, cms[i]]);
      }
    } else {
      if (table !== null) variants.push(["tbl", table, null, false, null]);
      if (mask !== null) variants.push(["shp", null, mask, false, null]);
    }
    if (!variants.length) return res;
    var mods = _modules(), m, hyps, hit, hp;
    for (m = 0; m < mods.length; m++) {
      if (nowMs() > sub.deadline) break;
      try { hyps = hypcacheGenerate(mods[m], sub); } catch (e) { continue; }
      hit = null;
      for (i = 0; i < hyps.length; i++) { hp = hyps[i]; if (hp.fits(sub.train)) { hit = hp; break; } }
      if (hit === null) continue;
      for (j = 0; j < variants.length; j++)
        res.push(new Hyp("panel[" + dname + "|" + variants[j][0] + "]>>" + hit.name,
                         _panelRule(dec, bg, hit, variants[j][1], variants[j][2],
                                    variants[j][3], variants[j][4]),
                         dcost + 2.0 + hit.cost, "partition"));
      if (res.length > 30) break;
    }
    return res;
  }

  function generate(ctx) {
    if (!ctx.same_shape()) return [];
    var deadline = ctx.deadline === null ? (nowMs() + 4000) : ctx.deadline;
    var bg = ctx.bg(), res = [], decs = PART.decompositions(ctx).slice(0, 4), d, roles = [false, true], i;
    for (d = 0; d < decs.length; d++)
      for (i = 0; i < 2; i++) {
        if (nowMs() > deadline || res.length > 30) break;
        res = res.concat(_one(ctx, decs[d][0], decs[d][1], decs[d][2], bg, roles[i], deadline));
      }
    return res;
  }

  PANELABS.positions = _positions;
  PANELABS.abstract = _abstract;
  PANELABS.code = _code;

  defSolver("panelabs", "partition", generate, 2);
})();
