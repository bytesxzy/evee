/* Port of engine/solvers/colormap.py -- colour rules that generalise beyond
 * the colours actually seen.
 *
 * A literal lookup table fails the moment a test grid introduces a colour that
 * never appeared in training. These rules key on rank, frequency and role, so
 * they transfer.
 */

(function () {
  var _h = mkHyp("colormap");

  function _rankOrder(g, bg, asc) {
    var hist = G.histogram(g), items = [], k;
    for (k = 0; k < G.NCOLORS; k++) if (hist[k] > 0 && k !== bg) items.push([k, hist[k]]);
    items.sort(function (a, b) { return (a[1] - b[1]) || (a[0] - b[0]); });
    if (!asc) items.reverse();
    var out = [];
    for (k = 0; k < items.length; k++) out.push(items[k][0]);
    return out;
  }

  function _fitRankPerm(ctx, bg, asc) {
    var perm = new Map(), t, a, b, order, idx, r, c, x, y, i;
    for (t = 0; t < ctx.train.length; t++) {
      a = ctx.train[t][0]; b = ctx.train[t][1];
      if (a.length !== b.length || a[0].length !== b[0].length) return null;
      order = _rankOrder(a, bg, asc);
      idx = new Map();
      for (i = 0; i < order.length; i++) idx.set(order[i], i);
      for (r = 0; r < a.length; r++) for (c = 0; c < a[r].length; c++) {
        x = a[r][c]; y = b[r][c];
        if (x === bg) { if (y !== bg) return null; continue; }
        i = idx.get(x);
        if (i === undefined) return null;
        if (!perm.has(i)) perm.set(i, y);
        else if (perm.get(i) !== y) return null;
      }
    }
    return perm.size ? perm : null;
  }

  function _applyRank(g, perm, bg, asc) {
    var order = _rankOrder(g, bg, asc), m = new Map(), i;
    for (i = 0; i < order.length; i++) {
      if (perm.has(i)) m.set(order[i], perm.get(i));
      else return null;
    }
    return G.applyCmap(g, m);
  }

  function _swapTop(g, bg, i, j) {
    var order = _rankOrder(g, bg, false);
    if (order.length <= Math.max(i, j)) return null;
    var a = order[i], b = order[j], m = new Map();
    m.set(a, b); m.set(b, a);
    return G.applyCmap(g, m);
  }

  function _swapMinmax(g, bg) {
    var order = _rankOrder(g, bg, false);
    if (order.length < 2) return null;
    var a = order[0], b = order[order.length - 1], m = new Map();
    m.set(a, b); m.set(b, a);
    return G.applyCmap(g, m);
  }

  function _keep(g, c, bg) {
    var out = [], r, j, row, nr;
    for (r = 0; r < g.length; r++) {
      row = g[r]; nr = new Array(row.length);
      for (j = 0; j < row.length; j++) nr[j] = row[j] === c ? c : bg;
      out.push(nr);
    }
    return out;
  }

  function _keepRank(g, bg, rarest) {
    var order = _rankOrder(g, bg, rarest);
    if (!order.length) return null;
    return _keep(g, order[0], bg);
  }

  function _allTo(g, bg, rarest) {
    var order = _rankOrder(g, bg, rarest);
    if (!order.length) return null;
    var c = order[0], out = [], r, j, row, nr;
    for (r = 0; r < g.length; r++) {
      row = g[r]; nr = new Array(row.length);
      for (j = 0; j < row.length; j++) nr[j] = row[j] !== bg ? c : bg;
      out.push(nr);
    }
    return out;
  }

  var GRID_PROPS = [
    ["ncolors", function (g, bg) { return String(G.csSize(G.palette(g))); }],
    ["nnz_parity", function (g, bg) {
      var n = 0, r, c;
      for (r = 0; r < g.length; r++) for (c = 0; c < g[r].length; c++) if (g[r][c] !== bg) n++;
      return String(n % 2);
    }],
    ["shape", function (g, bg) { return g.length + "x" + g[0].length; }],
    ["maxcolor", function (g, bg) {
      var p = G.csList(G.palette(g));
      return String(p[p.length - 1]);
    }],
    ["nobj", function (g, bg) { return String(G.floodRegions(g, bg, true, true).length); }],
    ["symm", function (g, bg) { return G.symmetries(g).join(","); }]
  ];

  /* out = const(P(input)) -- the answer is a lookup on a global property. A
     table keyed so finely that every training pair gets its own row has
     memorised the data, so real compression is required. */
  function _conditionalMaps(ctx, bg) {
    var res = [], i, t, table, ok, k, a, b;
    for (i = 0; i < GRID_PROPS.length; i++) {
      table = new Map(); ok = true;
      for (t = 0; t < ctx.train.length; t++) {
        a = ctx.train[t][0]; b = ctx.train[t][1];
        try { k = GRID_PROPS[i][1](a, bg); } catch (e) { ok = false; break; }
        if (!table.has(k)) table.set(k, b);
        else if (!G.gEq(table.get(k), b)) { ok = false; break; }
      }
      if (ok && table.size > 1 && table.size <= ctx.train.length - 2) {
        res.push(_h("lookup_" + GRID_PROPS[i][0], (function (tb, p) {
          return function (g) {
            var key;
            try { key = p(g, bg); } catch (e) { return null; }
            var v = tb.get(key);
            return v === undefined ? null : v;
          };
        })(table, GRID_PROPS[i][1]), 9.0));
      }
    }
    return res;
  }

  function generate(ctx) {
    var res = [], bg = ctx.bg(), i, pal;

    if (ctx.same_shape()) {
      var ascs = [false, true], a;
      for (a = 0; a < 2; a++) {
        var perm = _fitRankPerm(ctx, bg, ascs[a]);
        if (perm !== null)
          res.push(_h("rankmap_" + (ascs[a] === false ? "desc" : "asc"),
                      (function (p, asc) { return function (g) { return _applyRank(g, p, bg, asc); }; })(perm, ascs[a]),
                      4.0));
      }
      res.push(_h("swap_top2", function (g) { return _swapTop(g, bg, 0, 1); }, 4.5));
      res.push(_h("swap_minmax", function (g) { return _swapMinmax(g, bg); }, 4.5));
    }

    pal = G.csList(ctx.out_palette());
    for (i = 0; i < pal.length; i++)
      res.push(_h("solid#" + pal[i],
                  (function (c) { return function (g) { return G.constGrid(g.length, g[0].length, c); }; })(pal[i]), 6.0));

    pal = G.csList(ctx.in_palette());
    for (i = 0; i < pal.length; i++) {
      (function (c) {
        res.push(_h("keep#" + c, function (g) { return _keep(g, c, bg); }, 4.5));
        res.push(_h("drop#" + c, function (g) { return G.replaceColor(g, c, bg); }, 4.0));
      })(pal[i]);
    }

    res.push(_h("keep_rarest", function (g) { return _keepRank(g, bg, true); }, 5.0));
    res.push(_h("keep_commonest", function (g) { return _keepRank(g, bg, false); }, 5.0));
    res.push(_h("recolor_all_rarest", function (g) { return _allTo(g, bg, true); }, 5.5));
    res.push(_h("recolor_all_commonest", function (g) { return _allTo(g, bg, false); }, 5.5));

    return res.concat(_conditionalMaps(ctx, bg));
  }

  defSolver("colormap", "colormap", generate);
})();
