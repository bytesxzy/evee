/* Port of engine/task.py -- the solver context and the no-leak boundary.
 *
 * A Ctx is built from the train pairs and the test *inputs* only. Test outputs
 * never reach it, so a solver cannot read the answer it is scored on; the
 * harness holds the answers and compares afterwards.
 */

function nowMs() { return Date.now(); }

function Ctx(train, testInputs, deadline) {
  var i;
  this.train = [];
  for (i = 0; i < train.length; i++)
    this.train.push([G.asGrid(train[i][0]), G.asGrid(train[i][1])]);
  if (!this.train.length) throw new Error("at least one training pair is required");
  this.test_inputs = [];
  for (i = 0; i < testInputs.length; i++) this.test_inputs.push(G.asGrid(testInputs[i]));
  this.deadline = (deadline === undefined) ? null : deadline;
  this._cache = {};
  this.op_prior = null;
}

Ctx.prototype.memo = function (key, fn) {
  if (!(key in this._cache)) this._cache[key] = fn();
  return this._cache[key];
};

Ctx.prototype.inputs = function () {
  return this.memo("inputs", (function (self) {
    return function () { var o = [], i; for (i = 0; i < self.train.length; i++) o.push(self.train[i][0]); return o; };
  })(this));
};

Ctx.prototype.outputs = function () {
  return this.memo("outputs", (function (self) {
    return function () { var o = [], i; for (i = 0; i < self.train.length; i++) o.push(self.train[i][1]); return o; };
  })(this));
};

Ctx.prototype.all_inputs = function () {
  return this.memo("all_inputs", (function (self) {
    return function () { return self.inputs().concat(self.test_inputs); };
  })(this));
};

Ctx.prototype.same_shape = function () {
  return this.memo("same_shape", (function (self) {
    return function () {
      var i, a, b;
      for (i = 0; i < self.train.length; i++) {
        a = self.train[i][0]; b = self.train[i][1];
        if (a.length !== b.length || a[0].length !== b[0].length) return false;
      }
      return true;
    };
  })(this));
};

Ctx.prototype.const_out_shape = function () {
  return this.memo("cshape", (function (self) {
    return function () {
      var seen = {}, keys = [], i, b, k;
      for (i = 0; i < self.train.length; i++) {
        b = self.train[i][1]; k = b.length + "x" + b[0].length;
        if (!seen[k]) { seen[k] = [b.length, b[0].length]; keys.push(k); }
      }
      return keys.length === 1 ? seen[keys[0]] : null;
    };
  })(this));
};

Ctx.prototype.shape_ratio = function () {
  return this.memo("ratio", (function (self) {
    return function () {
      var seen = {}, keys = [], i, a, b, ky, kx, k;
      for (i = 0; i < self.train.length; i++) {
        a = self.train[i][0]; b = self.train[i][1];
        if (b.length % a.length || b[0].length % a[0].length) return null;
        ky = b.length / a.length; kx = b[0].length / a[0].length;
        k = ky + "," + kx;
        if (!seen[k]) { seen[k] = [ky, kx]; keys.push(k); }
      }
      return keys.length === 1 ? seen[keys[0]] : null;
    };
  })(this));
};

Ctx.prototype.inv_shape_ratio = function () {
  return this.memo("iratio", (function (self) {
    return function () {
      var seen = {}, keys = [], i, a, b, ky, kx, k;
      for (i = 0; i < self.train.length; i++) {
        a = self.train[i][0]; b = self.train[i][1];
        if (a.length % b.length || a[0].length % b[0].length) return null;
        ky = a.length / b.length; kx = a[0].length / b[0].length;
        k = ky + "," + kx;
        if (!seen[k]) { seen[k] = [ky, kx]; keys.push(k); }
      }
      return keys.length === 1 ? seen[keys[0]] : null;
    };
  })(this));
};

/* out = a * in + b with small integral coefficients: a law, not a curve fitted
   through three points. */
function _affineFit(pairs) {
  var a, b, i, ok, distinct;
  for (a = 0; a <= 4; a++) {
    for (b = -6; b <= 12; b++) {
      ok = true;
      for (i = 0; i < pairs.length; i++)
        if (pairs[i][1] !== a * pairs[i][0] + b) { ok = false; break; }
      if (ok) {
        if (a === 0) {
          distinct = {};
          for (i = 0; i < pairs.length; i++) distinct[pairs[i][0]] = 1;
          if (Object.keys(distinct).length < 2) return null;
        }
        return [a, b];
      }
    }
  }
  return null;
}

Ctx.prototype.affine_shape = function () {
  return this.memo("affine", (function (self) {
    return function () {
      var hs = [], ws = [], i, a, b;
      for (i = 0; i < self.train.length; i++) {
        a = self.train[i][0]; b = self.train[i][1];
        hs.push([a.length, b.length]);
        ws.push([a[0].length, b[0].length]);
      }
      var fy = _affineFit(hs), fx = _affineFit(ws);
      return (fy && fx) ? [fy, fx] : null;
    };
  })(this));
};

Ctx.prototype.bg = function () {
  return this.memo("bg", (function (self) {
    return function () {
      var cnt = new Int32Array(G.NCOLORS), order = [], i, v;
      for (i = 0; i < self.train.length; i++) {
        v = G.background(self.train[i][0]);
        if (!cnt[v]) order.push(v);
        cnt[v]++;
      }
      if (cnt[0]) return 0;
      var best = order[0];
      for (i = 0; i < order.length; i++) if (cnt[order[i]] > cnt[best]) best = order[i];
      return best;
    };
  })(this));
};

Ctx.prototype.bg_varies = function () {
  return this.memo("bgvar", (function (self) {
    return function () {
      var all = self.all_inputs(), m = 0, i;
      for (i = 0; i < all.length; i++) m |= 1 << G.background(all[i]);
      return G.csSize(m) > 1;
    };
  })(this));
};

Ctx.prototype.in_palette = function () {
  return this.memo("inpal", (function (self) {
    return function () {
      var all = self.all_inputs(), m = 0, i;
      for (i = 0; i < all.length; i++) m |= G.palette(all[i]);
      return m;
    };
  })(this));
};

Ctx.prototype.out_palette = function () {
  return this.memo("outpal", (function (self) {
    return function () {
      var outs = self.outputs(), m = 0, i;
      for (i = 0; i < outs.length; i++) m |= G.palette(outs[i]);
      return m;
    };
  })(this));
};

Ctx.prototype.new_colors = function () { return G.csDiff(this.out_palette(), this.in_palette()); };

Ctx.prototype.dropped_colors = function () {
  var ins = this.inputs(), m = 0, i;
  for (i = 0; i < ins.length; i++) m |= G.palette(ins[i]);
  return G.csDiff(m, this.out_palette());
};

Ctx.prototype.timed_out = function () {
  return this.deadline !== null && nowMs() > this.deadline;
};

/* A candidate rule: a total function from an input grid to an output grid. */
function Hyp(name, fn, cost, solver) {
  this.name = name;
  this.fn = fn;
  this.cost = (cost === undefined) ? 10.0 : cost;
  this.solver = solver || "";
}

Hyp.prototype.apply = function (g) {
  try {
    var r = this.fn(g);
    if (r === null || r === undefined) return null;
    return G.isGrid(r) ? r : G.asGrid(r);
  } catch (e) { return null; }
};

Hyp.prototype.fits = function (train) {
  var i, p;
  for (i = 0; i < train.length; i++) {
    p = this.apply(train[i][0]);
    if (p === null || !G.gEq(p, train[i][1])) return false;
  }
  return true;
};
