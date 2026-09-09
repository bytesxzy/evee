/* Ports of engine/solvers/compose.py and engine/solvers/enumerate_dsl.py.
 *
 * compose is a fast shallow pass (depth 2, small op set) that runs before the
 * full enumerator, so short programs are found and ranked cheaply.
 * enumerate_dsl is the general fallback: depth 4 with a binary combination
 * layer, consuming whatever budget is left, which is why it is registered last.
 */

(function () {
  function generateCompose(ctx) {
    var dl = ctx.deadline === null ? (nowMs() + 5000) : ctx.deadline;
    dl = Math.min(dl, nowMs() + 4000);
    var found = enumSearch(ctx, 2, 600, dl, "full", false), out = [], i;
    for (i = 0; i < found.length; i++)
      out.push(new Hyp(found[i][0], found[i][2], 2.0 + found[i][1], "compose"));
    return out;
  }
  defSolver("compose", "compose", generateCompose, 2);

  function generateEnum(ctx) {
    var dl = ctx.deadline === null ? (nowMs() + 10000) : ctx.deadline;
    var found = enumSearch(ctx, 4, 1500, dl, "full", true, ctx.op_prior), out = [], i;
    for (i = 0; i < found.length; i++)
      out.push(new Hyp(found[i][0], found[i][2], 3.0 + found[i][1], "enumerate"));
    return out;
  }
  defSolver("enumerate_dsl", "enumerate", generateEnum, 2);
})();
