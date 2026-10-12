// Check the JS port against PyTorch (models/<name>.ref.json from export_weights.py) and time it.
//   node web/check.js
const fs = require("fs");
const path = require("path");
const { Swarm } = require("./swarm.js");

const dir = path.join(__dirname, "models");
global.window = {};
for (const f of fs.readdirSync(dir).filter((f) => f.endsWith(".js"))) eval(fs.readFileSync(path.join(dir, f), "utf8"));

for (const w of window.SWARM_MODELS) {
  const sw = new Swarm(w);
  const refs = JSON.parse(fs.readFileSync(path.join(dir, `${w.name}.ref.json`), "utf8"));
  console.log(`== ${w.name}`);
  for (const r of refs) {
    const traj = sw.run(r.n, r.label, r.style);
    for (const [name, t, ref] of [["step 1", traj[1], r.step1], ["final", traj[sw.steps], r.final]]) {
      let m = 0;
      ref.forEach((row, i) => row.forEach((v, j) => (m = Math.max(m, Math.abs(v - t[i * sw.d + j])))));
      console.log(`digit ${r.label}, ${r.n} boids, ${name}: max abs error ${m.toExponential(2)}`);
    }
  }
}
const sw = new Swarm(window.SWARM_MODELS[0]);
for (const n of [64, 128, 256, 512]) {
  const t0 = performance.now();
  sw.run(n, 5, [0, 0]);
  console.log(`${n} boids: ${(performance.now() - t0).toFixed(0)} ms for ${sw.steps} steps`);
}
