// Check the JS port against PyTorch (reference.json from export_weights.py) and time it.
//   node web/check.js
const fs = require("fs");
const path = require("path");
const { Swarm } = require("./swarm.js");

const src = fs.readFileSync(path.join(__dirname, "weights.js"), "utf8");
const w = JSON.parse(src.slice(src.indexOf("{"), src.lastIndexOf("}") + 1));
const refs = JSON.parse(fs.readFileSync(path.join(__dirname, "reference.json"), "utf8"));
const sw = new Swarm(w);

for (const r of refs) {
  const traj = sw.run(r.n, r.label, r.style);
  for (const [name, t, ref] of [["step 1", traj[1], r.step1], ["final", traj[sw.steps], r.final]]) {
    let m = 0;
    ref.forEach((row, i) => row.forEach((v, j) => (m = Math.max(m, Math.abs(v - t[i * sw.d + j])))));
    console.log(`digit ${r.label}, ${r.n} boids, ${name}: max abs error ${m.toExponential(2)}`);
  }
}
for (const n of [64, 128, 256, 512]) {
  const t0 = performance.now();
  sw.run(n, 5, [0, 0]);
  console.log(`${n} boids: ${(performance.now() - t0).toFixed(0)} ms for ${sw.steps} steps`);
}
