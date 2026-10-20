// Runs the swarm off the main thread, so large flocks don't freeze the page.
// Protocol: {type: "init", scripts} loads swarm.js and the model files;
// {type: "run", id, model, n, x, from, to} steps state x from step `from` to `to`, posting
// {id, t, x} after every step; a newer run (or {type: "cancel", id}) stops the current one.
self.window = self; // the model files register themselves on window.SWARM_MODELS

let current = 0;
const built = {};

self.onmessage = (e) => {
  const m = e.data;
  if (m.type === "init") importScripts(...m.scripts);
  else if (m.type === "run") { current = m.id; run(m); }
  else if (m.type === "cancel" && current === m.id) current = 0;
};

async function run(m) {
  const sw = (built[m.model] ||= new self.SwarmModel.Swarm(self.SWARM_MODELS.find((w) => w.name === m.model)));
  const x = m.x;
  for (let t = m.from; t < m.to && current === m.id; ) {
    sw.step(x, m.n);
    t++;
    const out = x.slice();
    self.postMessage({ id: m.id, t, x: out }, [out.buffer]);
    await new Promise((r) => setTimeout(r, 0)); // let a newer run or a cancel arrive
  }
}
