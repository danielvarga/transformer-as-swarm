// A plain-JS port of swarm.py's Swarm.forward, for the browser app (and node, for checking against PyTorch).
// State is a Float64Array of n * d numbers, boid-major: boid i is x[i*d .. i*d+d).

(function (root) {
  function decode(b64) {
    const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  // Exact GELU needs erf; Abramowitz-Stegun 7.1.26 is good to ~1.5e-7.
  function erf(x) {
    const s = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x * Math.SQRT1_2));

  function sunflower(n, radius) {
    const xy = new Float64Array(2 * n);
    const golden = Math.PI * (1 + Math.sqrt(5));
    for (let k = 0; k < n; k++) {
      const i = k + 0.5;
      const r = radius * Math.sqrt(i / n);
      xy[2 * k] = r * Math.cos(golden * i);
      xy[2 * k + 1] = r * Math.sin(golden * i);
    }
    return xy;
  }

  class Swarm {
    constructor(w) {
      this.d = w.d;
      this.nhead = w.nhead;
      this.steps = w.steps;
      this.scale = w.step_scale;
      this.diskRadius = w.disk_radius;
      this.mask = w.update_mask;
      this.codes = w.codes;
      this.trainedN = w.trained_n_boids;
      this.qkv = decode(w.qkv);
      this.outW = decode(w.out_w);
      this.outB = decode(w.out_b);
      this.ffn = w.ffn.map((l) => ({ w: decode(l.w), b: decode(l.b), nin: l.in, nout: l.out }));
      this.maxWidth = Math.max(...this.ffn.map((l) => Math.max(l.nin, l.nout)));
    }

    initialState(n, label, style) {
      const d = this.d;
      const xy = sunflower(n, this.diskRadius);
      const x = new Float64Array(n * d);
      const hidden = [...this.codes[label], ...style];
      for (let i = 0; i < n; i++) {
        x[i * d] = xy[2 * i];
        x[i * d + 1] = xy[2 * i + 1];
        for (let j = 0; j < hidden.length; j++) x[i * d + 2 + j] = hidden[j];
      }
      return x;
    }

    // One application of the block, in place: x += flock(x); x += solo(x).
    step(x, n) {
      const d = this.d, H = this.nhead, W = this.qkv, mask = this.mask, sc = this.scale;
      // q, k, v per head: qkv row index = s*H*d + h*d + j (matches view(b, n, 3, H, d)).
      const q = new Float64Array(H * n * d), k = new Float64Array(H * n * d), v = new Float64Array(H * n * d);
      const bufs = [q, k, v];
      for (let i = 0; i < n; i++) {
        const xi = i * d;
        for (let s = 0; s < 3; s++) {
          const buf = bufs[s];
          for (let h = 0; h < H; h++) {
            for (let j = 0; j < d; j++) {
              const row = ((s * H + h) * d + j) * d;
              let acc = 0;
              for (let c = 0; c < d; c++) acc += W[row + c] * x[xi + c];
              buf[(h * n + i) * d + j] = acc;
            }
          }
        }
      }
      const msg = new Float64Array(n * d);
      const scores = new Float64Array(n);
      for (let h = 0; h < H; h++) {
        const base = h * n * d;
        for (let i = 0; i < n; i++) {
          const qi = base + i * d;
          let mx = -Infinity;
          for (let jj = 0; jj < n; jj++) {
            const kj = base + jj * d;
            let s = 0;
            for (let c = 0; c < d; c++) s += q[qi + c] * k[kj + c];
            scores[jj] = s;
            if (s > mx) mx = s;
          }
          let z = 0;
          for (let jj = 0; jj < n; jj++) z += (scores[jj] = Math.exp(scores[jj] - mx));
          const wz = 1 / (z * H);
          for (let jj = 0; jj < n; jj++) {
            const a = scores[jj] * wz, vj = base + jj * d;
            for (let c = 0; c < d; c++) msg[i * d + c] += a * v[vj + c];
          }
        }
      }
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
          let acc = this.outB[j];
          for (let c = 0; c < d; c++) acc += this.outW[j * d + c] * msg[i * d + c];
          x[i * d + j] += sc * acc * mask[j];
        }
      }
      let a = new Float64Array(this.maxWidth), b = new Float64Array(this.maxWidth);
      const last = this.ffn.length - 1;
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < d; c++) a[c] = x[i * d + c];
        for (let li = 0; li <= last; li++) {
          const L = this.ffn[li];
          for (let o = 0; o < L.nout; o++) {
            let acc = L.b[o];
            const row = o * L.nin;
            for (let c = 0; c < L.nin; c++) acc += L.w[row + c] * a[c];
            b[o] = li < last ? gelu(acc) : acc;
          }
          const t = a; a = b; b = t;
        }
        for (let j = 0; j < d; j++) x[i * d + j] += sc * a[j] * mask[j];
      }
      return x;
    }

    // Everything one block application does to boid i in state x, without changing x:
    // its input, each head's attention over the swarm, the flocking move, the FFN's hidden
    // activations (after GELU), the raw FFN output and the alone-time move. flock + solo is exactly the step.
    inspect(x, n, i) {
      const d = this.d, H = this.nhead, W = this.qkv, sc = this.scale, mask = this.mask;
      const proj = (s, h, b) => {
        const o = new Float64Array(d);
        for (let j = 0; j < d; j++) {
          const row = ((s * H + h) * d + j) * d;
          let acc = 0;
          for (let c = 0; c < d; c++) acc += W[row + c] * x[b * d + c];
          o[j] = acc;
        }
        return o;
      };
      const msg = new Float64Array(d), attn = [];
      for (let h = 0; h < H; h++) {
        const q = proj(0, h, i), a = new Float64Array(n), vs = [];
        let mx = -Infinity;
        for (let b = 0; b < n; b++) {
          const k = proj(1, h, b);
          let s = 0;
          for (let c = 0; c < d; c++) s += q[c] * k[c];
          a[b] = s;
          if (s > mx) mx = s;
          vs.push(proj(2, h, b));
        }
        let z = 0;
        for (let b = 0; b < n; b++) z += (a[b] = Math.exp(a[b] - mx));
        for (let b = 0; b < n; b++) {
          a[b] /= z;
          for (let c = 0; c < d; c++) msg[c] += (a[b] * vs[b][c]) / H;
        }
        attn.push(a);
      }
      const input = x.slice(i * d, i * d + d), flock = new Float64Array(d);
      for (let j = 0; j < d; j++) {
        let acc = this.outB[j];
        for (let c = 0; c < d; c++) acc += this.outW[j * d + c] * msg[c];
        flock[j] = sc * acc * mask[j];
      }
      const mid = Float64Array.from(input, (v, j) => v + flock[j]);
      let a = mid;
      const hidden = [], last = this.ffn.length - 1;
      for (let li = 0; li <= last; li++) {
        const L = this.ffn[li], b = new Float64Array(L.nout);
        for (let o = 0; o < L.nout; o++) {
          let acc = L.b[o];
          for (let c = 0; c < L.nin; c++) acc += L.w[o * L.nin + c] * a[c];
          b[o] = li < last ? gelu(acc) : acc;
        }
        if (li < last) hidden.push(b);
        a = b;
      }
      const solo = Float64Array.from(a, (v, j) => sc * v * mask[j]);
      return { input, attn, flock, mid, hidden, out: a, solo };
    }

    // Full trajectory: an array of steps+1 states, each a Float64Array(n*d).
    run(n, label, style, steps = this.steps) {
      let x = this.initialState(n, label, style);
      const traj = [x.slice()];
      for (let t = 0; t < steps; t++) traj.push(this.step(x, n).slice());
      return traj;
    }
  }

  const api = { Swarm, sunflower };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SwarmModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
