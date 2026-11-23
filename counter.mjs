// 跳绳计数核心：用躯干（双肩+双髋）的上下起伏数跳数。
// 输入每帧的 MediaPipe 姿态关键点（归一化坐标，y 向下为正）。
// 思路：
//   1. 用躯干长度做尺度，离镜头远近不影响阈值；
//   2. 慢速基线（约 0.7 秒）跟住人整体移动、蹲起、走动，只留下跳跃引起的快速起伏；
//   3. 阈值随最近 1.5 秒的起伏幅度自适应，再加一个最小值，站着不动时噪声数不上去；
//   4. 施密特触发（上阈值计数、必须回落过下阈值才允许下一次）防止一跳数两次；
//   5. 起跳必须是快速上升：400 毫秒内从低点升过整个阈值带，身体慢慢晃、慢慢站直不算；
//   6. 两次起跳至少间隔 150 毫秒（最快约 400 次/分钟）。
//
// 看绳子模式（ropeMode，默认开）：身体蹦一下不一定算一个。
//   绳子每转一圈从头顶扫过一次（app.js 检测后调用 pushRope）。
//   - 起跳前绳子刚转过来 → 算一个；没转过来（一圈蹦两下、绳子绊住还在蹦）→ 不算；
//   - 绳子扫过头顶比起跳晚一点点（120 毫秒内）也算，补回来；
//   - 算了之后绳子在一个周期内（≥1 秒）再没转过来 → 说明这一跳绊住了，撤回。

const SH_L = 11, SH_R = 12, HIP_L = 23, HIP_R = 24;

export class JumpCounter {
  constructor({ sensitivity = 1, ropeMode = true, p = {} } = {}) {
    this.ropeMode = ropeMode;
    this.p = Object.assign({ smTau: 30, baseTau: 700, minHi: 0.02, minLo: 0.012, kHi: 0.3, kLo: 0.2, riseMs: 400 }, p);
    this.sens = sensitivity;
    this.reset();
  }

  setSensitivity(s) { this.sens = s; }

  reset() {
    this.count = 0;
    this.times = [];        // 每次计数的时间（毫秒，调用方给的时间轴）
    this.scale = null;
    this.sm = null;
    this.base = null;
    this.state = 0;         // 0 = 落地段，1 = 腾空段
    this.lastCount = -1e9;
    this.lastT = null;
    this.hist = [];
    this.lostSince = null;
    this.hops = 0;          // 身体蹦的次数
    this.hopTimes = [];
    this.passes = 0;        // 绳子过头顶次数
    this.avail = false;     // 上次算数之后绳子转过来了
    this.pending = null;    // 等绳子再转过来确认的那一跳 {t, deadline}
    this.rejected = null;   // 刚被拒的起跳时间（等 120 毫秒看绳子是否稍晚扫过）
  }

  _period() {
    const h = this.hopTimes, n = h.length;
    if (n < 3) return 600;
    const d = [];
    for (let i = Math.max(1, n - 6); i < n; i++) d.push(h[i] - h[i - 1]);
    d.sort((a, b) => a - b);
    return d[d.length >> 1];
  }

  _add(t) { this.count++; this.times.push(t); }

  _hop(t) {
    this.hops++; this.hopTimes.push(t);
    if (this.hopTimes.length > 20) this.hopTimes.shift();
    if (!this.ropeMode) { this._add(t); return true; }
    if (this.avail) {
      this.avail = false;
      this._add(t);
      this.pending = { t, deadline: t + Math.min(1600, Math.max(1000, 1.6 * this._period())) };
      return true;
    }
    this.rejected = t;
    return false;
  }

  // 绳子从头顶扫过一次
  pushRope(t) {
    this.passes++;
    if (!this.ropeMode) return 0;
    if (this.pending && t <= this.pending.deadline) this.pending = null;
    if (this.rejected != null && t - this.rejected <= 80) {
      const rt = this.rejected;
      this.rejected = null;
      this._add(rt);
      this.pending = { t: rt, deadline: rt + Math.min(1600, Math.max(1000, 1.6 * this._period())) };
      return 1;
    }
    this.avail = true;
    return 0;
  }

  // 每帧调用：处理超时撤回。返回 -1 表示撤回了一个。
  tick(t) {
    if (this.rejected != null && t - this.rejected > 80) this.rejected = null;
    if (this.pending && t > this.pending.deadline) {
      const i = this.times.lastIndexOf(this.pending.t);
      if (i >= 0) { this.times.splice(i, 1); this.count--; }
      this.pending = null;
      this.avail = false;
      return -1;
    }
    return 0;
  }

  // t: 毫秒；lm: 33 个关键点数组或 null。返回本帧的诊断信息。
  push(t, lm) {
    const ok = lm && [SH_L, SH_R, HIP_L, HIP_R].every(i => lm[i] && (lm[i].visibility ?? 1) > 0.5);
    if (!ok) {
      if (this.lostSince == null) this.lostSince = t;
      if (t - this.lostSince > 500) {   // 丢人超过半秒：重新起算基线，避免回到画面时误计
        this.sm = this.base = null;
        this.state = 0;
        this.hist = [];
      }
      this.lastT = t;
      return { ok: false, counted: false };
    }
    this.lostSince = null;

    const sy = (lm[SH_L].y + lm[SH_R].y) / 2;
    const hy = (lm[HIP_L].y + lm[HIP_R].y) / 2;
    const torso = Math.abs(hy - sy);
    if (torso < 0.02) return { ok: false, counted: false };

    this.scale = this.scale == null ? torso : this.scale + 0.03 * (torso - this.scale);
    const s = -((sy + hy) / 2) / this.scale;          // 向上为正，单位：躯干长度

    const dt = this.lastT == null ? 33 : Math.max(1, Math.min(200, t - this.lastT));
    this.lastT = t;
    this.sm = this.sm == null ? s : this.sm + (1 - Math.exp(-dt / this.p.smTau)) * (s - this.sm);
    this.base = this.base == null ? this.sm : this.base + (1 - Math.exp(-dt / this.p.baseTau)) * (this.sm - this.base);
    const d = this.sm - this.base;

    this.hist.push(t, d);
    while (this.hist.length > 2 && t - this.hist[0] > 1500) this.hist.splice(0, 2);
    let mx = -Infinity, mn = Infinity, recentMin = Infinity;
    for (let i = 1; i < this.hist.length; i += 2) {
      const v = this.hist[i]; if (v > mx) mx = v; if (v < mn) mn = v;
      if (t - this.hist[i - 1] <= this.p.riseMs && v < recentMin) recentMin = v;
    }
    const half = (mx - mn) / 2;

    const hi = Math.max(this.p.minHi / this.sens, this.p.kHi * half);
    const lo = -Math.max(this.p.minLo / this.sens, this.p.kLo * half);

    let counted = false, hop = false;
    if (this.state === 0 && d > hi && d - recentMin >= hi - lo && t - this.lastCount >= 150) {
      this.state = 1;
      this.lastCount = t;
      hop = true;
      counted = this._hop(t);
    } else if (this.state === 1 && d < lo) {
      this.state = 0;
    }
    return { ok: true, counted, hop, d, hi, lo };
  }
}

// 头顶横带的帧差特征。g/prev：灰度图（dw×dh），c0..c1：看绳子的列范围，其余（留 5% 边距）当对照。
export function ropeFeatures(g, prev, dw, dh, c0, c1) {
  const zone = c1 - c0, n = dw * dh;
  const margin = Math.round(0.05 * dw);
  let ch = 0, side = 0, sideN = 0, covered = 0, sumN = 0, run = 0, best = 0, gap = 0;
  for (let x = 0; x < dw; x++) {
    let nc = 0;
    for (let y = 0; y < dh; y++) { const k = y * dw + x; if (Math.abs(g[k] - prev[k]) > 18) nc++; }
    if (x >= c0 && x < c1) {
      ch += nc;
      if (nc > 0) { covered++; sumN += nc; run += 1 + gap; gap = 0; if (run > best) best = run; }
      else if (run > 0 && gap < 2) gap++;
      else { run = 0; gap = 0; }
    } else if (x < c0 - margin || x >= c1 + margin) { side += nc; sideN += dh; }
  }
  return {
    e: ch / (zone * dh),
    eSide: sideN > 0.1 * n ? side / sideN : null,
    lineLen: best / zone,                          // 最长连续变化的宽度占比：绳子是一条横线
    thick: covered ? sumN / covered / dh : 1,      // 每列变化厚度占比：绳子细，晃动/光线变化是整片
  };
}
