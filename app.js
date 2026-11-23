import { FilesetResolver, PoseLandmarker } from './lib/vision_bundle.mjs';
import { JumpCounter } from './counter.mjs';

const $ = id => document.getElementById(id);
const video = $('cam'), overlay = $('overlay'), ctx = overlay.getContext('2d');
const wave = $('wave'), wctx = wave.getContext('2d');
const stage = $('stage'), countEl = $('count'), timerEl = $('timer'), statusEl = $('status');
const startBtn = $('startBtn'), cdEl = $('countdown'), fpsEl = $('fps'), errEl = $('err');

const counter = new JumpCounter();
let landmarker = null;
let facing = 'user';
let stream = null;
let source = 'camera';          // camera | file
let phase = 'idle';             // idle | countdown | running | done
let modeSec = 60;
let runStart = 0, runEnd = 0;
let recorder = null, recChunks = [], recStart = 0, replayURL = null, fileURL = null;
let replayTimes = [];           // 每次计数在录像里的秒数
let autoCount = 0, finalCount = 0;
let lastDetectTs = 0, frames = 0, fpsT = performance.now(), fps = 0;
let waveBuf = [];
let audio = null, wakeLock = null;

// ---------- 设置 ----------
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
modeSec = store.get('jr-mode', 60);
$('sens').value = store.get('jr-sens', 1);
$('beep').checked = store.get('jr-beep', true);
$('rec').checked = store.get('jr-rec', true);
$('ropeMode').checked = store.get('jr-rope', true);
counter.ropeMode = $('ropeMode').checked;
counter.setSensitivity(+$('sens').value);
for (const b of $('mode').querySelectorAll('button')) {
  b.setAttribute('aria-checked', String(+b.dataset.sec === modeSec));
  b.onclick = () => {
    if (phase === 'running' || phase === 'countdown') return;
    modeSec = +b.dataset.sec; store.set('jr-mode', modeSec);
    for (const x of $('mode').querySelectorAll('button')) x.setAttribute('aria-checked', String(x === b));
    showTimer(0);
  };
}
$('sens').oninput = e => { counter.setSensitivity(+e.target.value); store.set('jr-sens', +e.target.value); };
$('beep').onchange = e => store.set('jr-beep', e.target.checked);
$('rec').onchange = e => store.set('jr-rec', e.target.checked);
$('ropeMode').onchange = e => { store.set('jr-rope', e.target.checked); if (phase !== 'running') counter.ropeMode = e.target.checked; };

// ---------- 模型 ----------
// 模型文件约 15MB。自己下载并显示进度；本站卡住超过 12 秒就换镜像。
const MIRRORS = ['./lib/', 'https://cdn.jsdelivr.net/gh/fewmichael/jump-rope@main/lib/'];
const NPM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/';

async function fetchProgress(urls, onProg) {
  let lastErr;
  for (const url of urls) {
    const ctrl = new AbortController();
    let timer;
    const arm = () => { clearTimeout(timer); timer = setTimeout(() => ctrl.abort(), 12000); };
    try {
      arm();
      const res = await fetch(url, { signal: ctrl.signal });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const reader = res.body.getReader();
      const chunks = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length; arm(); onProg(got);
      }
      clearTimeout(timer);
      return new Blob(chunks);
    } catch (e) { clearTimeout(timer); lastErr = e; onProg(0); }
  }
  throw lastErr || new Error('下载失败');
}

async function loadModel() {
  const simd = await FilesetResolver.isSimdSupported().catch(() => true);
  const w = simd ? 'vision_wasm_internal' : 'vision_wasm_nosimd_internal';
  const files = [
    { urls: MIRRORS.map(m => m + 'wasm/' + w + '.js').concat(NPM + 'wasm/' + w + '.js'), size: 0.21e6, type: 'text/javascript' },
    { urls: MIRRORS.map(m => m + 'wasm/' + w + '.wasm').concat(NPM + 'wasm/' + w + '.wasm'), size: 9.4e6, type: 'application/wasm' },
    { urls: MIRRORS.map(m => m + 'pose_landmarker_lite.task'), size: 5.78e6, type: 'application/octet-stream' },
  ];
  const total = files.reduce((a, f) => a + f.size, 0);
  const got = files.map(() => 0);
  const show = () => {
    const pct = Math.min(99, Math.round(got.reduce((a, b) => a + b, 0) / total * 100));
    statusEl.textContent = `下载识别模型 ${pct}%（只有第一次要下载）`;
  };
  show();
  const blobs = await Promise.all(files.map((f, k) => fetchProgress(f.urls, n => { got[k] = n; show(); })
    .then(b => new Blob([b], { type: f.type }))));
  statusEl.textContent = '正在启动识别…';
  const fileset = { wasmLoaderPath: URL.createObjectURL(blobs[0]), wasmBinaryPath: URL.createObjectURL(blobs[1]) };
  const model = new Uint8Array(await blobs[2].arrayBuffer());
  const opts = delegate => ({
    baseOptions: { modelAssetBuffer: model, delegate },
    runningMode: 'VIDEO', numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  const timeout = ms => new Promise((_, rej) => setTimeout(() => rej(new Error('GPU 启动超时')), ms));
  try { landmarker = await Promise.race([PoseLandmarker.createFromOptions(fileset, opts('GPU')), timeout(10000)]); }
  catch { landmarker = await PoseLandmarker.createFromOptions(fileset, opts('CPU')); }
}

// ---------- 摄像头 ----------
async function startCamera() {
  stopStream();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
    });
  } catch (e) {
    showErr('打不开摄像头：' + (e.name === 'NotAllowedError' ? '请在浏览器里允许使用摄像头。' : e.message) +
      (location.protocol !== 'https:' && location.hostname !== 'localhost' ? '（网页必须用 https 打开才能用摄像头）' : ''));
    return;
  }
  source = 'camera';
  video.srcObject = stream; video.src = '';
  video.muted = true;
  await video.play().catch(() => {});
  stage.classList.toggle('mirror', facing === 'user');
}
function stopStream() { if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; } }

video.addEventListener('loadedmetadata', () => {
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  stage.style.aspectRatio = video.videoWidth + ' / ' + video.videoHeight;
});

$('flip').onclick = async () => {
  if (phase === 'running' || phase === 'countdown') return;
  facing = facing === 'user' ? 'environment' : 'user';
  await startCamera();
};

// ---------- 看绳子：头顶上方区域的画面变化 ----------
// 头顶上方本来是静止背景，绳子每转一圈从这里扫过一次，画面会“闪”一下。
const ropeCv = document.createElement('canvas');
const rctx = ropeCv.getContext('2d', { willReadFrequently: true });
const rope = { box: null, prev: null, e: 0, bg: 0.002, last: -1e9, above: false, headTop: null, torso: null, room: 1 };

function ropeReset() { rope.box = null; rope.prev = null; rope.bg = 0.002; rope.last = -1e9; rope.above = false; rope.headTop = null; rope.torso = null; }
const sens = () => +$('sens').value;

function ropeStep(lm, t) {
  const vis = i => (lm[i].visibility ?? 1) > .5;
  if (![0, 11, 12, 23, 24].every(vis)) { rope.prev = null; return null; }
  const W = video.videoWidth, H = video.videoHeight;
  const sy = (lm[11].y + lm[12].y) / 2, hy = (lm[23].y + lm[24].y) / 2, torso = hy - sy;
  const headTop = lm[0].y - 0.45 * torso;
  // 用慢速位置定区域，跳起来时区域不跟着动
  const a = 0.05;
  rope.headTop = rope.headTop == null ? headTop : rope.headTop + a * (headTop - rope.headTop);
  rope.torso = rope.torso == null ? torso : rope.torso + a * (torso - rope.torso);
  const bottom = rope.headTop - 0.35 * rope.torso;
  const top = Math.max(0, bottom - 0.9 * rope.torso);
  rope.room = (bottom - top) / (0.9 * rope.torso);          // 头顶上方空间够不够（1 = 够）
  const cx = (lm[11].x + lm[12].x) / 2;
  const half = Math.max(0.9 * rope.torso * H / W, 1.4 * Math.abs(lm[11].x - lm[12].x));
  const x0 = Math.max(0, cx - half), x1 = Math.min(1, cx + half);
  const box = { x: Math.round(x0 * W), y: Math.round(top * H), w: Math.round((x1 - x0) * W), h: Math.round((bottom - top) * H) };
  if (box.w < 20 || box.h < 8) { rope.prev = null; return null; }
  // 区域明显移动了才换，换的时候丢掉上一帧
  const ob = rope.box;
  if (!ob || Math.abs(ob.x - box.x) > 6 || Math.abs(ob.y - box.y) > 6 || Math.abs(ob.w - box.w) > 10 || Math.abs(ob.h - box.h) > 10) { rope.box = box; rope.prev = null; }
  const b = rope.box;
  const sc = Math.min(1, 200 / b.w);
  const dw = Math.max(8, Math.round(b.w * sc)), dh = Math.max(4, Math.round(b.h * sc));
  if (ropeCv.width !== dw || ropeCv.height !== dh) { ropeCv.width = dw; ropeCv.height = dh; rope.prev = null; }
  rctx.drawImage(video, b.x, b.y, b.w, b.h, 0, 0, dw, dh);
  const px = rctx.getImageData(0, 0, dw, dh).data;
  const n = dw * dh, g = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) g[i] = (px[j] + 2 * px[j + 1] + px[j + 2]) >> 2;
  // 手举过头顶时这一帧不看
  const handsUp = (vis(15) && lm[15].y < bottom) || (vis(16) && lm[16].y < bottom);
  let e = 0;
  if (rope.prev && !handsUp) {
    let c = 0;
    for (let i = 0; i < n; i++) if (Math.abs(g[i] - rope.prev[i]) > 18) c++;
    e = c / n;
  }
  rope.prev = g;
  rope.e = e;
  // 背景变化水平（慢）+ 尖峰检测
  const th = Math.max(0.012 / sens(), 4 * rope.bg);
  let pass = false;
  if (!rope.above && e > th && t - rope.last > 140) { rope.above = true; rope.last = t; pass = true; }
  else if (rope.above && e < th * 0.5) rope.above = false;
  if (!rope.above) rope.bg += 0.03 * (Math.min(e, 2 * rope.bg + 0.001) - rope.bg);
  return { pass, e, th };
}

// ---------- 识别循环 ----------
const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32]];

function loop() {
  if (landmarker && video.readyState >= 2 && !video.paused) {
    let ts = performance.now();
    if (ts <= lastDetectTs) ts = lastDetectTs + 1;
    lastDetectTs = ts;
    const res = landmarker.detectForVideo(video, ts);
    const lm = res.landmarks && res.landmarks[0];
    frames++;
    draw(lm);
    const t = source === 'file' ? video.currentTime * 1000 : ts;
    const rs = lm ? ropeStep(lm, t) : null;
    if (phase === 'running') {
      let rescued = 0;
      if (rs && rs.pass) rescued = counter.pushRope(t);
      const r = counter.push(t, lm);
      const back = counter.tick(t);
      if (r.counted || rescued) onCount(t);
      if (back) onRetract();
      if (r.ok) { waveBuf.push({ t, d: r.d, hi: r.hi, lo: r.lo, c: r.counted || rescued > 0, h: r.hop, p: !!(rs && rs.pass), e: rs ? rs.e / Math.max(rs.th, 1e-4) : 0 }); }
      updateSub();
      while (waveBuf.length && t - waveBuf[0].t > 6000) waveBuf.shift();
      drawWave(t);
    }
    setStatus(lm);
  }
  const now = performance.now();
  if (now - fpsT > 1000) {
    fps = frames * 1000 / (now - fpsT); frames = 0; fpsT = now;
    fpsEl.textContent = landmarker ? fps.toFixed(0) + ' 帧/秒' : '';
    fpsEl.classList.toggle('low', fps > 0 && fps < 20);
  }
  if (phase === 'running') tickTimer();
  if (video.requestVideoFrameCallback && !video.paused) video.requestVideoFrameCallback(loop);
  else requestAnimationFrame(loop);
}

function draw(lm) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!lm) return;
  const W = overlay.width, H = overlay.height;
  ctx.lineWidth = Math.max(2, W / 160); ctx.strokeStyle = 'rgba(60,207,142,.85)';
  ctx.beginPath();
  for (const [a, b] of BONES) {
    if ((lm[a].visibility ?? 1) < .4 || (lm[b].visibility ?? 1) < .4) continue;
    ctx.moveTo(lm[a].x * W, lm[a].y * H); ctx.lineTo(lm[b].x * W, lm[b].y * H);
  }
  ctx.stroke();
}

let lastStatus = '';
function setStatus(lm) {
  let txt, cls;
  if (!lm) { txt = '没看到人'; cls = 'bad'; }
  else {
    const vis = i => (lm[i].visibility ?? 1) > .5;
    const torso = [11, 12, 23, 24].every(vis);
    const feet = [27, 28].every(vis) && lm[27].y < .98 && lm[28].y < .98;
    const head = lm[0].y > .02;
    if (!torso) { txt = '身体没拍全'; cls = 'bad'; }
    else if (!feet || !head) { txt = '再退后一点，头到脚都要进画面'; cls = 'bad'; }
    else if (counter.ropeMode && rope.room < 0.7) { txt = '头顶上方要留出空间'; cls = 'bad'; }
    else { txt = phase === 'running' ? '计数中' : '位置合适'; cls = 'ok'; }
  }
  const key = txt + cls;
  if (key !== lastStatus) { statusEl.textContent = txt; statusEl.className = 'status ' + cls; lastStatus = key; }
}

function drawWave(now) {
  const W = wave.width = wave.clientWidth * devicePixelRatio, H = wave.height = wave.clientHeight * devicePixelRatio;
  wctx.clearRect(0, 0, W, H);
  if (waveBuf.length < 2) return;
  const span = 6000, maxA = Math.max(0.08, ...waveBuf.map(p => Math.abs(p.d)), ...waveBuf.map(p => p.hi)) * 1.15;
  const x = t => W - (now - t) / span * W, y = v => H / 2 - v / maxA * (H / 2 - 4);
  const css = getComputedStyle(document.documentElement);
  wctx.lineWidth = devicePixelRatio;
  wctx.strokeStyle = css.getPropertyValue('--line'); wctx.setLineDash([4, 4]);
  for (const k of ['hi', 'lo']) { wctx.beginPath(); waveBuf.forEach((p, i) => i ? wctx.lineTo(x(p.t), y(p[k])) : wctx.moveTo(x(p.t), y(p[k]))); wctx.stroke(); }
  wctx.setLineDash([]); wctx.lineWidth = 2 * devicePixelRatio; wctx.strokeStyle = css.getPropertyValue('--fg');
  wctx.beginPath(); waveBuf.forEach((p, i) => i ? wctx.lineTo(x(p.t), y(p.d)) : wctx.moveTo(x(p.t), y(p.d))); wctx.stroke();
  // 绳子信号（蓝，底部）
  wctx.strokeStyle = css.getPropertyValue('--rope'); wctx.lineWidth = 1.5 * devicePixelRatio;
  wctx.beginPath(); waveBuf.forEach((p, i) => { const yy = H - 2 - Math.min(1.5, p.e) / 1.5 * (H * 0.4); i ? wctx.lineTo(x(p.t), yy) : wctx.moveTo(x(p.t), yy); }); wctx.stroke();
  for (const p of waveBuf) {
    if (p.h || p.c) { wctx.fillStyle = css.getPropertyValue(p.c ? '--accent' : '--dim'); wctx.beginPath(); wctx.arc(x(p.t), y(p.d), 4 * devicePixelRatio, 0, 7); wctx.fill(); }
    if (p.p) { wctx.fillStyle = css.getPropertyValue('--rope'); wctx.fillRect(x(p.t) - devicePixelRatio, H - 10 * devicePixelRatio, 2 * devicePixelRatio, 10 * devicePixelRatio); }
  }
}

// ---------- 计数与计时 ----------
function onCount(t) {
  countEl.textContent = counter.count;
  countEl.classList.add('bump'); setTimeout(() => countEl.classList.remove('bump'), 90);
  if ($('beep').checked) beep(1200, 0.04, 0.15);
}
function onRetract() { countEl.textContent = counter.count; }
function toReplay(t) { return source === 'file' ? t / 1000 : (t - recStart) / 1000; }
function updateSub() {
  $('sub').textContent = counter.ropeMode
    ? `身体蹦 ${counter.hops} 下 · 绳子转 ${counter.passes} 圈`
    : `身体蹦 ${counter.hops} 下`;
}

function fmt(ms) { const s = Math.max(0, Math.ceil(ms / 1000)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }
function showTimer(elapsed) { timerEl.textContent = modeSec ? fmt(modeSec * 1000 - elapsed) : fmt(elapsed); }
function tickTimer() {
  const el = source === 'file' ? video.currentTime * 1000 - runStart : performance.now() - runStart;
  showTimer(el);
  if (modeSec && el >= modeSec * 1000) finish();
}

function beep(freq, dur, vol = 0.25) {
  if (!audio) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = freq; g.gain.value = vol;
  g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + dur);
  o.connect(g).connect(audio.destination); o.start(); o.stop(audio.currentTime + dur);
}

startBtn.onclick = () => {
  if (phase === 'running' || phase === 'countdown') { finish(); return; }
  try { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); audio.resume(); } catch {}
  if (source !== 'camera') { startCamera().then(begin); return; }
  begin();
};

async function begin() {
  if (!stream) return;
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {}
  $('result').hidden = true; $('replay').hidden = true;
  phase = 'countdown'; startBtn.textContent = '停止'; startBtn.classList.add('stop');
  counter.reset(); ropeReset(); countEl.textContent = '0'; waveBuf = []; replayTimes = []; updateSub();
  showTimer(0);
  cdEl.hidden = false;
  for (const n of [3, 2, 1]) {
    if (phase !== 'countdown') return;
    cdEl.textContent = n; beep(660, 0.15);
    await new Promise(r => setTimeout(r, 1000));
  }
  if (phase !== 'countdown') return;
  cdEl.hidden = true; beep(990, 0.35);
  startRecording();
  phase = 'running'; runStart = performance.now();
}

function startRecording() {
  recorder = null; recChunks = [];
  if (!$('rec').checked || !window.MediaRecorder) return;
  const type = ['video/mp4', 'video/webm;codecs=vp9', 'video/webm'].find(t => MediaRecorder.isTypeSupported?.(t));
  try {
    recorder = new MediaRecorder(stream, type ? { mimeType: type, videoBitsPerSecond: 1_500_000 } : undefined);
    recorder.ondataavailable = e => e.data.size && recChunks.push(e.data);
    recorder.start(500);
    recStart = performance.now();
  } catch { recorder = null; }
}

function finish() {
  const wasCountdown = phase === 'countdown';
  if (phase !== 'running' && !wasCountdown) return;
  runEnd = source === 'file' ? video.currentTime * 1000 : performance.now();
  phase = 'done'; cdEl.hidden = true;
  startBtn.textContent = '开始'; startBtn.classList.remove('stop');
  try { wakeLock?.release(); } catch {}
  if (wasCountdown) { phase = 'idle'; return; }
  beep(520, 0.6);
  if (source === 'file') video.pause();
  autoCount = finalCount = counter.count;
  replayTimes = counter.times.map(toReplay);
  const secs = (runEnd - runStart) / 1000;
  $('resNum').textContent = finalCount;
  $('resMeta').textContent = `用时 ${secs.toFixed(0)} 秒 · 平均 ${secs > 3 ? Math.round(finalCount / secs * 60) : '—'} 次/分钟 · 身体蹦了 ${counter.hops} 下，绳子转了 ${counter.passes} 圈` + (counter.ropeMode && counter.hops >= 5 && counter.passes === 0 ? '。没看到绳子：头顶上方要留空，背景尽量干净浅色；或者关掉「看绳子」再跳' : '');
  $('result').hidden = false;
  const done = () => {
    if (replayURL && replayURL !== fileURL) URL.revokeObjectURL(replayURL);
    replayURL = source === 'file' ? fileURL : (recChunks.length ? URL.createObjectURL(new Blob(recChunks, { type: recorder?.mimeType || 'video/webm' })) : null);
    $('openReplay').hidden = !replayURL;
  };
  if (recorder && recorder.state !== 'inactive') { recorder.onstop = done; recorder.stop(); } else done();
  $('result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

$('minus').onclick = () => { finalCount = Math.max(0, finalCount - 1); $('resNum').textContent = finalCount; };
$('plus').onclick = () => { finalCount++; $('resNum').textContent = finalCount; };
$('save').onclick = () => {
  const h = store.get('jr-history', []);
  h.unshift({ at: Date.now(), mode: source === 'file' ? 'video' : modeSec, count: finalCount, auto: autoCount, hops: counter.hops });
  store.set('jr-history', h.slice(0, 100));
  renderHistory();
  $('save').textContent = '已保存'; setTimeout(() => $('save').textContent = '保存', 1500);
};

// ---------- 回放核对 ----------
const rv = $('rv'), rvCount = $('rvCount');
$('openReplay').onclick = () => {
  if (!replayURL) return;
  rv.src = replayURL; rv.playbackRate = 0.5;
  $('replay').hidden = false; rv.play().catch(() => {});
  $('replay').scrollIntoView({ behavior: 'smooth' });
  replayLoop();
};
for (const b of $('replay').querySelectorAll('[data-rate]')) b.onclick = () => { rv.playbackRate = +b.dataset.rate; rv.play().catch(() => {}); };
let lastRv = -1;
function replayLoop() {
  if ($('replay').hidden) return;
  const ct = rv.currentTime;
  let n = 0; while (n < replayTimes.length && replayTimes[n] <= ct) n++;
  if (n !== lastRv) {
    rvCount.textContent = n;
    if (n > lastRv && lastRv >= 0) { rvCount.classList.add('flash'); setTimeout(() => rvCount.classList.remove('flash'), 150); }
    lastRv = n;
  }
  requestAnimationFrame(replayLoop);
}

// ---------- 分析已有视频 ----------
$('file').onchange = async e => {
  const f = e.target.files[0]; if (!f || phase === 'running') return;
  e.target.value = '';
  try { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); audio.resume(); } catch {}
  stopStream();
  if (fileURL) URL.revokeObjectURL(fileURL);
  fileURL = URL.createObjectURL(f);
  source = 'file'; stage.classList.remove('mirror');
  video.srcObject = null; video.src = fileURL; video.muted = true;
  video.playbackRate = 0.5;   // 慢放分析，每一跳拿到更多帧
  $('result').hidden = true; $('replay').hidden = true;
  counter.reset(); ropeReset(); countEl.textContent = '0'; waveBuf = []; replayTimes = []; updateSub();
  video.onended = () => { if (phase === 'running') finish(); };
  await video.play().catch(() => {});
  video.playbackRate = 0.5;
  phase = 'running'; runStart = video.currentTime * 1000;
  startBtn.textContent = '停止'; startBtn.classList.add('stop');
  const prev = modeSec; modeSec = 0; showTimer(0);
  video.addEventListener('pause', () => { modeSec = prev; }, { once: true });
};

// ---------- 记录 ----------
function renderHistory() {
  const h = store.get('jr-history', []);
  const ol = $('history'); ol.innerHTML = '';
  $('histEmpty').hidden = h.length > 0;
  for (const r of h.slice(0, 30)) {
    const li = document.createElement('li');
    const d = new Date(r.at);
    const when = `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
    const mode = r.mode === 'video' ? '视频' : r.mode ? r.mode + ' 秒' : '自由';
    const fix = r.count !== r.auto ? `（程序 ${r.auto}）` : '';
    li.innerHTML = `<span class="when"></span><span></span><b></b>`;
    li.children[0].textContent = when; li.children[1].textContent = mode + fix; li.children[2].textContent = r.count;
    ol.appendChild(li);
  }
}

function showErr(m) { errEl.textContent = m; errEl.hidden = false; }

// ---------- 启动 ----------
renderHistory(); showTimer(0);
window.addEventListener('error', e => showErr('出错了：' + (e.message || e.error)));
window.addEventListener('unhandledrejection', e => showErr('出错了：' + (e.reason && e.reason.message || e.reason)));
(async () => {
  const cam = startCamera();            // 先开摄像头，模型边下边等
  loop();
  try { await loadModel(); }
  catch (e) { showErr('识别模型加载失败：' + (e.message || e) + '。请检查网络后刷新；在微信里打不开的话，点右上角「…」选「在浏览器打开」。'); statusEl.textContent = '模型加载失败'; return; }
  statusEl.textContent = '模型就绪';
  await cam;
  startBtn.disabled = false;
})();
