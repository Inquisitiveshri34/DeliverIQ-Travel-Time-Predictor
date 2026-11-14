/* DeliverIQ UI: one bidirectional Streamlit component holding both screens.
 *
 * Python -> here (args):  metrics, feature_importance, result
 * here -> Python (value): { request_id, inputs }
 *
 * Nothing the user enters is stored or logged; inputs live in memory and are
 * sent to Python only when Predict Travel Time is pressed.
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var app = $('app');

  // ───────────────────────── Streamlit component protocol ─────────────────────────

  function post(type, data) {
    var msg = { isStreamlitMessage: true, type: type };
    for (var k in data) msg[k] = data[k];
    window.parent.postMessage(msg, '*');
  }

  var lastHeight = -1;
  function syncHeight() {
    var h = Math.ceil(app.getBoundingClientRect().height);
    if (h !== lastHeight) { lastHeight = h; post('streamlit:setFrameHeight', { height: h }); }
  }

  // The iframe is as tall as its content, so vh inside it is useless. --pvh is
  // 1% of the host page's viewport height (the component is served same-origin).
  function syncViewport() {
    var h = 0;
    try { h = window.parent.innerHeight; } catch (e) { h = 0; }
    if (!h) h = (window.screen && window.screen.availHeight) || 800;
    document.documentElement.style.setProperty('--pvh', (h / 100) + 'px');
  }

  function scrollHostToTop() {
    try {
      for (var el = window.frameElement; el; el = el.parentElement) el.scrollTop = 0;
      window.parent.scrollTo(0, 0);
    } catch (e) { /* cross-origin host: nothing to scroll from here */ }
    window.scrollTo(0, 0);
  }

  function reducedMotion() {
    return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }

  // ───────────────────────── Small DOM helpers ─────────────────────────

  function bind(name, text) {
    var nodes = document.querySelectorAll('[data-bind="' + name + '"]');
    for (var i = 0; i < nodes.length; i++) nodes[i].textContent = text;
  }

  function el(tag, css, text) {
    var n = document.createElement(tag);
    if (css) n.style.cssText = css;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  // Index-keyed list sync: existing children are updated in place (so CSS
  // transitions run), extras are removed, missing ones are created.
  function syncList(container, items, create, update) {
    while (container.children.length > items.length) container.removeChild(container.lastChild);
    for (var i = 0; i < items.length; i++) {
      var node = container.children[i];
      if (!node) { node = create(items[i], i); container.appendChild(node); }
      update(node, items[i], i);
    }
  }

  var pad = function (n) { return (n < 10 ? '0' : '') + n; };
  var isNum = function (v) { return typeof v === 'number' && isFinite(v); };
  var fixed = function (v, dp, suffix) { return isNum(v) ? v.toFixed(dp) + (suffix || '') : '–'; };
  var grouped = function (v) { return isNum(v) ? v.toLocaleString('en-US') : '–'; };

  function segStyle(on, offBg, offFg, offBorder) {
    return on ? { bg: '#F5B83D', fg: '#1A1405', border: '#F5B83D' } : { bg: offBg, fg: offFg, border: offBorder };
  }
  function paintSeg(btn, on, st) {
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    btn.style.border = '1px solid ' + st.border;
    btn.style.background = st.bg;
    btn.style.color = st.fg;
  }

  // ───────────────────────── Data from Python ─────────────────────────

  var metrics = {};
  var featureImportance = [];
  var dataKey = null;

  function mae() { return isNum(metrics.MAE) ? metrics.MAE : 0; }
  function trees() {
    var p = metrics.best_params || {};
    return p.model__n_estimators;
  }
  // Python-style rendering, as the prototype shows "True" for model__bootstrap.
  function paramText(v) {
    if (v === true) return 'True';
    if (v === false) return 'False';
    if (v === null || v === undefined) return 'None';
    return String(v);
  }

  function renderData() {
    var m = metrics;
    bind('modelName', m.model_name || '–');
    bind('trees', trees() === undefined ? '–' : String(trees()));
    bind('trainRows', grouped(m.training_rows));
    bind('testRows', grouped(m.test_rows));
    bind('logTransform', m.log_transform ? 'Yes' : 'No');
    bind('mae', fixed(m.MAE, 2, ' min'));
    bind('rmse', fixed(m.RMSE, 2, ' min'));
    bind('r2', fixed(m.R2, 4));
    bind('mape', fixed(m.MAPE, 2, '%'));
    bind('baselineMae', fixed(m.baseline_MAE, 2, ' min'));
    var imp = m.improvement_pct;
    bind('improvement', isNum(imp) ? (imp >= 0 ? '−' : '+') + Math.abs(imp).toFixed(1) + '%' : '–');
    var share = isNum(m.MAE) && isNum(m.baseline_MAE) && m.baseline_MAE > 0 ? m.MAE / m.baseline_MAE : 0;
    $('model-bar').style.width = (Math.min(1, Math.max(0, share)) * 100).toFixed(1) + '%';

    // Best Params
    var params = m.best_params || {};
    var keys = Object.keys(params);
    var box = $('params');
    box.textContent = '';
    keys.forEach(function (k, i) {
      var row = el('div', 'display: flex; justify-content: space-between; gap: 12px; padding: 6px 0' + (i < keys.length - 1 ? '; border-bottom: 1px solid #1E2A42' : ''));
      row.appendChild(el('span', 'color: #9AA8C0', k));
      row.appendChild(el('span', '', paramText(params[k])));
      box.appendChild(row);
    });

    // Feature Importance (Top 15)
    var rows = featureImportance.filter(function (f) { return f && isNum(f.importance); });
    var top = rows.reduce(function (a, f) { return Math.max(a, f.importance); }, 0);
    var list = $('features');
    list.textContent = '';
    rows.forEach(function (f, i) {
      var row = el('div', 'display: grid; grid-template-columns: minmax(0, 12.5rem) minmax(0, 1fr) 3.4rem; align-items: center; gap: 12px');
      row.appendChild(el('span', "font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #C9D3E6; overflow-wrap: anywhere", f.feature));
      var track = el('div', 'height: 14px; border-radius: 3px; background: #172238');
      track.appendChild(el('div', 'height: 100%; width: ' + (top > 0 ? f.importance / top * 100 : 0).toFixed(1) + '%; min-width: 3px; border-radius: 3px; background: ' + (i === 0 ? '#F5B83D' : '#4FB3D4') + '; transform-origin: left; animation: grow .9s ' + (i * 0.05).toFixed(2) + 's cubic-bezier(.2,.8,.2,1) both'));
      row.appendChild(track);
      row.appendChild(el('span', "font-family: 'JetBrains Mono', monospace; font-size: 12px; text-align: right", (f.importance * 100).toFixed(1) + '%'));
      list.appendChild(row);
    });

    // Caption: the prototype's sentence, with the names and share read from the data.
    var cap = $('features-caption');
    cap.textContent = '';
    var mono = function (t) { return el('span', "font-family: 'JetBrains Mono', monospace", t); };
    var plain = function (name) { return String(name).replace(/^[a-z]+__/, ''); };
    if (rows.length >= 3) {
      cap.appendChild(mono(plain(rows[0].feature)));
      cap.appendChild(document.createTextNode(' dominates at ~' + Math.round(rows[0].importance * 100) + '%, consistent with how '));
      cap.appendChild(mono('travel_time_min'));
      cap.appendChild(document.createTextNode(' was generated. '));
      cap.appendChild(mono(plain(rows[1].feature)));
      cap.appendChild(document.createTextNode(' and '));
      cap.appendChild(mono(plain(rows[2].feature)));
      cap.appendChild(document.createTextNode(' are next most informative.'));
    }
  }

  // ───────────────────────── Screen 1: animated route scene ─────────────────────────

  var SCENARIOS = [
    { label: 'Morning rush, city', distance: 12, lanes: 2, signals: 14, road: 'urban', traffic: 'high', weather: 'clear', hour: 8, day: 'Monday' },
    { label: 'Foggy highway haul', distance: 60, lanes: 4, signals: 2, road: 'highway', traffic: 'low', weather: 'fog', hour: 5, day: 'Thursday' },
    { label: 'Rainy evening, residential', distance: 6, lanes: 1, signals: 4, road: 'residential', traffic: 'medium', weather: 'rain', hour: 18, day: 'Saturday' }
  ];
  var SCENERY_HEIGHTS = [52, 78, 40, 90, 64, 46, 84, 58, 72, 44, 88, 60, 50, 80, 42, 70, 56, 86, 48, 74, 62, 92, 44, 68];
  var CYCLE_MS = 7000;

  var scene = { idx: 0, tick: 0, timer: null };

  function armScene() {
    if (scene.timer) clearInterval(scene.timer);
    scene.timer = setInterval(function () {
      scene.idx = (scene.idx + 1) % SCENARIOS.length;
      scene.tick += 1;
      renderScene();
    }, CYCLE_MS);
  }
  function stopScene() {
    if (scene.timer) { clearInterval(scene.timer); scene.timer = null; }
  }

  function renderScene() {
    var s = SCENARIOS[scene.idx];
    var H = 380, mid = 236;
    var roadH = 18 + s.lanes * 8;
    var roadTop = mid - roadH / 2;
    var roadW = 22 + (s.distance - 1) / 99 * 66;
    var laneH = (roadH - 4) / s.lanes;
    var i;

    var laneLines = [];
    for (i = 1; i < s.lanes; i++) laneLines.push({ top: (i * laneH - 1).toFixed(1) + 'px' });

    var signalDots = [];
    for (i = 0; i < s.signals; i++) signalDots.push({ left: (6 + roadW * (i + 1) / (s.signals + 1)).toFixed(2) + '%', delay: '-' + (i * 0.9 % 4).toFixed(1) + 's' });

    var count = { low: 4, medium: 9, high: 15 }[s.traffic];
    var baseDur = { low: 4.5, medium: 8, high: 13 }[s.traffic];
    var vehicles = [];
    for (i = 0; i < count; i++) {
      var d = baseDur + (i % 3) * 0.8;
      vehicles.push({ top: ((i % s.lanes) * laneH + laneH / 2 - 3).toFixed(1) + 'px', dur: d.toFixed(1) + 's', delay: '-' + (i * d / count + (i % 2) * 1.3).toFixed(2) + 's' });
    }

    var scenery = SCENERY_HEIGHTS.map(function (h, n) {
      var left = (1 + n * 4.15).toFixed(2) + '%';
      if (s.road === 'urban') return { left: left, width: '3.2%', height: h + 'px', bg: '#1A2843', pattern: 'repeating-linear-gradient(0deg, transparent 0 9px, rgba(245,184,61,.16) 9px 12px)', clip: 'none' };
      if (s.road === 'residential') return { left: left, width: '2.7%', height: (24 + (h % 3) * 5) + 'px', bg: '#22324F', pattern: 'none', clip: 'polygon(0 40%, 50% 0, 100% 40%, 100% 100%, 0 100%)' };
      return { left: left, width: '0.35%', height: '14px', bg: '#4A5C7E', pattern: 'none', clip: 'none' };
    });

    var isDay = s.hour >= 6 && s.hour <= 18;
    var arc = isDay ? (s.hour - 6) / 12 : ((s.hour + 24 - 19) % 24) / 10;
    var sky = (s.hour <= 4 || s.hour >= 20) ? '#0A1222' : s.hour <= 6 ? '#2A2742' : s.hour <= 16 ? '#17335A' : '#3A2A3E';
    var ab = scene.tick % 2 === 0 ? 'A' : 'B';

    var sceneEl = $('scene');
    sceneEl.style.background = sky;
    sceneEl.setAttribute('aria-label', 'Animated example route: ' + s.label + ', ' + s.distance + ' km on a ' + s.road + ' road with ' + s.traffic + ' traffic in ' + s.weather + ' weather');

    var orb = $('orb');
    orb.style.left = (8 + arc * 80).toFixed(1) + '%';
    orb.style.top = (150 - Math.sin(Math.PI * Math.min(1, Math.max(0, arc))) * 70).toFixed(0) + 'px';
    orb.style.background = isDay ? '#F5B83D' : '#C9D4E8';
    orb.style.boxShadow = '0 0 46px 8px ' + (isDay ? 'rgba(245,184,61,.35)' : 'rgba(201,212,232,.22)');

    var sceneryBottom = (H - roadTop + 26) + 'px';
    syncList($('scenery'), scenery,
      function () { return el('div', 'position: absolute; transition: height .7s, width .7s, bottom .5s'); },
      function (node, b) {
        node.style.left = b.left;
        node.style.width = b.width;
        node.style.height = b.height;
        node.style.bottom = sceneryBottom;
        node.style.background = b.bg;
        node.style.backgroundImage = b.pattern;
        node.style.clipPath = b.clip;
      });

    var rail = $('rail');
    rail.hidden = s.road !== 'highway';
    rail.style.bottom = (H - roadTop + 30) + 'px';

    $('ground').style.top = (roadTop - 26) + 'px';

    var signalTop = (roadTop - 22) + 'px';
    syncList($('signals'), signalDots,
      function () {
        var pole = el('div', 'position: absolute; width: 2px; height: 22px; margin-left: -1px; background: #55678A; transition: left .7s, top .5s');
        pole.appendChild(el('div', 'position: absolute; left: -4px; top: -6px; width: 10px; height: 10px; border-radius: 50%; background: #7CE3A1'));
        return pole;
      },
      function (node, dot) {
        node.style.left = dot.left;
        node.style.top = signalTop;
        node.firstChild.style.animation = 'sig 4s linear ' + dot.delay + ' infinite';
      });

    var road = $('road');
    road.style.width = roadW.toFixed(2) + '%';
    road.style.top = roadTop + 'px';
    road.style.height = roadH + 'px';

    syncList($('lanes'), laneLines,
      function () { return el('div', 'position: absolute; left: 0; right: 0; border-top: 2px dashed #7385A6; opacity: .7'); },
      function (node, l) { node.style.top = l.top; });

    syncList($('vehicles'), vehicles,
      function () { return el('div', 'position: absolute; width: 14px; height: 6px; border-radius: 2px; background: #9FB0CC'); },
      function (node, v) {
        node.style.top = v.top;
        node.style.animation = 'drive ' + v.dur + ' linear ' + v.delay + ' infinite';
      });

    // Alternating between two identical keyframes restarts the run on every tick.
    $('trail').style.animation = 'trail' + ab + ' 7s cubic-bezier(.3,.1,.3,1) both';

    var pinTop = (roadTop + roadH + 6) + 'px';
    $('pin-start').style.top = pinTop;
    $('pin-end').style.top = pinTop;
    $('pin-end').style.left = (6 + roadW).toFixed(2) + '%';
    $('pin-end-km').textContent = s.distance.toFixed(1);

    var lane = $('truck-lane');
    lane.style.width = roadW.toFixed(2) + '%';
    lane.style.top = (roadTop - 8) + 'px';
    $('truck-run').style.animation = 'truck' + ab + ' 7s cubic-bezier(.3,.1,.3,1) both';

    $('rain').hidden = s.weather !== 'rain';
    $('fog').hidden = s.weather !== 'fog';

    var chips = [s.distance.toFixed(1) + ' km', s.road + ' · ' + s.lanes + (s.lanes === 1 ? ' lane' : ' lanes'), s.signals + ' signals', s.traffic + ' traffic · ' + s.weather, pad(s.hour) + ':00 ' + s.day];
    syncList($('chips'), chips,
      function () { return el('span', "padding: 4px 10px; border-radius: 6px; background: rgba(11,18,32,.84); font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #E8EDF6"); },
      function (node, c) { node.textContent = c; });

    syncList($('scenarios'), SCENARIOS,
      function (def, n) {
        var b = el('button', 'min-height: 44px; padding: 0 14px; border-radius: 8px; font-size: 13px; font-weight: 500; transition: background .2s, color .2s', def.label);
        b.type = 'button';
        b.className = 'seg';
        b.addEventListener('click', function () {
          scene.idx = n;
          scene.tick += 1;
          renderScene();
          armScene();
        });
        return b;
      },
      function (node, def, n) {
        var on = n === scene.idx;
        paintSeg(node, on, segStyle(on, 'rgba(11,18,32,.84)', '#E8EDF6', '#34486B'));
      });
  }

  // ───────────────────────── Screen 2: form state ─────────────────────────

  // Same lookups as _AVG_SPEED / _VISIBILITY in app.py, used only for the live
  // badges. The model always receives the values Python derives.
  var AVG_SPEED = { low: 60, medium: 35, high: 20 };
  var VISIBILITY = { clear: 10, rain: 4, fog: 1.5 };
  var RUSH_HOURS = [7, 8, 17, 18];
  var DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  var PRESETS = [
    { label: 'Morning rush, city', v: { distance: 12, lanes: 2, signals: 14, road: 'urban', traffic: 'high', weather: 'clear', temp: 28, hour: 8, dow: 0 } },
    { label: 'Foggy highway haul', v: { distance: 60, lanes: 4, signals: 2, road: 'highway', traffic: 'low', weather: 'fog', temp: 9, hour: 5, dow: 3 } },
    { label: 'Rainy evening, residential', v: { distance: 6, lanes: 1, signals: 4, road: 'residential', traffic: 'medium', weather: 'rain', temp: 22, hour: 18, dow: 5 } }
  ];
  var ANIM_MS = 2600;
  var LOADER_MS = 1700;
  var ONE_DP = { distance_km: 1, avg_speed_kmph: 1, temperature_c: 1, visibility_km: 1 };

  var form = { distance: 15, lanes: 2, signals: 5, road: 'urban', traffic: 'medium', weather: 'clear', temp: 28, hour: 9, dow: 0 };
  // phase: 'idle' | 'running' | 'done'. While 'running', pred is null until
  // Python answers; the truck bobs at the start in the meantime.
  var run = { phase: 'idle', progress: 0, pred: null, shown: 0, stale: false, requestId: null, inputRow: null, raf: null, summaryOpen: false };

  function newRequestId() {
    if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
    return Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
  }

  // Every edit goes through here, as in the prototype's change().
  function change(patch) {
    var changed = false;
    for (var k in patch) if (form[k] !== patch[k]) { form[k] = patch[k]; changed = true; }
    if (!changed) return;
    if (run.raf) { cancelAnimationFrame(run.raf); run.raf = null; }
    if (run.phase === 'running') {
      // Cancel: a late answer for this request is ignored.
      run.phase = 'idle'; run.progress = 0; run.shown = 0; run.pred = null;
      run.stale = false; run.requestId = null; run.inputRow = null;
    } else {
      run.stale = run.phase === 'done';
    }
    renderPredict();
  }

  function startPrediction() {
    if (run.phase === 'running') return;
    if (run.raf) { cancelAnimationFrame(run.raf); run.raf = null; }
    run.phase = 'running'; run.progress = 0; run.shown = 0; run.pred = null;
    run.stale = false; run.inputRow = null;
    run.requestId = newRequestId();
    post('streamlit:setComponentValue', {
      dataType: 'json',
      value: {
        request_id: run.requestId,
        inputs: {
          distance_km: form.distance,
          num_lanes: form.lanes,
          num_signals: form.signals,
          hour: form.hour,
          day_of_week: form.dow,
          traffic_level: form.traffic,
          road_type: form.road,
          weather_type: form.weather,
          temperature_c: form.temp
        }
      }
    });
    renderPredict();
    revealResult();
  }

  function revealResult() {
    var card = $('result');
    if (card.scrollIntoView) card.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'nearest' });
  }

  function onResult(result) {
    if (!result || run.phase !== 'running' || run.pred !== null) return;
    if (result.request_id !== run.requestId || !isNum(result.pred_minutes)) return;
    var pred = Math.max(0, result.pred_minutes);
    run.pred = pred;
    run.inputRow = result.input_row || null;

    var finish = function () {
      run.raf = null;
      run.phase = 'done'; run.progress = 1; run.shown = pred;
      renderPredict();
      // The card grows when the range bar appears; keep all of it in view.
      revealResult();
    };
    if (reducedMotion()) { finish(); return; }

    var t0 = performance.now();
    var step = function (now) {
      var t = Math.min(1, (now - t0) / ANIM_MS);
      var e = 1 - Math.pow(1 - t, 3);
      if (t < 1) {
        run.progress = e; run.shown = pred * e;
        renderProgress();
        run.raf = requestAnimationFrame(step);
      } else {
        finish();
      }
    };
    run.raf = requestAnimationFrame(step);
  }

  // ───────────────────────── Screen 2: rendering ─────────────────────────

  var SLIDERS = [
    { id: 'f-distance', key: 'distance', parse: parseFloat },
    { id: 'f-lanes', key: 'lanes', parse: function (v) { return parseInt(v, 10); } },
    { id: 'f-signals', key: 'signals', parse: function (v) { return parseInt(v, 10); } },
    { id: 'f-hour', key: 'hour', parse: function (v) { return parseInt(v, 10); } },
    { id: 'f-temp', key: 'temp', parse: parseFloat }
  ];
  var SEGMENTS = [
    { id: 'seg-road', key: 'road', options: ['highway', 'urban', 'residential'] },
    { id: 'seg-traffic', key: 'traffic', options: ['low', 'medium', 'high'] },
    { id: 'seg-weather', key: 'weather', options: ['clear', 'rain', 'fog'] }
  ];

  function buildPredict() {
    SLIDERS.forEach(function (s) {
      $(s.id).addEventListener('input', function (e) {
        var v = s.parse(e.target.value);
        if (isNum(v)) { var p = {}; p[s.key] = v; change(p); }
      });
    });

    SEGMENTS.forEach(function (g) {
      var box = $(g.id);
      g.options.forEach(function (opt) {
        var b = el('button', 'min-height: 44px; padding: 0 4px; border-radius: 8px; font-size: 14px; font-weight: 500; text-transform: capitalize; transition: background .2s, color .2s', opt);
        b.type = 'button';
        b.className = 'seg';
        b.addEventListener('click', function () { var p = {}; p[g.key] = opt; change(p); });
        box.appendChild(b);
      });
    });

    var dayBox = $('seg-day');
    DAYS.forEach(function (d, i) {
      var b = el('button', 'min-height: 44px; padding: 0; border-radius: 8px; font-size: 12px; font-weight: 500; transition: background .2s, color .2s', d.slice(0, 3));
      b.type = 'button';
      b.className = 'seg';
      b.setAttribute('aria-label', d);
      b.addEventListener('click', function () { change({ dow: i }); });
      dayBox.appendChild(b);
    });

    var presetBox = $('presets');
    PRESETS.forEach(function (p) {
      var b = el('button', 'min-height: 44px; padding: 0 14px; border: 1px dashed #34486B; border-radius: 8px; background: transparent; color: #C9D3E6; font-size: 13px', p.label);
      b.type = 'button';
      b.className = 'seg';
      b.addEventListener('click', function () { change(p.v); });
      presetBox.appendChild(b);
    });

    $('go').addEventListener('click', startPrediction);
    $('summary-toggle').addEventListener('click', function () {
      run.summaryOpen = !run.summaryOpen;
      renderPredict();
    });
  }

  // The frame-by-frame part: trail, truck and the counting number.
  function renderProgress() {
    $('track-trail').style.width = (run.progress * 100).toFixed(1) + '%';
    $('track-truck').style.left = (run.progress * 100).toFixed(2) + '%';
    $('eta').textContent = run.phase !== 'idle' ? run.shown.toFixed(1) : '––.–';
  }

  function summaryRows() {
    var row = run.inputRow;
    if (row) {
      // What the model actually received for the estimate on screen.
      return Object.keys(row).map(function (k) {
        var v = row[k];
        return { k: k, v: ONE_DP[k] && isNum(v) ? v.toFixed(1) : String(v) };
      });
    }
    // No estimate yet: preview of the row Predict Travel Time would send.
    var isRush = RUSH_HOURS.indexOf(form.hour) >= 0;
    return [
      { k: 'distance_km', v: form.distance.toFixed(1) }, { k: 'num_lanes', v: form.lanes }, { k: 'num_signals', v: form.signals },
      { k: 'avg_speed_kmph', v: AVG_SPEED[form.traffic].toFixed(1) }, { k: 'temperature_c', v: form.temp.toFixed(1) }, { k: 'visibility_km', v: VISIBILITY[form.weather].toFixed(1) },
      { k: 'hour', v: form.hour }, { k: 'day_of_week', v: form.dow }, { k: 'is_rush_hour', v: isRush ? 1 : 0 }, { k: 'is_weekend', v: form.dow >= 5 ? 1 : 0 },
      { k: 'traffic_level', v: form.traffic }, { k: 'road_type', v: form.road }, { k: 'weather_type', v: form.weather }
    ];
  }

  function renderPredict() {
    var s = form;
    var speed = AVG_SPEED[s.traffic];
    var vis = VISIBILITY[s.weather];
    var isRush = RUSH_HOURS.indexOf(s.hour) >= 0;

    // Sliders and their read-outs
    SLIDERS.forEach(function (sl) {
      var input = $(sl.id);
      if (sl.parse(input.value) !== s[sl.key]) input.value = String(s[sl.key]);
    });
    $('v-distance').textContent = s.distance.toFixed(1);
    $('v-lanes').textContent = String(s.lanes);
    $('v-signals').textContent = String(s.signals);
    $('v-hour').textContent = pad(s.hour) + ':00';
    $('v-temp').textContent = s.temp.toFixed(1) + '°';

    // Segmented controls
    SEGMENTS.forEach(function (g) {
      var kids = $(g.id).children;
      for (var i = 0; i < kids.length; i++) {
        var on = s[g.key] === g.options[i];
        paintSeg(kids[i], on, segStyle(on, 'transparent', '#C9D3E6', '#2A3A58'));
      }
    });
    var dayKids = $('seg-day').children;
    for (var d = 0; d < dayKids.length; d++) {
      paintSeg(dayKids[d], s.dow === d, segStyle(s.dow === d, 'transparent', '#C9D3E6', '#2A3A58'));
    }

    // Derived badges
    $('badge-rush').style.borderColor = isRush ? '#F5B83D' : '#1E2A42';
    $('badge-rush-label').textContent = isRush ? 'Rush hour' : 'Off-peak';
    $('badge-rush-label').style.color = isRush ? '#F5B83D' : '#E8EDF6';
    $('badge-day').textContent = s.dow >= 5 ? 'Weekend' : 'Weekday';
    bind('speed', String(speed));
    $('badge-vis').textContent = vis.toFixed(1);

    // Predict button
    var busy = run.phase === 'running';
    $('go').disabled = busy;
    $('go-label').textContent = busy ? 'Predicting…' : 'Predict Travel Time';

    // Result card
    var card = $('result');
    var done = run.phase === 'done';
    card.style.borderColor = done && !run.stale ? '#2E6E86' : '#1E2A42';
    card.setAttribute('aria-busy', busy ? 'true' : 'false');

    var signalDots = [];
    for (var i = 0; i < s.signals; i++) signalDots.push({ left: (100 * (i + 1) / (s.signals + 1)).toFixed(2) + '%', delay: '-' + (i * 0.9 % 4).toFixed(1) + 's' });
    syncList($('track-signals'), signalDots,
      function () { return el('div', 'position: absolute; top: 16px; width: 8px; height: 8px; margin-left: -4px; border-radius: 50%; background: #7CE3A1'); },
      function (node, dot) {
        node.style.left = dot.left;
        node.style.animation = 'sig 4s linear ' + dot.delay + ' infinite';
      });
    $('track-truck').style.animation = busy ? 'bob .35s ease-in-out infinite' : 'none';
    $('track-km').textContent = s.distance.toFixed(1);
    renderProgress();

    $('eta').style.color = run.phase === 'idle' ? '#55678A' : run.stale ? '#9AA8C0' : '#E8EDF6';

    var pred = run.pred || 0;
    var etaSub = 'Set the route, then press Predict Travel Time.';
    if (busy) etaSub = 'Running the route through ' + (trees() === undefined ? 'the' : trees()) + ' trees…';
    if (done) {
      // Departure hour of the estimate on screen, which may predate later edits.
      var hour = run.inputRow && isNum(run.inputRow.hour) ? Math.round(run.inputRow.hour) : s.hour;
      var mins = Math.round(pred);
      var arrive = hour * 60 + mins;
      etaSub = (mins >= 60 ? '≈ ' + Math.floor(mins / 60) + ' hr ' + (mins % 60) + ' min · ' : '') +
        'Leaves ' + pad(hour) + ':00, arrives around ' + pad(Math.floor(arrive / 60) % 24) + ':' + pad(arrive % 60);
    }
    $('eta-sub').textContent = etaSub;

    $('range').hidden = !done;
    if (done) {
      $('range-low').textContent = Math.max(0, pred - mae()).toFixed(1);
      $('range-pred').textContent = pred.toFixed(1);
      $('range-high').textContent = (pred + mae()).toFixed(1);
    }
    $('stale').hidden = !(run.stale && done);

    // Input Summary
    var toggle = $('summary-toggle');
    toggle.textContent = run.summaryOpen ? 'Hide Input Summary' : 'Show Input Summary';
    toggle.setAttribute('aria-expanded', run.summaryOpen ? 'true' : 'false');
    var summary = $('summary');
    summary.hidden = !run.summaryOpen;
    if (run.summaryOpen) {
      syncList(summary, summaryRows(),
        function () {
          var row = el('div', "display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid #1E2A42; font-family: 'JetBrains Mono', monospace; font-size: 12px");
          row.appendChild(el('span', 'color: #9AA8C0'));
          row.appendChild(el('span'));
          return row;
        },
        function (node, r) {
          node.firstChild.textContent = r.k;
          node.lastChild.textContent = String(r.v);
        });
    }
  }

  // ───────────────────────── Screen switching ─────────────────────────

  var bootTimer = null;

  function show(screen) {
    var toPredict = screen === 'predict';
    $('screen-landing').hidden = toPredict;
    $('screen-predict').hidden = !toPredict;
    if (bootTimer) { clearTimeout(bootTimer); bootTimer = null; }

    if (toPredict) {
      stopScene();
      // The loader plays on every entry.
      $('loader').hidden = false;
      $('predict-main').hidden = true;
      bootTimer = setTimeout(function () {
        bootTimer = null;
        $('loader').hidden = true;
        $('predict-main').hidden = false;
        $('h-predict').focus({ preventScroll: true });
      }, LOADER_MS);
    } else {
      scene.tick += 1;
      renderScene();
      armScene();
      $('h-landing').focus({ preventScroll: true });
    }
    scrollHostToTop();
    syncHeight();
  }

  // ───────────────────────── Boot ─────────────────────────

  var started = false;

  function onRender(args) {
    args = args || {};
    var key = JSON.stringify([args.metrics, args.feature_importance]);
    if (key !== dataKey) {
      dataKey = key;
      metrics = args.metrics || {};
      featureImportance = Array.isArray(args.feature_importance) ? args.feature_importance : [];
      renderData();
      if (started) renderPredict();
    }
    if (!started) {
      started = true;
      buildPredict();
      renderPredict();
      renderScene();
      armScene();
      app.hidden = false;
      syncHeight();
    }
    onResult(args.result);
  }

  window.addEventListener('message', function (event) {
    if (event.source !== window.parent) return;
    var data = event.data;
    if (data && data.type === 'streamlit:render') onRender(data.args);
  });

  document.addEventListener('click', function (event) {
    var nav = event.target.closest && event.target.closest('[data-nav]');
    if (!nav) return;
    event.preventDefault();
    show(nav.getAttribute('data-nav'));
  });

  syncViewport();
  window.addEventListener('resize', syncViewport);
  try { window.parent.addEventListener('resize', syncViewport); } catch (e) { /* cross-origin host */ }

  if (window.ResizeObserver) new ResizeObserver(syncHeight).observe(app);
  else setInterval(syncHeight, 250);

  post('streamlit:componentReady', { apiVersion: 1 });
})();
