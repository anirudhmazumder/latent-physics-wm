/* Interactive replacements for the matplotlib PNGs on the index page.
 *
 * No dependencies: everything is hand-written SVG. The data comes from
 * assets/figdata.json, built by scripts/build_figdata.py out of the evaluation
 * reports under runs/ -- so these charts show the same series, axes and
 * reference lines as the PNGs they replace, and the PNGs stay in place as the
 * <noscript> fallback.
 *
 * Markup contract, in index.html:
 *   <figure class="chart" data-fig="KEY" data-fallback="runs/.../x.png">
 *     <div class="chart-box"></div>
 *     <noscript><img src="runs/.../x.png" alt="..."></noscript>
 *     <figcaption>...</figcaption>
 *   </figure>
 */
(() => {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const SANS = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';

  // coral and blue are the page's own accents; the rest are muted companions.
  const PALETTE = ['#e05a46', '#3a8fe0', '#4f9d69', '#8a6fbf', '#d9a441',
                   '#3fa3a3', '#c2739b', '#7a8b99', '#8d9b3c', '#b07c54',
                   '#6d7fc1'];
  const GREY = '#8c8c8c', FAINT = 'rgba(0,0,0,.09)', AXIS = 'rgba(0,0,0,.28)';
  const REF_GREEN = '#2a9d4a', REF_CORAL = '#d55e3a';

  // ------------------------------------------------------------- primitives

  const mk = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const div = (cls, parent) => {
    const e = document.createElement('div');
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  };
  const lin = (d0, d1, r0, r1) => v => d1 === d0 ? r0 : r0 + (v - d0) / (d1 - d0) * (r1 - r0);

  function niceTicks(lo, hi, n) {
    if (!(hi > lo)) return [lo];
    const raw = (hi - lo) / Math.max(n, 2);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
      out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    return out;
  }
  const fmt = (v, d) => v == null || !isFinite(v) ? '–'
    : (d != null ? v.toFixed(d) : String(Math.round(v * 1e4) / 1e4));

  // one shared tooltip
  let tipEl = null;
  function tip() {
    if (!tipEl) { tipEl = div('chart-tip'); tipEl.hidden = true; document.body.appendChild(tipEl); }
    return tipEl;
  }
  function showTip(ev, html) {
    const t = tip();
    t.innerHTML = html;
    t.hidden = false;
    const pad = 12, r = t.getBoundingClientRect();
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
    t.style.left = (x + scrollX) + 'px';
    t.style.top = (y + scrollY) + 'px';
  }
  const hideTip = () => { if (tipEl) tipEl.hidden = true; };
  function hoverable(node, html) {
    node.addEventListener('mousemove', e => showTip(e, html));
    node.addEventListener('mouseleave', hideTip);
  }

  /** A panel: an <svg> plus an inner <g> offset by the margin. */
  function panel(parent, width, height, margin, aria) {
    const m = Object.assign({ t: 22, r: 12, b: 34, l: 46 }, margin || {});
    const svg = mk('svg', {
      width: '100%', height: height, viewBox: `0 0 ${width} ${height}`,
      role: 'img', 'aria-label': aria || '', preserveAspectRatio: 'xMidYMid meet'
    }, parent);
    const g = mk('g', { transform: `translate(${m.l},${m.t})` }, svg);
    return { svg, g, m, w: width, h: height,
             iw: Math.max(width - m.l - m.r, 10), ih: Math.max(height - m.t - m.b, 10) };
  }

  function title(p, text) {
    mk('text', {
      x: p.iw / 2, y: -8, 'text-anchor': 'middle', 'font-family': SANS,
      'font-size': 11.5, fill: 'rgba(0,0,0,.62)'
    }, p.g).textContent = text;
  }

  function yAxis(p, y, opts) {
    const o = opts || {}, ticks = o.ticks || niceTicks(y.lo, y.hi, 5);
    ticks.forEach(v => {
      const yy = y.s(v);
      if (yy < -1 || yy > p.ih + 1) return;
      mk('line', { x1: 0, x2: p.iw, y1: yy, y2: yy, stroke: FAINT }, p.g);
      mk('text', {
        x: -7, y: yy + 3.5, 'text-anchor': 'end', 'font-family': SANS,
        'font-size': 11, fill: 'rgba(0,0,0,.5)'
      }, p.g).textContent = o.fmt ? o.fmt(v) : fmt(v);
    });
    if (o.label) {
      mk('text', {
        transform: `translate(${-p.m.l + 11},${p.ih / 2}) rotate(-90)`,
        'text-anchor': 'middle', 'font-family': SANS, 'font-size': 11,
        fill: 'rgba(0,0,0,.55)'
      }, p.g).textContent = o.label;
    }
  }

  function xAxisNum(p, x, opts) {
    const o = opts || {}, ticks = o.ticks || niceTicks(x.lo, x.hi, 5);
    mk('line', { x1: 0, x2: p.iw, y1: p.ih, y2: p.ih, stroke: AXIS }, p.g);
    ticks.forEach(v => {
      const xx = x.s(v);
      if (xx < -1 || xx > p.iw + 1) return;
      mk('line', { x1: xx, x2: xx, y1: p.ih, y2: p.ih + 4, stroke: AXIS }, p.g);
      mk('text', {
        x: xx, y: p.ih + 16, 'text-anchor': 'middle', 'font-family': SANS,
        'font-size': 11, fill: 'rgba(0,0,0,.5)'
      }, p.g).textContent = o.fmt ? o.fmt(v) : fmt(v);
    });
    if (o.label) {
      mk('text', {
        x: p.iw / 2, y: p.ih + 31, 'text-anchor': 'middle', 'font-family': SANS,
        'font-size': 11, fill: 'rgba(0,0,0,.55)'
      }, p.g).textContent = o.label;
    }
  }

  /** Category labels along the bottom, optionally rotated. */
  function xAxisCat(p, labels, centers, rotate) {
    mk('line', { x1: 0, x2: p.iw, y1: p.ih, y2: p.ih, stroke: AXIS }, p.g);
    labels.forEach((lab, i) => {
      const xx = centers[i];
      const lines = String(lab).split('\n');
      if (rotate) {
        const t = mk('text', {
          transform: `translate(${xx},${p.ih + 9}) rotate(${-rotate})`,
          'text-anchor': 'end', 'font-family': SANS, 'font-size': 10.5,
          fill: 'rgba(0,0,0,.55)'
        }, p.g);
        t.textContent = lines.join(' ');
      } else {
        lines.forEach((ln, k) => {
          mk('text', {
            x: xx, y: p.ih + 15 + k * 12, 'text-anchor': 'middle',
            'font-family': SANS, 'font-size': 11, fill: 'rgba(0,0,0,.55)'
          }, p.g).textContent = ln;
        });
      }
    });
  }

  /** A dashed reference line with a small label at the left. */
  function refLine(p, yPix, text, color, dash) {
    mk('line', {
      x1: 0, x2: p.iw, y1: yPix, y2: yPix, stroke: color || GREY,
      'stroke-width': 1.4, 'stroke-dasharray': dash === false ? null : '5 4'
    }, p.g);
    if (text) {
      mk('text', {
        x: 3, y: yPix - 4, 'font-family': SANS, 'font-size': 10,
        'paint-order': 'stroke', stroke: '#fff', 'stroke-width': 3, 'stroke-linejoin': 'round',
        fill: color || GREY
      }, p.g).textContent = text;
    }
  }

  const path = (pts, x, y) => pts.map((d, i) => (i ? 'L' : 'M') + x(d[0]).toFixed(2) + ' ' + y(d[1]).toFixed(2)).join(' ');

  /** Clickable legend under a chart. `items`: {name, color, dash} */
  function legend(box, items, state, redraw) {
    const l = div('chart-legend sans', box);
    items.forEach(it => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chart-key' + (state.off.has(it.name) ? ' off' : '');
      b.setAttribute('aria-pressed', String(!state.off.has(it.name)));
      const sw = document.createElement('i');
      sw.style.background = it.color;
      if (it.dash) { sw.style.background = 'none'; sw.style.borderTop = '2px dashed ' + it.color; sw.style.height = '0'; }
      b.appendChild(sw);
      b.appendChild(document.createTextNode(it.label || it.name));
      b.addEventListener('click', () => {
        if (state.off.has(it.name)) state.off.delete(it.name); else state.off.add(it.name);
        redraw();
      });
      l.appendChild(b);
    });
    return l;
  }

  /** Split a chart-box into n side-by-side panel holders. */
  function panels(box, n) {
    const row = div('chart-panels' + (n > 2 ? ' three' : ''), box);
    const out = [];
    for (let i = 0; i < n; i++) out.push(div('chart-panel', row));
    return out;
  }
  const widthOf = el => Math.max(el.clientWidth || el.getBoundingClientRect().width || 300, 220);

  /* Centred rolling mean with edge padding -- the same smoother
     wm/eval_conservation.py:compare() applies before drawing. */
  function smooth(ys, w) {
    const h = Math.floor(w / 2), n = ys.length, out = new Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = -h; k <= h; k++) s += ys[Math.min(n - 1, Math.max(0, i + k))];
      out[i] = s / (2 * h + 1);
    }
    return out;
  }

  /** Nearest-index hover for line charts sharing one x index. */
  function indexHover(p, xs, xScale, rows, fmtRow) {
    const rule = mk('line', {
      y1: 0, y2: p.ih, stroke: 'rgba(0,0,0,.25)', 'stroke-width': 1, opacity: 0
    }, p.g);
    const rect = mk('rect', { x: 0, y: 0, width: p.iw, height: p.ih, fill: 'transparent' }, p.g);
    rect.addEventListener('mousemove', ev => {
      // The svg is width:100% over a fixed viewBox, so client px must be
      // rescaled into viewBox units before the margin is taken off.
      const r = p.svg.getBoundingClientRect();
      const scale = p.w / (r.width || p.w);
      const px = (ev.clientX - r.left) * scale - p.m.l;
      let best = 0, bd = Infinity;
      xs.forEach((v, i) => { const d = Math.abs(xScale(v) - px); if (d < bd) { bd = d; best = i; } });
      rule.setAttribute('x1', xScale(xs[best]));
      rule.setAttribute('x2', xScale(xs[best]));
      rule.setAttribute('opacity', 1);
      showTip(ev, fmtRow(best));
    });
    rect.addEventListener('mouseleave', () => { rule.setAttribute('opacity', 0); hideTip(); });
    void rows;
  }

  // ---------------------------------------------------------------- figures

  const CHARTS = {};

  /* 1. where_is_velocity -- grouped bars, held-out R^2 per state variable,
     read from z / h / z+h, one panel per probe family. Horizontal bars so the
     six variable names stay readable in the 648px column. */
  CHARTS.where_is_velocity = (box, d, state) => {
    const items = d.series.map((s, i) => ({ name: s, color: PALETTE[i] }));
    const draw = () => {
      box.innerHTML = '';
      const holders = panels(box, d.panels.length);
      const on = d.series.filter(s => !state.off.has(s));
      d.panels.forEach((pan, pi) => {
        const w = widthOf(holders[pi]);
        const rowH = 15, groupH = rowH * Math.max(on.length, 1) + 10;
        const h = 30 + d.vars.length * groupH + 40;
        const p = panel(holders[pi], w, h, { t: 26, r: 10, b: 36, l: 62 },
          `${pan.label}: held-out R-squared for six state variables read from z, h and z+h`);
        title(p, pan.label);
        const lo = -0.1, hi = 1.05;
        const x = lin(lo, hi, 0, p.iw);
        // vertical gridlines
        niceTicks(0, 1, 4).forEach(v => {
          mk('line', { x1: x(v), x2: x(v), y1: 0, y2: p.ih, stroke: FAINT }, p.g);
          mk('text', {
            x: x(v), y: p.ih + 14, 'text-anchor': 'middle', 'font-family': SANS,
            'font-size': 11, fill: 'rgba(0,0,0,.5)'
          }, p.g).textContent = fmt(v, 1);
        });
        mk('line', { x1: x(0), x2: x(0), y1: 0, y2: p.ih, stroke: AXIS }, p.g);
        mk('text', {
          x: p.iw / 2, y: p.ih + 30, 'text-anchor': 'middle', 'font-family': SANS,
          'font-size': 11, fill: 'rgba(0,0,0,.55)'
        }, p.g).textContent = d.ylabel;
        d.vars.forEach((v, vi) => {
          const y0 = vi * (p.ih / d.vars.length);
          mk('text', {
            x: -6, y: y0 + (p.ih / d.vars.length) / 2 + 3.5, 'text-anchor': 'end',
            'font-family': SANS, 'font-size': 11, fill: 'rgba(0,0,0,.6)'
          }, p.g).textContent = v;
          on.forEach((s, si) => {
            const raw = d.values[pan.key][s][vi];
            const shown = Math.max(raw, d.clip);
            const bh = Math.min(rowH - 3, (p.ih / d.vars.length - 8) / on.length);
            const yy = y0 + 4 + si * (bh + 2);
            const xa = x(Math.min(shown, 0)), xb = x(Math.max(shown, 0));
            const r = mk('rect', {
              x: xa, y: yy, width: Math.max(xb - xa, 0.8), height: bh,
              fill: PALETTE[d.series.indexOf(s)], rx: 1
            }, p.g);
            hoverable(r, `<b>${v}</b><br>${s} · ${pan.label}<br>R² ${fmt(raw, 3)}`);
          });
        });
      });
      legend(box, items, state, draw);
      const note = div('chart-note sans', box);
      note.textContent = 'Bars are clipped at −0.05; hover for the true value.';
    };
    draw();
  };

  /* 2. cold_start_speed. The original left panel is a scatter of per-start
     points that were never saved; what survives in report.json is the OLS fit,
     the correlation and the marginal mean/sd, so panel A draws those against
     the identity line. Panel B is the warm-up sweep from the same report. */
  CHARTS.cold_start_speed = (box, d, state) => {
    const cols = { rnn_v2: PALETTE[0], rnn_v2_nocolor: PALETTE[1] };
    const items = d.models.map(m => ({ name: m, color: cols[m], label: `${d.labels[m]} (${m})` }));
    const draw = () => {
      box.innerHTML = '';
      const holders = panels(box, 2);
      const on = d.models.filter(m => !state.off.has(m));

      // -- panel A: dreamed vs true speed, one-frame warm-up
      {
        const w = widthOf(holders[0]), h = 320;
        const p = panel(holders[0], w, h, { t: 26, r: 14, b: 42, l: 54 },
          'Dreamed speed against true speed after a one-frame warm-up: the fitted line for each model against the identity line');
        title(p, `cold start: 1 frame, ${d.cold_horizon} dreamed steps`);
        const lo = 0.008, hi = 0.046;
        const x = lin(lo, hi, 0, p.iw), y = lin(lo, hi, p.ih, 0);
        yAxis(p, { lo, hi, s: y, }, { ticks: [0.01, 0.02, 0.03, 0.04], fmt: v => v.toFixed(2), label: 'dreamed speed' });
        xAxisNum(p, { lo, hi, s: x }, { ticks: [0.01, 0.02, 0.03, 0.04], fmt: v => v.toFixed(2), label: 'true speed = 0.022 / mass' });
        // held-out mass band
        const hb = d.holdout_speed_range;
        mk('rect', {
          x: x(hb[0]), y: 0, width: x(hb[1]) - x(hb[0]), height: p.ih,
          fill: 'rgba(0,0,0,.035)'
        }, p.g);
        mk('text', {
          x: (x(hb[0]) + x(hb[1])) / 2, y: 11, 'text-anchor': 'middle',
          'font-family': SANS, 'font-size': 9.5, fill: 'rgba(0,0,0,.42)'
        }, p.g).textContent = 'held-out band';
        // identity
        const idl = mk('line', {
          x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), stroke: 'rgba(0,0,0,.5)',
          'stroke-width': 1.2, 'stroke-dasharray': '5 4'
        }, p.g);
        hoverable(idl, 'identity — what a model that knew the law would dream');
        mk('text', {
          x: p.iw - 4, y: y(hi * .94) + 2, 'text-anchor': 'end', 'font-family': SANS,
          'font-size': 10, fill: 'rgba(0,0,0,.45)'
        }, p.g).textContent = 'identity (the law)';
        const tr = d.true_speed_range;
        on.forEach(m => {
          [['val', 1], ['holdout', 0]].forEach(([split, solid]) => {
            const f = d.fits[m][split];
            const a = split === 'val' ? tr[0] : hb[0], b = split === 'val' ? tr[1] : hb[1];
            const ln = mk('line', {
              x1: x(a), y1: y(f.slope * a + f.intercept),
              x2: x(b), y2: y(f.slope * b + f.intercept),
              stroke: cols[m], 'stroke-width': solid ? 2 : 1.6,
              'stroke-dasharray': solid ? null : '3 3', 'stroke-linecap': 'round'
            }, p.g);
            hoverable(ln, `<b>${d.labels[m]}</b><br>${split === 'val' ? 'in-distribution' : 'held-out mass band'} (n=${f.n})<br>slope ${fmt(f.slope, 2)} · r ${fmt(f.pearson_r, 2)}`);
            // mean +/- sd cross
            const cx = x(f.true_speed_mean), cy = y(f.dreamed_speed_mean);
            mk('line', { x1: x(f.true_speed_mean - f.true_speed_std), x2: x(f.true_speed_mean + f.true_speed_std), y1: cy, y2: cy, stroke: cols[m], 'stroke-width': 1, opacity: .55 }, p.g);
            mk('line', { x1: cx, x2: cx, y1: y(f.dreamed_speed_mean - f.dreamed_speed_std), y2: y(f.dreamed_speed_mean + f.dreamed_speed_std), stroke: cols[m], 'stroke-width': 1, opacity: .55 }, p.g);
            const c = mk('circle', { cx, cy, r: solid ? 4 : 3, fill: solid ? cols[m] : '#fff', stroke: cols[m], 'stroke-width': 1.5 }, p.g);
            hoverable(c, `<b>${d.labels[m]}</b> · ${split === 'val' ? 'in-distribution' : 'held-out'}<br>mean true ${fmt(f.true_speed_mean, 4)} ± ${fmt(f.true_speed_std, 4)}<br>mean dreamed ${fmt(f.dreamed_speed_mean, 4)} ± ${fmt(f.dreamed_speed_std, 4)}`);
          });
        });
      }

      // -- panel B: correlation vs warm-up length
      {
        const w = widthOf(holders[1]), h = 320;
        const p = panel(holders[1], w, h, { t: 26, r: 14, b: 42, l: 44 },
          'Correlation between dreamed and true speed as the warm-up lengthens from one to eight real frames');
        title(p, 'and how fast motion replaces colour');
        const ks = d.warmup[d.models[0]].k;
        const x = lin(0, ks.length - 1, 0, p.iw), y = lin(-0.15, 1.02, p.ih, 0);
        yAxis(p, { lo: -0.15, hi: 1.02, s: y }, { ticks: [0, 0.25, 0.5, 0.75, 1], fmt: v => v.toFixed(2), label: 'pearson r (dreamed vs true speed)' });
        xAxisCat(p, ks.map(k => 'K=' + k), ks.map((_, i) => x(i)));
        mk('text', {
          x: p.iw / 2, y: p.ih + 32, 'text-anchor': 'middle', 'font-family': SANS,
          'font-size': 11, fill: 'rgba(0,0,0,.55)'
        }, p.g).textContent = 'warm-up length (true frames)';
        mk('line', { x1: 0, x2: p.iw, y1: y(0), y2: y(0), stroke: AXIS }, p.g);
        on.forEach(m => {
          const ys = d.warmup[m].r;
          mk('path', {
            d: path(ys.map((v, i) => [i, v]), x, y), fill: 'none',
            stroke: cols[m], 'stroke-width': 2
          }, p.g);
          ys.forEach((v, i) => {
            const c = mk('circle', { cx: x(i), cy: y(v), r: 3.5, fill: cols[m] }, p.g);
            hoverable(c, `<b>${d.labels[m]}</b><br>warm-up K=${ks[i]}<br>r ${fmt(v, 3)} · slope ${fmt(d.warmup[m].slope[i], 3)}`);
          });
        });
      }
      legend(box, items, state, draw);
      const note = div('chart-note sans', box);
      note.innerHTML = 'Left: solid line and filled dot are the in-distribution fit and its mean ± sd; dashed line and hollow dot are the held-out mass band. The original figure plotted the 325 individual starts behind these fits; those points were not saved with the report, so only the fits are drawn here.';
    };
    draw();
  };

  /* 3. fix_comparison -- conservation curves over a 200-step sampled dream. */
  CHARTS.fix_comparison = (box, d, state) => {
    const items = d.runs.map((r, i) => ({ name: r, color: PALETTE[i] }))
      .concat([{ name: 'probe floor', color: '#333', dash: true }]);
    const draw = () => {
      box.innerHTML = '';
      const holders = panels(box, d.panels.length);
      const on = d.runs.filter(r => !state.off.has(r));
      const floorOn = !state.off.has('probe floor');
      d.panels.forEach((pan, pi) => {
        const w = widthOf(holders[pi]), h = 330;
        const p = panel(holders[pi], w, h, { t: 28, r: 12, b: 42, l: 46 },
          `${pan.label} at temperature ${d.tau}, per dream step, for the baseline and four candidate fixes`);
        title(p, `${pan.label}, τ=${d.tau}`);
        const n = d.series[d.runs[0]][pan.key].length;
        const x = lin(0, n - 1, 0, p.iw), y = lin(-1.05, 1.05, p.ih, 0);
        yAxis(p, { lo: -1.05, hi: 1.05, s: y }, { ticks: [-1, -0.5, 0, 0.5, 1], fmt: v => v.toFixed(1), label: pi === 0 ? d.ylabel : null });
        xAxisNum(p, { lo: 0, hi: n - 1, s: x }, { ticks: [0, 50, 100, 150, 200], fmt: v => String(v), label: d.xlabel });
        mk('line', { x1: 0, x2: p.iw, y1: y(0), y2: y(0), stroke: AXIS }, p.g);
        refLine(p, y(d.good_line), 'conserved (0.8)', REF_GREEN);
        on.forEach(run => {
          const ys = d.series[run][pan.key], col = PALETTE[d.runs.indexOf(run)];
          mk('path', { d: path(ys.map((v, i) => [i, v]), x, y), fill: 'none', stroke: col, 'stroke-width': 0.7, opacity: .22 }, p.g);
          mk('path', { d: path(smooth(ys, d.smooth_window).map((v, i) => [i, v]), x, y), fill: 'none', stroke: col, 'stroke-width': 1.9 }, p.g);
        });
        if (floorOn) {
          const f = d.probe_floor[pan.key];
          mk('path', { d: path(f.map((v, i) => [i, v]), x, y), fill: 'none', stroke: '#333', 'stroke-width': 1.1, 'stroke-dasharray': '5 4' }, p.g);
        }
        const xs = []; for (let i = 0; i < n; i++) xs.push(i);
        indexHover(p, xs, x, null, i => {
          let s = `<b>dream step ${i}</b>`;
          on.forEach(run => {
            s += `<br><i style="background:${PALETTE[d.runs.indexOf(run)]}"></i>${run} ${fmt(d.series[run][pan.key][i], 2)}`;
          });
          const f = d.probe_floor[pan.key];
          if (floorOn && i < f.length) s += `<br><i style="background:#333"></i>probe floor ${fmt(f[i], 2)}`;
          return s;
        });
      });
      legend(box, items, state, draw);
      const note = div('chart-note sans', box);
      note.textContent = 'Faint line: the raw per-step correlation over 30 episodes. Bold line: a centred 11-step rolling mean, as in the original figure.';
    };
    draw();
  };

  /* 4. skill_by_mass -- grouped bars per mass tercile with bootstrap CIs and a
     per-tercile oracle segment. */
  CHARTS.skill_by_mass = (box, d, state) => {
    const colOf = b => d.baselines.includes(b.name)
      ? (b.name === 'stay' ? '#b4b4b4' : '#858585')
      : PALETTE[(d.bars.filter(x => !d.baselines.includes(x.name)).indexOf(b)) % PALETTE.length];
    const items = d.bars.map(b => ({ name: b.name, color: colOf(b) }));
    const draw = () => {
      box.innerHTML = '';
      const on = d.bars.filter(b => !state.off.has(b.name));
      const w = widthOf(box), h = 340;
      const p = panel(box, w, h, { t: 22, r: 14, b: 48, l: 52 },
        'Interceptions per floor visit for each controller in each mass tercile, with 95% bootstrap confidence intervals and the oracle ceiling per tercile');
      const hi = 1.6;
      const y = lin(0, hi, p.ih, 0);
      yAxis(p, { lo: 0, hi, s: y }, { ticks: [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5], fmt: v => v.toFixed(2), label: d.ylabel });
      const gw = p.iw / d.terciles.length;
      const bw = (gw * 0.82) / Math.max(on.length, 1);
      xAxisCat(p, d.tercile_labels, d.terciles.map((_, i) => gw * (i + 0.5)));
      d.terciles.forEach((t, ti) => {
        const x0 = gw * ti + gw * 0.09;
        on.forEach((b, bi) => {
          const v = b.values[ti], ci = b.ci[ti];
          const xx = x0 + bi * bw;
          const r = mk('rect', {
            x: xx, y: y(v), width: Math.max(bw * 0.88, 1), height: p.ih - y(v),
            fill: colOf(b), rx: 1
          }, p.g);
          hoverable(r, `<b>${b.name}</b><br>${t} balls<br>${fmt(v, 3)} per floor visit` +
            (ci ? `<br>95% CI ${fmt(ci[0], 3)} – ${fmt(ci[1], 3)}` : ''));
          if (ci) {
            const cx = xx + bw * 0.44;
            mk('line', { x1: cx, x2: cx, y1: y(ci[0]), y2: y(ci[1]), stroke: 'rgba(0,0,0,.62)', 'stroke-width': 1 }, p.g);
            [ci[0], ci[1]].forEach(c => mk('line', {
              x1: cx - 2.5, x2: cx + 2.5, y1: y(c), y2: y(c),
              stroke: 'rgba(0,0,0,.62)', 'stroke-width': 1
            }, p.g));
          }
        });
        const ov = d.oracle[ti];
        const seg = mk('line', {
          x1: gw * ti + gw * 0.05, x2: gw * (ti + 1) - gw * 0.05, y1: y(ov), y2: y(ov),
          stroke: REF_GREEN, 'stroke-width': 2.4
        }, p.g);
        hoverable(seg, `<b>oracle (ceiling)</b><br>${t} balls · ${fmt(ov, 3)}`);
      });
      mk('text', {
        x: p.iw - 2, y: Math.max(y(d.oracle[d.oracle.length - 1]) - 34, 11),
        'text-anchor': 'end', 'font-family': SANS, 'font-size': 10, fill: REF_GREEN
      }, p.g).textContent = 'oracle (ceiling)';
      legend(box, items, state, draw);
    };
    draw();
  };

  /* 5. permanence_decay -- position read out of h as the occlusion goes on. */
  CHARTS.permanence_decay = (box, d, state) => {
    const colOf = f => f === 'z' ? GREY : PALETTE[d.order.filter(k => k !== 'z').indexOf(f) % PALETTE.length];
    const items = d.order.map(f => ({ name: f, color: colOf(f), dash: f === 'z' }))
      .concat([{ name: 'no memory', color: '#333', dash: true, label: 'no memory (last visible)' }]);
    const draw = () => {
      box.innerHTML = '';
      const holders = panels(box, d.panels.length);
      const on = d.order.filter(f => !state.off.has(f));
      const nmOn = !state.off.has('no memory');
      d.panels.forEach((pan, pi) => {
        const w = widthOf(holders[pi]), h = 320;
        const p = panel(holders[pi], w, h, { t: 22, r: 12, b: 44, l: 48 },
          `${pan.label} for the ball's position read out of each representation, against how long the ball has been hidden`);
        const ks = d.series[d.order[0]].k;
        let lo = Infinity, hi = -Infinity;
        const cons = v => { if (v != null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
        on.forEach(f => d.series[f][pan.key].forEach(cons));
        if (pan.key === 'rmse' && nmOn) d.no_memory.rmse.forEach(cons);
        if (!isFinite(lo)) { lo = 0; hi = 1; }
        const pad = (hi - lo) * 0.08 || 0.05;
        lo -= pad; hi += pad;
        if (pan.key === 'r2_x') hi = Math.max(hi, 0.02);
        const x = lin(ks[0], ks[ks.length - 1], 0, p.iw), y = lin(lo, hi, p.ih, 0);
        yAxis(p, { lo, hi, s: y }, { fmt: v => v.toFixed(2), label: pan.label });
        xAxisNum(p, { lo: ks[0], hi: ks[ks.length - 1], s: x }, { ticks: [1, 5, 10, 15, 20], fmt: v => String(v), label: d.xlabel });
        if (pan.key === 'r2_x') mk('line', { x1: 0, x2: p.iw, y1: y(0), y2: y(0), stroke: AXIS }, p.g);
        if (pan.key === 'rmse' && nmOn) {
          mk('path', { d: path(d.no_memory.rmse.map((v, i) => [d.no_memory.k[i], v]), x, y), fill: 'none', stroke: '#333', 'stroke-width': 1.4, 'stroke-dasharray': '6 4' }, p.g);
        }
        on.forEach(f => {
          const col = colOf(f), ys = d.series[f][pan.key];
          mk('path', {
            d: path(ys.map((v, i) => [d.series[f].k[i], v]), x, y), fill: 'none',
            stroke: col, 'stroke-width': f === 'z' ? 1.3 : 1.8,
            'stroke-dasharray': f === 'z' ? '2 3' : null
          }, p.g);
          ys.forEach((v, i) => mk('circle', { cx: x(d.series[f].k[i]), cy: y(v), r: 2.4, fill: col }, p.g));
        });
        indexHover(p, ks, x, null, i => {
          let s = `<b>hidden for ${ks[i]} frame${ks[i] === 1 ? '' : 's'}</b>`;
          on.forEach(f => { s += `<br><i style="background:${colOf(f)}"></i>${f} ${fmt(d.series[f][pan.key][i], 3)}`; });
          if (pan.key === 'rmse' && nmOn) s += `<br><i style="background:#333"></i>no memory ${fmt(d.no_memory.rmse[i], 3)}`;
          return s;
        });
      });
      legend(box, items, state, draw);
    };
    draw();
  };

  /* 6. skill_vs_bound -- every v3.1 controller against the two oracles.
     Horizontal, so the long run names stay readable in the column. */
  CHARTS.skill_vs_bound = (box, d, state) => {
    const bound = d.refs.wait_and_see;
    const draw = () => {
      box.innerHTML = '';
      const w = widthOf(box);
      const rowH = 21, h = 34 + d.bars.length * rowH + 40;
      const p = panel(box, w, h, { t: 30, r: 14, b: 40, l: 182 },
        'Interceptions per floor visit for every v3.1 controller, sorted, against the wait-and-see oracle and the full oracle');
      const hi = 1.12;
      const x = lin(0, hi, 0, p.iw);
      niceTicks(0, hi, 5).forEach(v => {
        mk('line', { x1: x(v), x2: x(v), y1: 0, y2: p.ih, stroke: FAINT }, p.g);
        mk('text', { x: x(v), y: p.ih + 15, 'text-anchor': 'middle', 'font-family': SANS, 'font-size': 11, fill: 'rgba(0,0,0,.5)' }, p.g).textContent = fmt(v, 1);
      });
      mk('line', { x1: 0, x2: 0, y1: 0, y2: p.ih, stroke: AXIS }, p.g);
      mk('text', { x: p.iw / 2, y: p.ih + 31, 'text-anchor': 'middle', 'font-family': SANS, 'font-size': 11, fill: 'rgba(0,0,0,.55)' }, p.g).textContent = d.ylabel;
      d.bars.forEach((b, i) => {
        const yy = i * (p.ih / d.bars.length) + 3;
        const bh = p.ih / d.bars.length - 6;
        const col = d.baselines.includes(b.name) ? '#b4b4b4'
          : b.value > bound ? REF_GREEN : REF_CORAL;
        const r = mk('rect', { x: 0, y: yy, width: Math.max(x(b.value), 1), height: bh, fill: col, rx: 1 }, p.g);
        hoverable(r, `<b>${b.name}</b><br>${fmt(b.value, 3)} per floor visit` +
          (b.ci ? `<br>95% CI ${fmt(b.ci[0], 3)} – ${fmt(b.ci[1], 3)}` : '') +
          `<br>${b.value > bound ? 'above' : 'at or below'} the memoryless bound`);
        if (b.ci) {
          const cy = yy + bh / 2;
          mk('line', { x1: x(b.ci[0]), x2: x(b.ci[1]), y1: cy, y2: cy, stroke: 'rgba(0,0,0,.6)', 'stroke-width': 1 }, p.g);
          [b.ci[0], b.ci[1]].forEach(c => mk('line', { x1: x(c), x2: x(c), y1: cy - 3, y2: cy + 3, stroke: 'rgba(0,0,0,.6)', 'stroke-width': 1 }, p.g));
        }
        mk('text', { x: -8, y: yy + bh / 2 + 3.5, 'text-anchor': 'end', 'font-family': SANS, 'font-size': 10.5, fill: 'rgba(0,0,0,.6)' }, p.g).textContent = b.name;
      });
      [['oracle', REF_GREEN, 'oracle (vision + memory)', false],
       ['wait_and_see', REF_CORAL, 'wait-and-see oracle (vision, NO memory)', true]].forEach(([k, col, lab, dash]) => {
        if (d.refs[k] == null) return;
        const xx = x(d.refs[k]);
        const ln = mk('line', { x1: xx, x2: xx, y1: -6, y2: p.ih, stroke: col, 'stroke-width': 2, 'stroke-dasharray': dash ? '6 4' : null }, p.g);
        hoverable(ln, `<b>${lab}</b><br>${fmt(d.refs[k], 3)}`);
        mk('text', { x: xx, y: -12, 'text-anchor': k === 'oracle' ? 'end' : 'middle', 'font-family': SANS, 'font-size': 10, fill: col }, p.g).textContent = `${k === 'oracle' ? 'oracle' : 'memoryless bound'} ${fmt(d.refs[k], 2)}`;
      });
      const note = div('chart-note sans', box);
      note.innerHTML = `Green: above the memoryless bound (${fmt(bound, 2)}), i.e. using memory. Coral: at or below it. Grey: the do-nothing baselines. Whiskers are 95% bootstrap CIs.`;
      void state;
    };
    draw();
  };

  /* 7. two_seeds -- paired bars, one pair per controller. */
  CHARTS.two_seeds = (box, d, state) => {
    const C0 = '#3d6fb0', C1 = '#8fb4dd';
    const items = [{ name: 'seed 0', color: C0 }, { name: 'seed 1', color: C1 }];
    const draw = () => {
      box.innerHTML = '';
      const on = items.filter(i => !state.off.has(i.name));
      const w = widthOf(box), h = 350;
      const p = panel(box, w, h, { t: 26, r: 14, b: 96, l: 50 },
        'Interceptions per floor visit for two training seeds of each v3.1 controller, against the memoryless bound and the full oracle');
      const hi = Math.max(1.12, Math.max(...d.pairs.map(p2 => Math.max(p2.seed0, p2.seed1))) * 1.12);
      const y = lin(0, hi, p.ih, 0);
      yAxis(p, { lo: 0, hi, s: y }, { fmt: v => v.toFixed(2), label: d.ylabel });
      const gw = p.iw / d.pairs.length, bw = gw * 0.34;
      xAxisCat(p, d.pairs.map(q => q.name), d.pairs.map((_, i) => gw * (i + 0.5)), 32);
      d.pairs.forEach((q, i) => {
        const cx = gw * (i + 0.5);
        const spots = [[q.seed0, C0, 'seed 0', cx - bw * 1.02], [q.seed1, C1, 'seed 1', cx + bw * 0.02]];
        spots.forEach(([v, col, nm, xx]) => {
          if (state.off.has(nm)) return;
          const r = mk('rect', { x: xx, y: y(v), width: bw, height: p.ih - y(v), fill: col, rx: 1 }, p.g);
          hoverable(r, `<b>${q.name}</b><br>${nm} · ${fmt(v, 3)}<br>two-seed mean ${fmt(q.mean, 3)} · spread ${fmt(q.spread, 3)}`);
        });
        if (on.length === 2) {
          mk('line', {
            x1: cx - bw * 0.52, x2: cx + bw * 0.52, y1: y(q.seed0), y2: y(q.seed1),
            stroke: 'rgba(0,0,0,.55)', 'stroke-width': 1
          }, p.g);
        }
      });
      [['oracle', REF_GREEN, 'oracle (vision + memory)', false],
       ['wait_and_see', REF_CORAL, 'wait-and-see oracle (vision, NO memory)', true]].forEach(([k, col, lab, dash]) => {
        if (d.refs[k] == null) return;
        const ln = mk('line', { x1: 0, x2: p.iw, y1: y(d.refs[k]), y2: y(d.refs[k]), stroke: col, 'stroke-width': 2, 'stroke-dasharray': dash ? '6 4' : null }, p.g);
        hoverable(ln, `<b>${lab}</b><br>${fmt(d.refs[k], 3)}`);
        mk('text', { x: 3, y: y(d.refs[k]) - 4, 'font-family': SANS, 'font-size': 10, 'paint-order': 'stroke', stroke: '#fff', 'stroke-width': 3, 'stroke-linejoin': 'round', fill: col }, p.g)
          .textContent = `${k === 'oracle' ? 'oracle' : 'memoryless bound'} = ${fmt(d.refs[k], 2)}`;
      });
      legend(box, items, state, draw);
    };
    draw();
  };

  /* 8. v4_switch -- three panels: sign recall vs frames since the flip, what
     the model state adds over one frame, and memory against inference. */
  CHARTS.v4_switch = (box, d, state) => {
    const COLS = { lstm: '#c2543a', transformer: '#3a7bd5', ff: '#7a7a7a', noact: '#2a9d4a', tf_ctx32: '#9a6bd0' };
    const items = d.order.map(m => ({ name: m, color: COLS[m], label: `${m} (h, ${d.models[m].hidden_dim}-d)` }))
      .concat([{ name: 'z', color: '#222', dash: true, label: 'z (one frame: the null AND the leak)' }]);
    const draw = () => {
      box.innerHTML = '';
      const holders = panels(box, 3);
      const on = d.order.filter(m => !state.off.has(m));
      const zOn = !state.off.has('z');
      const xs = d.bins.map((_, i) => i);

      // (a)
      {
        const w = widthOf(holders[0]), h = 330;
        const p = panel(holders[0], w, h, { t: 34, r: 12, b: 44, l: 46 },
          'Balanced accuracy of a linear probe reading the gravity sign from each model state, against frames since the last flip');
        title(p, '(a) sign accuracy vs frames since the flip');
        const x = lin(0, xs.length - 1, 0, p.iw), y = lin(0.15, 1.02, p.ih, 0);
        // shuffled-label null band
        const nb = d.null_balanced;
        mk('rect', { x: 0, y: y(nb), width: p.iw, height: y(1 - nb) - y(nb), fill: 'rgba(0,0,0,.07)' }, p.g);
        yAxis(p, { lo: 0.15, hi: 1.02, s: y }, { ticks: [0.2, 0.4, 0.6, 0.8, 1.0], fmt: v => v.toFixed(1), label: 'held-out balanced sign accuracy' });
        xAxisCat(p, d.bins, xs.map(i => x(i)));
        mk('text', { x: p.iw / 2, y: p.ih + 31, 'text-anchor': 'middle', 'font-family': SANS, 'font-size': 11, fill: 'rgba(0,0,0,.55)' }, p.g).textContent = d.xlabel;
        refLine(p, y(0.5), null, 'rgba(0,0,0,.4)');
        mk('text', {
          x: p.iw - 2, y: y(1 - nb) + 12, 'text-anchor': 'end',
          'font-family': SANS, 'font-size': 9.5, fill: 'rgba(0,0,0,.45)'
        }, p.g).textContent = `shuffled-label null (up to ${fmt(nb, 3)})`;
        if (zOn) {
          mk('path', { d: path(d.z_balanced.map((v, i) => [i, v]), x, y), fill: 'none', stroke: '#222', 'stroke-width': 1.4, 'stroke-dasharray': '5 4' }, p.g);
          d.z_balanced.forEach((v, i) => mk('rect', { x: x(i) - 2.5, y: y(v) - 2.5, width: 5, height: 5, fill: '#222' }, p.g));
        }
        on.forEach(m => {
          const ys = d.models[m].balanced;
          mk('path', { d: path(ys.map((v, i) => [i, v]), x, y), fill: 'none', stroke: COLS[m], 'stroke-width': 1.8 }, p.g);
          ys.forEach((v, i) => {
            mk('circle', { cx: x(i), cy: y(v), r: 3.4, fill: COLS[m] }, p.g);
            if (d.models[m].n[i] < d.small_bin) {
              mk('circle', { cx: x(i), cy: y(v), r: 6, fill: 'none', stroke: '#000', 'stroke-width': 1 }, p.g);
            }
          });
        });
        indexHover(p, xs, x, null, i => {
          let s = `<b>${d.bins[i]} frames since the flip</b>`;
          on.forEach(m => { s += `<br><i style="background:${COLS[m]}"></i>${m} ${fmt(d.models[m].balanced[i], 3)} <span style="opacity:.6">(n=${d.models[m].n[i]})</span>`; });
          if (zOn) s += `<br><i style="background:#222"></i>z ${fmt(d.z_balanced[i], 3)}`;
          return s;
        });
      }

      // (b) what h adds over z
      {
        const w = widthOf(holders[1]), h = 330;
        const p = panel(holders[1], w, h, { t: 34, r: 12, b: 44, l: 46 },
          'Balanced sign accuracy from the model state minus the same probe on a single frame, per bin');
        title(p, 'what the state ADDS over one frame');
        let lo = 0, hi = 0;
        d.order.forEach(m => d.models[m].balanced.forEach((v, i) => {
          const dv = v - d.z_balanced[i]; lo = Math.min(lo, dv); hi = Math.max(hi, dv);
        }));
        lo -= 0.05; hi += 0.05;
        const x = lin(0, xs.length - 1, 0, p.iw), y = lin(lo, hi, p.ih, 0);
        yAxis(p, { lo, hi, s: y }, { fmt: v => v.toFixed(1), label: 'accuracy from h − from z' });
        xAxisCat(p, d.bins, xs.map(i => x(i)));
        mk('text', { x: p.iw / 2, y: p.ih + 31, 'text-anchor': 'middle', 'font-family': SANS, 'font-size': 11, fill: 'rgba(0,0,0,.55)' }, p.g).textContent = d.xlabel;
        mk('line', { x1: 0, x2: p.iw, y1: y(0), y2: y(0), stroke: 'rgba(0,0,0,.55)', 'stroke-width': 1 }, p.g);
        on.forEach(m => {
          const ys = d.models[m].balanced.map((v, i) => v - d.z_balanced[i]);
          mk('path', { d: path(ys.map((v, i) => [i, v]), x, y), fill: 'none', stroke: COLS[m], 'stroke-width': 1.8 }, p.g);
          ys.forEach((v, i) => {
            mk('circle', { cx: x(i), cy: y(v), r: 3.4, fill: COLS[m] }, p.g);
            if (d.models[m].n[i] < d.small_bin) mk('circle', { cx: x(i), cy: y(v), r: 6, fill: 'none', stroke: '#000', 'stroke-width': 1 }, p.g);
          });
        });
        indexHover(p, xs, x, null, i => {
          let s = `<b>${d.bins[i]} frames since the flip</b>`;
          on.forEach(m => { s += `<br><i style="background:${COLS[m]}"></i>${m} ${fmt(d.models[m].balanced[i] - d.z_balanced[i], 3)}`; });
          return s;
        });
      }

      // (c) memory vs inference
      {
        const w = widthOf(holders[2]), h = 330;
        const p = panel(holders[2], w, h, { t: 34, r: 12, b: 56, l: 44 },
          'Sign accuracy in the ten frames after a flip, where only memory could help, against accuracy fifty or more frames later from a cold start');
        title(p, '(b) remember the event, or read the motion?');
        const y = lin(0, 1.05, p.ih, 0);
        yAxis(p, { lo: 0, hi: 1.05, s: y }, { ticks: [0, 0.2, 0.4, 0.6, 0.8, 1.0], fmt: v => v.toFixed(1), label: 'sign accuracy' });
        const gw = p.iw / d.order.length, bw = gw * 0.34;
        xAxisCat(p, d.order, d.order.map((_, i) => gw * (i + 0.5)), 26);
        const kinds = [['memory', '#c2543a', '(i) memory: 0–10 after the flip'],
                       ['inference', '#3a7bd5', '(ii) inference: 50+, cold start']];
        d.order.forEach((m, i) => {
          const cx = gw * (i + 0.5);
          kinds.forEach(([key, col, lab], k) => {
            const v = d.models[m][key];
            if (v == null) return;
            const xx = cx + (k - 1) * bw + bw * 0.5;
            const r = mk('rect', { x: xx - bw / 2, y: y(v), width: bw, height: p.ih - y(v), fill: col, rx: 1 }, p.g);
            hoverable(r, `<b>${m}</b><br>${lab}<br>${fmt(v, 3)}`);
          });
        });
        refLine(p, y(0.5), null, 'rgba(0,0,0,.4)');
        refLine(p, y(d.z_ref_memory), `z alone, 0–10 frames (${fmt(d.z_ref_memory, 2)})`, '#c2543a');
        refLine(p, y(d.z_ref_inference), `z alone, 50–200 frames (${fmt(d.z_ref_inference, 2)})`, '#3a7bd5');
        const lg = div('chart-legend sans inline', holders[2]);
        kinds.forEach(([, col, lab]) => {
          const s = document.createElement('span');
          s.className = 'chart-key static';
          const i2 = document.createElement('i'); i2.style.background = col;
          s.appendChild(i2); s.appendChild(document.createTextNode(lab));
          lg.appendChild(s);
        });
      }
      legend(box, items, state, draw);
      const note = div('chart-note sans', box);
      note.textContent = 'Hollow rings in the first two panels mark bins with fewer than 400 held-out frames. The grey band in (a) is the shuffled-label null.';
    };
    draw();
  };

  // ------------------------------------------------------------------ boot

  function init(data) {
    const figs = Array.from(document.querySelectorAll('figure.chart'));
    const live = [];
    figs.forEach(fig => {
      const key = fig.dataset.fig, box = fig.querySelector('.chart-box');
      if (!box) return;
      if (!data[key] || !CHARTS[key]) {
        if (fig.dataset.fallback) box.innerHTML = `<img src="${fig.dataset.fallback}" alt="">`;
        return;
      }
      const state = { off: new Set() };
      const render = () => { try { CHARTS[key](box, data[key], state); } catch (e) { console.error(key, e); } };
      render();
      live.push({ render, box });
    });
    let t = null, last = innerWidth;
    addEventListener('resize', () => {
      if (innerWidth === last) return;
      last = innerWidth;
      clearTimeout(t);
      t = setTimeout(() => live.forEach(c => c.render()), 160);
    });
    addEventListener('scroll', hideTip, { passive: true });
  }

  fetch('assets/figdata.json')
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(init)
    .catch(err => {
      console.error('figdata', err);
      document.querySelectorAll('figure.chart').forEach(fig => {
        const box = fig.querySelector('.chart-box');
        if (box && fig.dataset.fallback) box.innerHTML = `<img src="${fig.dataset.fallback}" alt="">`;
      });
    });
})();
