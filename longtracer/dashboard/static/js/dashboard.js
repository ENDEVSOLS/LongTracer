/**
 * LongTracer Dashboard — HTMX + Chart.js logic.
 */

// ── HTMX configuration ────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function () {
  var apiKey = getApiKey();
  if (apiKey) {
    document.body.addEventListener('htmx:configRequest', function (event) {
      event.detail.headers['x-api-key'] = apiKey;
    });
  }
});

// ── API key management ─────────────────────────────────────────────

function getApiKey() {
  return localStorage.getItem('longtracer_api_key') || '';
}

function setApiKey(key) {
  localStorage.setItem('longtracer_api_key', key);
}

// ── Toast notifications ────────────────────────────────────────────

function showToast(message, duration) {
  duration = duration || 3000;
  var toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = message;
  toast.classList.remove('hidden');
  setTimeout(function () {
    toast.classList.add('hidden');
  }, duration);
}

// ── Metrics API calls ──────────────────────────────────────────────

var trustScoreChart = null;
var hallucinationChart = null;

function loadMetrics() {
  var project = '';
  var filterEl = document.getElementById('project-filter');
  if (filterEl) {
    project = filterEl.value || '';
  }

  var apiKey = getApiKey();
  var headers = {};
  if (apiKey) {
    headers['x-api-key'] = apiKey;
  }

  // Fetch summary + timeseries in parallel
  var summaryUrl = '/api/v1/metrics/summary';
  var tsUrl = '/api/v1/metrics/timeseries?interval=1d';
  if (project) {
    summaryUrl += '?project=' + encodeURIComponent(project);
    tsUrl += '&project=' + encodeURIComponent(project);
  }

  Promise.all([
    fetch(summaryUrl, { headers: headers }).then(function (r) {
      if (!r.ok) throw new Error('Summary fetch failed');
      return r.json();
    }).catch(function () { return null; }),
    fetch(tsUrl, { headers: headers }).then(function (r) {
      if (!r.ok) throw new Error('Timeseries fetch failed');
      return r.json();
    }).catch(function () { return { data_points: [] }; }),
  ]).then(function (results) {
    var summary = results[0];
    var timeseries = results[1];

    if (summary === null) {
      showToast('Failed to load metrics');
      return;
    }

    renderSummary(summary);
    renderVerdict(summary);
    renderCharts(timeseries);
  });
}

// ── Render summary stat cards ──────────────────────────────────────

function renderSummary(data) {
  if (!data) return;

  setText('stat-total', data.total_traces || 0);

  if (data.avg_trust_score !== null && data.avg_trust_score !== undefined) {
    setText('stat-avg-score', (data.avg_trust_score * 100).toFixed(1) + '%');
  } else {
    setText('stat-avg-score', '—');
  }

  if (data.pass_rate !== null && data.pass_rate !== undefined) {
    setText('stat-pass-rate', (data.pass_rate * 100).toFixed(1) + '%');
  } else {
    setText('stat-pass-rate', '—');
  }

  setText('stat-hallucinations', data.total_hallucinations || 0);
  if (data.hallucination_rate !== null && data.hallucination_rate !== undefined) {
    setText('stat-hall-rate', 'Rate: ' + (data.hallucination_rate * 100).toFixed(2) + '%');
  } else {
    setText('stat-hall-rate', '');
  }
}

// ── Render verdict summary ─────────────────────────────────────────

function renderVerdict(data) {
  if (!data || !data.total_traces) {
    setText('verdict-pass', '0');
    setText('verdict-fail', '0');
    setText('verdict-pct', '0%');
    return;
  }

  var total = data.total_traces;
  var passRate = data.pass_rate || 0;
  var passCount = Math.round(passRate * total);
  var failCount = total - passCount;

  setText('verdict-pass', passCount);
  setText('verdict-fail', failCount);
  setText('verdict-pct', (passRate * 100).toFixed(1) + '%');

  var barEl = document.getElementById('verdict-bar-pass');
  if (barEl) {
    barEl.style.width = (passRate * 100).toFixed(1) + '%';
  }
}

// ── Render charts ──────────────────────────────────────────────────

var chartColors = {
  accent: '#6366f1',
  accentBg: 'rgba(99,102,241,0.15)',
  green: '#22c55e',
  greenBg: 'rgba(34,197,94,0.15)',
  red: '#ef4444',
  redBg: 'rgba(239,68,68,0.15)',
  yellow: '#eab308',
  grid: '#2d3248',
  text: '#8b8fa3',
};

var chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#1a1d27',
      borderColor: '#2d3248',
      borderWidth: 1,
      titleColor: '#e4e6ef',
      bodyColor: '#e4e6ef',
      padding: 10,
      cornerRadius: 6,
    },
  },
  scales: {
    x: {
      grid: { color: chartColors.grid, drawBorder: false },
      ticks: { color: chartColors.text, font: { size: 11 } },
    },
    y: {
      grid: { color: chartColors.grid, drawBorder: false },
      ticks: { color: chartColors.text, font: { size: 11 } },
      beginAtZero: true,
    },
  },
};

function renderCharts(timeseries) {
  var points = (timeseries && timeseries.data_points) || [];

  var trustEmpty = document.getElementById('trust-score-empty');
  var hallEmpty = document.getElementById('hallucination-empty');

  if (points.length === 0) {
    // Show empty placeholders
    if (trustEmpty) trustEmpty.classList.remove('hidden');
    if (hallEmpty) hallEmpty.classList.remove('hidden');
    if (trustScoreChart) { trustScoreChart.destroy(); trustScoreChart = null; }
    if (hallucinationChart) { hallucinationChart.destroy(); hallucinationChart = null; }
    return;
  }

  if (trustEmpty) trustEmpty.classList.add('hidden');
  if (hallEmpty) hallEmpty.classList.add('hidden');

  var labels = points.map(function (p) {
    // Format bucket: "2026-05-14" → "May 14"
    var parts = p.bucket.split('-');
    if (parts.length === 3) {
      var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      return months[parseInt(parts[1], 10) - 1] + ' ' + parseInt(parts[2], 10);
    }
    return p.bucket;
  });

  // Trust Score line chart
  var trustData = points.map(function (p) {
    return p.avg_trust_score !== null ? +(p.avg_trust_score * 100).toFixed(1) : null;
  });

  if (trustScoreChart) trustScoreChart.destroy();

  var trustCtx = document.getElementById('trust-score-chart');
  if (trustCtx) {
    trustScoreChart = new Chart(trustCtx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Trust Score',
          data: trustData,
          borderColor: chartColors.accent,
          backgroundColor: chartColors.accentBg,
          fill: true,
          tension: 0.3,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: chartColors.accent,
          borderWidth: 2,
        }],
      },
      options: Object.assign({}, chartDefaults, {
        scales: {
          x: chartDefaults.scales.x,
          y: Object.assign({}, chartDefaults.scales.y, {
            min: 0,
            max: 100,
            ticks: {
              color: chartColors.text,
              font: { size: 11 },
              callback: function (v) { return v + '%'; },
            },
          }),
        },
        plugins: Object.assign({}, chartDefaults.plugins, {
          tooltip: Object.assign({}, chartDefaults.plugins.tooltip, {
            callbacks: {
              label: function (ctx) { return 'Trust Score: ' + ctx.parsed.y + '%'; },
            },
          }),
        }),
      }),
    });
  }

  // Hallucination bar chart
  var hallData = points.map(function (p) { return p.hallucination_count || 0; });

  if (hallucinationChart) hallucinationChart.destroy();

  var hallCtx = document.getElementById('hallucination-chart');
  if (hallCtx) {
    hallucinationChart = new Chart(hallCtx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Hallucinations',
          data: hallData,
          backgroundColor: hallData.map(function (v) {
            return v > 0 ? chartColors.redBg : chartColors.greenBg;
          }),
          borderColor: hallData.map(function (v) {
            return v > 0 ? chartColors.red : chartColors.green;
          }),
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: Object.assign({}, chartDefaults, {
        plugins: Object.assign({}, chartDefaults.plugins, {
          tooltip: Object.assign({}, chartDefaults.plugins.tooltip, {
            callbacks: {
              label: function (ctx) { return 'Hallucinations: ' + ctx.parsed.y; },
            },
          }),
        }),
      }),
    });
  }
}

// ── Helpers ────────────────────────────────────────────────────────

function setText(id, value) {
  var el = document.getElementById(id);
  if (el) el.textContent = value;
}
