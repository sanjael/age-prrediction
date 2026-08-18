/**
 * app.js
 * Main UI Controller & Component Renderer.
 * Mounts centralized dataset, sets up scroll spys, and connects `/demo` route.
 */

import {
  PROJECT_CONFIG,
  KPI_METRICS,
  PROBLEM_CARDS,
  DATASET_PIPELINE,
  DEMOGRAPHIC_COHORTS,
  MODEL_TOURNAMENT,
  EXPERIMENT_TIMELINE,
  TRAINING_CONVERGENCE,
  DEMOGRAPHIC_PERFORMANCE,
  ERROR_TOLERANCE,
  ARCHITECTURE_WHY,
  RESEARCH_INSIGHTS,
  LIMITATIONS
} from './data.js';

import {
  initChartDefaults,
  renderDemographicChart,
  renderAge100DistributionChart,
  renderTournamentChart,
  renderConvergenceChart,
  renderDemographicPerformanceChart
} from './charts.js';

document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize UI Elements from data
  renderKpis();
  renderProblemSection();
  renderPipeline();
  renderTournamentTable();
  renderTimeline();
  renderConvergenceDetails();
  renderDemographicTable();
  renderToleranceBars();
  renderWhyCards();
  renderInsights();
  renderLimitations();

  // 2. Initialize Charts
  initChartDefaults();
  renderDemographicChart('chart-demographics');
  renderAge100DistributionChart('chart-age-dist');
  renderTournamentChart('chart-tournament');
  renderConvergenceChart('chart-convergence');
  renderDemographicPerformanceChart('chart-demographic-perf');

  // 3. Navigation Spy & Smooth Scrolling
  setupNavSpy();
  setupRouteHandler();
});

function renderKpis() {
  const container = document.getElementById('kpi-container');
  if (!container) return;

  container.innerHTML = KPI_METRICS.map(kpi => `
    <div class="kpi-card ${kpi.highlight ? 'kpi-highlight' : ''}">
      <div>
        <div class="kpi-label">${kpi.label}</div>
        <div class="kpi-value-row">
          <div class="kpi-value">${kpi.value}</div>
          ${kpi.unit ? `<div class="kpi-unit">${kpi.unit}</div>` : ''}
        </div>
      </div>
      <div class="kpi-footer">
        <span class="kpi-direction">${kpi.direction}</span>
        <span class="kpi-delta ${kpi.status}">${kpi.delta}</span>
      </div>
    </div>
  `).join('');
}

function renderProblemSection() {
  const container = document.getElementById('problem-container');
  if (!container) return;

  container.innerHTML = PROBLEM_CARDS.map(card => `
    <div class="problem-card">
      <div>
        <div class="problem-card-top">
          <span class="problem-num">${card.num}</span>
          <span class="problem-metric-tag">${card.metric}</span>
        </div>
        <h3 class="problem-title">${card.title}</h3>
        <p class="problem-desc">${card.desc}</p>
      </div>
      <div style="border-top: 1px solid var(--border-subtle); padding-top: 10px; font-size: 12px; color: var(--text-muted);">
        <strong>Core challenge:</strong> ${card.summary}
      </div>
    </div>
  `).join('');
}

function renderPipeline() {
  const container = document.getElementById('pipeline-container');
  if (!container) return;

  container.innerHTML = DATASET_PIPELINE.map(node => `
    <div class="pipeline-node ${node.isMaster ? 'master-node' : ''}">
      <div class="pipeline-stage">${node.stage}</div>
      <div class="pipeline-name">${node.name}</div>
      <div class="pipeline-count">${node.count} <span style="font-size: 13px; font-weight: 500; color: var(--text-secondary);">${node.unit}</span></div>
      <div class="pipeline-role">${node.role}</div>
    </div>
  `).join('');
}

function renderTournamentTable() {
  const container = document.getElementById('tournament-table-body');
  if (!container) return;

  container.innerHTML = MODEL_TOURNAMENT.map(m => `
    <tr class="${m.isChampion ? 'champion-row' : ''}">
      <td><span class="badge-tag ${m.isChampion ? 'badge-champion' : ''}">${m.id}</span></td>
      <td><strong>${m.architecture}</strong></td>
      <td><code>${m.backbone}</code></td>
      <td><strong style="color: ${m.isChampion ? 'var(--accent-primary)' : 'var(--text-primary)'}; font-family: var(--font-family-mono); font-size: 14px;">${m.mae.toFixed(2)} yrs</strong></td>
      <td><span style="font-family: var(--font-family-mono); font-weight: 600;">${m.acc7}</span></td>
      <td>${m.isChampion ? '<span class="badge-tag badge-champion">Champion</span>' : '<span class="badge-tag">Evaluated</span>'}</td>
    </tr>
  `).join('');
}

function renderTimeline() {
  const container = document.getElementById('timeline-container');
  if (!container) return;

  container.innerHTML = EXPERIMENT_TIMELINE.map(item => `
    <div class="timeline-item ${item.isChampion ? 'champion' : ''}">
      <div class="timeline-dot">${item.id.replace('EXP-', '')}</div>
      <div class="timeline-content">
        <div class="timeline-top">
          <span class="timeline-phase">${item.phase}</span>
          <span class="timeline-mae">${item.mae} <small style="font-size: 11px; color: var(--text-muted);">(${item.delta})</small></span>
        </div>
        <div class="timeline-model">${item.model}</div>
        <p class="timeline-insight">${item.insight}</p>
      </div>
    </div>
  `).join('');
}

function renderConvergenceDetails() {
  const container = document.getElementById('convergence-details-container');
  if (!container) return;

  container.innerHTML = `
    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 14px;">
      ${TRAINING_CONVERGENCE.map(pt => `
        <div style="background-color: var(--surface-secondary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 10px; text-align: center;">
          <div style="font-family: var(--font-family-mono); font-size: 11px; color: var(--text-muted); font-weight: 600;">EPOCH ${pt.epoch}</div>
          <div style="font-size: 14px; font-weight: 700; color: var(--accent-primary); font-family: var(--font-family-mono); margin: 4px 0;">${pt.valMae.toFixed(2)} yrs</div>
          <div style="font-size: 11px; color: #10B981; font-weight: 600;">±5yr: ${pt.acc5}%</div>
          <div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">Train: ${pt.trainMae.toFixed(2)}y</div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderDemographicTable() {
  const container = document.getElementById('demographic-table-body');
  if (!container) return;

  container.innerHTML = DEMOGRAPHIC_PERFORMANCE.map(c => `
    <div class="cohort-item">
      <span class="cohort-bracket">${c.bracket}</span>
      <span class="cohort-name">${c.cohort}</span>
      <span class="cohort-samples">${c.testSamples} val</span>
      <span class="cohort-mae" style="color: ${c.isWeakness ? 'var(--color-danger)' : 'var(--accent-primary)'};">${c.mae.toFixed(2)} yrs</span>
      <span class="cohort-acc">${c.acc7}%</span>
      <span class="cohort-grade-badge ${c.isWeakness ? 'sparse' : c.highlight ? 'exceptional' : 'good'}">${c.grade}</span>
    </div>
  `).join('');
}

function renderToleranceBars() {
  const container = document.getElementById('tolerance-container');
  if (!container) return;

  container.innerHTML = ERROR_TOLERANCE.map(t => `
    <div class="tolerance-row ${t.isBenchmark ? 'benchmark' : ''}">
      <div class="tolerance-meta">
        <div>
          <span class="tolerance-label">${t.tolerance}</span>
          <span class="tolerance-sub">${t.label}</span>
        </div>
        <span class="tolerance-pct">${t.percentage.toFixed(2)}%</span>
      </div>
      <div class="tolerance-bar-track">
        <div class="tolerance-bar-fill" style="width: ${t.percentage}%;"></div>
      </div>
    </div>
  `).join('');
}

function renderWhyCards() {
  const container = document.getElementById('why-container');
  if (!container) return;

  container.innerHTML = ARCHITECTURE_WHY.map(item => `
    <div class="why-card">
      <div class="why-tag">${item.tag}</div>
      <h3 class="why-title">${item.name}</h3>
      <p class="why-desc">${item.desc}</p>
    </div>
  `).join('');
}

function renderInsights() {
  const container = document.getElementById('insights-container');
  if (!container) return;

  container.innerHTML = RESEARCH_INSIGHTS.map(item => `
    <div class="insight-card">
      <div class="insight-top">
        <span class="insight-num">INSIGHT ${item.num}</span>
      </div>
      <h4 class="insight-title">${item.title}</h4>
      <p class="insight-desc">${item.desc}</p>
    </div>
  `).join('');
}

function renderLimitations() {
  const container = document.getElementById('limitations-container');
  if (!container) return;

  container.innerHTML = LIMITATIONS.map(item => `
    <div class="limitation-item">
      <div class="limitation-title">${item.title}</div>
      <p class="limitation-desc">${item.desc}</p>
    </div>
  `).join('');
}

function setupNavSpy() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  window.addEventListener('scroll', () => {
    let current = '';
    const scrollY = window.pageYOffset;

    sections.forEach(section => {
      const sectionTop = section.offsetTop - 100;
      const sectionHeight = section.offsetHeight;
      if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}

function setupRouteHandler() {
  const demoButtons = document.querySelectorAll('[data-route="/demo"]');
  demoButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      window.location.href = 'demo.html';
    });
  });
}
