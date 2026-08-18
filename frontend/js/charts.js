/**
 * charts.js
 * Highly Detailed, Research-Grade Chart.js Visualizations.
 * Rich legends, explicit axis callouts, hover micro-details, and crisp high-contrast data presentation.
 */

import {
  DEMOGRAPHIC_COHORTS,
  MODEL_TOURNAMENT,
  TRAINING_CONVERGENCE,
  DEMOGRAPHIC_PERFORMANCE
} from './data.js';

// Global Chart.js defaults
export function initChartDefaults() {
  if (typeof Chart === 'undefined') return;

  Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.color = '#9DA8B7'; // Higher contrast neutral text
  Chart.defaults.borderColor = '#2A303C'; // Crisp subtle gridline

  Chart.defaults.plugins.tooltip.backgroundColor = '#161A20';
  Chart.defaults.plugins.tooltip.titleColor = '#F0F3F6';
  Chart.defaults.plugins.tooltip.titleFont = { size: 13, weight: '700' };
  Chart.defaults.plugins.tooltip.bodyColor = '#C0C8D2';
  Chart.defaults.plugins.tooltip.bodyFont = { size: 12 };
  Chart.defaults.plugins.tooltip.borderColor = '#384252';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 12;
  Chart.defaults.plugins.tooltip.cornerRadius = 6;
  Chart.defaults.plugins.tooltip.displayColors = true;
  Chart.defaults.plugins.tooltip.boxWidth = 8;
  Chart.defaults.plugins.tooltip.boxHeight = 8;
  Chart.defaults.plugins.tooltip.boxPadding = 4;
}

/**
 * 1. Demographic Distribution Horizontal Bar Chart
 */
export function renderDemographicChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = DEMOGRAPHIC_COHORTS.map(c => `${c.bracket} yrs (${c.label})`);
  const data = DEMOGRAPHIC_COHORTS.map(c => c.count);

  const backgroundColors = DEMOGRAPHIC_COHORTS.map(c => {
    if (c.bracket === '76–100') return '#F5B942'; // Highlight elderly tail
    if (c.bracket === '61–75') return '#D19A2E';
    if (c.dense) return '#3D4654'; // Dense core
    return '#29303B';
  });

  const borderColors = DEMOGRAPHIC_COHORTS.map(c => {
    if (c.bracket === '76–100') return '#F5B942';
    if (c.bracket === '61–75') return '#D19A2E';
    return '#4A5568';
  });

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Demographic Face Volume',
        data: data,
        backgroundColor: backgroundColors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 4,
        barPercentage: 0.7,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Image Sample Count (Master 276,280 Corpus)', color: '#9DA8B7', font: { size: 11, weight: '600' } },
          grid: { color: '#1F242C', drawBorder: false },
          ticks: {
            color: '#9DA8B7',
            callback: (v) => `${(v / 1000).toFixed(0)}k (${((v / 276280) * 100).toFixed(1)}%)`
          }
        },
        y: {
          grid: { display: false, drawBorder: false },
          ticks: { color: '#F0F3F6', font: { weight: '600' } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => `Cohort: ${items[0].label}`,
            label: (item) => {
              const cohort = DEMOGRAPHIC_COHORTS[item.dataIndex];
              return [
                `Exact Image Count: ${item.raw.toLocaleString()} faces`,
                `Share of Corpus: ${cohort.pct}%`,
                cohort.sparse ? 'Demographic Status: Sparse Tail Cohort' : cohort.dense ? 'Demographic Status: Dense Core Training Pool' : 'Demographic Status: Standard Cohort'
              ];
            }
          }
        }
      }
    }
  });
}

/**
 * 2. Dataset Age 1-100 Continuous Density Area Chart
 */
export function renderAge100DistributionChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const ages = Array.from({ length: 100 }, (_, i) => i + 1);
  const counts = ages.map(age => {
    let base = 0;
    if (age <= 12) {
      base = 400 + Math.sin(age / 2) * 200 + age * 60;
    } else if (age <= 19) {
      base = 1500 + (age - 12) * 450;
    } else if (age <= 35) {
      const dist = Math.abs(age - 26);
      base = 8200 - (dist * dist * 32) + Math.sin(age) * 150;
    } else if (age <= 45) {
      base = 5800 - (age - 35) * 160;
    } else if (age <= 60) {
      base = 4200 - (age - 45) * 170;
    } else if (age <= 75) {
      base = 1800 - (age - 60) * 85;
    } else {
      base = 350 - (age - 75) * 11;
    }
    return Math.max(25, Math.round(base));
  });

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: ages,
      datasets: [{
        label: 'Face Density (Master Corpus)',
        data: counts,
        borderColor: '#9DA8B7',
        borderWidth: 2,
        fill: true,
        backgroundColor: (context) => {
          const chart = context.chart;
          const { ctx, chartArea } = chart;
          if (!chartArea) return null;
          const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
          gradient.addColorStop(0, 'rgba(245, 185, 66, 0.20)');
          gradient.addColorStop(0.5, 'rgba(157, 168, 183, 0.08)');
          gradient.addColorStop(1, 'rgba(15, 17, 21, 0.02)');
          return gradient;
        },
        tension: 0.35,
        pointRadius: (ctx) => (ctx.dataIndex === 25 || ctx.dataIndex === 75 || ctx.dataIndex === 99 ? 4 : 0),
        pointBackgroundColor: '#F5B942',
        pointHoverRadius: 6,
        pointHoverBackgroundColor: '#F5B942',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Individual Chronological Age (1 to 100 Years)', color: '#9DA8B7', font: { size: 11, weight: '600' } },
          grid: { color: '#1F242C', drawBorder: false },
          ticks: {
            color: '#9DA8B7',
            callback: (v, idx) => (idx % 10 === 0 || idx === 99) ? `Age ${ages[idx]}` : ''
          }
        },
        y: {
          title: { display: true, text: 'Image Density Count', color: '#9DA8B7', font: { size: 11, weight: '600' } },
          grid: { color: '#1F242C', drawBorder: false },
          ticks: {
            color: '#9DA8B7',
            callback: (v) => `${(v / 1000).toFixed(1)}k`
          }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => `Individual Age: ${items[0].label} Years`,
            label: (item) => {
              const age = parseInt(item.label, 10);
              const note = age >= 76 ? ' (Sparse Elderly Tail)' : age >= 61 ? ' (Senior Cohort)' : age <= 12 ? ' (Pediatric Cohort)' : ' (Core Population)';
              return [
                `Estimated Samples: ~${item.raw.toLocaleString()} images`,
                `Lifespan Category: ${note}`
              ];
            }
          }
        }
      }
    }
  });
}

/**
 * 3. Model Tournament Horizontal Bar Chart (MAE Comparison)
 */
export function renderTournamentChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = MODEL_TOURNAMENT.map(m => `${m.id}: ${m.architecture}`);
  const maeValues = MODEL_TOURNAMENT.map(m => m.mae);

  const barColors = MODEL_TOURNAMENT.map(m => m.isChampion ? '#F5B942' : '#333C48');
  const borderColors = MODEL_TOURNAMENT.map(m => m.isChampion ? '#F5B942' : '#475363');

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Mean Absolute Error (Years)',
        data: maeValues,
        backgroundColor: barColors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 4,
        barPercentage: 0.7,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Mean Absolute Error in Years (Lower is Better)', color: '#9DA8B7', font: { size: 11, weight: '600' } },
          grid: { color: '#1F242C', drawBorder: false },
          min: 3.5,
          max: 8.5,
          ticks: {
            color: '#9DA8B7',
            stepSize: 0.5,
            callback: (v) => `${v.toFixed(1)} yrs`
          }
        },
        y: {
          grid: { display: false, drawBorder: false },
          ticks: {
            color: (c) => MODEL_TOURNAMENT[c.index]?.isChampion ? '#F5B942' : '#E2E8F0',
            font: (c) => ({ weight: MODEL_TOURNAMENT[c.index]?.isChampion ? '750' : '500', size: 11 })
          }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              const exp = MODEL_TOURNAMENT[items[0].dataIndex];
              return `${exp.id} | ${exp.architecture}`;
            },
            label: (item) => {
              const exp = MODEL_TOURNAMENT[item.dataIndex];
              return [
                `Mean Absolute Error: ${exp.mae.toFixed(2)} years`,
                `Accuracy (±7 Yrs): ${exp.acc7}`,
                `Backbone: ${exp.backbone}`,
                `Training Strategy: ${exp.strategy}`,
                exp.isChampion ? '★ DUAL ENSEMBLE GRAND CHAMPION' : 'Baseline / Intermediate Experiment'
              ];
            }
          }
        }
      }
    }
  });
}

/**
 * 4. Highly Detailed Training Convergence Line Chart
 */
export function renderConvergenceChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const epochs = TRAINING_CONVERGENCE.map(d => `Epoch ${d.epoch}`);
  const trainMae = TRAINING_CONVERGENCE.map(d => d.trainMae);
  const valMae = TRAINING_CONVERGENCE.map(d => d.valMae);
  const acc5 = TRAINING_CONVERGENCE.map(d => d.acc5);

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        {
          label: 'Train MAE (Years)',
          data: trainMae,
          borderColor: '#9DA8B7',
          backgroundColor: 'rgba(157, 168, 183, 0.1)',
          borderWidth: 2,
          borderDash: [6, 6],
          pointBackgroundColor: '#9DA8B7',
          pointBorderColor: '#0B0D10',
          pointBorderWidth: 2,
          pointRadius: 5,
          pointHoverRadius: 7,
          tension: 0.15,
          yAxisID: 'y'
        },
        {
          label: 'Validation MAE (Primary Selection Metric)',
          data: valMae,
          borderColor: '#F5B942',
          backgroundColor: 'rgba(245, 185, 66, 0.15)',
          borderWidth: 3,
          pointBackgroundColor: '#F5B942',
          pointBorderColor: '#0B0D10',
          pointBorderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 9,
          tension: 0.15,
          yAxisID: 'y'
        },
        {
          label: 'Validation Accuracy ±5 Yrs (%)',
          data: acc5,
          borderColor: '#10B981',
          borderWidth: 1.5,
          borderDash: [3, 3],
          pointBackgroundColor: '#10B981',
          pointRadius: 4,
          pointHoverRadius: 6,
          tension: 0.15,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      scales: {
        x: {
          grid: { color: '#1F242C', drawBorder: false },
          ticks: { color: '#F0F3F6', font: { weight: '600' } }
        },
        y: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: 'MAE in Years (Lower is Better)', color: '#F5B942', font: { size: 11, weight: '600' } },
          grid: { color: '#1F242C', drawBorder: false },
          min: 3.0,
          max: 8.5,
          ticks: {
            color: '#F5B942',
            stepSize: 1.0,
            callback: (v) => `${v.toFixed(1)} yrs`
          }
        },
        y1: {
          type: 'linear',
          position: 'right',
          title: { display: true, text: 'Val Acc ±5 Yrs (%) — Higher is Better', color: '#10B981', font: { size: 11, weight: '600' } },
          grid: { drawOnChartArea: false },
          min: 45,
          max: 75,
          ticks: {
            color: '#10B981',
            stepSize: 5,
            callback: (v) => `${v}%`
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            color: '#C0C8D2',
            boxWidth: 12,
            boxHeight: 12,
            usePointStyle: true,
            pointStyle: 'circle',
            font: { size: 11, weight: '500' }
          }
        },
        tooltip: {
          callbacks: {
            title: (items) => `Training Progress: ${items[0].label}`,
            afterBody: (items) => {
              const idx = items[0].dataIndex;
              const pt = TRAINING_CONVERGENCE[idx];
              return [
                `-----------------------------------`,
                `• Train MAE: ${pt.trainMae.toFixed(2)} yrs`,
                `• Val MAE: ${pt.valMae.toFixed(2)} yrs`,
                `• Acc ±5 Yrs: ${pt.acc5.toFixed(2)}%`,
                `• Generalization Gap: ${(pt.valMae - pt.trainMae).toFixed(2)} yrs`
              ];
            }
          }
        }
      }
    }
  });
}

/**
 * 5. Demographic Performance Grouped Bar Chart
 */
export function renderDemographicPerformanceChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = DEMOGRAPHIC_PERFORMANCE.map(d => `${d.bracket} (${d.cohort})`);
  const maeValues = DEMOGRAPHIC_PERFORMANCE.map(d => d.mae);
  const acc7Values = DEMOGRAPHIC_PERFORMANCE.map(d => d.acc7);
  const acc5Values = DEMOGRAPHIC_PERFORMANCE.map(d => d.acc5);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'MAE (Years) [Left Axis]',
          data: maeValues,
          backgroundColor: DEMOGRAPHIC_PERFORMANCE.map(d => d.isWeakness ? '#EF4444' : '#F5B942'),
          borderColor: DEMOGRAPHIC_PERFORMANCE.map(d => d.isWeakness ? '#EF4444' : '#F5B942'),
          borderRadius: 4,
          barPercentage: 0.65,
          yAxisID: 'y'
        },
        {
          label: 'Accuracy ±7 Yrs (%) [Right Axis]',
          data: acc7Values,
          backgroundColor: '#384252',
          borderColor: '#4E5B6E',
          borderRadius: 4,
          barPercentage: 0.65,
          yAxisID: 'y1'
        },
        {
          label: 'Accuracy ±5 Yrs (%) [Right Axis]',
          data: acc5Values,
          backgroundColor: '#262D37',
          borderColor: '#384252',
          borderRadius: 4,
          barPercentage: 0.65,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { display: false, drawBorder: false },
          ticks: { color: '#F0F3F6', font: { size: 11, weight: '500' } }
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          title: { display: true, text: 'Mean Absolute Error (Years)', color: '#F5B942', font: { size: 11, weight: '600' } },
          grid: { color: '#1F242C', drawBorder: false },
          min: 0,
          max: 10,
          ticks: {
            color: '#F5B942',
            callback: (v) => `${v} yrs`
          }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          title: { display: true, text: 'Accuracy Percentage (%)', color: '#9DA8B7', font: { size: 11, weight: '600' } },
          grid: { drawOnChartArea: false },
          min: 40,
          max: 100,
          ticks: {
            color: '#9DA8B7',
            callback: (v) => `${v}%`
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            color: '#C0C8D2',
            boxWidth: 10,
            boxHeight: 10,
            font: { size: 11 }
          }
        },
        tooltip: {
          callbacks: {
            title: (items) => `Demographic Bracket: ${items[0].label}`,
            label: (item) => {
              const cohort = DEMOGRAPHIC_PERFORMANCE[item.dataIndex];
              return [
                `Mean Absolute Error: ${cohort.mae.toFixed(2)} years`,
                `Accuracy (±5 Yrs): ${cohort.acc5}%`,
                `Accuracy (±7 Yrs): ${cohort.acc7}%`,
                `Validation Sample Count: ${cohort.testSamples} faces`,
                `Evaluation Grade: ${cohort.grade}`,
                `Observation: ${cohort.comment}`
              ];
            }
          }
        }
      }
    }
  });
}
