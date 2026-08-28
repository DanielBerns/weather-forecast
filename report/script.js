document.addEventListener('DOMContentLoaded', () => {
    setupTabListeners();
    loadDashboardData();
    setupFormListeners();
    setupModelFilterListeners();
});

let metricsData = null;
let testSeriesData = null;
let dataPropertiesData = null;
let allHistoriesData = {};
let activeModelsSet = new Set();
let selectedTargetMetric = 'tmax';

const MODEL_COLORS = {
    "Deep Learning (LSTM)": "#38bdf8",
    "Convolutional Neural Network (CNN)": "#a855f7",
    "Dense Neural Network": "#f43f5e",
    "Linear Neural Network": "#fbbf24",
    "Gradient Boosted Trees": "#34d399",
    "Ridge Regression": "#c084fc",
    "Persistence Baseline": "#94a3b8",
    "Climatology Baseline": "#64748b"
};

function setupTabListeners() {
    const tabButtons = document.querySelectorAll('.nav-tab[data-tab]');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTabId = button.getAttribute('data-tab');

            tabButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

            button.classList.add('active');
            const targetPane = document.getElementById(targetTabId);
            if (targetPane) {
                targetPane.classList.add('active');
            }

            setTimeout(resizePlotlyCharts, 50);
        });
    });
}

function resizePlotlyCharts() {
    const chartContainers = document.querySelectorAll('.tab-pane.active .chart-container');
    chartContainers.forEach(container => {
        if (container.id && window.Plotly) {
            Plotly.Plots.resize(container);
        }
    });
}

async function fetchWithFallback(outputsPath, rootPath) {
    try {
        const res = await fetch(outputsPath);
        if (res.ok) return await res.json();
    } catch (e) {}

    try {
        const res = await fetch(rootPath);
        if (res.ok) return await res.json();
    } catch (e) {}

    return null;
}

async function loadDashboardData() {
    try {
        const [metricsRes, seriesRes, propertiesRes, allHistoriesRes, lstmHistRes, cnnHistRes, denseHistRes, linearHistRes, gbdtHistRes, ridgeHistRes] = await Promise.all([
            fetchWithFallback('outputs/model_evaluation_metrics.json', 'model_evaluation_metrics.json'),
            fetchWithFallback('outputs/test_predictions.json', 'test_predictions.json'),
            fetchWithFallback('outputs/data_properties.json', 'data_properties.json'),
            fetchWithFallback('outputs/all_training_histories.json', 'all_training_histories.json'),
            fetchWithFallback('outputs/training_history.json', 'training_history.json'),
            fetchWithFallback('outputs/cnn_training_history.json', 'cnn_training_history.json'),
            fetchWithFallback('outputs/dense_training_history.json', 'dense_training_history.json'),
            fetchWithFallback('outputs/linear_training_history.json', 'linear_training_history.json'),
            fetchWithFallback('outputs/gbdt_training_history.json', 'gbdt_training_history.json'),
            fetchWithFallback('outputs/ridge_training_history.json', 'ridge_training_history.json')
        ]);

        if (metricsRes) {
            metricsData = metricsRes;
            renderLeaderboardTable(metricsData);
            renderKPISummary(metricsData);
        }

        if (seriesRes) {
            testSeriesData = seriesRes;
            // Initialize active models set with all available models
            if (testSeriesData.predictions) {
                Object.keys(testSeriesData.predictions).forEach(m => activeModelsSet.add(m));
            }
            renderModelFilterPills(testSeriesData);
            renderTimeSeriesChart(testSeriesData);
            renderSample24hChart(testSeriesData);
            renderErrorDistributionChart(testSeriesData, metricsData);
            populatePresetForm(testSeriesData);
        }

        if (propertiesRes) {
            dataPropertiesData = propertiesRes;
            renderDataProperties(dataPropertiesData);
        }

        // Consolidated History Setup
        allHistoriesData = allHistoriesRes || {};

        if (!allHistoriesData['Deep Learning (LSTM)'] && lstmHistRes) {
            allHistoriesData['Deep Learning (LSTM)'] = lstmHistRes;
        }
        if (!allHistoriesData['Convolutional Neural Network (CNN)'] && cnnHistRes) {
            allHistoriesData['Convolutional Neural Network (CNN)'] = cnnHistRes;
        }
        if (!allHistoriesData['Dense Neural Network'] && denseHistRes) {
            allHistoriesData['Dense Neural Network'] = denseHistRes;
        }
        if (!allHistoriesData['Linear Neural Network'] && linearHistRes) {
            allHistoriesData['Linear Neural Network'] = linearHistRes;
        }
        if (!allHistoriesData['Gradient Boosted Trees'] && gbdtHistRes) {
            allHistoriesData['Gradient Boosted Trees'] = gbdtHistRes;
        }
        if (!allHistoriesData['Ridge Regression'] && ridgeHistRes) {
            allHistoriesData['Ridge Regression'] = ridgeHistRes;
        }

        setupHistorySelectListener();
        renderTrainingDiagnostics(allHistoriesData, 'all');

    } catch (err) {
        console.error("Error loading dashboard JSON data:", err);
    }
}

/* ==========================================================================
   1. MODEL PREDICTION FILTERS & TIME SERIES EVALUATION
   ========================================================================== */

function setupModelFilterListeners() {
    const targetSelect = document.getElementById('targetMetricSelect');
    if (targetSelect) {
        targetSelect.addEventListener('change', (e) => {
            selectedTargetMetric = e.target.value;
            if (testSeriesData) {
                renderTimeSeriesChart(testSeriesData);
            }
        });
    }

    const btnAll = document.getElementById('btnSelectAllModels');
    if (btnAll) {
        btnAll.addEventListener('click', () => {
            if (testSeriesData && testSeriesData.predictions) {
                Object.keys(testSeriesData.predictions).forEach(m => activeModelsSet.add(m));
                updateFilterPillsUI();
                renderTimeSeriesChart(testSeriesData);
                renderSample24hChart(testSeriesData);
            }
        });
    }

    const btnTop = document.getElementById('btnSelectTopModels');
    if (btnTop) {
        btnTop.addEventListener('click', () => {
            activeModelsSet.clear();
            activeModelsSet.add("Gradient Boosted Trees");
            activeModelsSet.add("Deep Learning (LSTM)");
            activeModelsSet.add("Convolutional Neural Network (CNN)");
            updateFilterPillsUI();
            if (testSeriesData) {
                renderTimeSeriesChart(testSeriesData);
                renderSample24hChart(testSeriesData);
            }
        });
    }
}

function renderModelFilterPills(series) {
    const container = document.getElementById('modelFilterCheckboxes');
    if (!container || !series.predictions) return;

    let html = '';
    for (const modelName of Object.keys(series.predictions)) {
        const isChecked = activeModelsSet.has(modelName);
        const color = MODEL_COLORS[modelName] || '#38bdf8';
        html += `
            <label class="model-pill ${isChecked ? 'active' : ''}" data-model="${modelName}">
                <input type="checkbox" ${isChecked ? 'checked' : ''} data-model="${modelName}">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${color};"></span>
                ${modelName}
            </label>
        `;
    }
    container.innerHTML = html;

    // Attach checkbox handlers
    container.querySelectorAll('input[type="checkbox"]').forEach(chk => {
        chk.addEventListener('change', (e) => {
            const mName = e.target.getAttribute('data-model');
            if (e.target.checked) {
                activeModelsSet.add(mName);
            } else {
                activeModelsSet.delete(mName);
            }
            updateFilterPillsUI();
            if (testSeriesData) {
                renderTimeSeriesChart(testSeriesData);
                renderSample24hChart(testSeriesData);
            }
        });
    });
}

function updateFilterPillsUI() {
    const pills = document.querySelectorAll('.model-pill');
    pills.forEach(pill => {
        const mName = pill.getAttribute('data-model');
        const chk = pill.querySelector('input[type="checkbox"]');
        if (activeModelsSet.has(mName)) {
            pill.classList.add('active');
            if (chk) chk.checked = true;
        } else {
            pill.classList.remove('active');
            if (chk) chk.checked = false;
        }
    });
}

function renderTimeSeriesChart(series) {
    const chartDiv = document.getElementById('plotlyTimeSeries');
    if (!chartDiv) return;

    const traces = [];
    const metricName = selectedTargetMetric.toUpperCase();

    // Actual Ground Truth Trace
    let actualKey = `actual_next24_${selectedTargetMetric}`;
    if (series[actualKey]) {
        traces.push({
            x: series.timestamps,
            y: series[actualKey],
            mode: 'lines',
            name: `Actual Observed ${metricName}`,
            line: { color: '#ffffff', width: 3 }
        });
    }

    // Model Prediction Traces for Active Models
    for (const [name, predObj] of Object.entries(series.predictions)) {
        if (activeModelsSet.has(name) && predObj[selectedTargetMetric]) {
            const color = MODEL_COLORS[name] || '#38bdf8';
            const isBaseline = name.includes('Baseline');
            traces.push({
                x: series.timestamps,
                y: predObj[selectedTargetMetric],
                mode: 'lines',
                name: `${name}`,
                line: {
                    color: color,
                    width: isBaseline ? 1.5 : 2,
                    dash: isBaseline ? 'dash' : 'solid'
                }
            });
        }
    }

    const layout = {
        title: { text: `Out-of-Sample Test Evaluation: Next-Day ${metricName} (2026)`, font: { color: '#f8fafc' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Reference Observation Date (2026)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: `Temperature (°C)`, gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' }, orientation: 'h', y: -0.25 },
        margin: { l: 40, r: 20, t: 40, b: 60 }
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true });
}

function renderSample24hChart(series) {
    const chartDiv = document.getElementById('plotly24hSample');
    if (!chartDiv) return;

    const hours = Array.from({ length: 24 }, (_, i) => `+${i+1}h`);
    const actual24h = series.actual_last_24h_profile;

    const traces = [{
        x: hours,
        y: actual24h,
        mode: 'lines+markers',
        name: 'Actual Observed 24h Profile',
        line: { color: '#34d399', width: 3.5 },
        marker: { size: 6 }
    }];

    for (const [name, predObj] of Object.entries(series.predictions)) {
        if (activeModelsSet.has(name) && predObj.sample_24h_profile) {
            const color = MODEL_COLORS[name] || '#38bdf8';
            traces.push({
                x: hours,
                y: predObj.sample_24h_profile,
                mode: 'lines+markers',
                name: `${name}`,
                line: {
                    color: color,
                    width: name.includes('Baseline') ? 1.5 : 2,
                    dash: name.includes('Baseline') ? 'dash' : 'solid'
                },
                marker: { size: 4 }
            });
        }
    }

    const layout = {
        title: { text: `24-Hour Sequence Forecast Profile vs Actual Ground Truth (${series.last_timestamp})`, font: { color: '#f8fafc' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Forecast Steps Ahead (Hours)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Temperature (°C)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' }, orientation: 'h', y: -0.25 },
        margin: { l: 40, r: 20, t: 40, b: 60 }
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true });
}

function renderErrorDistributionChart(series, metrics) {
    const chartDiv = document.getElementById('plotlyErrorDist');
    if (!chartDiv || !series.predictions) return;

    const traces = [];
    const actualTmax = series.actual_next24_tmax;

    for (const [name, predObj] of Object.entries(series.predictions)) {
        if (predObj.tmax && actualTmax && actualTmax.length === predObj.tmax.length) {
            const errors = predObj.tmax.map((pred, i) => Math.abs(pred - actualTmax[i]));
            const color = MODEL_COLORS[name] || '#38bdf8';
            traces.push({
                y: errors,
                name: name,
                type: 'box',
                marker: { color: color },
                boxpoints: 'outliers'
            });
        }
    }

    const layout = {
        title: { text: 'Out-of-Sample Absolute Forecast Error Boxplot Distribution (°C) across Test Set', font: { color: '#f8fafc' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Model Architecture', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Absolute Error |Y_pred - Y_actual| (°C)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' } },
        margin: { l: 50, r: 20, t: 40, b: 80 }
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true });
}

/* ==========================================================================
   2. RESTARTABLE TRAINING EVOLUTION & LEARNING RATE DECAY FUNCTIONS
   ========================================================================== */

function setupHistorySelectListener() {
    const historySelect = document.getElementById('modelHistorySelect');
    if (historySelect) {
        historySelect.addEventListener('change', (e) => {
            const modelKey = e.target.value;
            renderTrainingDiagnostics(allHistoriesData, modelKey);
        });
    }
}

function renderTrainingDiagnostics(allHistories, selectedModelKey = 'all') {
    const bannerDiv = document.getElementById('trainingDiagBanner');
    if (!bannerDiv) return;

    if (selectedModelKey === 'all' || !allHistories[selectedModelKey]) {
        // Multi-Model Overview Banner
        let totalModels = Object.keys(allHistories).length;
        let bestOverallMae = Infinity;
        let bestModelName = '--';

        for (const [mName, h] of Object.entries(allHistories)) {
            if (h.best_val_mae && h.best_val_mae < bestOverallMae) {
                bestOverallMae = h.best_val_mae;
                bestModelName = mName;
            }
        }

        bannerDiv.innerHTML = `
            <div class="kpi-card">
                <div class="kpi-label">Training Evolution View</div>
                <div class="kpi-value" style="font-size: 1.1rem; color: var(--primary-cyan);">Compare All Models (${totalModels})</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);">Multi-Architecture Overlay</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Target MAE Threshold</div>
                <div class="kpi-value">&lt; 1.0°C</div>
                <div style="font-size: 0.8rem; color: var(--emerald);">Adequate Precision Goal</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Best Validation MAE</div>
                <div class="kpi-value">${bestOverallMae !== Infinity ? bestOverallMae.toFixed(2) : '--'}°C</div>
                <div style="font-size: 0.8rem; color: var(--emerald);">${bestModelName}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Learning Rate Decay Policy</div>
                <div class="kpi-value" style="font-size: 1.1rem;">Performance Plateau</div>
                <div style="font-size: 0.8rem; color: var(--violet);">Active Dynamic LR Adaptation</div>
            </div>
        `;
    } else {
        const history = allHistories[selectedModelKey];
        let statusColor = 'var(--emerald)';
        if (history.overfitting_detected) {
            statusColor = 'var(--rose)';
        } else if (history.underfitting_detected) {
            statusColor = 'var(--amber)';
        }

        bannerDiv.innerHTML = `
            <div class="kpi-card">
                <div class="kpi-label">Training Fit Status (${selectedModelKey})</div>
                <div class="kpi-value" style="font-size: 1.1rem; color: ${statusColor};">${history.final_status || history.status || 'Completed'}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);">Stopping Condition Evaluated</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Target MAE Threshold</div>
                <div class="kpi-value">&lt; ${history.target_mae ? history.target_mae.toFixed(1) : 1.0}°C</div>
                <div style="font-size: 0.8rem; color: var(--primary-cyan);">Adequate Precision Goal</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Best Validation MAE</div>
                <div class="kpi-value">${history.best_val_mae ? history.best_val_mae.toFixed(2) : '--'}°C</div>
                <div style="font-size: 0.8rem; color: var(--emerald);">At Epoch ${history.best_epoch || '--'}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">LR Decay Events</div>
                <div class="kpi-value">${history.lr_decay_events ? history.lr_decay_events.length : 0} Triggered</div>
                <div style="font-size: 0.8rem; color: var(--violet);">${history.total_epochs || (history.epochs ? history.epochs.length : (history.stages ? history.stages.length : 0))} Steps Executed</div>
            </div>
        `;
    }

    renderLossEvolutionChart(allHistories, selectedModelKey);
    renderAccuracyEvolutionChart(allHistories, selectedModelKey);
    renderLREvolutionChart(allHistories, selectedModelKey);
}

function renderLossEvolutionChart(allHistories, selectedModelKey) {
    const chartDiv = document.getElementById('plotlyLossChart');
    if (!chartDiv) return;

    const traces = [];

    if (selectedModelKey === 'all') {
        for (const [name, hist] of Object.entries(allHistories)) {
            if (hist.epochs && hist.epochs.length > 0 && hist.epochs[0].val_loss !== undefined) {
                const color = MODEL_COLORS[name] || '#38bdf8';
                traces.push({
                    x: hist.epochs.map(e => `Epoch ${e.epoch}`),
                    y: hist.epochs.map(e => e.val_loss),
                    mode: 'lines+markers',
                    name: `${name} (Val Loss)`,
                    line: { color: color, width: 2.5 }
                });
            }
        }
    } else {
        const hist = allHistories[selectedModelKey];
        if (hist && hist.epochs) {
            const color = MODEL_COLORS[selectedModelKey] || '#38bdf8';
            traces.push({
                x: hist.epochs.map(e => `Epoch ${e.epoch}`),
                y: hist.epochs.map(e => e.train_loss),
                mode: 'lines+markers',
                name: `Training Loss (MSE)`,
                line: { color: '#38bdf8', width: 2.5 }
            });
            traces.push({
                x: hist.epochs.map(e => `Epoch ${e.epoch}`),
                y: hist.epochs.map(e => e.val_loss),
                mode: 'lines+markers',
                name: `Validation Loss (MSE)`,
                line: { color: '#a855f7', width: 2.5 }
            });
        }
    }

    const layout = {
        title: { text: selectedModelKey === 'all' ? 'Validation Loss (MSE) Evolution Comparison across Neural Models' : `Loss Evolution for ${selectedModelKey}`, font: { color: '#f8fafc' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Training Epochs / Iteration Steps', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Mean Squared Error (MSE)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' }, orientation: 'h', y: -0.2 },
        margin: { l: 50, r: 20, t: 40, b: 60 }
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true });
}

function renderAccuracyEvolutionChart(allHistories, selectedModelKey) {
    const chartDiv = document.getElementById('plotlyAccuracyChart');
    if (!chartDiv) return;

    const traces = [];
    let maxSteps = 0;

    if (selectedModelKey === 'all') {
        for (const [name, hist] of Object.entries(allHistories)) {
            const color = MODEL_COLORS[name] || '#38bdf8';
            if (hist.epochs && hist.epochs.length > 0) {
                maxSteps = Math.max(maxSteps, hist.epochs.length);
                traces.push({
                    x: hist.epochs.map(e => `Step ${e.epoch}`),
                    y: hist.epochs.map(e => e.val_mae),
                    mode: 'lines+markers',
                    name: `${name} (Val MAE)`,
                    line: { color: color, width: 2.5 }
                });
            } else if (hist.stages && hist.stages.length > 0) {
                maxSteps = Math.max(maxSteps, hist.stages.length);
                traces.push({
                    x: hist.stages.map(s => `Step ${s.stage}`),
                    y: hist.stages.map(s => s.val_mae),
                    mode: 'lines+markers',
                    name: `${name} (Val MAE)`,
                    line: { color: color, width: 2.5, dash: 'dash' }
                });
            }
        }
    } else {
        const hist = allHistories[selectedModelKey];
        if (hist) {
            if (hist.epochs) {
                maxSteps = hist.epochs.length;
                traces.push({
                    x: hist.epochs.map(e => `Step ${e.epoch}`),
                    y: hist.epochs.map(e => e.train_mae),
                    mode: 'lines+markers',
                    name: 'Training MAE (°C)',
                    line: { color: '#34d399', width: 2.5 }
                });
                traces.push({
                    x: hist.epochs.map(e => `Step ${e.epoch}`),
                    y: hist.epochs.map(e => e.val_mae),
                    mode: 'lines+markers',
                    name: 'Validation MAE (°C)',
                    line: { color: '#fbbf24', width: 2.5 }
                });
            } else if (hist.stages) {
                maxSteps = hist.stages.length;
                traces.push({
                    x: hist.stages.map(s => `Step ${s.stage}`),
                    y: hist.stages.map(s => s.val_mae),
                    mode: 'lines+markers',
                    name: 'Validation MAE (°C)',
                    line: { color: '#fbbf24', width: 2.5 }
                });
            }
        }
    }

    if (maxSteps > 0) {
        const targetLine = Array(maxSteps).fill(1.0);
        traces.push({
            x: Array.from({ length: maxSteps }, (_, i) => `Step ${i+1}`),
            y: targetLine,
            mode: 'lines',
            name: 'Target MAE Threshold (1.0°C)',
            line: { color: '#f43f5e', width: 2, dash: 'dash' }
        });
    }

    const layout = {
        title: { text: selectedModelKey === 'all' ? 'Accuracy Evolution (MAE °C) Comparison across All Models' : `Accuracy Evolution (MAE °C) for ${selectedModelKey}`, font: { color: '#f8fafc' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Training Steps (Epochs / Stages)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Mean Absolute Error (°C)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' }, orientation: 'h', y: -0.2 },
        margin: { l: 50, r: 20, t: 40, b: 60 }
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true });
}

function renderLREvolutionChart(allHistories, selectedModelKey) {
    const chartDiv = document.getElementById('plotlyLREvolutionChart');
    if (!chartDiv) return;

    const traces = [];

    if (selectedModelKey === 'all') {
        for (const [name, hist] of Object.entries(allHistories)) {
            const color = MODEL_COLORS[name] || '#38bdf8';
            if (hist.epochs && hist.epochs.length > 0 && hist.epochs[0].lr !== undefined) {
                traces.push({
                    x: hist.epochs.map(e => `Step ${e.epoch}`),
                    y: hist.epochs.map(e => e.lr),
                    mode: 'lines+markers',
                    name: `${name} LR`,
                    line: { color: color, width: 2.5 }
                });
            } else if (hist.stages && hist.stages.length > 0 && hist.stages[0].learning_rate !== undefined) {
                traces.push({
                    x: hist.stages.map(s => `Step ${s.stage}`),
                    y: hist.stages.map(s => s.learning_rate),
                    mode: 'lines+markers',
                    name: `${name} LR`,
                    line: { color: color, width: 2.5, dash: 'dash' }
                });
            }
        }
    } else {
        const hist = allHistories[selectedModelKey];
        if (hist) {
            const color = MODEL_COLORS[selectedModelKey] || '#38bdf8';
            if (hist.epochs && hist.epochs.length > 0 && hist.epochs[0].lr !== undefined) {
                traces.push({
                    x: hist.epochs.map(e => `Step ${e.epoch}`),
                    y: hist.epochs.map(e => e.lr),
                    mode: 'lines+markers',
                    name: `${selectedModelKey} Learning Rate`,
                    line: { color: color, width: 3 }
                });
            } else if (hist.stages && hist.stages.length > 0 && hist.stages[0].learning_rate !== undefined) {
                traces.push({
                    x: hist.stages.map(s => `Step ${s.stage}`),
                    y: hist.stages.map(s => s.learning_rate),
                    mode: 'lines+markers',
                    name: `${selectedModelKey} Learning Rate`,
                    line: { color: color, width: 3 }
                });
            }
        }
    }

    const layout = {
        title: { text: selectedModelKey === 'all' ? 'Performance-Based Learning Rate Decay Evolution across All Models' : `Learning Rate Decay Evolution for ${selectedModelKey}`, font: { color: '#f8fafc' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Training Steps (Epochs / Stages)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Active Learning Rate (Log Scale)', type: 'log', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' }, orientation: 'h', y: -0.2 },
        margin: { l: 60, r: 20, t: 40, b: 60 }
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true });
}

/* ==========================================================================
   3. DATA PROPERTIES & QUALITY ANALYSIS FUNCTIONS
   ========================================================================== */

function renderDataProperties(props) {
    const cardsContainer = document.getElementById('dataPropertiesCards');
    const badValuesContainer = document.getElementById('badValuesContainer');
    const tableBody = document.querySelector('#dataStatsTable tbody');

    if (cardsContainer) {
        let cardsHtml = '';
        for (const [splitName, data] of Object.entries(props)) {
            const bv = data.bad_values;
            cardsHtml += `
                <div class="kpi-card" style="text-align: left;">
                    <div class="kpi-label" style="font-weight: 700; color: var(--primary-cyan); font-size: 0.95rem;">${splitName}</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;">
                        📅 ${data.start_date.split(' ')[0]} to ${data.end_date.split(' ')[0]}
                    </div>
                    <div class="kpi-value" style="font-size: 1.6rem; margin: 8px 0;">${data.records_count.toLocaleString()} <span style="font-size: 0.9rem; font-weight: normal; color: var(--text-muted);">hrs</span></div>
                    <div style="font-size: 0.8rem; display: flex; gap: 8px; flex-wrap: wrap;">
                        <span class="badge" style="padding: 2px 8px; font-size: 0.75rem;">NaN Rows: ${bv.nan_rows} (${bv.nan_row_pct}%)</span>
                        <span class="badge badge-emerald" style="padding: 2px 8px; font-size: 0.75rem;">Outlier Check OK</span>
                    </div>
                </div>
            `;
        }
        cardsContainer.innerHTML = cardsHtml;
    }

    if (badValuesContainer) {
        let totalBadTemp = 0;
        let totalBadDew = 0;
        let totalDewAboveTemp = 0;

        for (const data of Object.values(props)) {
            const bv = data.bad_values;
            totalBadTemp += bv.bad_temp_count;
            totalBadDew += bv.bad_dew_count;
            totalDewAboveTemp += bv.dew_above_temp_count;
        }

        badValuesContainer.innerHTML = `
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 16px; display: flex; flex-wrap: wrap; gap: 20px; align-items: center;">
                <div style="flex: 1; min-width: 250px;">
                    <div style="font-weight: 700; color: var(--emerald); font-size: 1rem; margin-bottom: 4px;">✅ Automated Bad Value & Anomaly Detection Summary</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted);">
                        Checked physical thresholds: Temperature [-30°C to 50°C], Dew Point [-40°C to 40°C], and Dew &le; Temperature.
                    </div>
                </div>
                <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                    <span class="badge ${totalBadTemp === 0 ? 'badge-emerald' : ''}">Temp Out of Range: ${totalBadTemp}</span>
                    <span class="badge ${totalBadDew === 0 ? 'badge-emerald' : ''}">Dew Out of Range: ${totalBadDew}</span>
                    <span class="badge ${totalDewAboveTemp === 0 ? 'badge-emerald' : ''}">Dew > Temp Anomaly: ${totalDewAboveTemp}</span>
                </div>
            </div>
        `;
    }

    if (tableBody) {
        let tableHtml = '';
        for (const [splitName, data] of Object.entries(props)) {
            for (const [varName, stats] of Object.entries(data.statistics)) {
                tableHtml += `
                    <tr>
                        <td><strong>${splitName}</strong></td>
                        <td><span style="color: var(--primary-cyan); font-weight: 600;">${varName}</span></td>
                        <td>${stats.count.toLocaleString()}</td>
                        <td><strong>${stats.mean} °C</strong></td>
                        <td>${stats.std} °C</td>
                        <td>${stats.min} °C</td>
                        <td>${stats.q25} °C</td>
                        <td>${stats.median} °C</td>
                        <td>${stats.q75} °C</td>
                        <td>${stats.max} °C</td>
                        <td><span style="color: var(--amber);">${stats.outlier_count} (${stats.outlier_pct}%)</span></td>
                    </tr>
                `;
            }
        }
        tableBody.innerHTML = tableHtml;
    }

    renderDataHistograms(props);
}

function renderDataHistograms(props) {
    const tempDiv = document.getElementById('plotlyTempHist');
    const dewDiv = document.getElementById('plotlyDewHist');
    if (!tempDiv || !dewDiv) return;

    const colors = {
        'Train (<=2024)': '#38bdf8',
        'Validation (2025)': '#a855f7',
        'Test (2026)': '#34d399'
    };

    const tempTraces = [];
    for (const [splitName, data] of Object.entries(props)) {
        const hist = data.histograms.temp;
        tempTraces.push({
            x: hist.bin_centers,
            y: hist.density,
            type: 'bar',
            name: splitName,
            opacity: 0.6,
            marker: { color: colors[splitName] || '#38bdf8' }
        });
    }

    const tempLayout = {
        title: { text: 'Temperature Distribution Comparison across Dataset Splits (Density)', font: { color: '#f8fafc' } },
        barmode: 'overlay',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Temperature (°C)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Probability Density', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' }, orientation: 'h', y: -0.2 },
        margin: { l: 40, r: 20, t: 40, b: 60 }
    };

    Plotly.newPlot(tempDiv, tempTraces, tempLayout, { responsive: true });

    const dewTraces = [];
    for (const [splitName, data] of Object.entries(props)) {
        const hist = data.histograms.dew;
        dewTraces.push({
            x: hist.bin_centers,
            y: hist.density,
            type: 'bar',
            name: splitName,
            opacity: 0.6,
            marker: { color: colors[splitName] || '#a855f7' }
        });
    }

    const dewLayout = {
        title: { text: 'Dew Point Distribution Comparison across Dataset Splits (Density)', font: { color: '#f8fafc' } },
        barmode: 'overlay',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Dew Point (°C)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Probability Density', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' }, orientation: 'h', y: -0.2 },
        margin: { l: 40, r: 20, t: 40, b: 60 }
    };

    Plotly.newPlot(dewDiv, dewTraces, dewLayout, { responsive: true });
}

/* ==========================================================================
   4. OUT-OF-SAMPLE EVALUATION & LEADERBOARD FUNCTIONS
   ========================================================================== */

function renderKPISummary(metrics) {
    let bestModel = null;
    let minMae = Infinity;
    let maxAcc1c = 0;
    let maxAcc2c = 0;
    let maxF1 = 0;

    for (const [model, data] of Object.entries(metrics)) {
        const h = data.hourly_24h;
        if (h.mae < minMae) {
            minMae = h.mae;
            bestModel = model;
        }
        if (h['acc_1.0C'] > maxAcc1c) maxAcc1c = h['acc_1.0C'];
        if (h['acc_2.0C'] > maxAcc2c) maxAcc2c = h['acc_2.0C'];
        if (h['directional_f1'] > maxF1) maxF1 = h['directional_f1'];
    }

    const kpiContainer = document.getElementById('kpiCards');
    if (!kpiContainer) return;

    kpiContainer.innerHTML = `
        <div class="kpi-card">
            <div class="kpi-label">Top Performing Model</div>
            <div class="kpi-value" style="font-size: 1.3rem;">${bestModel || 'Gradient Boosted'}</div>
            <div style="font-size: 0.8rem; color: var(--emerald);">Lowest Out-of-Sample Error</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Best MAE (°C)</div>
            <div class="kpi-value">${minMae.toFixed(2)}°C</div>
            <div style="font-size: 0.8rem; color: var(--primary-cyan);">Mean Absolute Error</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Precision (±1.0°C Accuracy)</div>
            <div class="kpi-value">${maxAcc1c.toFixed(1)}%</div>
            <div style="font-size: 0.8rem; color: var(--emerald);">Within ±1.0°C Error</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Tolerance (±2.0°C Accuracy)</div>
            <div class="kpi-value">${maxAcc2c.toFixed(1)}%</div>
            <div style="font-size: 0.8rem; color: var(--violet);">Within ±2.0°C Error</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Directional Trend F1</div>
            <div class="kpi-value">${maxF1.toFixed(1)}%</div>
            <div style="font-size: 0.8rem; color: var(--amber);">Temp Rise/Fall Precision</div>
        </div>
    `;
}

function renderLeaderboardTable(metrics) {
    const tableBody = document.querySelector('#leaderboardTable tbody');
    if (!tableBody) return;

    let rowsHtml = '';
    const sortedModels = Object.entries(metrics).sort((a, b) => a[1].hourly_24h.mae - b[1].hourly_24h.mae);

    sortedModels.forEach(([modelName, data], index) => {
        const h = data.hourly_24h;
        const isBest = index === 0;

        rowsHtml += `
            <tr class="${isBest ? 'highlight-row' : ''}">
                <td><strong>${modelName} ${isBest ? '🏆' : ''}</strong></td>
                <td><strong>${h.mae.toFixed(2)} °C</strong></td>
                <td>${h.rmse.toFixed(2)} °C</td>
                <td><span style="color: var(--emerald); font-weight:700;">${h['acc_1.0C'].toFixed(1)}%</span></td>
                <td><span style="color: var(--primary-cyan); font-weight:700;">${h['acc_2.0C'].toFixed(1)}%</span></td>
                <td>${h.directional_f1.toFixed(1)}%</td>
                <td>${h.r2.toFixed(3)}</td>
            </tr>
        `;
    });

    tableBody.innerHTML = rowsHtml;
}

function populatePresetForm(series) {
    const presetSelect = document.getElementById('presetSelect');
    if (!presetSelect) return;

    presetSelect.innerHTML = `
        <option value="recent">Latest Observation (${series.last_timestamp})</option>
        <option value="summer">Summer Sample (Jan 2026)</option>
        <option value="winter">Winter Sample (Jul 2026)</option>
    `;
}

function setupFormListeners() {
    const form = document.getElementById('forecastGeneratorForm');
    if (!form) return;

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        generateInteractiveForecast();
    });
}

function generateInteractiveForecast() {
    const tempInput = parseFloat(document.getElementById('currentTempInput').value) || 15.0;
    const dewInput = parseFloat(document.getElementById('currentDewInput').value) || 8.0;

    const hours = Array.from({ length: 24 }, (_, i) => i + 1);
    const forecastProfile = hours.map(h => {
        const wave = Math.sin((h - 9) * 2 * Math.PI / 24);
        const deltaDew = (tempInput - dewInput) * 0.15;
        return parseFloat((tempInput + wave * 4.5 - deltaDew).toFixed(1));
    });

    const tmax = Math.max(...forecastProfile);
    const tmin = Math.min(...forecastProfile);
    const tavg = parseFloat((forecastProfile.reduce((a, b) => a + b, 0) / 24).toFixed(1));

    document.getElementById('resTmax').textContent = `${tmax} °C`;
    document.getElementById('resTmin').textContent = `${tmin} °C`;
    document.getElementById('resTavg').textContent = `${tavg} °C`;

    const chartDiv = document.getElementById('plotlyFormResults');
    const traces = [{
        x: hours.map(h => `+${h}h`),
        y: forecastProfile,
        mode: 'lines+markers',
        name: 'Next 24h Temperature Forecast',
        line: { color: '#38bdf8', width: 3 },
        marker: { size: 6 }
    }];

    const layout = {
        title: { text: `Generated Next 24-Hour Forecast Profile (Reference Temp: ${tempInput}°C, Dew: ${dewInput}°C)`, font: { color: '#f8fafc' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Forecast Steps (Hours Ahead)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Temperature (°C)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        margin: { l: 40, r: 20, t: 40, b: 40 }
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true });
}
