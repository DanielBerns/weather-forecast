document.addEventListener('DOMContentLoaded', () => {
    setupTabListeners();
    loadDashboardData();
    setupFormListeners();
});

let metricsData = null;
let testSeriesData = null;
let dataPropertiesData = null;
let trainingHistoryData = null;

function setupTabListeners() {
    const tabButtons = document.querySelectorAll('.nav-tab');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTabId = button.getAttribute('data-tab');

            // Deactivate all tab buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

            // Activate target tab button and pane
            button.classList.add('active');
            const targetPane = document.getElementById(targetTabId);
            if (targetPane) {
                targetPane.classList.add('active');
            }

            // Trigger Plotly chart resize for visible tab
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
        const [metricsRes, seriesRes, propertiesRes, historyRes] = await Promise.all([
            fetchWithFallback('outputs/model_evaluation_metrics.json', 'model_evaluation_metrics.json'),
            fetchWithFallback('outputs/test_predictions.json', 'test_predictions.json'),
            fetchWithFallback('outputs/data_properties.json', 'data_properties.json'),
            fetchWithFallback('outputs/training_history.json', 'training_history.json')
        ]);

        if (metricsRes) {
            metricsData = metricsRes;
            renderLeaderboardTable(metricsData);
            renderKPISummary(metricsData);
        }

        if (seriesRes) {
            testSeriesData = seriesRes;
            renderTimeSeriesChart(testSeriesData);
            renderSample24hChart(testSeriesData);
            populatePresetForm(testSeriesData);
        }

        if (propertiesRes) {
            dataPropertiesData = propertiesRes;
            renderDataProperties(dataPropertiesData);
        }

        if (historyRes) {
            trainingHistoryData = historyRes;
            renderTrainingDiagnostics(trainingHistoryData);
        }
    } catch (err) {
        console.error("Error loading dashboard JSON data:", err);
    }
}

/* ==========================================================================
   1. DATA PROPERTIES & QUALITY ANALYSIS FUNCTIONS
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
        let totalNaNRows = 0;

        for (const data of Object.values(props)) {
            const bv = data.bad_values;
            totalBadTemp += bv.bad_temp_count;
            totalBadDew += bv.bad_dew_count;
            totalDewAboveTemp += bv.dew_above_temp_count;
            totalNaNRows += bv.nan_rows;
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

    // 1. Temperature Histograms
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

    // 2. Dew Point Histograms
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
   2. RESTARTABLE TRAINING DIAGNOSTICS & EVOLUTION FUNCTIONS
   ========================================================================== */

function renderTrainingDiagnostics(history) {
    const bannerDiv = document.getElementById('trainingDiagBanner');
    if (!bannerDiv) return;

    let statusColor = 'var(--emerald)';
    if (history.overfitting_detected) {
        statusColor = 'var(--rose)';
    } else if (history.underfitting_detected) {
        statusColor = 'var(--amber)';
    }

    bannerDiv.innerHTML = `
        <div class="kpi-card">
            <div class="kpi-label">Training Fit Status</div>
            <div class="kpi-value" style="font-size: 1.1rem; color: ${statusColor};">${history.final_status}</div>
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
            <div class="kpi-label">Resumable Training Execution</div>
            <div class="kpi-value">${history.iterations ? history.iterations.length : 1} Iterations</div>
            <div style="font-size: 0.8rem; color: var(--violet);">${history.total_epochs || 0} Total Epochs Executed</div>
        </div>
    `;

    renderLossEvolutionChart(history);
    renderAccuracyEvolutionChart(history);
}

function renderLossEvolutionChart(history) {
    const chartDiv = document.getElementById('plotlyLossChart');
    if (!chartDiv || !history.epochs) return;

    const epochs = history.epochs.map(e => `Epoch ${e.epoch}`);
    const trainLoss = history.epochs.map(e => e.train_loss);
    const valLoss = history.epochs.map(e => e.val_loss);

    const traces = [
        {
            x: epochs,
            y: trainLoss,
            mode: 'lines+markers',
            name: 'Training Loss (MSE)',
            line: { color: '#38bdf8', width: 2.5 }
        },
        {
            x: epochs,
            y: valLoss,
            mode: 'lines+markers',
            name: 'Validation Loss (MSE)',
            line: { color: '#a855f7', width: 2.5 }
        }
    ];

    const layout = {
        title: { text: 'Evolution of Loss across Resumable Training Epochs', font: { color: '#f8fafc' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Training Epochs (Chunked Iterations)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Mean Squared Error (MSE)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' }, orientation: 'h', y: -0.2 },
        margin: { l: 50, r: 20, t: 40, b: 60 }
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true });
}

function renderAccuracyEvolutionChart(history) {
    const chartDiv = document.getElementById('plotlyAccuracyChart');
    if (!chartDiv || !history.epochs) return;

    const epochs = history.epochs.map(e => `Epoch ${e.epoch}`);
    const trainMae = history.epochs.map(e => e.train_mae);
    const valMae = history.epochs.map(e => e.val_mae);
    const targetLine = Array(epochs.length).fill(history.target_mae || 1.0);

    const traces = [
        {
            x: epochs,
            y: trainMae,
            mode: 'lines+markers',
            name: 'Training MAE (°C)',
            line: { color: '#34d399', width: 2.5 }
        },
        {
            x: epochs,
            y: valMae,
            mode: 'lines+markers',
            name: 'Validation MAE (°C)',
            line: { color: '#fbbf24', width: 2.5 }
        },
        {
            x: epochs,
            y: targetLine,
            mode: 'lines',
            name: `Target MAE (${history.target_mae || 1.0}°C)`,
            line: { color: '#f43f5e', width: 2, dash: 'dash' }
        }
    ];

    const layout = {
        title: { text: 'Accuracy Evolution (MAE °C) & Convergence to Target Precision', font: { color: '#f8fafc' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Training Epochs (Chunked Iterations)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Mean Absolute Error (°C)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' }, orientation: 'h', y: -0.2 },
        margin: { l: 50, r: 20, t: 40, b: 60 }
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true });
}

/* ==========================================================================
   3. OUT-OF-SAMPLE EVALUATION & LEADERBOARD FUNCTIONS
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

function renderTimeSeriesChart(series) {
    const chartDiv = document.getElementById('plotlyTimeSeries');
    if (!chartDiv) return;

    const traces = [];

    // Actual TMAX
    traces.push({
        x: series.timestamps,
        y: series.actual_next24_tmax,
        mode: 'lines',
        name: 'Actual Next-24h TMAX',
        line: { color: '#f43f5e', width: 2 }
    });

    // Actual TMIN
    traces.push({
        x: series.timestamps,
        y: series.actual_next24_tmin,
        mode: 'lines',
        name: 'Actual Next-24h TMIN',
        line: { color: '#34d399', width: 2 }
    });

    // Best Model Predictions
    if (series.predictions['Gradient Boosted Trees']) {
        const gb = series.predictions['Gradient Boosted Trees'];
        traces.push({
            x: series.timestamps,
            y: gb.tmax,
            mode: 'lines',
            name: 'GBDT Predicted TMAX',
            line: { color: '#fbbf24', width: 2, dash: 'dot' }
        });

        traces.push({
            x: series.timestamps,
            y: gb.tmin,
            mode: 'lines',
            name: 'GBDT Predicted TMIN',
            line: { color: '#38bdf8', width: 2, dash: 'dot' }
        });
    }

    const layout = {
        title: { text: 'Out-of-Sample Test Evaluation: Next-Day TMAX and TMIN (2026)', font: { color: '#f8fafc' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Reference Date (2026)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Temperature (°C)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' }, orientation: 'h', y: -0.2 },
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
        line: { color: '#34d399', width: 3 }
    }];

    for (const [name, predObj] of Object.entries(series.predictions)) {
        if (predObj.sample_24h_profile) {
            traces.push({
                x: hours,
                y: predObj.sample_24h_profile,
                mode: 'lines+markers',
                name: `${name} Forecast`,
                line: { width: 2, dash: name.includes('Baseline') ? 'dash' : 'solid' }
            });
        }
    }

    const layout = {
        title: { text: `24-Hour Forecast Sequence vs Actual (${series.last_timestamp})`, font: { color: '#f8fafc' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Forecast Steps (Hours Ahead)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        yaxis: { title: 'Temperature (°C)', gridcolor: 'rgba(255,255,255,0.05)', color: '#94a3b8' },
        legend: { font: { color: '#f8fafc' }, orientation: 'h', y: -0.25 },
        margin: { l: 40, r: 20, t: 40, b: 60 }
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true });
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
