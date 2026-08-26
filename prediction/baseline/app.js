const canvas = document.getElementById('histogramCanvas');
const ctx = canvas.getContext('2d');
const timeSlider = document.getElementById('timeSlider');
const playPauseBtn = document.getElementById('playPauseBtn');
const playIcon = document.getElementById('playIcon');
const pauseIcon = document.getElementById('pauseIcon');
const currentDateDisplay = document.getElementById('currentDateDisplay');
const currentHourInfo = document.getElementById('currentHourInfo');
const maxCountLabel = document.getElementById('maxCountLabel');
const accumulateCheckbox = document.getElementById('accumulateCheckbox');
const speedSelect = document.getElementById('speedSelect');

let histogramData = null;
let currentHour = 0;
let isPlaying = false;
let animationId = null;
let lastFrameTime = 0;

// To store accumulated grids
let accumulatedGrid = null;
let maxCount = 1;

// Color gradient mapping
// Returns an rgb string based on a normalized value 0.0 to 1.0
function getColor(val) {
    // 0 -> Blue (240)
    // 0.5 -> Green (120)
    // 1 -> Red (0)
    // This is a simple HSV-like mapping but using HSL
    if (val === 0) return 'transparent';
    const h = (1.0 - val) * 240; 
    return `hsl(${h}, 100%, 50%)`;
}

// Draw the gradient bar
function setupGradientBar() {
    const bar = document.querySelector('.gradient-bar');
    bar.style.background = 'linear-gradient(to right, hsl(240, 100%, 50%), hsl(120, 100%, 50%), hsl(0, 100%, 50%))';
}

function formatDateFromHour(hourOfYear) {
    // Treat as non-leap year 1999 for simplicity
    const d = new Date(1999, 0, 1);
    d.setHours(hourOfYear);
    const options = { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' };
    return d.toLocaleString('en-US', options).replace(',', '');
}

function initData(data) {
    histogramData = data;
    
    // Set labels
    document.getElementById('xMinLabel').textContent = data.tmp_min;
    document.getElementById('xMaxLabel').textContent = data.tmp_max;
    document.getElementById('yMinLabel').textContent = data.dew_min;
    document.getElementById('yMaxLabel').textContent = data.dew_max;
    
    // Initialize accumulated grid
    accumulatedGrid = new Int32Array(data.grid_size * data.grid_size);
    
    // Calculate global max count for a single hour to scale colors nicely
    let globalMax = 0;
    for (let frame of data.histograms) {
        for (let pt of frame) {
            if (pt[2] > globalMax) globalMax = pt[2];
        }
    }
    // Give a slightly lower scale max to make colors pop more
    maxCount = Math.max(1, globalMax);
    maxCountLabel.textContent = maxCount;
    
    setupGradientBar();
    renderFrame(currentHour);
}

function computeAccumulatedGrid(upToHour) {
    accumulatedGrid.fill(0);
    let accMax = 0;
    for (let h = 0; h <= upToHour; h++) {
        const frame = histogramData.histograms[h];
        for (let pt of frame) {
            const idx = pt[1] * histogramData.grid_size + pt[0];
            accumulatedGrid[idx] += pt[2];
            if (accumulatedGrid[idx] > accMax) accMax = accumulatedGrid[idx];
        }
    }
    return accMax;
}

function renderFrame(hour) {
    if (!histogramData) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const gs = histogramData.grid_size;
    const cellW = canvas.width / gs;
    const cellH = canvas.height / gs;
    
    let currentMax = maxCount;
    
    if (accumulateCheckbox.checked) {
        currentMax = computeAccumulatedGrid(hour);
        maxCountLabel.textContent = currentMax;
        
        for (let y = 0; y < gs; y++) {
            for (let x = 0; x < gs; x++) {
                const count = accumulatedGrid[y * gs + x];
                if (count > 0) {
                    const norm = Math.min(1.0, count / currentMax);
                    ctx.fillStyle = getColor(norm);
                    // y-axis is inverted (0 is bottom)
                    ctx.fillRect(x * cellW, canvas.height - (y + 1) * cellH, cellW, cellH);
                }
            }
        }
    } else {
        maxCountLabel.textContent = maxCount;
        const frame = histogramData.histograms[hour];
        for (let pt of frame) {
            const x = pt[0];
            const y = pt[1];
            const count = pt[2];
            const norm = Math.min(1.0, count / maxCount);
            ctx.fillStyle = getColor(norm);
            ctx.fillRect(x * cellW, canvas.height - (y + 1) * cellH, cellW, cellH);
        }
    }
    
    currentDateDisplay.textContent = formatDateFromHour(hour);
    currentHourInfo.textContent = `Hour ${hour} / 8759`;
    timeSlider.value = hour;
}

function togglePlay() {
    isPlaying = !isPlaying;
    if (isPlaying) {
        playIcon.style.display = 'none';
        pauseIcon.style.display = 'block';
        lastFrameTime = performance.now();
        animationId = requestAnimationFrame(animate);
    } else {
        playIcon.style.display = 'block';
        pauseIcon.style.display = 'none';
        cancelAnimationFrame(animationId);
    }
}

function animate(time) {
    if (!isPlaying) return;
    
    const speedMs = parseInt(speedSelect.value, 10);
    const dt = time - lastFrameTime;
    
    if (dt >= speedMs) {
        currentHour++;
        if (currentHour > 8759) {
            currentHour = 0;
            if (accumulateCheckbox.checked) {
                togglePlay(); // stop at end if accumulating
                return;
            }
        }
        renderFrame(currentHour);
        lastFrameTime = time;
    }
    
    animationId = requestAnimationFrame(animate);
}

// Event Listeners
playPauseBtn.addEventListener('click', togglePlay);

timeSlider.addEventListener('input', (e) => {
    currentHour = parseInt(e.target.value, 10);
    renderFrame(currentHour);
    if (isPlaying) togglePlay(); // Pause on scrub
});

accumulateCheckbox.addEventListener('change', () => {
    renderFrame(currentHour);
});

// Load data
fetch('histogram_data.json')
    .then(res => res.json())
    .then(data => initData(data))
    .catch(err => {
        console.error("Failed to load histogram_data.json", err);
        ctx.fillStyle = "white";
        ctx.fillText("Failed to load data.", 20, 20);
    });
