// ==========================================================================
// Cognizant AgeVision AI™ - Frontend Logic & Chart.js Integration v3.0
// ==========================================================================

let currentTab = 'review';
let currentUseCase = 'KYC_Verification';
let currentInputMode = 'upload';
let webcamStream = null;
let lossChart = null;
let ageGroupChart = null;

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    fetchReviewMetrics();
    fetchAuditLogs();
    setupDragAndDrop();
});

// Tab Switcher
function switchTab(tabName) {
    currentTab = tabName;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-view').forEach(view => view.classList.remove('active'));
    
    if (tabName === 'review') {
        document.getElementById('tab-review-btn').classList.add('active');
        document.getElementById('tab-review').classList.add('active');
    } else {
        document.getElementById('tab-scanner-btn').classList.add('active');
        document.getElementById('tab-scanner').classList.add('active');
        fetchAuditLogs();
    }
}

// Enterprise Use Case Selector
function selectUseCase(useCase) {
    currentUseCase = useCase;
    document.querySelectorAll('.radio-card').forEach(card => card.classList.remove('active'));
    const targetRadio = document.querySelector(`input[value="${useCase}"]`);
    if (targetRadio) {
        targetRadio.checked = true;
        targetRadio.closest('.radio-card').classList.add('active');
    }
}

// Input Mode (Upload vs Webcam)
function setInputMode(mode) {
    currentInputMode = mode;
    document.getElementById('btn-mode-upload').classList.toggle('active', mode === 'upload');
    document.getElementById('btn-mode-webcam').classList.toggle('active', mode === 'webcam');
    
    const dropZone = document.getElementById('drop-zone');
    const webcamBox = document.getElementById('webcam-box');
    
    if (mode === 'upload') {
        dropZone.style.display = 'block';
        webcamBox.style.display = 'none';
        stopWebcam();
    } else {
        dropZone.style.display = 'none';
        webcamBox.style.display = 'flex';
        startWebcam();
    }
}

// Webcam Stream
async function startWebcam() {
    try {
        const video = document.getElementById('webcam-video');
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = webcamStream;
    } catch (err) {
        alert("Could not access webcam: " + err.message);
        setInputMode('upload');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
}

function captureWebcam() {
    const video = document.getElementById('webcam-video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(blob => {
        const file = new File([blob], "webcam_snapshot.png", { type: "image/png" });
        sendPredictionRequest(file);
    }, 'image/png');
}

// Drag & Drop Setup
function setupDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });
    
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            sendPredictionRequest(files[0]);
        }
    });
}

function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        sendPredictionRequest(files[0]);
    }
}



// Send Prediction API Call with Strict Face Validation
async function sendPredictionRequest(file) {
    const preview = document.getElementById('image-preview');
    const placeholder = document.getElementById('result-placeholder');
    const activeContent = document.getElementById('result-active-content');
    
    // Clear previous bounding box & errors
    const faceBox = document.getElementById('face-box');
    faceBox.style.display = 'none';
    
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    placeholder.style.display = 'none';
    activeContent.style.display = 'block';
    
    // Set loading state
    document.getElementById('res-age').textContent = "...";
    document.getElementById('res-category').textContent = "Scanning Biometrics...";
    document.getElementById('res-confidence').textContent = "Running Tri-Model...";
    document.getElementById('res-latency').textContent = "Detecting Face...";
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('use_case', currentUseCase);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Handle No Face Error Gracefully
        if (data.status === "ERROR_NO_FACE") {
            showNoFaceError(data.message);
            return;
        }
        
        renderPredictionOutcome(data);
        fetchAuditLogs();
        
    } catch (err) {
        showNoFaceError("Prediction failed: " + err.message);
    }
}

// Show No Face Error Card
function showNoFaceError(msg) {
    document.getElementById('res-age').textContent = "N/A";
    document.getElementById('res-category').textContent = "Validation: Failed";
    document.getElementById('res-confidence').textContent = "No Human Face";
    document.getElementById('res-latency').textContent = "0 ms";
    
    document.getElementById('res-m1').textContent = "N/A";
    document.getElementById('res-m2').textContent = "N/A";
    document.getElementById('res-m3').textContent = "N/A";
    
    const insightBox = document.getElementById('insight-body');
    insightBox.innerHTML = `<span class="text-danger"><i class="fa-solid fa-triangle-exclamation"></i> <strong>Face Detection Error:</strong> ${msg}</span>`;
    
    document.getElementById('face-box').style.display = 'none';
}

// Render Prediction Result
function renderPredictionOutcome(data) {
    document.getElementById('res-age').textContent = data.predicted_age.toFixed(1);
    document.getElementById('res-category').textContent = `Category: ${data.age_category}`;
    document.getElementById('res-confidence').textContent = `Confidence: ${data.confidence_range}`;
    document.getElementById('res-latency').textContent = `Latency: ${data.latency_ms} ms`;
    
    // Consensus
    document.getElementById('res-m1').textContent = `${data.models_breakdown.model_1_dex.toFixed(1)} yrs`;
    document.getElementById('res-m2').textContent = `${data.models_breakdown.model_2_hybrid.toFixed(1)} yrs`;
    document.getElementById('res-m3').textContent = `${data.models_breakdown.model_3_convnext.toFixed(1)} yrs`;
    
    // Face Bounding Box
    const preview = document.getElementById('image-preview');
    const faceBox = document.getElementById('face-box');
    if (data.bounding_box && preview.naturalWidth > 0) {
        const scaleX = preview.clientWidth / preview.naturalWidth;
        const scaleY = preview.clientHeight / preview.naturalHeight;
        
        faceBox.style.left = `${data.bounding_box.x * scaleX}px`;
        faceBox.style.top = `${data.bounding_box.y * scaleY}px`;
        faceBox.style.width = `${data.bounding_box.width * scaleX}px`;
        faceBox.style.height = `${data.bounding_box.height * scaleY}px`;
        faceBox.style.display = 'block';
    } else {
        faceBox.style.display = 'none';
    }
    
    // Insights
    const insightBox = document.getElementById('insight-body');
    const insight = data.client_insights;
    if (insight.status) {
        insightBox.innerHTML = `<strong>Status:</strong> <span class="badge badge-info">${insight.status}</span> <br> <strong>Recommended Action:</strong> ${insight.action || insight.ad_theme || insight.department}`;
    }
}

// Initialize Charts
function initCharts() {
    // 1. Loss & MAE Chart
    const ctx1 = document.getElementById('lossChart').getContext('2d');
    lossChart = new Chart(ctx1, {
        type: 'line',
        data: {
            labels: ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5'],
            datasets: [
                {
                    label: 'Training MAE (Years)',
                    data: [7.84, 5.30, 4.61, 3.99, 3.62],
                    borderColor: '#8B5CF6',
                    backgroundColor: 'rgba(139, 92, 246, 0.12)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Validation MAE (Years)',
                    data: [5.73, 4.90, 4.82, 4.69, 4.64],
                    borderColor: '#00E5FF',
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94A3B8', font: { family: 'Inter', size: 11 } } }
            },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94A3B8' } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94A3B8' }, min: 3.0, max: 8.5 }
            }
        }
    });

    // 2. Age Group Chart
    const ctx2 = document.getElementById('ageGroupChart').getContext('2d');
    ageGroupChart = new Chart(ctx2, {
        type: 'bar',
        data: {
            labels: ['01-12 (Child)', '20-30 (Young)', '31-45 (Adult)', '46-60 (Mid)', '61-75 (Senior)', '76-100 (Elder)'],
            datasets: [
                {
                    label: 'MAE Error (Years)',
                    data: [2.63, 3.74, 3.98, 5.13, 5.97, 8.59],
                    backgroundColor: [
                        '#10B981',
                        '#00E5FF',
                        '#0072CE',
                        '#F59E0B',
                        '#8B5CF6',
                        '#EF4444'
                    ],
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94A3B8', font: { size: 10 } } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94A3B8' }, min: 0, max: 10 }
            }
        }
    });
}

// Fetch Official Benchmark Review Metrics
async function fetchReviewMetrics() {
    try {
        const res = await fetch('/api/review-metrics');
        if (res.ok) {
            const data = await res.json();
            document.getElementById('kpi-mae').innerHTML = `${data.champion_metrics.overall_mae.toFixed(2)} <span class="unit">Yrs</span>`;
            document.getElementById('kpi-acc7').textContent = `${data.champion_metrics.accuracy_pm_7.toFixed(1)}%`;
            document.getElementById('kpi-acc10').textContent = `${data.champion_metrics.accuracy_pm_10.toFixed(1)}%`;
            document.getElementById('kpi-core-mae').innerHTML = `${data.champion_metrics.core_population_mae.toFixed(2)} <span class="unit">Yrs</span>`;
        }
    } catch (e) {
        console.log("Using initial metrics cache");
    }
}

// Fetch Audit Logs from MySQL / SQLite
async function fetchAuditLogs() {
    try {
        const res = await fetch('/api/audit-logs');
        if (!res.ok) return;
        const logs = await res.json();
        const tbody = document.getElementById('audit-table-body');
        
        if (logs.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No biometric scans recorded yet.</td></tr>`;
            return;
        }
        
        tbody.innerHTML = logs.map(l => `
            <tr>
                <td><strong>${l.scan_id}</strong></td>
                <td>${l.timestamp}</td>
                <td>
                    ${l.image_base64 ? `<img src="${l.image_base64}" style="width:36px;height:36px;border-radius:6px;object-fit:cover;border:1px solid rgba(0,229,255,0.3);">` : `<i class="fa-solid fa-image text-muted"></i>`}
                </td>
                <td><strong class="text-accent">${l.predicted_age.toFixed(1)} yrs</strong></td>
                <td><span class="badge badge-info">${l.age_category}</span></td>
                <td>${l.client_use_case}</td>
                <td><code>${l.latency_ms} ms</code></td>
            </tr>
        `).join('');
    } catch (e) {
        console.log("Audit log fetch error:", e);
    }
}
