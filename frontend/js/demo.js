/**
 * demo.js
 * Live Biometric Age Prediction Controller with Clear High-Contrast Face Targeting Frame
 * Features:
 *  - Ergonomic, spacious Black Border Face Detection Box.
 *  - Single-shot face capture on detection (freezes on result, no endless recalculations).
 *  - Instant "Scan Again" & "Capture Snapshot" buttons for new scans with fresh live camera frames.
 */

// Local Backend API Endpoint
const BACKEND_API_URL = "http://localhost:8000/api/predict";

let webcamStream = null;
let isWebcamRunning = false;
let isInferring = false;
let isScanCompleted = false;
let blazefaceModel = null;
let reArmTimestamp = 0;
let auditLogs = [];

document.addEventListener("DOMContentLoaded", async () => {
  await initFaceDetectorModel();
  startWebcam();
  checkBackendHealth();
});

/**
 * 1. Initialize BlazeFace / YOLO Edge Face Detector
 */
async function initFaceDetectorModel() {
  const fpsBadge = document.getElementById("cam-fps-badge");
  if (fpsBadge) fpsBadge.textContent = "AI Detector: Loading...";

  try {
    if (typeof blazeface !== "undefined") {
      blazefaceModel = await blazeface.load();
      console.log("[+] BlazeFace Neural Face Detector Loaded!");
      if (fpsBadge) fpsBadge.textContent = "AI Face Lock: READY";
    }
  } catch (err) {
    console.warn("BlazeFace initialization note:", err);
    if (fpsBadge) fpsBadge.textContent = "Optical Tracker";
  }
}

/**
 * 2. Mode Switcher (Webcam vs Image Upload)
 */
window.switchInputMode = function (mode) {
  const btnCamera = document.getElementById("btn-mode-camera");
  const btnUpload = document.getElementById("btn-mode-upload");
  const panelWebcam = document.getElementById("panel-webcam");
  const panelUpload = document.getElementById("panel-upload");

  if (mode === "camera") {
    btnCamera.classList.add("active");
    btnUpload.classList.remove("active");
    panelWebcam.style.display = "block";
    panelUpload.style.display = "none";
    if (!isWebcamRunning) startWebcam();
  } else {
    btnCamera.classList.remove("active");
    btnUpload.classList.add("active");
    panelWebcam.style.display = "none";
    panelUpload.style.display = "block";
    pauseWebcam();
  }
};

/**
 * 3. Webcamera Stream Lifecycle
 */
window.startWebcam = async function () {
  const video = document.getElementById("webcam-video");
  const placeholder = document.getElementById("camera-placeholder");
  const laser = document.getElementById("scanner-laser");

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });
    video.srcObject = webcamStream;
    await video.play();

    isWebcamRunning = true;
    isScanCompleted = false;
    reArmTimestamp = performance.now() + 1000;
    placeholder.style.display = "none";
    laser.style.display = "block";
    document.getElementById("btn-toggle-cam").textContent = "Pause Camera";

    // Launch continuous real-time face tracking loop
    startFaceTrackingLoop();
  } catch (err) {
    console.warn("Webcam access error:", err);
    placeholder.style.display = "flex";
    placeholder.querySelector("p").textContent = "Webcam access denied or camera not found.";
  }
};

window.pauseWebcam = function () {
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  isWebcamRunning = false;
  document.getElementById("scanner-laser").style.display = "none";
  document.getElementById("camera-placeholder").style.display = "flex";
  document.getElementById("btn-toggle-cam").textContent = "Resume Camera";
  clearOverlay();
};

window.toggleWebcam = function () {
  if (isWebcamRunning) {
    pauseWebcam();
  } else {
    startWebcam();
  }
};

/**
 * Reset & Scan Again (Re-arms detector for fresh live frame)
 */
window.resetAndScanAgain = function () {
  isScanCompleted = false;
  isInferring = false;
  reArmTimestamp = performance.now() + 800; // 800ms re-arm delay to capture a fresh frame

  const scanAgainBtn = document.getElementById("btn-scan-again");
  if (scanAgainBtn) scanAgainBtn.style.display = "none";

  updateStatusBadge("SCANNING FOR NEW FACE...");
  document.getElementById("out-cohort-badge").textContent = "Align Face Inside Black Frame...";
};

/**
 * 4. High-Contrast Black Border Face Tracking & Auto-Capture
 */
function startFaceTrackingLoop() {
  const video = document.getElementById("webcam-video");
  const overlay = document.getElementById("overlay-canvas");
  if (!overlay) return;
  const ctx = overlay.getContext("2d");

  async function renderFrame() {
    if (isWebcamRunning && video.readyState >= 2) {
      overlay.width = video.videoWidth || 640;
      overlay.height = video.videoHeight || 480;
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      // Ergonomic, spacious Face Target Frame (Black border, easy to fit face)
      const frameW = overlay.width * 0.65;
      const frameH = overlay.height * 0.72;
      const frameX = (overlay.width - frameW) / 2;
      const frameY = (overlay.height - frameH) / 2;

      let detectedFace = null;

      // 1. Detect Face Coordinates with high sensitivity
      if (blazefaceModel) {
        try {
          const preds = await blazefaceModel.estimateFaces(video, false, 0.65);
          if (preds.length > 0) {
            const p = preds[0];
            const start = p.topLeft;
            const end = p.bottomRight;
            const w = end[0] - start[0];
            const h = end[1] - start[1];
            detectedFace = {
              x: start[0],
              y: start[1],
              width: w,
              height: h,
              landmarks: p.landmarks || [],
              confidence: p.probability ? (p.probability[0] * 100).toFixed(0) : 98
            };
          }
        } catch (e) {
          detectedFace = null;
        }
      }

      // 2. Draw Solid Black Detection Border & Corners
      ctx.lineWidth = 4;
      ctx.strokeStyle = "#000000"; // Solid Black Border

      // Outer Black Guide Box
      drawRoundedRect(ctx, frameX, frameY, frameW, frameH, 16);
      ctx.stroke();

      // Inner subtle contrast line so border is visible on dark clothes
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = detectedFace ? "#10B981" : "rgba(245, 185, 66, 0.7)";
      drawRoundedRect(ctx, frameX + 2, frameY + 2, frameW - 4, frameH - 4, 14);
      ctx.stroke();

      // Draw Heavy Solid Black Corner HUD Brackets
      drawHeavyCornerBrackets(ctx, frameX, frameY, frameW, frameH, 32);

      // 3. Render Status Header on the Box
      if (detectedFace) {
        // Black Background Badge with Bright Text
        ctx.fillStyle = "#000000";
        ctx.fillRect(frameX + 10, frameY - 24, 170, 22);
        ctx.strokeStyle = "#10B981";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(frameX + 10, frameY - 24, 170, 22);

        ctx.fillStyle = "#10B981";
        ctx.font = "bold 11px JetBrains Mono, monospace";
        ctx.fillText(`✓ FACE LOCKED (${detectedFace.confidence}%)`, frameX + 18, frameY - 9);

        // Draw facial landmarks if available
        if (detectedFace.landmarks) {
          ctx.fillStyle = "#F5B942";
          detectedFace.landmarks.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], 2.5, 0, 2 * Math.PI);
            ctx.fill();
          });
        }

        // 4. Automatic One-Shot Capture Trigger
        const now = performance.now();
        if (!isScanCompleted && !isInferring && now > reArmTimestamp) {
          isScanCompleted = true; // Freeze trigger after 1 capture
          // Auto-trigger fresh prediction from video
          captureAndPredictAuto(video);
        }
      } else {
        // Guide text when no face is locked
        ctx.fillStyle = "#000000";
        ctx.fillRect(overlay.width / 2 - 110, frameY - 24, 220, 22);
        ctx.strokeStyle = "#F5B942";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(overlay.width / 2 - 110, frameY - 24, 220, 22);

        ctx.fillStyle = "#F5B942";
        ctx.font = "bold 11px JetBrains Mono, monospace";
        ctx.textAlign = "center";
        ctx.fillText("ALIGN FACE IN BLACK BOX", overlay.width / 2, frameY - 9);
        ctx.textAlign = "left";

        // Auto-capture fallback after 2.5s of camera startup if detector didn't lock
        const now = performance.now();
        if (!isScanCompleted && !isInferring && now > (reArmTimestamp + 2000)) {
          isScanCompleted = true;
          captureAndPredictAuto(video);
        }
      }
    }
    requestAnimationFrame(renderFrame);
  }
  renderFrame();
}

function drawRoundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawHeavyCornerBrackets(ctx, x, y, w, h, len) {
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 6;
  // Top-Left
  ctx.beginPath(); ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y); ctx.stroke();
  // Top-Right
  ctx.beginPath(); ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len); ctx.stroke();
  // Bottom-Left
  ctx.beginPath(); ctx.moveTo(x, y + h - len); ctx.lineTo(x, y); ctx.lineTo(x + len, y + h); ctx.stroke();
  // Bottom-Right
  ctx.beginPath(); ctx.moveTo(x + w - len, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - len); ctx.stroke();
}

function clearOverlay() {
  const overlay = document.getElementById("overlay-canvas");
  if (overlay) {
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }
}

/**
 * 5. One-Shot Auto-Capture (Extracts FRESH video frame)
 */
async function captureAndPredictAuto(videoElement) {
  const canvas = document.createElement("canvas");
  canvas.width = videoElement.videoWidth || 640;
  canvas.height = videoElement.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  const freshFrameBase64 = canvas.toDataURL("image/jpeg", 0.95);
  await executeInference(freshFrameBase64, "Live Auto-Scan");
}

/**
 * 6. Manual Snapshot Trigger (Extracts FRESH video frame immediately)
 */
window.captureAndPredict = async function () {
  if (!isWebcamRunning || isInferring) return;

  const video = document.getElementById("webcam-video");
  if (video.videoWidth === 0) return;

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const freshFrameBase64 = canvas.toDataURL("image/jpeg", 0.95);
  isScanCompleted = true; // Freeze on manual capture
  await executeInference(freshFrameBase64, "Manual Snapshot");
};

/**
 * 7. Handle Image File Upload
 */
window.handleFileUpload = async function (event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async function (e) {
    const rawBase64 = e.target.result;

    // Display upload preview
    document.getElementById("dropzone").style.display = "none";
    const previewBox = document.getElementById("upload-preview-box");
    const previewImg = document.getElementById("upload-preview-img");
    previewImg.src = rawBase64;
    previewBox.style.display = "flex";

    await executeInference(rawBase64, file.name);
  };
  reader.readAsDataURL(file);
};

window.clearUpload = function () {
  document.getElementById("file-input").value = "";
  document.getElementById("upload-preview-box").style.display = "none";
  document.getElementById("dropzone").style.display = "flex";
};

/**
 * 8. Quick Demographic Presets
 */
window.loadSampleFace = async function (cohort) {
  switchInputMode("upload");

  const samples = {
    "child": { age: 8.4, name: "Sample Child (01–12 yrs)" },
    "teen": { age: 17.2, name: "Sample Teen (13–19 yrs)" },
    "young-adult": { age: 26.5, name: "Sample Young Adult (20–35 yrs)" },
    "adult": { age: 42.1, name: "Sample Adult (36–45 yrs)" },
    "senior": { age: 68.7, name: "Sample Senior (61–75 yrs)" },
  };

  const sample = samples[cohort] || samples["young-adult"];

  const canvas = document.createElement("canvas");
  canvas.width = 320;
  canvas.height = 320;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#15181D";
  ctx.fillRect(0, 0, 320, 320);
  ctx.fillStyle = "#F5B942";
  ctx.font = "bold 15px Inter";
  ctx.textAlign = "center";
  ctx.fillText("Demographic Test Sample", 160, 150);
  ctx.fillStyle = "#9DA8B7";
  ctx.font = "12px Inter";
  ctx.fillText(sample.name, 160, 175);

  const base64Data = canvas.toDataURL("image/jpeg");
  document.getElementById("dropzone").style.display = "none";
  const previewBox = document.getElementById("upload-preview-box");
  const previewImg = document.getElementById("upload-preview-img");
  previewImg.src = base64Data;
  previewBox.style.display = "flex";

  await executeInference(base64Data, sample.name);
};

/**
 * 9. Send Image to CUDA Backend API
 */
async function executeInference(imageData, sourceName = "Face Input") {
  isInferring = true;
  updateStatusBadge("PROCESSING FACE (CUDA)...");

  let result = null;
  let isNoFaceFound = false;

  try {
    const response = await fetch(BACKEND_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData, tta: true }),
    });

    if (response.ok) {
      result = await response.json();
    } else if (response.status === 422) {
      isNoFaceFound = true;
      result = await response.json();
    } else {
      console.warn("Backend error status:", response.status);
    }
  } catch (apiErr) {
    console.error("Backend connection error:", apiErr);
  }

  if (isNoFaceFound || (result && result.status === "NO_FACE_DETECTED")) {
    updateStatusBadge("NO HUMAN FACE DETECTED");
    document.getElementById("out-age-value").textContent = "--.-";
    document.getElementById("out-cohort-badge").textContent = "⚠️ No Human Face Detected in Image";
    document.getElementById("out-model-a").innerHTML = `--.- <small>yrs</small>`;
    document.getElementById("out-model-b").innerHTML = `--.- <small>yrs</small>`;
    
    // Show thumbnail of rejected image
    const thumbImg = document.getElementById("out-face-thumb");
    if (thumbImg && result && result.face_thumbnail) {
      thumbImg.src = result.face_thumbnail;
    }
  } else if (result && result.predicted_age !== undefined) {
    renderPredictionResult(result, sourceName);
    const scanAgainBtn = document.getElementById("btn-scan-again");
    if (scanAgainBtn) {
      scanAgainBtn.style.display = "inline-flex";
    }
  } else {
    updateStatusBadge("BACKEND OFFLINE");
    document.getElementById("out-cohort-badge").textContent = "⚠️ Backend offline on port 8000";
  }

  isInferring = false;
}

/**
 * 10. Render Prediction in UI
 */
function renderPredictionResult(res, sourceName) {
  const age = res.predicted_age;
  const cohort = getAgeCohort(age);

  // 1. Output Age & Cohort
  document.getElementById("out-age-value").textContent = age.toFixed(1);
  document.getElementById("out-cohort-badge").textContent = res.age_group || cohort.label;

  // Render Detected Cropped Face Thumbnail
  const thumbImg = document.getElementById("out-face-thumb");
  if (thumbImg && res.face_thumbnail) {
    thumbImg.src = res.face_thumbnail;
  }

  // 2. Sub-Models Breakdown
  document.getElementById("out-model-a").innerHTML = `${res.model_a_dex.toFixed(1)} <small>yrs</small>`;
  document.getElementById("out-model-b").innerHTML = `${res.model_b_hybrid.toFixed(1)} <small>yrs</small>`;

  // 3. Tolerance Bounds
  document.getElementById("out-bound-3").textContent = `${Math.max(1, (age - 3)).toFixed(1)} – ${(age + 3).toFixed(1)} yrs`;
  document.getElementById("out-bound-5").textContent = `${Math.max(1, (age - 5)).toFixed(1)} – ${(age + 5).toFixed(1)} yrs`;
  document.getElementById("out-bound-7").textContent = `${Math.max(1, (age - 7)).toFixed(1)} – ${(age + 7).toFixed(1)} yrs`;

  // 4. Telemetry
  document.getElementById("out-latency").textContent = `${res.latency_ms || 28} ms`;
  document.getElementById("out-disagreement").textContent = `±${res.disagreement || 0.4} yrs`;
  updateStatusBadge("AGE PREDICTED (STABLE)");

  // 5. Log Audit Entry
  logAuditEntry({
    timestamp: new Date().toLocaleTimeString(),
    source: sourceName,
    age: age.toFixed(1),
    cohort: res.age_group || cohort.label
  });
}

function getAgeCohort(age) {
  if (age <= 12) return { label: "👶 Child (01–12 yrs)", color: "#10B981" };
  if (age <= 19) return { label: "🧑 Teen (13–19 yrs)", color: "#60A5FA" };
  if (age <= 35) return { label: "👨 Young Adult (20–35 yrs)", color: "#F5B942" };
  if (age <= 45) return { label: "👩 Adult (36–45 yrs)", color: "#F5B942" };
  if (age <= 60) return { label: "👨 Middle Age (46–60 yrs)", color: "#9DA8B7" };
  if (age <= 75) return { label: "👴 Senior (61–75 yrs)", color: "#F59E0B" };
  return { label: "👵 Elderly (76–100 yrs)", color: "#EF4444" };
}

function updateStatusBadge(text) {
  const tag = document.getElementById("result-status-tag");
  if (tag) tag.textContent = text;
}

function logAuditEntry(entry) {
  auditLogs.unshift(entry);
  if (auditLogs.length > 6) auditLogs.pop();

  const container = document.getElementById("audit-logs-container");
  if (!container) return;

  container.innerHTML = auditLogs.map(log => `
    <div class="audit-log-item">
      <div class="audit-left">
        <span class="audit-time">${log.timestamp}</span>
        <span class="audit-src">${log.source}</span>
      </div>
      <div class="audit-right">
        <strong style="color: var(--accent-primary); font-family: var(--font-family-mono); font-size: 14px;">${log.age} yrs</strong>
        <span style="color: var(--text-secondary); font-size: 11px;">${log.cohort}</span>
      </div>
    </div>
  `).join('');
}

window.clearLogs = function () {
  auditLogs = [];
  document.getElementById("audit-logs-container").innerHTML = `
    <div class="audit-log-empty">Session audit logs cleared.</div>
  `;
};

async function checkBackendHealth() {
  try {
    const res = await fetch("http://localhost:8000/api/health");
    if (res.ok) {
      document.getElementById("backend-status-text").textContent = "DUAL ENSEMBLE CUDA ACTIVE";
    }
  } catch (e) {
    document.getElementById("backend-status-text").textContent = "BACKEND OFFLINE (PORT 8000)";
  }
}
