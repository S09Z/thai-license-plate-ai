"use strict";

// A plain client of POST /recognize. Recognized text is untrusted input: it
// only ever reaches the DOM through textContent, never innerHTML, and is never
// evaluated. See "Never execute OCR output" in CLAUDE.md.

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const submitButton = document.getElementById("submit-button");
const statusLine = document.getElementById("status");
const previewPanel = document.getElementById("preview-panel");
const canvas = document.getElementById("preview");
const results = document.getElementById("results");
const resultsBody = document.getElementById("results-body");
const cameraFeed = document.getElementById("camera-feed");
const cameraStage = document.getElementById("camera-stage");
const trackingOverlay = document.getElementById("tracking-overlay");
const startCameraButton = document.getElementById("start-camera-button");
const stopCameraButton = document.getElementById("stop-camera-button");
const uploadModeButton = document.getElementById("upload-mode-button");
const cameraModeButton = document.getElementById("camera-mode-button");
const cameraPanel = document.getElementById("camera-panel");
const faceModeSelect = document.getElementById("face-mode");

const BOX_COLOUR = "#e0245e";
// Faces are a different kind of thing from plates, so they get a different
// colour rather than a different line style — colour survives a glance.
const FACE_BOX_COLOUR = "#00b8d4";
const BOX_WIDTH = 3;
const FEATURE_WIDTH = 2;

// Each group is stroked as a polyline in its own colour rather than as a cloud
// of anonymous dots, so the overlay reads as "eyes, eyebrows, nose, mouth" at a
// glance. `split` is the index where a group breaks into a second line: the
// nose is a bridge then a nostril line, and the mouth is an outer then an inner
// lip loop, so drawing either as one continuous path would zigzag across them.
const FEATURE_STROKES = [
  { group: "right_eyebrow", colour: "#ffb300", closed: false },
  { group: "left_eyebrow", colour: "#ffb300", closed: false },
  { group: "nose", colour: "#f5f5f5", closed: false, split: 4 },
  { group: "right_eye", colour: "#7cfc00", closed: true },
  { group: "left_eye", colour: "#7cfc00", closed: true },
  { group: "mouth", colour: "#ff4081", closed: true, split: 12 },
];

// A face 503 steps the control down one level rather than off, because the
// server cannot say which of the two models is missing; the next tick settles
// it. Falling straight to "off" would hide working face boxes.
const FACE_MODE_FALLBACK = { features: "boxes", boxes: "off" };
const CAPTURE_INTERVAL_MS = 1500;
// Detection alone is ~25ms (docs/benchmark/detect-v0.1.md), so boxes can be
// refreshed far more often than the ~400ms full recognize pipeline allows.
const TRACK_INTERVAL_MS = 200;

// Only 503 is reworded: the API's "Detector model is not available" is accurate
// but does not tell someone looking at a browser what to do about it. Every
// other failure already carries a usable detail string from the server.
const WEIGHTS_MISSING =
  "Detector model is not installed, so recognition cannot run yet. " +
  "Install trained plate weights at the configured detector path to enable it.";

const FACE_MODEL_MISSING = {
  boxes:
    "A face model is not installed, so facial features are unavailable. " +
    "Showing face boxes only; fetch it with `make fetch-face-landmark-model`.",
  off:
    "Face detection model is not installed, so face overlays are unavailable. " +
    "Fetch it with `make fetch-face-model` to enable them.",
};

let loadedImage = null;
let objectUrl = null;

/** Show a message, or hide the line when text is empty. */
function setStatus(text, kind) {
  statusLine.textContent = text;
  statusLine.className = `status ${kind || ""}`.trim();
  statusLine.hidden = !text;
}

/** Decode a chosen file into an Image, replacing any previous one. */
function loadImage(file) {
  return new Promise((resolve, reject) => {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
    }
    objectUrl = URL.createObjectURL(file);

    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("That file could not be displayed."));
    image.src = objectUrl;
  });
}

/**
 * Draw the image at its natural size, then stroke each box over it.
 *
 * Box coordinates from the API are source-image pixels, so drawing at natural
 * size keeps the mapping 1:1 — the canvas is scaled down by CSS, not by us.
 */
function drawScene(plates) {
  if (!loadedImage) {
    return;
  }

  canvas.width = loadedImage.naturalWidth;
  canvas.height = loadedImage.naturalHeight;

  const context = canvas.getContext("2d");
  context.drawImage(loadedImage, 0, 0);

  context.strokeStyle = BOX_COLOUR;
  context.fillStyle = BOX_COLOUR;
  context.lineWidth = BOX_WIDTH;
  context.font = `${Math.max(16, canvas.width / 40)}px sans-serif`;

  plates.forEach((plate, index) => {
    const { x1, y1, x2, y2 } = plate.box;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    // The row number, not recognized text: it ties a box to a table row, and
    // keeps Thai glyph rendering out of the canvas.
    context.fillText(String(index + 1), x1, Math.max(y1 - 6, 16));
  });

  previewPanel.hidden = false;
}

/** Append one cell to a row. */
function addCell(row, text, className) {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) {
    cell.className = className;
  }
  row.appendChild(cell);
  return cell;
}

/** Render a confidence as a percentage, or a dash when there is none. */
function formatConfidence(value) {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

/** Build the results table from the API response. */
function renderRows(plates) {
  resultsBody.replaceChildren();

  plates.forEach((plate, index) => {
    const row = document.createElement("tr");

    addCell(row, `${index + 1}.`, "index");

    // A number that failed the plate pattern is shown exactly as read and
    // marked, never cleaned up into something plausible.
    const plateCell = addCell(row, plate.plate_text || "—");
    if (!plate.is_well_formed) {
      plateCell.classList.add("unverified");
      plateCell.title = "Does not match the Thai plate pattern; shown as read.";
    }

    addCell(row, formatConfidence(plate.plate_confidence));

    // A null province means the knowledge base would not vouch for one. That
    // is unknown, not failure, and it is not a guess.
    const province = plate.province;
    const provinceCell = addCell(row, province === null ? "Unknown" : province);
    if (province === null) {
      provinceCell.classList.add("unverified");
      provinceCell.title = "No province matched confidently.";
    }

    addCell(row, formatConfidence(plate.province_confidence));
    addCell(row, plate.province_candidates.join(" · ") || "—", "raw");

    resultsBody.appendChild(row);
  });

  results.hidden = plates.length === 0;
}

/** Post the file to /recognize, raising a readable error on failure. */
async function recognize(file) {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch("/recognize", { method: "POST", body });
  if (response.ok) {
    return response.json();
  }

  if (response.status === 503) {
    const error = new Error(WEIGHTS_MISSING);
    error.status = response.status;
    throw error;
  }

  let detail = `Recognition failed (HTTP ${response.status}).`;
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      detail = payload.detail;
    }
  } catch {
    // A non-JSON error body is not worth reporting over the status code.
  }
  const error = new Error(detail);
  error.status = response.status;
  throw error;
}

/** Post the file to /detect, raising a readable error on failure.
 *
 * Deliberately parallel to recognize() above: same 503 wording, same status
 * carried on the error so a caller can tell "stop" from "retry next frame".
 */
async function detectOnly(file) {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch("/detect", { method: "POST", body });
  if (response.ok) {
    return response.json();
  }

  if (response.status === 503) {
    const error = new Error(WEIGHTS_MISSING);
    error.status = response.status;
    throw error;
  }

  let detail = `Detection failed (HTTP ${response.status}).`;
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      detail = payload.detail;
    }
  } catch {
    // A non-JSON error body is not worth reporting over the status code.
  }
  const error = new Error(detail);
  error.status = response.status;
  throw error;
}

/** Post the file to /detect/faces, raising a readable error on failure.
 *
 * Parallel to detectOnly() above, but every error it raises is tagged
 * `faces: true`: a missing face model must not be mistaken for a missing plate
 * detector, since only one of those is worth stopping the camera over.
 *
 * @param {File} file The captured frame.
 * @param {boolean} landmarks Whether to ask for feature points as well.
 */
async function detectFaces(file, landmarks) {
  const body = new FormData();
  body.append("file", file);

  const url = landmarks ? "/detect/faces?landmarks=true" : "/detect/faces";
  const response = await fetch(url, { method: "POST", body });
  if (response.ok) {
    return response.json();
  }

  if (response.status === 503) {
    const error = new Error("A face model is not available.");
    error.status = response.status;
    error.faces = true;
    throw error;
  }

  let detail = `Face detection failed (HTTP ${response.status}).`;
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      detail = payload.detail;
    }
  } catch {
    // A non-JSON error body is not worth reporting over the status code.
  }
  const error = new Error(detail);
  error.status = response.status;
  error.faces = true;
  throw error;
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  results.hidden = true;
  resultsBody.replaceChildren();

  if (!file) {
    previewPanel.hidden = true;
    setStatus("");
    return;
  }

  try {
    loadedImage = await loadImage(file);
    drawScene([]);
    setStatus("");
  } catch (error) {
    loadedImage = null;
    previewPanel.hidden = true;
    setStatus(error.message, "error");
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    setStatus("Choose an image first.", "error");
    return;
  }

  submitButton.disabled = true;
  setStatus("Recognizing…");

  try {
    const result = await recognize(file);
    drawScene(result.plates);
    renderRows(result.plates);
    setStatus(
      result.count === 0
        ? "No plates detected."
        : `${result.count} plate${result.count === 1 ? "" : "s"} detected.`,
    );
  } catch (error) {
    results.hidden = true;
    drawScene([]);
    setStatus(error.message, "error");
  } finally {
    submitButton.disabled = false;
  }
});

// Live camera capture. Two loops run while the camera is on:
//
//   - the track loop hits /detect every 200ms and strokes the boxes onto a
//     transparent canvas over the still-playing video, so a box follows a
//     moving plate;
//   - the recognize loop hits /recognize every 1.5s and refreshes the results
//     table, which is all the slow full pipeline is needed for here.
//
// Camera mode therefore never draws to #preview-panel — drawScene() and the
// preview canvas belong to the upload flow alone.
let cameraStream = null;
let captureTimer = null;
let trackTimer = null;
let capturing = false;

/** Request the camera, show the live feed, and start the capture loop. */
async function startCamera() {
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  cameraFeed.srcObject = cameraStream;
  cameraStage.hidden = false;
  stopCameraButton.hidden = false;
  startCameraButton.disabled = true;
  submitButton.disabled = true; // don't race the upload flow's own /recognize call

  // videoWidth/videoHeight are 0 until metadata loads; capturing before then
  // would send an empty frame and draw a guaranteed, misleading 400.
  if (cameraFeed.readyState < HTMLMediaElement.HAVE_METADATA) {
    await new Promise((resolve) => {
      cameraFeed.addEventListener("loadedmetadata", resolve, { once: true });
    });
  }

  // Box coordinates come back in source-frame pixels, so the overlay is sized
  // to the video's native resolution and left for CSS to scale — the same 1:1
  // trick drawScene() uses for uploads.
  trackingOverlay.width = cameraFeed.videoWidth;
  trackingOverlay.height = cameraFeed.videoHeight;

  capturing = true;
  captureAndRecognize();
  trackLoop();
}

/** Stop both loops and release the camera. */
function stopCamera() {
  capturing = false;
  clearTimeout(captureTimer);
  clearTimeout(trackTimer);
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraStage.hidden = true;
  stopCameraButton.hidden = true;
  startCameraButton.disabled = false;
  submitButton.disabled = false;
}

/** Copy the current video frame into an in-memory canvas. */
function captureFrame() {
  const frame = document.createElement("canvas");
  frame.width = cameraFeed.videoWidth;
  frame.height = cameraFeed.videoHeight;
  frame.getContext("2d").drawImage(cameraFeed, 0, 0);
  return frame;
}

/** Encode a captured frame as the JPEG upload both endpoints expect. */
async function frameToFile(frame) {
  const blob = await new Promise((resolve) => frame.toBlob(resolve, "image/jpeg", 0.85));
  return new File([blob], "frame.jpg", { type: "image/jpeg" });
}

/**
 * Replace the overlay's boxes with this frame's.
 *
 * No row numbers here, unlike drawScene(): at this cadence a box has no stable
 * correspondence to a row in the (much slower) results table, so a number would
 * point at the wrong plate more often than the right one.
 */
function drawTrackingBoxes(plates, faces) {
  const context = trackingOverlay.getContext("2d");
  // One clear for both kinds of box. Clearing per-kind would mean the second
  // draw erased the first, leaving only whichever ran last on screen.
  context.clearRect(0, 0, trackingOverlay.width, trackingOverlay.height);
  context.lineWidth = BOX_WIDTH;

  context.strokeStyle = BOX_COLOUR;
  plates.forEach(({ x1, y1, x2, y2 }) => {
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  });

  context.strokeStyle = FACE_BOX_COLOUR;
  faces.forEach(({ box }) => {
    context.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
  });

  faces.forEach(({ landmarks }) => {
    if (landmarks) {
      drawLandmarks(context, landmarks);
    }
  });
}

/** Stroke one run of landmark points as a single path.
 *
 * @param {CanvasRenderingContext2D} context Overlay context.
 * @param {Array<Array<number>>} points `[x, y]` pairs in frame coordinates.
 * @param {boolean} closed Whether to join the last point back to the first.
 */
function strokePolyline(context, points, closed) {
  if (points.length < 2) {
    return;
  }
  context.beginPath();
  context.moveTo(points[0][0], points[0][1]);
  points.slice(1).forEach(([x, y]) => context.lineTo(x, y));
  if (closed) {
    context.closePath();
  }
  context.stroke();
}

/** Draw the named feature groups of one face onto the overlay.
 *
 * Called from inside drawTrackingBoxes(), after the single clearRect() that
 * tick is allowed — clearing again here would erase the plate boxes.
 */
function drawLandmarks(context, landmarks) {
  context.lineWidth = FEATURE_WIDTH;
  FEATURE_STROKES.forEach(({ group, colour, closed, split }) => {
    const points = landmarks[group] ?? [];
    context.strokeStyle = colour;
    const runs = split ? [points.slice(0, split), points.slice(split)] : [points];
    runs.forEach((run) => strokePolyline(context, run, closed));
  });
}

/**
 * Redraw the tracking boxes from the current frame, then schedule the next tick.
 *
 * The next tick is scheduled only once this one has resolved, so a slow response
 * stretches the cadence instead of stacking requests behind each other.
 */
async function trackLoop() {
  try {
    // The frame is built once and shared by both requests, which are issued
    // concurrently so a tick costs one round trip's latency rather than two.
    const file = await frameToFile(captureFrame());
    // Facial features ride along on the face request as a query parameter, so
    // the richer mode still costs one face round trip, not two.
    const faceMode = faceModeSelect.value;
    const [plates, faces] = await Promise.all([
      detectOnly(file),
      faceMode === "off"
        ? Promise.resolve({ faces: [] })
        : detectFaces(file, faceMode === "features"),
    ]);
    drawTrackingBoxes(plates.boxes, faces.faces);
  } catch (error) {
    if (error.status === 503) {
      // A missing face model disables only the overlay it powers; plate
      // tracking still works, so the session keeps running one mode lower.
      // A missing plate detector still stops everything.
      if (error.faces) {
        const fallback = FACE_MODE_FALLBACK[faceModeSelect.value] ?? "off";
        faceModeSelect.value = fallback;
        setStatus(FACE_MODEL_MISSING[fallback], "error");
      } else {
        stopCamera();
        return;
      }
    }
    // One dropped tick is not worth taking over the status line the recognize
    // loop owns; the boxes simply stay put until the next frame lands.
  }

  if (capturing) {
    trackTimer = setTimeout(trackLoop, TRACK_INTERVAL_MS);
  }
}

/**
 * Recognize the current video frame into the results table, then schedule the
 * next one.
 *
 * A 503 means the detector isn't installed and every future frame would fail
 * the same way, so the loop stops itself instead of hammering a known-broken
 * endpoint. Any other error (a single garbled frame, a network blip) is shown
 * and the loop keeps going — the next frame is likely fine.
 */
async function captureAndRecognize() {
  try {
    const result = await recognize(await frameToFile(captureFrame()));
    renderRows(result.plates);
    setStatus(
      result.count === 0
        ? "No plates detected."
        : `${result.count} plate${result.count === 1 ? "" : "s"} detected.`,
    );
  } catch (error) {
    setStatus(error.message, "error");
    if (error.status === 503) {
      stopCamera();
      return;
    }
  }

  if (capturing) {
    captureTimer = setTimeout(captureAndRecognize, CAPTURE_INTERVAL_MS);
  }
}

startCameraButton.addEventListener("click", () => {
  startCamera().catch((error) => {
    setStatus(error.message, "error");
  });
});

stopCameraButton.addEventListener("click", stopCamera);

/** Show the upload form or the camera panel, never both. */
function setMode(mode) {
  const isCamera = mode === "camera";
  if (!isCamera) {
    stopCamera();
  }
  form.hidden = isCamera;
  cameraPanel.hidden = !isCamera;
  uploadModeButton.setAttribute("aria-pressed", String(!isCamera));
  cameraModeButton.setAttribute("aria-pressed", String(isCamera));
}

uploadModeButton.addEventListener("click", () => setMode("upload"));
cameraModeButton.addEventListener("click", () => setMode("camera"));
