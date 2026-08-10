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
const faceOverlay = document.getElementById("face-overlay");
const startCameraButton = document.getElementById("start-camera-button");
const stopCameraButton = document.getElementById("stop-camera-button");
const uploadModeButton = document.getElementById("upload-mode-button");
const cameraPlateButton = document.getElementById("camera-plate-button");
const cameraFaceButton = document.getElementById("camera-face-button");
const cameraPanel = document.getElementById("camera-panel");
const faceControls = document.getElementById("face-controls");
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

// The mesh is one colour, not seven: it is a surface rather than a set of named
// parts, and colouring the triangles by which feature they overlap would imply
// a grouping the triangulation does not have.
const MESH_COLOUR = "#00e5ff";

// A face 503 steps the control down one level rather than off, because the
// server cannot say which of the two models is missing; the next tick settles
// it. Falling straight to "off" would hide working face boxes.
const FACE_MODE_FALLBACK = {
  mesh: "features",
  // Attributes step down to plain features rather than off: a 503 there usually
  // means an expression/gender model is missing, not the landmark model, and
  // features still work on their own. If the landmark model is the one missing,
  // the next tick fails again and steps features -> boxes, so it self-heals.
  attributes: "features",
  features: "boxes",
  boxes: "off",
};
const CAPTURE_INTERVAL_MS = 1500;
// Detection alone is ~25ms (docs/benchmark/detect-v0.1.md), so boxes can be
// refreshed far more often than the ~400ms full recognize pipeline allows.
const TRACK_INTERVAL_MS = 200;
// The fast face path asks the server to downscale first, which brings a 720p
// face detection from ~18ms down to ~3ms (docs/benchmark/face-fast-phase13.md).
// At that cost a tick can target the camera's 60fps cadence, so boxes follow a
// moving face instead of updating at the 200ms plate cadence.
const FACE_FAST_MS = 16;

// Only 503 is reworded: the API's "Detector model is not available" is accurate
// but does not tell someone looking at a browser what to do about it. Every
// other failure already carries a usable detail string from the server.
const WEIGHTS_MISSING =
  "Detector model is not installed, so recognition cannot run yet. " +
  "Install trained plate weights at the configured detector path to enable it.";

// Keyed by the mode being stepped *down to*, so each message describes what is
// still working rather than what just failed.
const FACE_MODEL_MISSING = {
  features:
    "A face model is not installed, so the face mesh is unavailable. " +
    "Showing facial features only; fetch it with `make fetch-face-landmark-model`.",
  boxes:
    "A face model is not installed, so facial features are unavailable. " +
    "Showing face boxes only; fetch it with `make fetch-face-landmark-model`.",
  off:
    "Face detection model is not installed, so face overlays are unavailable. " +
    "Fetch it with `make fetch-face-model` to enable them.",
};

// Attributes lean on two extra models the other modes never touch, so a 503
// there gets its own message rather than borrowing a landmark-model one.
const FACE_ATTRIBUTES_MISSING =
  "An expression or gender model is not installed, so those labels are " +
  "unavailable. Showing facial features only; fetch them with " +
  "`make fetch-face-attribute-models`.";

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

/** Append a crop-thumbnail cell, or a dash when no crop was returned. */
function addCropCell(row, cropPng) {
  const cell = document.createElement("td");
  cell.className = "crop";
  if (cropPng) {
    const image = document.createElement("img");
    image.src = cropPng;
    image.alt = "Rectified plate crop";
    cell.appendChild(image);
  } else {
    cell.textContent = "—";
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

    // The rectified crop, so a reader can see what the OCR was looking at.
    // `crop_png` is a data: URI the server built from our own perspective
    // correction — image bytes, never markup, and never recognized text — so
    // it is safe as an <img> src. Absent on the live path, shown as a dash.
    addCropCell(row, plate.crop_png);

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

/** Post the file to /recognize, raising a readable error on failure.
 *
 * @param {File} file The image to recognize.
 * @param {boolean} includeCrops Whether to ask the server to attach each
 *   plate's rectified crop. The upload flow does; the live loop does not, to
 *   keep its per-frame payloads small.
 */
async function recognize(file, includeCrops) {
  const body = new FormData();
  body.append("file", file);

  const url = includeCrops ? "/recognize?include_crops=true" : "/recognize";
  const response = await fetch(url, { method: "POST", body });
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
 * @param {boolean} mesh Whether to ask for the whole-face triangulation.
 * @param {boolean} fast Whether to ask the server to downscale for speed.
 * @param {boolean} attributes Whether to ask for expression and gender.
 */
async function detectFaces(file, landmarks, mesh, fast, attributes) {
  const body = new FormData();
  body.append("file", file);

  // Each mode maps to exactly one flag: `mesh` and `attributes` both imply
  // fitting server-side, so they win outright rather than being combined with
  // `landmarks`. `fast` is the boxes-only path: the only time a client wants
  // the downscale trade is when it only needs a box.
  let url = "/detect/faces";
  if (mesh) {
    url = "/detect/faces?mesh=true";
  } else if (landmarks) {
    url = "/detect/faces?landmarks=true";
  } else if (attributes) {
    url = "/detect/faces?attributes=true";
  } else if (fast) {
    url = "/detect/faces?fast=true";
  }
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
    const result = await recognize(file, true);
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

// Live camera capture. Three self-scheduling loops run while the camera is on,
// each posting from a fresh captured frame and each throwing - never touching
// another loop's overlay:
//
//   - the track loop hits /detect every 200ms and strokes the plate boxes onto
//     the plate overlay, so a box follows a moving plate;
//   - the face loop hits /detect/faces and strokes onto its own overlay, on top
//     of the plates'. It stays at the 200ms cadence for the rich feature/mesh
//     modes and speeds up to ~16ms (targeting 60fps) for plain face boxes,
//     where the server's ?fast=true downscale keeps each tick cheap;
//   - the recognize loop hits /recognize every 1.5s and refreshes the results
//     table, which is all the slow full pipeline is needed for here.
//
// Camera mode therefore never draws to #preview-panel — drawScene() and the
// preview canvas belong to the upload flow alone.
let cameraStream = null;
let captureTimer = null;
let trackTimer = null;
let faceTimer = null;
let capturing = false;
// Which camera tab is active: "plate" runs the recognize + plate-box loops and
// fills the table; "face" runs the face loop alone. Each camera tab drives
// exactly one pipeline, so the two overlays never fight over a frame.
let cameraTab = "plate";

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
  // trick drawScene() uses for uploads. Both sheets share the frame size.
  trackingOverlay.width = cameraFeed.videoWidth;
  trackingOverlay.height = cameraFeed.videoHeight;
  faceOverlay.width = cameraFeed.videoWidth;
  faceOverlay.height = cameraFeed.videoHeight;

  capturing = true;
  // Each tab runs only its own loops: the plate tab reads the number into the
  // table and tracks plate boxes; the face tab draws the face overlay. Running
  // the other tab's loops would waste frames and cross the two surfaces.
  if (cameraTab === "face") {
    faceLoop();
  } else {
    captureAndRecognize();
    trackLoop();
  }
}

/** Stop all three loops and release the camera. */
function stopCamera() {
  capturing = false;
  clearTimeout(captureTimer);
  clearTimeout(trackTimer);
  clearTimeout(faceTimer);
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
 * Replace the plate overlay's boxes with the latest detection.
 *
 * No row numbers here, unlike drawScene(): at this cadence a box has no stable
 * correspondence to a row in the (much slower) results table, so a number would
 * point at the wrong plate more often than the right one.
 */
function drawPlateBoxes(plates) {
  const context = trackingOverlay.getContext("2d");
  context.clearRect(0, 0, trackingOverlay.width, trackingOverlay.height);
  context.lineWidth = BOX_WIDTH;
  context.strokeStyle = BOX_COLOUR;
  plates.forEach(({ x1, y1, x2, y2 }) => {
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  });
}

/**
 * Replace the face overlay's contents — boxes, and any requested landmarks or
 * mesh — with the frame's.
 *
 * Drawn on its own overlay, so the fast (boxes) branch can clear and redraw
 * this layer without touching the plate boxes on the sheet below.
 *
 * Which renderer runs is decided by what the response carries, not by the
 * dropdown: the mode that produced the data is already encoded in it, and
 * reading the control here could disagree with the frame in hand.
 */
function drawFaces(faces) {
  const context = faceOverlay.getContext("2d");
  context.clearRect(0, 0, faceOverlay.width, faceOverlay.height);
  context.lineWidth = BOX_WIDTH;
  context.strokeStyle = FACE_BOX_COLOUR;

  faces.forEach(({ box }) => {
    context.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
  });

  faces.forEach(({ landmarks }) => {
    if (!landmarks) {
      return;
    }
    if (landmarks.triangles) {
      drawMesh(context, landmarks);
    } else {
      drawLandmarks(context, landmarks);
    }
  });

  // Labels last, so their backing plate sits over the box and any landmarks
  // rather than under them. These are English tokens ("female", "happy"), not
  // recognized plate text, so drawing them on the canvas is safe — unlike the
  // Thai plate glyphs the results table keeps out of the canvas on purpose.
  faces.forEach(({ box, attributes }) => {
    if (attributes) {
      drawAttributeLabel(context, box, attributes);
    }
  });
}

/** Compose one face's attributes into a short label, abstentions as "?". */
function formatAttributes(attributes) {
  const gender = attributes.apparent_gender ?? "?";
  const expression = attributes.expression ?? "?";
  return `${gender} · ${expression}`;
}

/** Draw an attribute label on a dark plate just above the face box.
 *
 * Runs after the face overlay's own clearRect() (in drawFaces), so it never
 * clears the box and landmarks it is meant to sit on top of.
 */
function drawAttributeLabel(context, box, attributes) {
  const text = formatAttributes(attributes);
  const fontSize = Math.max(14, faceOverlay.width / 45);
  context.font = `${fontSize}px sans-serif`;
  context.textBaseline = "bottom";

  const padding = Math.round(fontSize * 0.3);
  const plateWidth = context.measureText(text).width + padding * 2;
  const plateHeight = fontSize + padding * 2;
  // Sit the plate just above the box, but never let it clip off the top edge.
  const bottom = Math.max(box.y1, plateHeight);

  context.fillStyle = "rgba(0, 0, 0, 0.6)";
  context.fillRect(box.x1, bottom - plateHeight, plateWidth, plateHeight);
  context.fillStyle = FACE_BOX_COLOUR;
  context.fillText(text, box.x1 + padding, bottom - padding);
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
 * Called from inside drawFaces(), after the face overlay's own clearRect() —
 * clearing again here would erase the box drawn above these points.
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

/** Draw one face as a Delaunay triangle wireframe.
 *
 * The server sends triangles as index triples into `points` rather than as
 * coordinates, so the topology arrives at a quarter of the payload.
 *
 * Like drawLandmarks(), this runs after the face overlay's own clearRect().
 */
function drawMesh(context, landmarks) {
  const points = landmarks.points ?? [];
  context.lineWidth = FEATURE_WIDTH;
  context.strokeStyle = MESH_COLOUR;
  (landmarks.triangles ?? []).forEach((triangle) => {
    const corners = triangle.map((index) => points[index]);
    if (corners.some((corner) => corner === undefined)) {
      return;
    }
    strokePolyline(context, corners, true);
  });
}

/**
 * Refresh the plate boxes from the current frame, then schedule the next tick.
 *
 * The next tick is scheduled only once this one has resolved, so a slow response
 * stretches the cadence instead of stacking requests behind each other. Faces
 * live on their own overlay and their own loop, so they never share this
 * cadence and never risk being erased by it.
 */
async function trackLoop() {
  try {
    const file = await frameToFile(captureFrame());
    const plates = await detectOnly(file);
    drawPlateBoxes(plates.boxes);
  } catch (error) {
    if (error.status === 503) {
      // A missing plate detector makes every future tick fail the same way,
      // so stop the whole session rather than hammer a known-broken endpoint.
      stopCamera();
      return;
    }
    // One dropped tick is not worth taking over the status line the recognize
    // loop owns; the plate boxes simply stay put until the next frame lands.
  }

  if (capturing) {
    trackTimer = setTimeout(trackLoop, TRACK_INTERVAL_MS);
  }
}

/**
 * Refresh the face overlay from the current frame, then schedule the next tick.
 *
 * The cadence follows the control: plain boxes ask the server to downscale
 * (`?fast=true`, ~3ms detection — see docs/benchmark/face-fast-phase13.md) so
 * they can run toward the camera's 60fps rate, while the richer feature and
 * mesh modes stay at the 200ms plate cadence because their precision depends on
 * full-resolution fitting. One loop covers all three so the overlay is never
 * drawn from two loops that could fight over it.
 */
function faceLoopIntervalMs(mode) {
  return mode === "boxes" ? FACE_FAST_MS : TRACK_INTERVAL_MS;
}

async function faceLoop() {
  const mode = faceModeSelect.value;

  if (mode !== "off") {
    try {
      const file = await frameToFile(captureFrame());
      const result = await detectFaces(
        file,
        mode === "features",
        mode === "mesh",
        mode === "boxes",
        mode === "attributes",
      );
      drawFaces(result.faces);
    } catch (error) {
      if (error.status === 503) {
        const fallback = FACE_MODE_FALLBACK[mode] ?? "off";
        // "off" means even plain boxes failed, so the base face detector is
        // missing and nothing on this face-only tab can render — stop rather
        // than leave a dead feed. (On the plate tab a face 503 can't occur.)
        if (fallback === "off") {
          setStatus(FACE_MODEL_MISSING.off, "error");
          stopCamera();
          return;
        }
        // Otherwise step down one level: the select settles on the highest
        // working mode by itself. Attributes name their own models to fetch.
        faceModeSelect.value = fallback;
        setStatus(
          mode === "attributes"
            ? FACE_ATTRIBUTES_MISSING
            : FACE_MODEL_MISSING[fallback],
          "error",
        );
      }
      // One dropped tick is left alone; the boxes stay put until the next frame.
    }
  }

  if (capturing) {
    faceTimer = setTimeout(faceLoop, faceLoopIntervalMs(mode));
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
    const result = await recognize(await frameToFile(captureFrame()), false);
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

/** Switch between the three tabs: upload, live plate, live face.
 *
 * @param {"upload"|"plate"|"face"} mode The tab to show.
 */
function setMode(mode) {
  // Always release the camera on any switch, including plate<->face: the two
  // camera tabs run different loops, so the feed is restarted for the new one.
  stopCamera();

  const isUpload = mode === "upload";
  form.hidden = !isUpload;
  cameraPanel.hidden = isUpload;
  // The face-mode selector belongs to the face tab alone; the plate tabs show
  // no face UI at all.
  faceControls.hidden = mode !== "face";
  // Remember which camera tab a later Start should drive.
  if (!isUpload) {
    cameraTab = mode;
  }

  uploadModeButton.setAttribute("aria-pressed", String(mode === "upload"));
  cameraPlateButton.setAttribute("aria-pressed", String(mode === "plate"));
  cameraFaceButton.setAttribute("aria-pressed", String(mode === "face"));

  // A result belongs to the tab that produced it. Without this, the last
  // upload's photo and plate table stay on screen under a live feed, reading
  // as if they described what the camera is looking at. Cleared in every
  // direction, since rows are equally stale once any other tab is shown.
  previewPanel.hidden = true;
  results.hidden = true;
  resultsBody.replaceChildren();
  setStatus("");

  // The file input still names a file, so the upload tab should still show its
  // preview — without boxes, since those results were just discarded.
  if (isUpload && loadedImage) {
    drawScene([]);
  }
}

uploadModeButton.addEventListener("click", () => setMode("upload"));
cameraPlateButton.addEventListener("click", () => setMode("plate"));
cameraFaceButton.addEventListener("click", () => setMode("face"));
