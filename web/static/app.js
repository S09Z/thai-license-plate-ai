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

const BOX_COLOUR = "#e0245e";
const BOX_WIDTH = 3;

// Only 503 is reworded: the API's "Detector model is not available" is accurate
// but does not tell someone looking at a browser what to do about it. Every
// other failure already carries a usable detail string from the server.
const WEIGHTS_MISSING =
  "Detector model is not installed, so recognition cannot run yet. " +
  "Install trained plate weights at the configured detector path to enable it.";

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
    throw new Error(WEIGHTS_MISSING);
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
  throw new Error(detail);
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
