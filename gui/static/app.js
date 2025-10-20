const filesInput = document.getElementById('files');
const fileList = document.getElementById('fileList');
const predictButton = document.getElementById('predict');
const resultImage = document.getElementById('resultImage');
const placeholder = document.getElementById('placeholder');
const controls = document.getElementById('controls');
const counter = document.getElementById('counter');
const log = document.getElementById('log');

let uploadedFiles = [];
let resultImages = [];
let currentResult = 0;

function writeLog(message, isError = false) {
  const time = new Date().toLocaleTimeString();
  log.textContent += `\n[${time}] ${message}`;
  log.classList.toggle('error', isError);
  log.scrollTop = log.scrollHeight;
}

async function readResponse(response) {
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error('The server returned an invalid response');
  }

  if (!response.ok) {
    throw new Error(data.error || 'The request failed');
  }
  return data;
}

function resetResults() {
  resultImages = [];
  currentResult = 0;
  resultImage.removeAttribute('src');
  resultImage.style.display = 'none';
  placeholder.style.display = 'block';
  controls.style.display = 'none';
  document.getElementById('metrics').replaceChildren();
}

function showResult(index) {
  if (!resultImages.length) return;
  currentResult = (index + resultImages.length) % resultImages.length;
  resultImage.src = `${resultImages[currentResult]}?v=${Date.now()}`;
  resultImage.style.display = 'block';
  placeholder.style.display = 'none';
  controls.style.display = resultImages.length > 1 ? 'flex' : 'none';
  counter.textContent = `${currentResult + 1} of ${resultImages.length}`;
}

function showMetrics(values) {
  const fields = [
    ['Images', 'num_images'],
    ['Total', 'total_detections'],
    ['Average', 'mean_detections_per_image'],
    ['Maximum', 'max_detections'],
    ['Minimum', 'min_detections']
  ];
  const metrics = document.getElementById('metrics');
  metrics.replaceChildren();

  for (const [label, key] of fields) {
    const item = document.createElement('div');
    item.className = 'metric';
    item.textContent = label;
    const value = document.createElement('strong');
    value.textContent = values[key] ?? 0;
    item.appendChild(value);
    metrics.appendChild(item);
  }
}

filesInput.addEventListener('change', async () => {
  if (!filesInput.files.length) return;

  const form = new FormData();
  for (const file of filesInput.files) {
    form.append('files[]', file);
  }

  predictButton.disabled = true;
  resetResults();
  writeLog('Uploading images');

  try {
    const response = await fetch('/upload', { method: 'POST', body: form });
    const data = await readResponse(response);
    uploadedFiles = data.uploaded;
    fileList.replaceChildren();

    for (const filename of uploadedFiles) {
      const item = document.createElement('div');
      item.textContent = filename;
      fileList.appendChild(item);
    }

    predictButton.disabled = false;
    writeLog(`Uploaded ${uploadedFiles.length} image(s)`);
  } catch (error) {
    uploadedFiles = [];
    writeLog(error.message, true);
  }
});

predictButton.addEventListener('click', async () => {
  predictButton.disabled = true;
  writeLog('Running prediction');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: document.getElementById('model').value,
        conf: Number(document.getElementById('confidence').value),
        iou: Number(document.getElementById('iou').value),
        files: uploadedFiles
      })
    });
    const data = await readResponse(response);
    resultImages = data.images || [];
    showMetrics(data.metrics || {});

    if (resultImages.length) {
      showResult(0);
    } else {
      writeLog('Prediction finished but no result images were saved', true);
    }
    writeLog(`Prediction finished with ${resultImages.length} result(s)`);
  } catch (error) {
    writeLog(error.message, true);
  } finally {
    predictButton.disabled = uploadedFiles.length === 0;
  }
});

document.getElementById('previous').addEventListener('click', () => {
  showResult(currentResult - 1);
});

document.getElementById('next').addEventListener('click', () => {
  showResult(currentResult + 1);
});

window.addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft') showResult(currentResult - 1);
  if (event.key === 'ArrowRight') showResult(currentResult + 1);
});
