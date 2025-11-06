const dz = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const statusEl = document.getElementById("status");
const imgEl = document.getElementById("resultImg");
const metaEl = document.getElementById("meta");

function setStatus(text) {
  statusEl.textContent = text;
}

browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files?.length) upload(fileInput.files[0]);
});

["dragenter","dragover"].forEach(evt => {
  dz.addEventListener(evt, e => {
    e.preventDefault(); e.stopPropagation();
    dz.classList.add("hover");
  });
});
["dragleave","drop"].forEach(evt => {
  dz.addEventListener(evt, e => {
    e.preventDefault(); e.stopPropagation();
    dz.classList.remove("hover");
  });
});
dz.addEventListener("drop", e => {
  const file = e.dataTransfer.files?.[0];
  if (file) upload(file);
});

async function upload(file) {
  setStatus("Uploading…");
  const fd = new FormData();
  fd.append("image", file);

  try {
    const resp = await fetch("/predict", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Upload failed");

    setStatus(`Done. Device: ${data.device}. ${data.detections.length} detections.`);
    imgEl.src = data.result_url + "?t=" + Date.now();

    // Render detections
    metaEl.innerHTML = "";
    if (data.detections.length) {
      const ul = document.createElement("ul");
      data.detections.forEach(d => {
        const li = document.createElement("li");
        li.textContent = `${d.cls_name} (${d.conf}) [${d.box.map(n => Math.round(n)).join(", ")}]`;
        ul.appendChild(li);
      });
      metaEl.appendChild(ul);
    } else {
      metaEl.textContent = "No objects detected.";
    }
  } catch (err) {
    console.error(err);
    setStatus("Error: " + err.message);
  }
}
