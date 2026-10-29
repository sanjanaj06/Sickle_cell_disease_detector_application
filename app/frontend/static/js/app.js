document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("image-input");
  const previewImg = document.getElementById("image-preview");
  const dropzonePrompt = document.getElementById("dropzone-prompt");
  const form = document.getElementById("diagnostic-form");
  const submitBtn = document.getElementById("submit-btn");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");

  const placeholderView = document.getElementById("placeholder-view");
  const reportView = document.getElementById("report-view");

  // Click dropzone to open file picker
  dropzone.addEventListener("click", () => fileInput.click());

  // Handle drag & drop
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#2563eb";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "#cbd5e1";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#cbd5e1";
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(fileInput.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  function handleFileSelected(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewImg.classList.remove("hidden");
      dropzonePrompt.classList.add("hidden");
    };
    reader.readAsDataURL(file);
  }

  // Handle Form Submit
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!fileInput.files || fileInput.files.length === 0) {
      alert("Please select or drop a blood smear image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    formData.append("age", document.getElementById("age-input").value);
    formData.append("gender", document.getElementById("gender-input").value);
    formData.append("pain_crises", document.getElementById("pain-input").value);
    formData.append("spo2", document.getElementById("spo2-input").value);

    // UI Loading state
    btnText.textContent = "Analyzing Smear & Clinical Vitals...";
    btnSpinner.classList.remove("hidden");
    submitBtn.disabled = true;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const resData = await response.json();

      if (!response.ok || !resData.success) {
        throw new Error(resData.error || "Failed to analyze image.");
      }

      displayResults(resData.data);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      btnText.textContent = "Run Diagnostic Assessment";
      btnSpinner.classList.add("hidden");
      submitBtn.disabled = false;
    }
  });

    function displayResults(data) {
    placeholderView.classList.add("hidden");
    reportView.classList.remove("hidden");

    const pred = data.prediction;
    const cell = data.cell_analysis;
    const clin = data.clinical_assessment;

    // 1. Diagnosis Banner
    const banner = document.getElementById("diagnosis-banner");
    const diagText = document.getElementById("diagnosis-text");
    const confVal = document.getElementById("confidence-val");

    if (pred.is_positive) {
      banner.className = "diagnosis-banner positive";
      diagText.textContent = "POSITIVE: SICKLE CELL DISEASE DETECTED";
    } else {
      banner.className = "diagnosis-banner negative";
      diagText.textContent = "NEGATIVE: NORMAL BLOOD SMEAR";
    }
    confVal.textContent = pred.confidence_percent + "%";

    // 2. Visualizations
    document.getElementById("gradcam-img").src = data.visualizations.gradcam_overlay;
    document.getElementById("cell-img").src = data.visualizations.cell_overlay;

    // 3. Numerical Metrics
    document.getElementById("m-total-rbc").textContent = cell.total_rbc;
    document.getElementById("m-sickle-count").textContent = cell.sickle_count;
    document.getElementById("m-sci").textContent = cell.sci + "%";
    document.getElementById("m-cd").textContent = cell.cd + "%";

        // 3b. Patient Summary
    document.getElementById("ps-age").textContent = document.getElementById("age-input").value;
    document.getElementById("ps-gender").textContent = document.getElementById("gender-input").value;
    document.getElementById("ps-spo2").textContent = document.getElementById("spo2-input").value;
    document.getElementById("ps-pain").textContent = document.getElementById("pain-input").value;
    document.getElementById("ps-date").textContent = new Date().toLocaleDateString("en-IN", {
      day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit"
    });

    // 4. Clinical Gate: Check if disease was detected
    const healthyCard = document.getElementById("healthy-notice-card");
    const severityCard = document.getElementById("clinical-report-card");

    if (!pred.disease_detected || !clin) {
      // Normal Smear / 0% Sickles: Hide severity and show clean healthy notice
      healthyCard.classList.remove("hidden");
      severityCard.classList.add("hidden");
    } else {
      // Disease Present: Show severity, Hb range, and VORS risk breakdown
      healthyCard.classList.add("hidden");
      severityCard.classList.remove("hidden");

      const vors = clin.vors;
      document.getElementById("ibsi-val").textContent = clin.ibsi;
      const sevBadge = document.getElementById("severity-badge");
      sevBadge.textContent = clin.severity;
      sevBadge.className = "badge " + clin.severity.toLowerCase();

      const hbRange = clin.estimated_hb_range_g_dl;
      document.getElementById("hb-range-val").textContent = `${hbRange[0]} - ${hbRange[1]} g/dL`;

      document.getElementById("vors-val").textContent = vors.vors;
      const vorsBadge = document.getElementById("vors-badge");
      vorsBadge.textContent = vors.risk + " Risk";
      vorsBadge.className = "badge " + vors.risk.toLowerCase();

      // VORS Breakdown Bars
      const comp = vors.component_scores;
      document.getElementById("bar-spo2").style.width = Math.min(100, comp.spo2) + "%";
      document.getElementById("bar-ibsi").style.width = Math.min(100, comp.ibsi) + "%";
      document.getElementById("bar-hb").style.width = Math.min(100, comp.hb) + "%";
      document.getElementById("bar-pain").style.width = Math.min(100, comp.pain) + "%";
    }
  }
});


function downloadPDF() {
  const report = document.getElementById("report-view");
  const btn = document.getElementById("download-pdf-btn");

  // Temporarily hide the download button itself from the PDF
  btn.style.display = "none";

  const opt = {
    margin:       [10, 10, 10, 10],
    filename:     "SickleCell_Diagnostic_Report.pdf",
    image:        { type: "jpeg", quality: 0.95 },
    html2canvas:  { scale: 2, useCORS: true, scrollY: 0 },
    jsPDF:        { unit: "mm", format: "a4", orientation: "portrait" },
    pagebreak:    { mode: ["avoid-all", "css", "legacy"] }
  };

  html2pdf().set(opt).from(report).save().then(() => {
    btn.style.display = "";
  });
}