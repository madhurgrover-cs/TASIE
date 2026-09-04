// Same-origin: the dashboard is served by the API app itself, so requests go to
// "/api/..." on whatever host/port the page was loaded from. Override by setting
// window.API_BASE before this script if the API lives elsewhere.
const API = (typeof window !== "undefined" && window.API_BASE) || "";

// ─── Toast ─────────────────────────────────────────────────────────────────
function toast(msg, type = "default") {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = `toast show ${type}`;
  setTimeout(() => { t.className = "toast"; }, 3500);
}

// ─── API helpers ───────────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

// ─── Dashboard Metrics ─────────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const data = await apiFetch("/api/dashboard/metrics");

    // Cards
    animateValue("totalScans", data.total_scans);
    animateValue("fpRate", data.false_positive_rate_pct, "%");
    animateValue("acceptRate", data.developer_acceptance_rate_pct, "%");

    document.getElementById("fpCount").textContent =
      `${data.false_positive_count} of ${data.total_labelled} labelled`;

    document.getElementById("activeModel").textContent =
      data.active_model || "Not trained";

    document.getElementById("modelVersions").textContent =
      `${data.model_versions} version${data.model_versions !== 1 ? "s" : ""} stored`;

    // Weekly trend bars
    renderTrendChart(data.weekly_trends);

    // Vuln recurrence horizontal bars
    renderVulnBars(data.vulnerability_recurrence);

  } catch (e) {
    console.warn("Metrics unavailable:", e.message);
    document.getElementById("statusBadge").textContent = "Offline";
    document.getElementById("statusBadge").style.background = "rgba(248,113,113,0.15)";
    document.getElementById("statusBadge").style.color = "var(--red)";
    document.getElementById("statusBadge").style.borderColor = "rgba(248,113,113,0.3)";
    document.getElementById("statusBadge").style.animation = "none";
  }
}

// ─── Count-up animation ─────────────────────────────────────────────────────
function animateValue(id, to, suffix = "") {
  const el = document.getElementById(id);
  if (el === null) return;
  const from = 0;
  const duration = 800;
  const start = performance.now();
  const update = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = (Math.round(from + (to - from) * ease * 10) / 10) + suffix;
    if (t < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

// ─── Trend Chart ───────────────────────────────────────────────────────────
function renderTrendChart(trends) {
  const el = document.getElementById("trendChart");
  if (!trends || trends.length === 0) {
    el.innerHTML = '<div class="chart-empty">No trend data yet. Run a scan to begin.</div>';
    return;
  }
  const max = Math.max(...trends.map(t => t.count), 1);
  el.innerHTML = trends.map(t => `
    <div class="bar-wrap">
      <div class="bar" style="height:${Math.max(8, (t.count / max) * 150)}px"
           data-tip="Week ${t.week}: ${t.count}"></div>
      <span class="bar-label">W${t.week}</span>
    </div>
  `).join("");
}

// ─── Vulnerability Recurrence ──────────────────────────────────────────────
function renderVulnBars(vulns) {
  const el = document.getElementById("vulnBars");
  if (!vulns || vulns.length === 0) {
    el.innerHTML = '<div class="chart-empty">No data yet.</div>';
    return;
  }
  const sorted = [...vulns].sort((a, b) => b.count - a.count).slice(0, 8);
  const max = sorted[0].count;
  el.innerHTML = sorted.map(v => `
    <div class="h-bar-row">
      <span class="h-bar-name" title="${escHtml(v.type || 'Unknown')}">${escHtml(v.type || "Unknown")}</span>
      <div class="h-bar-track">
        <div class="h-bar-fill" style="width:${Math.max(4, v.count / max * 100)}%"></div>
      </div>
      <span class="h-bar-count">${v.count}</span>
    </div>
  `).join("");
}

// ─── Findings ──────────────────────────────────────────────────────────────
async function loadFindings() {
  const el = document.getElementById("findingsList");
  try {
    const findings = await apiFetch("/api/feedback");
    if (!findings.length) {
      el.innerHTML = '<div class="empty-state">No pending findings. Scan a repository to get started.</div>';
      return;
    }
    el.innerHTML = findings.map(f => `
      <div class="finding-card" id="finding-${escHtml(f.code_hash)}">
        <div class="finding-info">
          <div class="finding-top">
            <span class="finding-type">${escHtml(f.vulnerability_type || "Unknown")}</span>
            <span class="severity-badge ${getSeverityClass(f.confidence_score)}">
              ${getSeverityLabel(f.confidence_score)}
            </span>
          </div>
          <div class="finding-file">${escHtml(f.file_path || "unknown file")}</div>
          <div class="finding-snippet">${escHtml(f.code_snippet || "")}</div>
          <div class="confidence">Confidence: ${f.confidence_score != null ? (f.confidence_score * 100).toFixed(0) + "%" : "N/A"}</div>
        </div>
        <div class="finding-actions">
          <button class="label-btn label-valid" onclick="submitFeedback('${f.code_hash}', 'valid_vulnerability', this)">✓ Valid</button>
          <button class="label-btn label-fp" onclick="submitFeedback('${f.code_hash}', 'false_positive', this)">✗ False Positive</button>
          <button class="label-btn label-review" onclick="submitFeedback('${f.code_hash}', 'needs_review', this)">? Review</button>
        </div>
      </div>
    `).join("");
  } catch (e) {
    el.innerHTML = `<div class="empty-state">Could not load findings: ${e.message}</div>`;
  }
}

function getSeverityClass(score) {
  if (score == null) return "sev-medium";
  if (score >= 0.75) return "sev-high";
  if (score >= 0.4) return "sev-medium";
  return "sev-low";
}

function getSeverityLabel(score) {
  if (score == null) return "UNKNOWN";
  if (score >= 0.75) return "HIGH";
  if (score >= 0.4) return "MEDIUM";
  return "LOW";
}

function escHtml(s) {
  return String(s ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

async function submitFeedback(hash, label, btn) {
  btn.disabled = true;
  try {
    await apiFetch("/api/feedback", {
      method: "POST",
      body: JSON.stringify({ code_hash: hash, developer_label: label }),
    });
    const card = document.getElementById(`finding-${hash}`);
    if (card) {
      card.style.transition = "opacity 0.4s, transform 0.4s";
      card.style.opacity = "0";
      card.style.transform = "translateX(20px)";
      setTimeout(() => card.remove(), 400);
    }
    toast(`Marked as ${label.replace(/_/g, " ")}`, "success");
    setTimeout(loadMetrics, 600);
  } catch (e) {
    toast("Error: " + e.message, "error");
    btn.disabled = false;
  }
}

// ─── Smart Memory ──────────────────────────────────────────────────────────
async function loadMemory() {
  const el = document.getElementById("memoryList");
  try {
    const rows = await apiFetch("/api/smart-memory");
    if (!rows.length) {
      el.innerHTML = '<div class="empty-state">No smart memory patterns added yet.</div>';
      return;
    }
    el.innerHTML = rows.map(r => `
      <div class="memory-row">
        <span class="memory-pattern">${escHtml(r.pattern)}</span>
        <span class="memory-type">${escHtml(r.pattern_type)}</span>
        <span style="font-size:0.75rem;color:var(--text-3)">${escHtml(r.description || "")}</span>
      </div>
    `).join("");
  } catch (e) {
    el.innerHTML = `<div class="empty-state">Unavailable: ${e.message}</div>`;
  }
}

document.getElementById("addPatternBtn").addEventListener("click", async () => {
  const pattern = document.getElementById("smPattern").value.trim();
  const type = document.getElementById("smType").value;
  const desc = document.getElementById("smDesc").value.trim();
  if (!pattern) { toast("Pattern is required", "error"); return; }
  try {
    await apiFetch("/api/smart-memory", {
      method: "POST",
      body: JSON.stringify({ pattern, pattern_type: type, description: desc }),
    });
    toast("Pattern added to Smart Memory", "success");
    document.getElementById("smPattern").value = "";
    document.getElementById("smDesc").value = "";
    loadMemory();
  } catch (e) {
    toast("Error: " + e.message, "error");
  }
});

// ─── Model Versions ─────────────────────────────────────────────────────────
async function loadModels() {
  const el = document.getElementById("modelList");
  try {
    const versions = await apiFetch("/api/model/versions");
    if (!versions.length) {
      el.innerHTML = '<div class="empty-state">No models trained yet. Label at least 10 findings then retrain.</div>';
      return;
    }
    const activeVersion = versions[versions.length - 1].version;
    el.innerHTML = [...versions].reverse().map(v => `
      <div class="model-row">
        <span class="model-version-tag">${escHtml(v.version)}</span>
        ${v.version === activeVersion ? '<span class="model-active-badge">ACTIVE</span>' : ""}
        <span style="font-size:0.78rem;color:var(--text-3)">
          ${v.metrics?.samples_trained ?? "?"} samples · threshold ${v.metrics?.threshold?.toFixed(2) ?? "?"}
        </span>
        <span class="model-date">${new Date(v.created_at).toLocaleDateString()}</span>
      </div>
    `).join("");
  } catch (e) {
    el.innerHTML = `<div class="empty-state">Unavailable: ${e.message}</div>`;
  }
}

document.getElementById("retrainBtn").addEventListener("click", async () => {
  const btn = document.getElementById("retrainBtn");
  btn.disabled = true;
  btn.textContent = "⟳ Training…";
  try {
    const res = await apiFetch("/api/model/retrain", { method: "POST" });
    if (res.status === "skipped") {
      toast(res.reason, "error");
    } else {
      toast(`Model ${res.version} trained successfully!`, "success");
      loadModels();
      loadMetrics();
    }
  } catch (e) {
    toast("Retraining failed: " + e.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "⟳ Retrain Model";
  }
});

// ─── Scan Modal ─────────────────────────────────────────────────────────────
const scanModal = document.getElementById("scanModal");
document.getElementById("scanBtn").addEventListener("click", () => {
  scanModal.classList.add("open");
});
document.getElementById("cancelScan").addEventListener("click", () => {
  scanModal.classList.remove("open");
});
document.getElementById("confirmScan").addEventListener("click", async () => {
  const path = document.getElementById("repoPath").value.trim();
  const mode = document.querySelector('input[name="scanMode"]:checked').value;
  if (!path) { toast("Enter a repository path", "error"); return; }
  scanModal.classList.remove("open");
  toast("Scan started…");
  try {
    const res = await apiFetch("/api/scan", {
      method: "POST",
      body: JSON.stringify({ repo_path: path, mode }),
    });
    toast(`Scan complete — ${res.total} finding(s) detected`, "success");
    loadFindings();
    loadMetrics();
  } catch (e) {
    toast("Scan failed: " + e.message, "error");
  }
});
scanModal.addEventListener("click", (e) => {
  if (e.target === scanModal) scanModal.classList.remove("open");
});

// ─── Refresh ──────────────────────────────────────────────────────────────
document.getElementById("refreshBtn").addEventListener("click", () => {
  loadMetrics();
  loadFindings();
  loadMemory();
  loadModels();
  toast("Refreshed");
});

// ─── Nav smooth scroll ─────────────────────────────────────────────────────
document.querySelectorAll(".nav-link").forEach(link => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute("href"));
    if (target) target.scrollIntoView({ behavior: "smooth" });
    document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
    link.classList.add("active");
  });
});

// ─── Init ─────────────────────────────────────────────────────────────────
loadMetrics();
loadFindings();
loadMemory();
loadModels();
