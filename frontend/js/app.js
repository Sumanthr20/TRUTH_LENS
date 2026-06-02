/**
 * TruthLens — AI Fake News Detection System
 * Powered by Claude AI (claude-sonnet-4)
 */

// ── Sample texts ──────────────────────────────────────────────────────────────
const SAMPLE_FAKE = `SHOCKING EXPOSED: Government secretly installing mind-control microchips inside COVID-19 vaccines! Bill Gates admitted in leaked audio that 5G towers activate the chips. Deep state elites hiding this truth from public. Doctors who speak out are being silenced by big pharma!! SHARE BEFORE THEY DELETE THIS!!!`;

const SAMPLE_REAL = `Scientists at the Indian Space Research Organisation (ISRO) successfully launched the GSAT-20 communication satellite from the Satish Dhawan Space Centre in Sriharikota. The satellite will enhance broadband connectivity across India, particularly in remote and rural regions. Officials confirmed all systems are functioning nominally following the launch.`;

// ── DOM References ─────────────────────────────────────────────────────────────
const newsInput       = document.getElementById('newsInput');
const charCount       = document.getElementById('charCount');
const analyseBtn      = document.getElementById('analyseBtn');
const btnText         = document.getElementById('btnText');
const clearBtn        = document.getElementById('clearBtn');
const sampleFakeBtn   = document.getElementById('sampleFakeBtn');
const sampleRealBtn   = document.getElementById('sampleRealBtn');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent   = document.getElementById('resultContent');
const batchInput      = document.getElementById('batchInput');
const batchBtn        = document.getElementById('batchBtn');
const batchBtnText    = document.getElementById('batchBtnText');
const batchResults    = document.getElementById('batchResults');

// ── Character counter ─────────────────────────────────────────────────────────
newsInput.addEventListener('input', () => {
  const len = newsInput.value.length;
  charCount.textContent = `${len} character${len !== 1 ? 's' : ''}`;
});

clearBtn.addEventListener('click', () => {
  newsInput.value = '';
  charCount.textContent = '0 characters';
  showPlaceholder();
});

sampleFakeBtn.addEventListener('click', () => {
  newsInput.value = SAMPLE_FAKE;
  charCount.textContent = `${SAMPLE_FAKE.length} characters`;
  newsInput.focus();
});

sampleRealBtn.addEventListener('click', () => {
  newsInput.value = SAMPLE_REAL;
  charCount.textContent = `${SAMPLE_REAL.length} characters`;
  newsInput.focus();
});

// ── Claude API Call ───────────────────────────────────────────────────────────
async function callClaudeAPI(text) {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text })
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(`Error ${response.status}: ${err.error || response.statusText}`);
  }
  return await response.json();
}

async function callClaudeBatch(texts) {
  const response = await fetch('/api/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts: texts })
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(`Error ${response.status}: ${err.error || response.statusText}`);
  }
  return await response.json();
}

// ── Show/hide result ──────────────────────────────────────────────────────────
function showPlaceholder() {
  resultPlaceholder.style.display = 'flex';
  resultContent.style.display = 'none';
  resultContent.innerHTML = '';
}

function showResult(data) {
  resultPlaceholder.style.display = 'none';
  resultContent.style.display = 'block';

  const isFake = data.is_fake;
  const cls = isFake ? 'fake' : 'real';
  const icon = isFake ? '⚠' : '✓';
  const verdict = isFake ? 'FAKE NEWS' : 'REAL NEWS';

  const indicatorsHTML = (data.indicators && data.indicators.length > 0)
    ? `<div class="indicators-title">LINGUISTIC INDICATORS</div>` +
      data.indicators.map(ind => `
        <div class="indicator ${ind.type}">
          <div class="indicator-dot"></div>
          <span>${ind.text}</span>
        </div>
      `).join('')
    : '';

  resultContent.innerHTML = `
    <div class="verdict ${cls}">
      <div class="verdict-icon">${icon}</div>
      <div>
        <div class="verdict-label">${verdict}</div>
        <div class="verdict-sub">Confidence: ${data.confidence_label} · ${data.confidence}%</div>
      </div>
    </div>

    <div class="confidence-bar-wrap">
      <div class="conf-labels">
        <span>Confidence Score</span>
        <span>${data.confidence}%</span>
      </div>
      <div class="conf-track">
        <div class="conf-fill ${cls}" id="confFill" style="width: 0%"></div>
      </div>
    </div>

    <div class="proba-row">
      <div class="proba-item real-p">
        <div class="proba-value">${data.probabilities.real}%</div>
        <div class="proba-type">Real Probability</div>
      </div>
      <div class="proba-item fake-p">
        <div class="proba-value">${data.probabilities.fake}%</div>
        <div class="proba-type">Fake Probability</div>
      </div>
    </div>

    ${indicatorsHTML}

    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);">
      <span style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim);">
        Model: ${data.model || 'Claude AI'} · Accuracy: ${data.model_accuracy || 96}% · Words analysed: ${data.word_count}
      </span>
    </div>
  `;

  requestAnimationFrame(() => {
    setTimeout(() => {
      const fill = document.getElementById('confFill');
      if (fill) fill.style.width = data.confidence + '%';
    }, 80);
  });
}

// ── Single Analysis ───────────────────────────────────────────────────────────
analyseBtn.addEventListener('click', async () => {
  const text = newsInput.value.trim();
  if (!text) { pulse(newsInput); return; }
  if (text.length < 10) {
    showError('Please enter at least 10 characters for a reliable analysis.');
    return;
  }

  setLoading(analyseBtn, btnText, true, 'Analysing...');

  try {
    const data = await callClaudeAPI(text);
    showResult(data);
  } catch (err) {
    showError(`Error: ${err.message}`);
  } finally {
    setLoading(analyseBtn, btnText, false, 'Analyse with AI');
  }
});

// ── Batch Analysis ────────────────────────────────────────────────────────────
batchBtn.addEventListener('click', async () => {
  const raw = batchInput.value.trim();
  if (!raw) { pulse(batchInput); return; }

  const texts = raw.split('\n').map(t => t.trim()).filter(t => t.length > 5);
  if (texts.length === 0) {
    batchResults.innerHTML = `<div class="error-msg">No valid lines found.</div>`;
    return;
  }
  if (texts.length > 20) {
    batchResults.innerHTML = `<div class="error-msg">Maximum 20 items per batch. Found ${texts.length}.</div>`;
    return;
  }

  setLoading(batchBtn, batchBtnText, true, `Analysing ${texts.length} items...`);
  batchResults.innerHTML = '';

  try {
    const data = await callClaudeBatch(texts);
    renderBatchResults(data);
  } catch (err) {
    batchResults.innerHTML = `<div class="error-msg">Error: ${err.message}</div>`;
  } finally {
    setLoading(batchBtn, batchBtnText, false, 'Analyse Batch');
  }
});

function renderBatchResults(data) {
  const summary = data.summary;
  let html = `
    <div class="batch-summary" style="margin-bottom: 1rem;">
      <div class="summary-item"><strong>${summary.total}</strong><span>Total</span></div>
      <div class="summary-item real"><strong>${summary.real_count}</strong><span>Real</span></div>
      <div class="summary-item fake"><strong>${summary.fake_count}</strong><span>Fake</span></div>
    </div>
  `;
  data.results.forEach((item, i) => {
    const cls = item.is_fake ? 'fake' : 'real';
    const label = item.is_fake ? '⚠ FAKE' : '✓ REAL';
    html += `
      <div class="batch-item" style="animation-delay: ${i * 0.05}s">
        <div class="batch-badge ${cls}">${label}</div>
        <div class="batch-text">${escapeHtml(item.text)}</div>
        <div class="batch-conf">${item.confidence}%</div>
      </div>
    `;
  });
  batchResults.innerHTML = html;
}

// ── Dashboard Metrics (static — no backend needed) ────────────────────────────
function renderFallbackBars() {
  const barsEl = document.getElementById('modelBars');
  if (!barsEl) return;
  const models = [
    ['Logistic Regression', 100.0],
    ['Naive Bayes', 94.4],
    ['Random Forest', 83.3],
  ];
  barsEl.innerHTML = models.map(([name, pct]) => `
    <div class="bar-row">
      <div class="bar-label">${name}</div>
      <div class="bar-track">
        <div class="bar-fill" data-pct="${pct}" style="width: 0%">
          <span class="bar-pct">${pct}%</span>
        </div>
      </div>
    </div>
  `).join('');

  const observer = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) {
      document.querySelectorAll('.bar-fill').forEach(bar => {
        const pct = bar.dataset.pct;
        setTimeout(() => { bar.style.width = pct + '%'; }, 100);
      });
      observer.disconnect();
    }
  }, { threshold: 0.3 });
  observer.observe(barsEl);
}

// ── Error display ─────────────────────────────────────────────────────────────
function showError(msg) {
  resultPlaceholder.style.display = 'none';
  resultContent.style.display = 'block';
  resultContent.innerHTML = `
    <div style="
      background: var(--danger-dim);
      border: 1px solid rgba(255,107,107,0.3);
      border-radius: var(--radius);
      padding: 1.5rem; color: var(--danger);
      font-size: 0.875rem; white-space: pre-line;
    ">
      <div style="font-family: var(--font-display); font-weight: 700; margin-bottom: 0.5rem;">⚠ Error</div>
      ${escapeHtml(msg)}
    </div>
  `;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setLoading(btn, textEl, loading, label) {
  btn.disabled = loading;
  if (loading) {
    textEl.innerHTML = `<span class="spinner"></span> ${label}`;
  } else {
    textEl.textContent = label;
  }
}

function pulse(el) {
  el.style.borderColor = 'var(--danger)';
  el.style.boxShadow = '0 0 0 3px var(--danger-dim)';
  setTimeout(() => {
    el.style.borderColor = '';
    el.style.boxShadow = '';
  }, 700);
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

newsInput.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') analyseBtn.click();
});

document.addEventListener('DOMContentLoaded', () => {
  renderFallbackBars();
});
