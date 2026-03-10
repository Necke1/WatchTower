const API = '';

// ── Tabs ──────────────────────────────────────────────────────────────────
function switchTab(name) {
  const order = ['analyze','demo','dictionary','corrections','patterns'];
  document.querySelectorAll('.tab-btn').forEach((b,i) => b.classList.toggle('active', order[i] === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'dictionary')  loadDictionary();
  if (name === 'corrections') loadCorrections();
  if (name === 'patterns')    loadPatterns();
}

// ── Option checkboxes ─────────────────────────────────────────────────────
function syncOption(chk, id) {
  document.getElementById(id).classList.toggle('selected', chk.checked);
}

// ── Input mode toggle ─────────────────────────────────────────────────────
let inputMode       = 'text';
let loadedFile      = null;
let _loadedFileText = '';   // full raw text of the loaded file, used by context IC mode

function setInputMode(mode) {
  inputMode = mode;
  document.getElementById('mode-text-btn').classList.toggle('active', mode === 'text');
  document.getElementById('mode-file-btn').classList.toggle('active', mode === 'file');
  document.getElementById('input-text-wrap').classList.toggle('hidden', mode !== 'text');
  document.getElementById('input-file-wrap').classList.toggle('hidden', mode !== 'file');
  if (mode === 'text') clearFile();
}

// ── Drag & drop ───────────────────────────────────────────────────────────
function onDragOver(e)  { e.preventDefault(); document.getElementById('drop-zone').classList.add('dragover'); }
function onDragLeave()  { document.getElementById('drop-zone').classList.remove('dragover'); }
function onDrop(e)      { e.preventDefault(); document.getElementById('drop-zone').classList.remove('dragover'); const f = e.dataTransfer.files[0]; if (f) processFile(f); }
function onFileChosen(e){ const f = e.target.files[0]; if (f) processFile(f); }

function processFile(f) {
  const ext = f.name.split('.').pop().toLowerCase();
  if (!['txt','json'].includes(ext)) { alert('Samo .txt i .json fajlovi su podržani.'); return; }
  if (f.size > 5 * 1024 * 1024)     { alert('Fajl je prevelik. Maksimalna veličina je 5 MB.'); return; }
  loadedFile      = f;
  _loadedFileText = '';
  document.getElementById('drop-zone').classList.add('hidden');
  document.getElementById('file-banner').classList.remove('hidden');
  document.getElementById('file-ext-badge').textContent  = ext.toUpperCase();
  document.getElementById('file-name-label').textContent = f.name;
  document.getElementById('file-size-label').textContent = formatBytes(f.size);

  // Always read the full raw text — used by context IC mode
  const rawReader = new FileReader();
  rawReader.onload = ev => { _loadedFileText = ev.target.result || ''; };
  rawReader.readAsText(f);

  if (ext === 'json') {
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const obj = JSON.parse(ev.target.result);
        const extracted = extractJsonText(obj);
        document.getElementById('json-preview-text').value = extracted || '(Nije pronađen tekst)';
      } catch { document.getElementById('json-preview-text').value = '(Neispravan JSON)'; }
      document.getElementById('json-preview-wrap').classList.remove('hidden');
    };
    reader.readAsText(f);
  } else {
    document.getElementById('json-preview-wrap').classList.add('hidden');
  }
}

function extractJsonText(obj) {
  const KEYS = ['text','tekst','content','sadrzaj','message','poruka','body','telo','description','opis'];
  function find(o) {
    if (!o || typeof o !== 'object') return null;
    if (Array.isArray(o)) { for (const i of o) { const r = find(i); if (r) return r; } return null; }
    for (const k of KEYS) if (k in o && typeof o[k] === 'string' && o[k].trim()) return o[k];
    for (const v of Object.values(o)) { const r = find(v); if (r) return r; }
    return null;
  }
  function collect(o, d=0) {
    if (d > 10) return [];
    if (typeof o === 'string' && o.trim()) return [o];
    if (Array.isArray(o)) return o.flatMap(i => collect(i, d+1));
    if (o && typeof o === 'object') return Object.values(o).flatMap(v => collect(v, d+1));
    return [];
  }
  return find(obj) || collect(obj).join('\n');
}

function clearFile() {
  loadedFile      = null;
  _loadedFileText = '';
  document.getElementById('file-input').value = '';
  document.getElementById('drop-zone').classList.remove('hidden');
  document.getElementById('file-banner').classList.add('hidden');
  document.getElementById('json-preview-wrap').classList.add('hidden');
  document.getElementById('json-preview-text').value = '';
}

function formatBytes(b) {
  if (b < 1024)       return b + ' B';
  if (b < 1048576)    return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(2) + ' MB';
}

// ── Loading overlay ───────────────────────────────────────────────────────
function showLoading(msg = 'Analizira se…') {
  document.getElementById('loading-msg').textContent = msg;
  document.getElementById('loading-overlay').classList.remove('hidden');
}
function hideLoading() {
  document.getElementById('loading-overlay').classList.add('hidden');
}

// ── Submit ────────────────────────────────────────────────────────────────
async function submitAnalysis() {
  if (inputMode === 'file') await analyzeFile();
  else                       await analyzeText();
}

// ── Analyze: plain text ───────────────────────────────────────────────────
async function analyzeText() {
  const text = document.getElementById('input-text').value.trim();
  if (!text) { alert('Unesite tekst za analizu.'); return; }
  const btn = document.getElementById('btn-analyze');
  btn.disabled = true;
  showLoading('Analizira se tekst…');
  let data = null;
  try {
    const resp = await fetch(`${API}/api/analyze`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        text,
        use_spellcheck: document.getElementById('chk-spellcheck').checked,
        auto_correct:   document.getElementById('chk-autocorrect').checked,
        output_json:    document.getElementById('chk-json').checked,
      }),
    });
    data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Greška servera');
    renderResults(data, null);
  } catch(e) { alert('Greška: ' + e.message); }
  finally    { hideLoading(); btn.disabled = false; }

  if (!data) return;

  // ── Post-analysis interactive modes ──────────────────────────────────────
  if (document.getElementById('chk-ic-spell').checked) {
    const errors = data?.spell_checking?.error_details || [];
    if (errors.length > 0) icStart(errors);
  }
  if (document.getElementById('chk-ic-context').checked) {
    const items = data?.context_items || [];
    if (items.length > 0) ctxStart(items);
  }
}

// ── Analyze: file upload ──────────────────────────────────────────────────
async function analyzeFile() {
  if (!loadedFile) { alert('Izaberite fajl za analizu.'); return; }
  const btn = document.getElementById('btn-analyze');
  btn.disabled = true;
  showLoading('Analizira se fajl…');
  let data = null;
  try {
    const form = new FormData();
    form.append('file',           loadedFile);
    form.append('use_spellcheck', document.getElementById('chk-spellcheck').checked  ? 'true' : 'false');
    form.append('auto_correct',   document.getElementById('chk-autocorrect').checked ? 'true' : 'false');
    form.append('output_json',    document.getElementById('chk-json').checked        ? 'true' : 'false');
    const resp = await fetch(`${API}/api/analyze/file`, {method:'POST', body: form});
    data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Greška servera');
    renderResults(data, data.file_info);
  } catch(e) { alert('Greška: ' + e.message); }
  finally    { hideLoading(); btn.disabled = false; }

  if (!data) return;

  if (document.getElementById('chk-ic-spell').checked) {
    const errors = data?.spell_checking?.error_details || [];
    if (errors.length > 0) icStart(errors);
  }
  if (document.getElementById('chk-ic-context').checked) {
    const items = data?.context_items || [];
    if (items.length > 0) ctxStart(items);
  }
}

function clearAnalysis() {
  document.getElementById('input-text').value = '';
  clearFile();
  document.getElementById('results-section').classList.add('hidden');
}

// ── Risk helpers ──────────────────────────────────────────────────────────
function riskClass(level) {
  if (level.includes('VISOK'))     return 'risk-VISOK';
  if (level.includes('SREDNJI'))   return 'risk-SREDNJI';
  if (level.includes('NIZAK'))     return 'risk-NIZAK';
  if (level.includes('MINIMALAN')) return 'risk-MINIMALAN';
  return 'risk-BEZ';
}
function riskLvClass(level) {
  if (level.includes('VISOK'))   return 'lv-VISOK';
  if (level.includes('SREDNJI')) return 'lv-SREDNJI';
  if (level.includes('NIZAK'))   return 'lv-NIZAK';
  return 'lv-MIN';
}

// ── Render results ────────────────────────────────────────────────────────
function renderResults(data, fileInfo) {
  const an = data.analysis;
  const sp = data.spell_checking;
  document.getElementById('results-section').classList.remove('hidden');
  document.getElementById('results-section').scrollIntoView({behavior:'smooth', block:'start'});

  // ── File info ────────────────────────────────────────────────────────────
  const fileCard = document.getElementById('res-file-info-card');
  if (fileInfo) {
    fileCard.classList.remove('hidden');
    const notes = (fileInfo.extraction_notes || []).join(' ');
    document.getElementById('res-file-info-body').innerHTML = `
      <div class="file-info-section">
        <div class="file-info-row"><span class="file-info-key">Naziv fajla</span><span class="file-info-val">${esc(fileInfo.filename)}</span></div>
        <div class="file-info-row"><span class="file-info-key">Tip</span><span class="file-info-val">${esc((fileInfo.extension||'').toUpperCase())}</span></div>
        <div class="file-info-row"><span class="file-info-key">Veličina</span><span class="file-info-val">${formatBytes(fileInfo.size_bytes)}</span></div>
        <div class="file-info-row"><span class="file-info-key">Karaktera izvučeno</span><span class="file-info-val">${fileInfo.chars_extracted}</span></div>
        ${notes ? `<div class="file-info-row"><span class="file-info-key">Napomena</span><span class="file-info-val" style="font-family:inherit;">${esc(notes)}</span></div>` : ''}
      </div>`;
  } else {
    fileCard.classList.add('hidden');
  }

  // ── Chat analysis ────────────────────────────────────────────────────────
  const chatCard = document.getElementById('res-chat-card');
  if (data.chat_analysis) {
    chatCard.classList.remove('hidden');
    renderChatAnalysis(data.chat_analysis);
  } else {
    chatCard.classList.add('hidden');
  }

  // ── Hero — for chat exports show chat-level risk, not combined-text risk ─
  const heroLevel = (data.chat_analysis?.overall_risk_level) || an.risk_level;
  const heroDesc  = (data.chat_analysis?.overall_risk_description) || an.risk_description;
  const heroScore = (data.chat_analysis?.stats?.weighted_score_sum != null)
                    ? data.chat_analysis.stats.weighted_score_sum
                    : an.total_score;

  document.getElementById('res-risk-badge').innerHTML =
    `<span class="risk-badge ${riskClass(heroLevel)}">${heroLevel}</span>`;
  document.getElementById('res-risk-desc').textContent = heroDesc;
  document.getElementById('res-score').textContent     = data.chat_analysis ? heroScore : an.total_score;
  document.getElementById('res-words').textContent     = an.total_words;
  document.getElementById('res-unique').textContent    = an.unique_terms_count;
  document.getElementById('res-density').textContent   = an.term_density.toFixed(2) + '%';
  document.getElementById('res-time').textContent      = data.statistics.processing_time_seconds.toFixed(3) + 's';

  // ── Terms ─────────────────────────────────────────────────────────────────
  const termsBody = document.getElementById('res-terms-body');
  const weights   = data.term_weights || {};
  document.getElementById('res-terms-count').textContent = an.unique_terms_count;

  if (Object.keys(an.term_frequencies).length === 0) {
    termsBody.innerHTML = '<div class="empty-state"><div class="empty-icon">✅</div>Nisu pronađeni rizični termini.</div>';
  } else {
    const maxW = Math.max(...Object.values(weights), 1);
    const rows = Object.entries(an.term_frequencies)
      .sort((a,b) => b[1] - a[1])
      .map(([term, count]) => {
        const w = weights[term] || 1;
        return `<tr>
          <td><span class="term-word">${esc(term)}</span></td>
          <td>${count}×</td>
          <td><div>${w}</div><div class="weight-bar" style="width:${Math.round((w/maxW)*100)}%"></div></td>
        </tr>`;
      }).join('');
    termsBody.innerHTML = `<table class="terms-table"><thead><tr><th>Term</th><th>Pojave</th><th>Težina</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  // ── Recommendations ──────────────────────────────────────────────────────
  const recs = (data.chat_analysis?.overall_recommendations) || an.recommendations;
  document.getElementById('res-recs').innerHTML = recs
    .map((r,i) => `<li><span class="rec-num">${i+1}</span>${esc(r)}</li>`).join('');

  // ── Spell check ──────────────────────────────────────────────────────────
  const spBadge = document.getElementById('res-spell-badge');
  const spBody  = document.getElementById('res-spell-body');
  if (!data.metadata.spell_check_enabled) {
    spBadge.className = 'pill pill-muted'; spBadge.textContent = 'Isključeno';
    spBody.innerHTML  = '<p style="color:var(--muted);font-size:.875rem;">Provera pravopisa je isključena.</p>';
  } else if (sp.errors_found === 0) {
    spBadge.className = 'pill pill-green'; spBadge.textContent = 'Bez grešaka';
    spBody.innerHTML  = '<p style="color:var(--muted);font-size:.875rem;">Nisu pronađene pravopisne greške.</p>';
  } else {
    spBadge.className = 'pill pill-red'; spBadge.textContent = `${sp.errors_found} greška`;
    const corrInfo = sp.corrections_made > 0
      ? `<p style="font-size:.82rem;color:var(--muted);margin-bottom:.75rem;">Automatski ispravljeno: <strong>${sp.corrections_made}</strong> reči. Ispravljeni tekst: <code style="font-family:'DM Mono',monospace;">${esc(sp.corrected_text)}</code></p>` : '';
    const errHtml = sp.error_details.map(e => {
      const suggs = (e.suggestions||[]).slice(0,4).map(s=>`<span class="suggestion-chip">${esc(s)}</span>`).join('');
      return `<div style="margin-bottom:.6rem;"><span class="error-chip">${esc(e.word)}</span>${suggs ? '<span style="font-size:.78rem;color:var(--muted);margin:0 .3rem;">→</span>'+suggs:''}</div>`;
    }).join('');
    spBody.innerHTML = corrInfo + errHtml;
  }

  // ── Console + report ──────────────────────────────────────────────────────
  document.getElementById('res-console').textContent = data.console_output    || '(nema izlaza)';
  document.getElementById('res-report').textContent  = data.formatted_report  || '';
}

// ── Render chat analysis panel ────────────────────────────────────────────
function renderChatAnalysis(ca) {
  const stats   = ca.stats     || {};
  const meta    = ca.chat_meta  || {};
  const flagged = ca.flagged_messages || [];

  // Banner
  document.getElementById('res-chat-banner-text').innerHTML =
    `Chat export: <strong>${esc(meta.chat_name || 'Nepoznat chat')}</strong>
     &nbsp;·&nbsp; Tip: <strong>${esc(meta.chat_type || '—')}</strong>
     &nbsp;·&nbsp; ID: <code style="font-family:'DM Mono',monospace;">${esc(String(meta.chat_id || '—'))}</code>`;

  // Overall chat risk (exponential model)
  const overall = ca.overall_risk_level;
  if (overall) {
    document.getElementById('res-chat-overall').innerHTML = `
      <div class="chat-overall-risk">
        <div>
          <div class="chat-overall-label">Ukupni rizik chata (eksponencijalni model)</div>
          <span class="risk-badge ${riskClass(overall)}">${esc(overall)}</span>
          <div class="chat-weighted-info">
            Ponderisani skor: <strong>${stats.weighted_score_sum ?? '—'}</strong>
            &nbsp;·&nbsp; Ponderisani prosek: <strong>${stats.weighted_avg ?? '—'}</strong>
            &nbsp;·&nbsp; Maks. poruka: <strong>${stats.max_score ?? '—'}</strong>
          </div>
        </div>
      </div>`;
  } else {
    document.getElementById('res-chat-overall').innerHTML = '';
  }

  // Stats row
  document.getElementById('res-chat-stats').innerHTML = `
    <div class="chat-stat"><div class="chat-stat-val">${stats.total_messages_in_export ?? '—'}</div><div class="chat-stat-lbl">Ukupno poruka</div></div>
    <div class="chat-stat"><div class="chat-stat-val">${stats.analysable_messages ?? '—'}</div><div class="chat-stat-lbl">Analizirano</div></div>
    <div class="chat-stat"><div class="chat-stat-val" style="color:var(--red)">${stats.flagged_messages_count ?? '—'}</div><div class="chat-stat-lbl">Sumnjivih</div></div>
    <div class="chat-stat"><div class="chat-stat-val">${stats.average_score ?? '—'}</div><div class="chat-stat-lbl">Prosečan skor</div></div>`;

  // Top users table
  const users = stats.top_users_by_score || [];
  if (users.length === 0) {
    document.getElementById('res-top-users').innerHTML =
      '<p style="font-size:.82rem;color:var(--muted);">Nema podataka.</p>';
  } else {
    const rows = users.map(u => `<tr>
      <td><strong>${esc(u.user_name)}</strong></td>
      <td style="font-family:'DM Mono',monospace;font-size:.78rem;color:var(--muted);">${esc(u.user_id)}</td>
      <td>${u.messages}</td>
      <td style="color:var(--red);font-weight:600;">${u.flagged}</td>
      <td><strong>${u.total_score}</strong></td>
    </tr>`).join('');
    document.getElementById('res-top-users').innerHTML =
      `<table class="top-users-table">
        <thead><tr><th>Korisnik</th><th>ID</th><th>Poruka</th><th>Sumnjivih</th><th>Ukupan skor</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  // Flagged count badge
  document.getElementById('res-flagged-count').textContent =
    flagged.length + (flagged.length === 1 ? ' poruka' : ' poruka');

  // Flagged messages list — already sorted by backend (VISOK first)
  const listEl = document.getElementById('res-flagged-list');
  if (flagged.length === 0) {
    listEl.innerHTML = `<div class="empty-state"><div class="empty-icon">✅</div>Nisu pronađene sumnjive poruke.</div>`;
    return;
  }

  listEl.innerHTML = flagged.map(msg => {
    const termsHtml = Object.entries(msg.term_frequencies || {})
      .sort((a,b) => b[1]-a[1])
      .map(([t, c]) => {
        const w = (msg.term_weights||{})[t] || 1;
        return `<span class="fterm">${esc(t)} <span class="fterm-w">×${c} w${w}</span></span>`;
      }).join('');

    const verdictHtml = msg.verdict
      ? `<div class="flagged-verdict">${esc(msg.verdict)}</div>` : '';

    return `<div class="flagged-card ${riskLvClass(msg.risk_level)}">
      <div class="flagged-head">
        <div class="flagged-who">
          <span class="flagged-name">${esc(msg.user_name)}</span>
          <span class="flagged-id">${esc(msg.user_id ? 'ID: ' + msg.user_id : '')}</span>
          <span class="flagged-date">${esc(msg.date)}</span>
        </div>
        <span class="risk-badge ${riskClass(msg.risk_level)}">${esc(msg.risk_level)} · skor ${msg.total_score}</span>
      </div>
      <div class="flagged-body">
        <div class="flagged-text">${esc(msg.text)}</div>
        ${verdictHtml}
        <div class="flagged-terms">${termsHtml}</div>
      </div>
    </div>`;
  }).join('');
}

// ══════════════════════════════════════════════════════════════════
// INTERACTIVE SPELL-CHECK MODE
// Walks through spell errors one by one — mirrors CLI --interactive-correct
// ══════════════════════════════════════════════════════════════════

let _icErrors  = [];
let _icIndex   = 0;
let _icRating  = 0;
let _icChoice  = null;

function icStart(errors) {
  // Only show dialog for errors that have suggestions or are correctable
  _icErrors = (errors || []).filter(e => e.word && e.word.length > 1);
  _icIndex  = 0;
  _icRating = 0;
  _icChoice = null;
  if (_icErrors.length === 0) return;
  document.getElementById('ic-overlay').classList.remove('hidden');
  icShowCurrent();
}

function icShowCurrent() {
  if (_icIndex >= _icErrors.length) { icFinish(); return; }
  const err = _icErrors[_icIndex];
  const pct = Math.round(((_icIndex + 1) / _icErrors.length) * 100);

  document.getElementById('ic-progress').textContent =
    `${_icIndex + 1} / ${_icErrors.length}`;
  document.getElementById('ic-prog-fill').style.width = pct + '%';
  document.getElementById('ic-word').textContent = err.word;
  document.getElementById('ic-manual-input').value = '';
  document.getElementById('ic-rating-row').classList.add('hidden');
  document.getElementById('ic-confirm-btn').disabled = true;
  _icChoice = null;
  _icRating = 0;
  icUpdateStars();

  const suggs = (err.suggestions || []).slice(0, 5);
  document.getElementById('ic-sugg-list').innerHTML = suggs.length
    ? suggs.map((s, i) => `
        <button class="ic-sugg-btn" onclick='icChooseSugg(${JSON.stringify(s)})' data-word="${esc(s)}">
          <span>${esc(s)}</span>
          <span class="ic-sugg-num">${i + 1}</span>
        </button>`).join('')
    : `<div style="font-size:.82rem;color:var(--muted);padding:.35rem 0;">Nema predloga za ovu reč.</div>`;
}

function icChooseSugg(word) {
  _icChoice = word;
  document.querySelectorAll('.ic-sugg-btn').forEach(b =>
    b.classList.toggle('chosen', b.dataset.word === word)
  );
  document.getElementById('ic-rating-row').classList.remove('hidden');
  document.getElementById('ic-confirm-btn').disabled = false;
}

function icApplyManual() {
  const val = document.getElementById('ic-manual-input').value.trim();
  if (!val) return;
  _icChoice = val;
  document.querySelectorAll('.ic-sugg-btn').forEach(b => b.classList.remove('chosen'));
  document.getElementById('ic-rating-row').classList.remove('hidden');
  document.getElementById('ic-confirm-btn').disabled = false;
}

function icSetRating(n) {
  _icRating = n;
  icUpdateStars();
}

function icUpdateStars() {
  document.querySelectorAll('.ic-star').forEach((s, i) =>
    s.classList.toggle('lit', i < _icRating)
  );
}

async function icConfirm() {
  if (!_icChoice) return;
  const err = _icErrors[_icIndex];
  try {
    await fetch(`${API}/api/corrections`, {
      method:  'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        original:  err.word,
        corrected: _icChoice,
        rating:    _icRating || 2,
      }),
    });
  } catch(e) { console.warn('Could not save correction:', e); }
  _icIndex++;
  icShowCurrent();
}

function icSkip() {
  _icIndex++;
  icShowCurrent();
}

function icFinish() {
  document.getElementById('ic-overlay').classList.add('hidden');
  // Refresh the corrections tab if it's currently showing
  loadCorrections();
}


// ══════════════════════════════════════════════════════════════════
// INTERACTIVE CONTEXT-CHECK MODE
// All context data comes from the backend (context_items in API response).
// JS only renders — no text processing, no regex, no message map building.
// ══════════════════════════════════════════════════════════════════

let _ctxItems  = [];   // context_items array from API response
let _ctxIndex  = 0;

function ctxStart(items) {
  _ctxItems = items || [];
  _ctxIndex = 0;
  if (_ctxItems.length === 0) return;
  document.getElementById('ctx-overlay').classList.remove('hidden');
  ctxShowCurrent();
}

function ctxShowCurrent() {
  if (_ctxIndex >= _ctxItems.length) { ctxFinish(); return; }
  const item = _ctxItems[_ctxIndex];
  const pct  = Math.round(((_ctxIndex + 1) / _ctxItems.length) * 100);

  document.getElementById('ctx-progress').textContent =
    `${_ctxIndex + 1} / ${_ctxItems.length}`;
  document.getElementById('ctx-prog-fill').style.width = pct + '%';
  document.getElementById('ctx-term').textContent   = item.term;
  document.getElementById('ctx-count').textContent  =
    item.occ_total > 1 ? `pojava ${item.occurrence_n}/${item.occ_total}` : '';
  document.getElementById('ctx-weight').textContent = `težina: ${item.weight}`;
  document.getElementById('ctx-score-input').value  = '';

  // Build the context box — snippet comes pre-built from backend,
  // just convert [term] brackets to <mark> tags for HTML display.
  let snippetHtml = esc(item.snippet)
    .replace(/\[([^\]]+)\]/g, '<mark>$1</mark>');

  let header = '';
  if (item.label) {
    const riskTag = item.risk
      ? ` <span class="risk-badge ${riskClass(item.risk)}" style="font-size:.7rem;padding:.1rem .45rem;">${esc(item.risk)} · ${item.score}</span>`
      : '';
    header = `<div style="font-size:.75rem;color:var(--muted);margin-bottom:.35rem;">${esc(item.label)}${riskTag}</div>`;
  }

  document.getElementById('ctx-message').innerHTML = header + snippetHtml;
}

async function ctxFeedback(feedbackType) {
  const item = _ctxItems[_ctxIndex];
  const customMultiplierRaw = document.getElementById('ctx-score-input').value.trim();
  const customMultiplier    = customMultiplierRaw !== '' ? parseFloat(customMultiplierRaw) : null;

  if (customMultiplierRaw !== '' && (isNaN(customMultiplier) || customMultiplier < 0 || customMultiplier > 10)) {
    alert('Multiplikator mora biti broj između 0.0 i 10.0.');
    return;
  }

  try {
    // unit_text is the exact message/sentence text — correct context for the backend
    const body = { text: item.unit_text, term: item.term, feedback: feedbackType };
    if (customMultiplier !== null) body.custom_multiplier = customMultiplier;
    await fetch(`${API}/api/feedback`, {
      method:  'POST',
      headers: {'Content-Type':'application/json'},
      body:    JSON.stringify(body),
    });
  } catch(e) { console.warn('Could not save context feedback:', e); }

  _ctxIndex++;
  ctxShowCurrent();
}

function ctxSkip() {
  _ctxIndex++;
  ctxShowCurrent();
}

function ctxFinish() {
  document.getElementById('ctx-overlay').classList.add('hidden');
}


// ── Demo ──────────────────────────────────────────────────────────────────
async function runDemo() {
  const btn = document.getElementById('btn-demo');
  btn.disabled = true; btn.textContent = 'Pokrenuto…';
  const container = document.getElementById('demo-results');
  container.innerHTML = '<div class="loading"><div class="spinner"></div> Analizira se…</div>';
  try {
    const resp = await fetch(`${API}/api/demo`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        use_spellcheck: document.getElementById('chk-demo-spellcheck').checked,
        auto_correct:   document.getElementById('chk-demo-autocorrect').checked,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail);
    const html = data.results.map(r => {
      const an = r.analysis;
      const termsHtml = Object.entries(an.term_frequencies||{})
        .map(([t,c])=>`<span class="error-chip" style="background:var(--accent-l);color:var(--accent);">${esc(t)} ×${c}</span>`)
        .join('') || '<span style="color:var(--muted);font-size:.8rem;">Bez termina</span>';
      return `<div class="demo-card">
        <div class="demo-card-head"><span class="demo-card-name">${esc(r.name)}</span><span class="risk-badge ${riskClass(an.risk_level)}">${an.risk_level}</span></div>
        <div class="demo-card-body">
          <div class="demo-card-text">"${esc(r.text)}"</div>
          <div style="margin-bottom:.5rem;">${termsHtml}</div>
          <div class="demo-stats"><span>Skor: <strong>${an.total_score}</strong></span><span>Termina: <strong>${an.unique_terms_count}</strong></span><span>Reči: <strong>${an.total_words}</strong></span><span>Gustina: <strong>${an.term_density.toFixed(2)}%</strong></span></div>
        </div></div>`;
    }).join('');
    container.innerHTML = `<div class="demo-grid">${html}</div>`;
  } catch(e) { container.innerHTML = `<div class="empty-state" style="color:var(--red);">Greška: ${e.message}</div>`; }
  finally    { btn.disabled = false; btn.textContent = 'Pokreni demo'; }
}

// ── Dictionary ────────────────────────────────────────────────────────────
let dictLoaded = false;
async function loadDictionary() {
  if (dictLoaded) return;
  const content = document.getElementById('dict-content');
  content.innerHTML = '<div class="loading"><div class="spinner"></div> Učitavanje…</div>';
  try {
    const r = await fetch(`${API}/api/dictionary`);
    const d = await r.json();
    document.getElementById('dict-info').textContent = `${d.size} termina  •  ${d.file_path}`;
    const chips = d.terms.map(t=>`<div class="dict-term-chip"><span class="word">${esc(t.term)}</span><span class="wt">w:${t.weight}</span></div>`).join('');
    content.innerHTML = `<div class="dict-terms-grid" style="max-height:420px;overflow-y:auto;padding:1rem;">${chips}</div>`;
    dictLoaded = true;
  } catch(e) { content.innerHTML = `<div class="empty-state" style="color:var(--red);">Greška pri učitavanju rečnika.</div>`; }
}

async function reloadDictionary() {
  dictLoaded = false;
  await fetch(`${API}/api/dictionary/reload`, {method:'POST'});
  await loadDictionary();
  await checkHealth();
}

// ── Corrections ───────────────────────────────────────────────────────────
async function loadCorrections() {
  const content = document.getElementById('corrections-content');
  content.innerHTML = '<div class="loading"><div class="spinner"></div> Učitavanje…</div>';
  try {
    const r = await fetch(`${API}/api/corrections`);
    const d = await r.json();
    if (d.corrections.length === 0) {
      content.innerHTML = `<div class="empty-state"><div class="empty-icon">✏️</div>Nema naučenih korekcija.<br><span style="font-size:.8rem;">Koristite interaktivni mod korekcije pravopisa ili <code style="font-family:'DM Mono',monospace;">--interactive-correct</code> u CLI.</span></div>`;
      return;
    }
    const rows = d.corrections.map(c => {
      const stars = '★'.repeat(c.rating) + '☆'.repeat(4 - c.rating);
      return `<tr><td><span class="term-word">${esc(c.original)}</span></td><td><span class="term-word">${esc(c.corrected)}</span></td><td>${stars}</td><td>${c.count}</td><td style="color:var(--muted);font-size:.78rem;">${esc(c.last_used)}</td></tr>`;
    }).join('');
    content.innerHTML = `<table class="corrections-table"><thead><tr><th>Original</th><th>Korekcija</th><th>Ocena</th><th>Br. upotreba</th><th>Poslednja upotreba</th></tr></thead><tbody>${rows}</tbody></table>`;
  } catch(e) { content.innerHTML = `<div class="empty-state" style="color:var(--red);">Greška pri učitavanju.</div>`; }
}

async function clearCorrections() {
  if (!confirm('Obrisati sve naučene korekcije? Ova akcija je nepovratna.')) return;
  try {
    const r = await fetch(`${API}/api/corrections`, { method: 'DELETE' });
    const d = await r.json();
    alert(d.message || 'Korekcije obrisane.');
  } catch(e) { alert('Greška pri brisanju korekcija.'); }
  loadCorrections();
}

// ── Patterns ──────────────────────────────────────────────────────────────

const _MULTIPLIER_META = {
  ignore:    { label: 'Ignoriši',          cls: 'pat-ignore',  icon: '🚫' },
  dampen:    { label: 'Prigušivač',        cls: 'pat-dampen',  icon: '📉' },
  amplify:   { label: 'Pojačivač',         cls: 'pat-amplify', icon: '📈' },
  confirmed: { label: 'Potvrđena pretnja', cls: 'pat-confirm', icon: '⚠️'  },
  neutral:   { label: 'Neutralno',         cls: 'pat-neutral', icon: '➖' },
};

function _patMeta(mult) {
  if (mult <= 0.1)  return _MULTIPLIER_META.ignore;
  if (mult < 1.0)   return _MULTIPLIER_META.dampen;
  if (mult >= 1.5)  return _MULTIPLIER_META.amplify;
  if (mult >= 1.99) return _MULTIPLIER_META.confirmed;
  return _MULTIPLIER_META.neutral;
}

function _confidenceBar(confidence) {
  const pct  = Math.round(confidence * 100);
  const col  = pct >= 80 ? 'var(--green)' : pct >= 50 ? '#f59e0b' : 'var(--muted)';
  return `<div class="pat-conf-wrap">
    <div class="pat-conf-bar" style="width:${pct}%;background:${col};"></div>
    <span class="pat-conf-label">${pct}%</span>
  </div>`;
}

async function loadPatterns() {
  const content = document.getElementById('patterns-content');
  const summary = document.getElementById('patterns-summary');
  content.innerHTML = '<div class="loading"><div class="spinner"></div> Učitavanje…</div>';
  summary.classList.add('hidden');
  try {
    const r = await fetch(`${API}/api/patterns`);
    const d = await r.json();
    const patterns = d.patterns || [];

    if (patterns.length === 0) {
      content.innerHTML = `<div class="empty-state">
        <div class="empty-icon">🧠</div>
        Nema naučenih paterna.<br>
        <span style="font-size:.8rem;">Koristite interaktivni mod analize konteksta da dodate paterne.</span>
      </div>`;
      return;
    }

    // Summary strip
    const counts = { ignore: 0, dampen: 0, amplify: 0, neutral: 0 };
    for (const p of patterns) {
      const m = p.multiplier;
      if (m <= 0.1)       counts.ignore++;
      else if (m < 1.0)   counts.dampen++;
      else if (m >= 1.5)  counts.amplify++;
      else                counts.neutral++;
    }
    summary.classList.remove('hidden');
    summary.innerHTML = `
      <div class="pat-summary-row">
        <div class="pat-summary-chip pat-ignore-chip">
          ${_MULTIPLIER_META.ignore.icon} ${counts.ignore} Ignoriši
        </div>
        <div class="pat-summary-chip pat-dampen-chip">
          ${_MULTIPLIER_META.dampen.icon} ${counts.dampen} Prigušivač
        </div>
        <div class="pat-summary-chip pat-amplify-chip">
          ${_MULTIPLIER_META.amplify.icon} ${counts.amplify} Pojačivač
        </div>
        <div class="pat-summary-chip pat-neutral-chip">
          ${_MULTIPLIER_META.neutral.icon} ${counts.neutral} Neutralno
        </div>
        <div class="pat-summary-chip" style="margin-left:auto;color:var(--muted);background:var(--bg);border-color:var(--border);">
          Ukupno: <strong>${patterns.length}</strong>
        </div>
      </div>`;

    // Pattern cards grouped by term
    const byTerm = {};
    for (const p of patterns) {
      if (!byTerm[p.term]) byTerm[p.term] = [];
      byTerm[p.term].push(p);
    }

    const html = Object.entries(byTerm).map(([term, plist]) => {
      const rows = plist.map(p => {
        const meta = _patMeta(p.multiplier);
        const ctx  = (p.context || []).map(w =>
          `<span class="pat-ctx-word">${esc(w)}</span>`).join('');
        return `<div class="pat-row">
          <div class="pat-row-left">
            <span class="pat-type-badge ${meta.cls}">${meta.icon} ${meta.label}</span>
            <div class="pat-ctx-words">${ctx || '<span style="color:var(--muted);font-size:.75rem;">bez kontekstnih reči</span>'}</div>
          </div>
          <div class="pat-row-right">
            <div class="pat-mult">×${p.multiplier.toFixed(2)}</div>
            <div class="pat-eff" title="Efektivni multiplikator na osnovu broja upotreba">
              ef. ×${p.effective_at_full.toFixed(2)}
            </div>
            ${_confidenceBar(p.confidence)}
            <div class="pat-meta">
              ${p.count}× · ${esc(p.last_used)}
            </div>
          </div>
        </div>`;
      }).join('');

      return `<div class="pat-group">
        <div class="pat-group-header">
          <span class="pat-term">${esc(term)}</span>
          <span class="pat-count-badge">${plist.length} patern${plist.length === 1 ? '' : 'a'}</span>
        </div>
        <div class="pat-rows">${rows}</div>
      </div>`;
    }).join('');

    content.innerHTML = `<div class="pat-list">${html}</div>`;

  } catch(e) {
    content.innerHTML = `<div class="empty-state" style="color:var(--red);">Greška pri učitavanju paterna.</div>`;
  }
}

async function clearPatterns() {
  if (!confirm('Obrisati sve naučene paterne konteksta? Ova akcija je nepovratna.')) return;
  try {
    const r = await fetch(`${API}/api/patterns`, { method: 'DELETE' });
    const d = await r.json();
    alert(d.message || 'Paterne obrisane.');
  } catch(e) { alert('Greška pri brisanju paterna.'); }
  loadPatterns();
}

// ── Health ────────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/api/health`);
    const d = await r.json();
    const setPill = (id, ok) => {
      document.getElementById(id).classList.toggle('ok',   ok);
      document.getElementById(id).classList.toggle('warn', !ok);
    };
    setPill('pill-api',       d.status === 'ok');
    setPill('pill-classla',   d.classla_available);
    setPill('pill-phunspell', d.phunspell_available);
    document.getElementById('dict-size-label').textContent = `${d.dictionary_size} termina`;
  } catch { document.getElementById('pill-api').classList.add('warn'); }
}

checkHealth();

// ── Helpers ───────────────────────────────────────────────────────────────
function esc(s) {
  return String(s ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function toggleConsole() { document.getElementById('console-wrap').classList.toggle('hidden'); }
function copyReport()    { navigator.clipboard.writeText(document.getElementById('res-report').textContent).then(() => alert('Izveštaj kopiran u clipboard.')); }

// Ctrl+Enter to submit
document.getElementById('input-text').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) submitAnalysis();
});