const API = '';

// ── Tabs ──────────────────────────────────────────────────────────────────
function switchTab(name) {
  const order = ['analyze','demo','dictionary','corrections'];
  document.querySelectorAll('.tab-btn').forEach((b,i) => b.classList.toggle('active', order[i] === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'dictionary')  loadDictionary();
  if (name === 'corrections') loadCorrections();
}

// ── Option checkboxes ─────────────────────────────────────────────────────
function syncOption(chk, id) {
  document.getElementById(id).classList.toggle('selected', chk.checked);
}

// ── Input mode toggle ─────────────────────────────────────────────────────
let inputMode  = 'text';
let loadedFile = null;

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
  loadedFile = f;
  document.getElementById('drop-zone').classList.add('hidden');
  document.getElementById('file-banner').classList.remove('hidden');
  document.getElementById('file-ext-badge').textContent  = ext.toUpperCase();
  document.getElementById('file-name-label').textContent = f.name;
  document.getElementById('file-size-label').textContent = formatBytes(f.size);
  if (ext === 'json') {
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const extracted = extractJsonText(JSON.parse(ev.target.result));
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
  loadedFile = null;
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
  btn.disabled = true; btn.textContent = 'Analizira se…';
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
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Greška servera');
    renderResults(data, null);
  } catch(e) { alert('Greška: ' + e.message); }
  finally    { btn.disabled = false; btn.textContent = 'Analiziraj'; }
}

// ── Analyze: file upload ──────────────────────────────────────────────────
async function analyzeFile() {
  if (!loadedFile) { alert('Izaberite fajl za analizu.'); return; }
  const btn = document.getElementById('btn-analyze');
  btn.disabled = true; btn.textContent = 'Analizira se…';
  try {
    const form = new FormData();
    form.append('file',           loadedFile);
    form.append('use_spellcheck', document.getElementById('chk-spellcheck').checked  ? 'true' : 'false');
    form.append('auto_correct',   document.getElementById('chk-autocorrect').checked ? 'true' : 'false');
    form.append('output_json',    document.getElementById('chk-json').checked        ? 'true' : 'false');
    const resp = await fetch(`${API}/api/analyze/file`, {method:'POST', body: form});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Greška servera');
    renderResults(data, data.file_info);
  } catch(e) { alert('Greška: ' + e.message); }
  finally    { btn.disabled = false; btn.textContent = 'Analiziraj'; }
}

function clearAnalysis() {
  document.getElementById('input-text').value = '';
  clearFile();
  document.getElementById('results-section').classList.add('hidden');
}

// ── Render results ────────────────────────────────────────────────────────
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

function renderResults(data, fileInfo) {
  const an = data.analysis;
  const sp = data.spell_checking;
  document.getElementById('results-section').classList.remove('hidden');
  document.getElementById('results-section').scrollIntoView({behavior:'smooth', block:'start'});

  // ── File info card ──────────────────────────────────────────────────────
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

  // ── Chat analysis panel (only for chat exports) ─────────────────────────
  const chatCard = document.getElementById('res-chat-card');
  if (data.chat_analysis) {
    chatCard.classList.remove('hidden');
    renderChatAnalysis(data.chat_analysis);
  } else {
    chatCard.classList.add('hidden');
  }

  // ── Hero ────────────────────────────────────────────────────────────────
  document.getElementById('res-risk-badge').innerHTML =
    `<span class="risk-badge ${riskClass(an.risk_level)}">${an.risk_level}</span>`;
  document.getElementById('res-risk-desc').textContent = an.risk_description;
  document.getElementById('res-score').textContent     = an.total_score;
  document.getElementById('res-words').textContent     = an.total_words;
  document.getElementById('res-unique').textContent    = an.unique_terms_count;
  document.getElementById('res-density').textContent   = an.term_density.toFixed(2) + '%';
  document.getElementById('res-time').textContent      = data.statistics.processing_time_seconds.toFixed(3) + 's';

  // ── Terms ───────────────────────────────────────────────────────────────
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

  // ── Recommendations ─────────────────────────────────────────────────────
  document.getElementById('res-recs').innerHTML = an.recommendations
    .map((r,i) => `<li><span class="rec-num">${i+1}</span>${esc(r)}</li>`).join('');

  // ── Spell check ─────────────────────────────────────────────────────────
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

  // ── Console + report ────────────────────────────────────────────────────
  document.getElementById('res-console').textContent = data.console_output    || '(nema izlaza)';
  document.getElementById('res-report').textContent  = data.formatted_report  || '';
}

// ── Render chat analysis panel ────────────────────────────────────────────
function renderChatAnalysis(ca) {
  const stats    = ca.stats    || {};
  const meta     = ca.chat_meta || {};
  const flagged  = ca.flagged_messages || [];

  // Banner
  document.getElementById('res-chat-banner-text').innerHTML =
    `Chat export: <strong>${esc(meta.chat_name || 'Nepoznat chat')}</strong>
     &nbsp;·&nbsp; Tip: <strong>${esc(meta.chat_type || '—')}</strong>
     &nbsp;·&nbsp; ID: <code style="font-family:'DM Mono',monospace;">${esc(String(meta.chat_id || '—'))}</code>`;

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

  // Flagged messages list
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
      content.innerHTML = `<div class="empty-state"><div class="empty-icon">✏️</div>Nema naučenih korekcija.<br><span style="font-size:.8rem;">Koristite <code style="font-family:'DM Mono',monospace;">--interactive-correct</code> u CLI.</span></div>`;
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
  const r = await fetch(`${API}/api/corrections`, {
    method: 'DELETE', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({confirm: true}),
  });
  const d = await r.json();
  alert(d.message);
  loadCorrections();
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
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function toggleConsole() { document.getElementById('console-wrap').classList.toggle('hidden'); }
function copyReport()    { navigator.clipboard.writeText(document.getElementById('res-report').textContent).then(() => alert('Izveštaj kopiran u clipboard.')); }

// Ctrl+Enter to submit
document.getElementById('input-text').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) submitAnalysis();
});