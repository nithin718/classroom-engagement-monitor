// AI Classroom Engagement Portal - Client Application
let isRunning = false;
let currentMode = "SIMULATION";
let simState = null;

// Format seconds into MM:SS
function formatTime(sec) {
  const m = Math.floor(sec / 60).toString().padStart(2, '0');
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

// Fetch live state from backend
async function fetchState() {
  try {
    const res = await fetch('/api/simulation/state');
    if (!res.ok) return;
    simState = await res.json();
    renderDashboard(simState);
  } catch (err) {
    console.error("Polling error:", err);
  }
}

// Render Dashboard View
function renderDashboard(state) {
  if (!state) return;

  // 1. Clock & Mode
  document.getElementById('sim-clock').textContent = formatTime(state.sim_time);
  isRunning = state.running;
  
  const playBtn = document.getElementById('btn-play-pause');
  document.getElementById('play-icon').textContent = isRunning ? '⏸' : '▶';
  document.getElementById('play-text').textContent = isRunning ? 'Pause' : 'Play';
  if (isRunning) {
    playBtn.classList.add('btn-primary');
    playBtn.classList.remove('btn-secondary');
  } else {
    playBtn.classList.remove('btn-primary');
    playBtn.classList.add('btn-secondary');
  }

  // 2. Classroom Score & Level
  const c = state.classroom;
  document.getElementById('classroom-score').textContent = c.score.toFixed(1);
  
  const lvlBadge = document.getElementById('level-badge');
  lvlBadge.textContent = c.level;
  lvlBadge.className = `level-tag level-${c.level.toLowerCase()}`;

  const progFill = document.getElementById('classroom-progress-fill');
  progFill.style.width = `${Math.min(100, Math.max(0, c.score))}%`;
  progFill.className = `progress-fill level-${c.level.toLowerCase()}-fill`;

  // Aggregates
  document.getElementById('count-focused').textContent = c.focused_count;
  document.getElementById('count-device').textContent = c.device_count;
  document.getElementById('count-sleep').textContent = c.sleep_count;
  document.getElementById('count-stand').textContent = c.stand_count;
  document.getElementById('count-turn').textContent = c.turn_head_count;

  // UART telemetry
  document.getElementById('uart-packet-output').textContent = c.uart_packet;

  // 3. Render Simulated Camera Overlays
  state.students.forEach(s => {
    const actEl = document.getElementById(`cam-act-${s.student_id}`);
    const scoreEl = document.getElementById(`cam-score-${s.student_id}`);
    const bboxEl = document.getElementById(`bbox-s${s.student_id}`);
    if (actEl && scoreEl && bboxEl) {
      actEl.textContent = s.current_activity;
      scoreEl.textContent = s.current_score.toFixed(1);
      
      // Update bbox color by activity
      if (['look_forward', 'read', 'write', 'handrise'].includes(s.current_activity)) {
        bboxEl.style.borderColor = '#10b981';
      } else if (s.current_activity === 'sleep') {
        bboxEl.style.borderColor = '#3b82f6';
      } else {
        bboxEl.style.borderColor = '#f59e0b';
      }
    }
  });

  // 4. Render 3 Student Cards
  const container = document.getElementById('student-cards-container');
  container.innerHTML = state.students.map(s => `
    <div class="glass-card student-card" onclick="openStudentModal(${s.student_id})">
      <div class="student-top">
        <div class="student-identity">
          <h4>${s.name}</h4>
          <span class="track-badge">TRACK ID #${s.track_id}</span>
        </div>
        <span class="act-badge act-${s.current_activity}">${s.current_activity}</span>
      </div>

      <div class="score-display">
        <div class="score-number-wrapper">
          <span class="score-number" style="font-size: 2.2rem;">${s.current_score.toFixed(1)}</span>
          <span class="level-tag level-${s.level.toLowerCase()}">${s.level}</span>
        </div>
      </div>

      <div class="duration-stats">
        <div>
          <span class="d-label">Focused</span>
          <span class="d-val">${formatTime(s.focused_time)}</span>
        </div>
        <div>
          <span class="d-label">Device</span>
          <span class="d-val">${formatTime(s.device_time)}</span>
        </div>
        <div>
          <span class="d-label">Sleep</span>
          <span class="d-val">${formatTime(s.sleep_time)}</span>
        </div>
      </div>

      <div class="mini-timeline-bar" title="Recent activity timeline">
        ${renderMiniTimeline(s.timeline, state.sim_time)}
      </div>
    </div>
  `).join('');

  // 5. Render Chart SVG
  renderChart(state.history);

  // 6. Render Alerts
  const alertsEl = document.getElementById('alerts-container');
  if (state.alerts && state.alerts.length > 0) {
    alertsEl.innerHTML = state.alerts.map(a => `
      <div class="alert-row">⚠️ ${a.message} <span style="opacity: 0.6; font-size: 0.7rem;">(${a.timestamp})</span></div>
    `).join('');
  } else {
    alertsEl.innerHTML = '';
  }

  // 7. Render Recent Events Stream
  const eventsEl = document.getElementById('events-stream');
  document.getElementById('event-count').textContent = `${state.recent_events.length} Events`;
  eventsEl.innerHTML = state.recent_events.map(e => `
    <div class="event-row">
      <span class="event-time">${formatTime(e.sim_time)}</span>
      <span class="event-msg"><strong>${e.student_name}:</strong> ${e.from_activity} → <span style="color: #38bdf8;">${e.to_activity}</span></span>
    </div>
  `).join('');
}

// Mini timeline bar generator
function renderMiniTimeline(timeline, maxTime) {
  if (!timeline || timeline.length === 0) return '';
  const total = Math.max(1, maxTime);
  return timeline.map(seg => {
    const dur = Math.max(1, (seg.end - seg.start));
    const pct = Math.min(100, (dur / total) * 100);
    let color = '#10b981';
    if (['using_device', 'turn_head', 'stand'].includes(seg.activity)) color = '#f59e0b';
    if (seg.activity === 'sleep') color = '#3b82f6';
    return `<div class="timeline-segment" style="width: ${pct}%; background-color: ${color};" title="${seg.activity}"></div>`;
  }).join('');
}

// Render Historical Score SVG Chart
function renderChart(history) {
  if (!history || history.length < 2) return;
  const width = 500;
  const height = 180;
  
  const minTime = history[0].time;
  const maxTime = Math.max(minTime + 1, history[history.length - 1].time);
  
  const points = history.map(h => {
    const x = ((h.time - minTime) / (maxTime - minTime)) * width;
    const y = height - (h.score / 100.0) * (height - 20) - 10;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const pathD = `M ${points.join(' L ')}`;
  document.getElementById('chart-path').setAttribute('d', pathD);
  
  const firstX = 0;
  const lastX = width;
  const fillD = `M 0,${height} L ${points.join(' L ')} L ${lastX},${height} Z`;
  document.getElementById('chart-fill').setAttribute('d', fillD);
}

// Student Detail Modal
async function openStudentModal(studentId) {
  try {
    const res = await fetch(`/api/simulation/student/${studentId}`);
    if (!res.ok) return;
    const data = await res.json();
    const s = data.student;
    
    document.getElementById('modal-content').innerHTML = `
      <div style="margin-bottom: 1.25rem;">
        <span class="track-badge">TRACK ID #${s.track_id}</span>
        <h2 style="font-size: 1.5rem; margin-top: 0.2rem;">${s.name} Detailed Analytics</h2>
        <span class="level-tag level-${s.level.toLowerCase()}">${s.level} Engagement (${s.current_score.toFixed(1)}/100)</span>
      </div>

      <div class="duration-stats" style="margin-bottom: 1.25rem; padding: 1rem;">
        <div><span class="d-label">Focused Time</span><span class="d-val" style="font-size: 1rem;">${formatTime(s.focused_time)}</span></div>
        <div><span class="d-label">Device Usage</span><span class="d-val" style="font-size: 1rem;">${formatTime(s.device_time)}</span></div>
        <div><span class="d-label">Sleep Duration</span><span class="d-val" style="font-size: 1rem;">${formatTime(s.sleep_time)}</span></div>
        <div><span class="d-label">Turned Head</span><span class="d-val" style="font-size: 1rem;">${formatTime(s.turn_head_time)}</span></div>
        <div><span class="d-label">Standing Time</span><span class="d-val" style="font-size: 1rem;">${formatTime(s.stand_time)}</span></div>
        <div><span class="d-label">Handrises</span><span class="d-val" style="font-size: 1rem;">${s.handrise_count} times</span></div>
      </div>

      <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem; color: var(--text-muted);">Recent Activity History</h4>
      <div class="events-list" style="max-height: 180px;">
        ${data.events.map(e => `
          <div class="event-row">
            <span class="event-time">${formatTime(e.sim_time)}</span>
            <span class="event-msg">${e.from_activity} → <strong style="color: #38bdf8;">${e.to_activity}</strong></span>
          </div>
        `).join('') || '<div style="color: var(--text-dim); font-size: 0.85rem;">No activity changes recorded yet.</div>'}
      </div>
    `;
    document.getElementById('student-modal').classList.remove('hidden');
  } catch (err) {
    console.error(err);
  }
}

function closeModal() {
  document.getElementById('student-modal').classList.add('hidden');
}

// Simulation Control Actions
async function togglePlayPause() {
  const url = isRunning ? '/api/simulation/pause' : '/api/simulation/start';
  await fetch(url, { method: 'POST' });
  fetchState();
}

async function resetSimulation() {
  await fetch('/api/simulation/reset', { method: 'POST' });
  fetchState();
}

async function jumpTime(sec) {
  await fetch('/api/simulation/jump', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seconds: sec })
  });
  fetchState();
}

async function setSpeed(val) {
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.classList.toggle('active', parseFloat(btn.textContent) === val);
  });
  await fetch('/api/simulation/speed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ speed: val })
  });
}

async function changeScenario(key) {
  await fetch('/api/simulation/scenario', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario: key })
  });
  fetchState();
}

async function switchMode(mode) {
  document.getElementById('btn-mode-sim').classList.toggle('active', mode === 'SIMULATION');
  document.getElementById('btn-mode-live').classList.toggle('active', mode === 'LIVE');
  const badge = document.getElementById('mode-badge');
  badge.className = `mode-badge ${mode.toLowerCase()}`;
  document.getElementById('mode-text').textContent = `${mode} MODE`;

  await fetch('/api/mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: mode })
  });
}

function exportData(format) {
  window.open(`/api/simulation/export/${format}`, '_blank');
}

// Initialize Polling Loop
setInterval(fetchState, 500);
fetchState();
