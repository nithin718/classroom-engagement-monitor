/**
 * simulation_canvas.js
 * 
 * High-performance 2.5D Procedural Classroom Simulation Engine
 * Visualizes 3 distinct animated students with anatomical articulation for all 8 behaviors:
 * 1. look_forward: Upright posture, breathing idle, direct forward gaze
 * 2. read: Torso leaned forward, head down, eyes scanning open textbook
 * 3. write: Pen in hand, rhythmic hand movement across notebook
 * 4. handrise: Arm raised high above shoulder/head, slight natural waiting sway
 * 5. sleep: Slumped forward, head resting on folded arms on desk, slow deep breath
 * 6. stand: Full body rises from chair, taller vertical profile, bounding box stretches
 * 7. turn_head: Head rotated 45deg sideways, gaze shifted away from blackboard
 * 8. using_device: Glowing smartphone held near chest/desk, thumb tapping motion
 * 
 * Features:
 * - Dynamic AI Vision bounding box overlays with corner reticles and confidence ratings
 * - Engagement score tags with exact color semantics (HIGH: Bright/White-Green, MED: Yellow, LOW: Blue)
 * - Deterministic 90-second storyboard demonstrating all 8 behaviors across 3 students
 * - Minimal Camera HUD, Toast event alerts, Floating dock controls, and Presentation Mode
 */

(function() {
    'use strict';

    // --- Configuration & Constants ---
    const BEHAVIOR_WEIGHTS = {
        'handrise': 100,
        'look_forward': 100,
        'read': 85,
        'write': 80,
        'stand': 60,
        'turn_head': 40,
        'using_device': 20,
        'sleep': 5
    };

    function getEngagementLevel(score) {
        if (score >= 66.0) return 'HIGH';
        if (score >= 33.0) return 'MEDIUM';
        return 'LOW';
    }

    function getLevelColor(level) {
        switch (level) {
            case 'HIGH': return '#10b981'; // Vibrant emerald white-green
            case 'MEDIUM': return '#f59e0b'; // Amber yellow
            case 'LOW': return '#3b82f6'; // Clean cool blue
            default: return '#94a3b8';
        }
    }

    // --- 90-Second Deterministic Storyboard (Spec Section 22) ---
    // All 8 classes are demonstrated across the 3 students
    const STORYBOARD_90S = [
        { start: 0,  end: 10, s1: 'look_forward', s2: 'read',         s3: 'look_forward' },
        { start: 10, end: 20, s1: 'write',        s2: 'read',         s3: 'turn_head' },
        { start: 20, end: 30, s1: 'handrise',     s2: 'using_device', s3: 'turn_head' },
        { start: 30, end: 40, s1: 'look_forward', s2: 'using_device', s3: 'sleep' },
        { start: 40, end: 50, s1: 'read',         s2: 'look_forward', s3: 'sleep' },
        { start: 50, end: 60, s1: 'write',        s2: 'write',        s3: 'stand' },
        { start: 60, end: 70, s1: 'look_forward', s2: 'turn_head',    s3: 'stand' },
        { start: 70, end: 80, s1: 'handrise',     s2: 'look_forward', s3: 'look_forward' },
        { start: 80, end: 90, s1: 'write',        s2: 'read',         s3: 'look_forward' }
    ];

    const SCENARIOS = {
        'classroom_demo': {
            name: '90s Capstone Demo (All 8 Behaviors)',
            duration: 90,
            timeline: STORYBOARD_90S
        },
        'distraction_event': {
            name: 'Distraction & Device Misuse',
            duration: 60,
            timeline: [
                { start: 0,  end: 15, s1: 'write',        s2: 'read',         s3: 'look_forward' },
                { start: 15, end: 35, s1: 'look_forward', s2: 'using_device', s3: 'using_device' },
                { start: 35, end: 45, s1: 'handrise',     s2: 'turn_head',    s3: 'sleep' },
                { start: 45, end: 60, s1: 'write',        s2: 'look_forward', s3: 'look_forward' }
            ]
        },
        'question_session': {
            name: 'Interactive Q&A Session',
            duration: 60,
            timeline: [
                { start: 0,  end: 15, s1: 'look_forward', s2: 'look_forward', s3: 'look_forward' },
                { start: 15, end: 30, s1: 'handrise',     s2: 'handrise',     s3: 'turn_head' },
                { start: 30, end: 45, s1: 'stand',        s2: 'read',         s3: 'handrise' },
                { start: 45, end: 60, s1: 'write',        s2: 'write',        s3: 'look_forward' }
            ]
        },
        'low_engagement': {
            name: 'Afternoon Fatigue & Disengagement',
            duration: 60,
            timeline: [
                { start: 0,  end: 15, s1: 'read',         s2: 'turn_head',    s3: 'using_device' },
                { start: 15, end: 35, s1: 'turn_head',    s2: 'using_device', s3: 'sleep' },
                { start: 35, end: 50, s1: 'read',         s2: 'sleep',        s3: 'sleep' },
                { start: 50, end: 60, s1: 'look_forward', s2: 'look_forward', s3: 'stand' }
            ]
        }
    };

    // --- Student State Model ---
    class SimulatedStudent {
        constructor(id, name, seatLabel, colorTheme, deskX) {
            this.id = id;
            this.name = name;
            this.seatLabel = seatLabel;
            this.colorTheme = colorTheme; // { shirt, pants, hair, skin }
            this.deskX = deskX;           // 0 to 1 normalized horizontal position
            this.currentBehavior = 'look_forward';
            this.targetBehavior = 'look_forward';
            this.transitionProgress = 1.0; // 0 to 1
            this.score = 100.0;
            this.targetScore = 100.0;
            this.confidence = 0.94;
            this.targetConfidence = 0.94;
            
            // Procedural animation parameters
            this.animTime = Math.random() * 10;
            this.armRaiseProgress = 0.0;
            this.sleepSlumpProgress = 0.0;
            this.standProgress = 0.0;
            this.headTurnProgress = 0.0;
            this.deviceProgress = 0.0;
            this.readLeanProgress = 0.0;
            this.writeMoveProgress = 0.0;
            
            // Physical bounding box in canvas coordinates (calculated dynamically)
            this.bbox = { x: 0, y: 0, w: 0, h: 0 };
        }

        setBehavior(behavior) {
            if (this.targetBehavior !== behavior) {
                this.currentBehavior = this.targetBehavior;
                this.targetBehavior = behavior;
                this.transitionProgress = 0.0;
                this.targetScore = BEHAVIOR_WEIGHTS[behavior] || 80;
                this.targetConfidence = 0.88 + Math.random() * 0.09; // 0.88 - 0.97
            }
        }

        update(dt) {
            this.animTime += dt;
            
            // Smooth score & confidence transition
            this.score += (this.targetScore - this.score) * Math.min(1.0, dt * 3.5);
            this.confidence += (this.targetConfidence - this.confidence) * Math.min(1.0, dt * 2.0);

            // Smooth transition progress
            if (this.transitionProgress < 1.0) {
                this.transitionProgress = Math.min(1.0, this.transitionProgress + dt * 2.5);
            }

            // Target pose interpolations (0 = not active, 1 = fully in pose)
            const targetArm = (this.targetBehavior === 'handrise') ? 1.0 : 0.0;
            const targetSleep = (this.targetBehavior === 'sleep') ? 1.0 : 0.0;
            const targetStand = (this.targetBehavior === 'stand') ? 1.0 : 0.0;
            const targetTurn = (this.targetBehavior === 'turn_head') ? 1.0 : 0.0;
            const targetDevice = (this.targetBehavior === 'using_device') ? 1.0 : 0.0;
            const targetRead = (this.targetBehavior === 'read') ? 1.0 : 0.0;
            const targetWrite = (this.targetBehavior === 'write') ? 1.0 : 0.0;

            const easeRate = dt * 4.0;
            this.armRaiseProgress += (targetArm - this.armRaiseProgress) * easeRate;
            this.sleepSlumpProgress += (targetSleep - this.sleepSlumpProgress) * (easeRate * 0.8);
            this.standProgress += (targetStand - this.standProgress) * easeRate;
            this.headTurnProgress += (targetTurn - this.headTurnProgress) * (easeRate * 1.2);
            this.deviceProgress += (targetDevice - this.deviceProgress) * easeRate;
            this.readLeanProgress += (targetRead - this.readLeanProgress) * easeRate;
            this.writeMoveProgress += (targetWrite - this.writeMoveProgress) * easeRate;
        }
    }

    // --- Simulation Controller & Canvas Renderer ---
    class ClassroomSimulation {
        constructor() {
            this.canvas = document.getElementById('classroomCanvas');
            this.ctx = this.canvas.getContext('2d');
            
            this.simTime = 0.0;
            this.isPlaying = true;
            this.speed = 1.0;
            this.currentScenarioKey = 'classroom_demo';
            this.currentScenario = SCENARIOS[this.currentScenarioKey];
            this.lastFrameTime = performance.now();
            
            // Presentation & HUD Mode
            this.presentationMode = false;
            this.showDetectionBoxes = true;
            this.showScoreTags = true;
            this.showCamHUD = true;

            // Initialize 3 Students with distinct characteristics
            this.students = [
                new SimulatedStudent(1, 'Student 01', 'Desk L1 (Front-Left)', {
                    shirt: '#1e3a8a', // Navy blue
                    pants: '#1e293b',
                    hair: '#332211',
                    skin: '#fbd38d'
                }, 0.22),
                new SimulatedStudent(2, 'Student 02', 'Desk C1 (Center)', {
                    shirt: '#166534', // Forest green
                    pants: '#334155',
                    hair: '#1a1a1a',
                    skin: '#e2b17a'
                }, 0.50),
                new SimulatedStudent(3, 'Student 03', 'Desk R2 (Right-Rear)', {
                    shirt: '#881337', // Crimson / Wine
                    pants: '#0f172a',
                    hair: '#c27803',
                    skin: '#fcd34d'
                }, 0.78)
            ];

            // Real-time Event Toast Buffer
            this.toasts = [];
            this.lastReportedBehaviors = { 1: '', 2: '', 3: '' };

            this.initDOM();
            this.handleResize();
            window.addEventListener('resize', () => this.handleResize());
            
            // Start render & tick loop
            requestAnimationFrame((t) => this.renderLoop(t));
        }

        initDOM() {
            // Control buttons
            const playBtn = document.getElementById('btnPlay');
            const pauseBtn = document.getElementById('btnPause');
            const restartBtn = document.getElementById('btnRestart');
            const speedSelect = document.getElementById('speedSelect');
            const scenarioSelect = document.getElementById('scenarioSelect');
            const timelineScrubber = document.getElementById('timelineScrubber');
            const btnPresentation = document.getElementById('btnPresentation');
            const btnFullscreen = document.getElementById('btnFullscreen');
            const toggleBoxes = document.getElementById('toggleBoxes');

            if (playBtn) playBtn.onclick = () => { this.isPlaying = true; updatePlayState(true); };
            if (pauseBtn) pauseBtn.onclick = () => { this.isPlaying = false; updatePlayState(false); };
            if (restartBtn) restartBtn.onclick = () => this.restart();
            
            if (speedSelect) {
                speedSelect.onchange = (e) => {
                    this.speed = parseFloat(e.target.value);
                };
            }

            if (scenarioSelect) {
                scenarioSelect.onchange = (e) => {
                    this.setScenario(e.target.value);
                };
            }

            if (timelineScrubber) {
                timelineScrubber.oninput = (e) => {
                    this.simTime = (parseFloat(e.target.value) / 100) * this.currentScenario.duration;
                    this.syncStudentsToTimeline();
                };
            }

            if (btnPresentation) {
                btnPresentation.onclick = () => {
                    this.togglePresentationMode();
                };
            }

            if (btnFullscreen) {
                btnFullscreen.onclick = () => {
                    if (!document.fullscreenElement) {
                        document.documentElement.requestFullscreen().catch(err => console.log(err));
                    } else {
                        document.exitFullscreen();
                    }
                };
            }

            if (toggleBoxes) {
                toggleBoxes.onchange = (e) => {
                    this.showDetectionBoxes = e.target.checked;
                };
            }

            function updatePlayState(playing) {
                if (playBtn) playBtn.classList.toggle('active', playing);
                if (pauseBtn) pauseBtn.classList.toggle('active', !playing);
            }
            updatePlayState(true);
        }

        togglePresentationMode() {
            this.presentationMode = !this.presentationMode;
            document.body.classList.toggle('presentation-mode', this.presentationMode);
            const btn = document.getElementById('btnPresentation');
            if (btn) btn.classList.toggle('active', this.presentationMode);
            this.addToast('Presentation Mode ' + (this.presentationMode ? 'Enabled' : 'Disabled'));
        }

        setScenario(key) {
            if (SCENARIOS[key]) {
                this.currentScenarioKey = key;
                this.currentScenario = SCENARIOS[key];
                this.simTime = 0.0;
                this.lastReportedBehaviors = { 1: '', 2: '', 3: '' };
                this.syncStudentsToTimeline();
                this.addToast(`Scenario loaded: ${this.currentScenario.name}`);
            }
        }

        restart() {
            this.simTime = 0.0;
            this.lastReportedBehaviors = { 1: '', 2: '', 3: '' };
            this.syncStudentsToTimeline();
            this.addToast('Simulation timeline restarted');
        }

        handleResize() {
            const dpr = window.devicePixelRatio || 1;
            const w = this.canvas.parentElement.clientWidth;
            const h = this.canvas.parentElement.clientHeight;
            this.canvas.width = w * dpr;
            this.canvas.height = h * dpr;
            this.canvas.style.width = `${w}px`;
            this.canvas.style.height = `${h}px`;
            this.ctx.scale(dpr, dpr);
            this.viewportWidth = w;
            this.viewportHeight = h;
        }

        syncStudentsToTimeline() {
            const t = this.simTime % this.currentScenario.duration;
            for (const seg of this.currentScenario.timeline) {
                if (t >= seg.start && t < seg.end) {
                    this.students[0].setBehavior(seg.s1);
                    this.students[1].setBehavior(seg.s2);
                    this.students[2].setBehavior(seg.s3);
                    break;
                }
            }
        }

        triggerBehaviorEvents() {
            // Check for noteworthy events to toast
            this.students.forEach(s => {
                if (s.targetBehavior !== this.lastReportedBehaviors[s.id]) {
                    this.lastReportedBehaviors[s.id] = s.targetBehavior;
                    
                    if (s.targetBehavior === 'using_device') {
                        this.addToast(`⚠️ TRACK #${s.id} ${s.name}: Unauthorized Device Detected!`);
                    } else if (s.targetBehavior === 'sleep') {
                        this.addToast(`💤 TRACK #${s.id} ${s.name}: Sleeping Detected on Desk!`);
                    } else if (s.targetBehavior === 'handrise') {
                        this.addToast(`✋ TRACK #${s.id} ${s.name}: Hand Raised for Question`);
                    } else if (s.targetBehavior === 'stand') {
                        this.addToast(`🧍 TRACK #${s.id} ${s.name}: Standing Position Detected`);
                    }
                }
            });
        }

        addToast(msg) {
            this.toasts.push({ text: msg, expiry: performance.now() + 3800 });
            if (this.toasts.length > 3) this.toasts.shift();
        }

        renderLoop(now) {
            const dt = Math.min(0.1, (now - this.lastFrameTime) / 1000);
            this.lastFrameTime = now;

            if (this.isPlaying) {
                this.simTime += dt * this.speed;
                if (this.simTime >= this.currentScenario.duration) {
                    this.simTime = 0.0; // Loop automatically
                }
                this.syncStudentsToTimeline();
                this.triggerBehaviorEvents();
            }

            // Update student skeletal state
            this.students.forEach(s => s.update(dt));

            // Render complete frame
            this.draw();
            this.updateHUD();

            requestAnimationFrame((t) => this.renderLoop(t));
        }

        updateHUD() {
            // Timeline scrubber update
            const scrubber = document.getElementById('timelineScrubber');
            const timeLabel = document.getElementById('timeDisplay');
            if (scrubber) {
                const pct = (this.simTime / this.currentScenario.duration) * 100;
                scrubber.value = pct;
            }
            if (timeLabel) {
                const curM = Math.floor(this.simTime / 60);
                const curS = Math.floor(this.simTime % 60);
                const totM = Math.floor(this.currentScenario.duration / 60);
                const totS = Math.floor(this.currentScenario.duration % 60);
                timeLabel.textContent = `${String(curM).padStart(2, '0')}:${String(curS).padStart(2, '0')} / ${String(totM).padStart(2, '0')}:${String(totS).padStart(2, '0')}`;
            }

            // --- 1. Classroom Overall Metrics ---
            const avgScore = (this.students[0].score + this.students[1].score + this.students[2].score) / 3.0;
            const level = getEngagementLevel(avgScore);
            const focused = this.students.filter(s => ['handrise', 'look_forward', 'read', 'write'].includes(s.targetBehavior)).length;
            const device = this.students.filter(s => s.targetBehavior === 'using_device').length;
            const sleep = this.students.filter(s => s.targetBehavior === 'sleep').length;
            const uartPacket = `ENG,${Math.round(avgScore)},${level},${focused},${device},${sleep}`;

            // --- 2. Live Update Teacher Smartphone Frame ---
            const phoneClassScore = document.getElementById('phoneClassScore');
            const phoneClassLevel = document.getElementById('phoneClassLevel');
            const phoneClock = document.getElementById('phoneClock');
            if (phoneClassScore) phoneClassScore.textContent = Math.round(avgScore);
            if (phoneClassLevel) {
                phoneClassLevel.textContent = level;
                phoneClassLevel.className = 'phone-level-tag level-' + (level === 'HIGH' ? 'high' : (level === 'MEDIUM' ? 'med' : 'low'));
            }
            if (phoneClock) {
                const now = new Date();
                phoneClock.textContent = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;
            }

            // Update 3 Students in Phone
            this.students.forEach((s, idx) => {
                const sId = s.id;
                const actEl = document.getElementById(`phoneAct${sId}`);
                const scoreEl = document.getElementById(`phoneScore${sId}`);
                const lvlEl = document.getElementById(`phoneLvl${sId}`);
                const sLvl = getEngagementLevel(s.score);

                if (actEl) actEl.textContent = s.targetBehavior.toUpperCase();
                if (scoreEl) scoreEl.textContent = Math.round(s.score);
                if (lvlEl) {
                    lvlEl.textContent = sLvl;
                    lvlEl.className = 'p-student-lvl level-' + (sLvl === 'HIGH' ? 'high' : (sLvl === 'MEDIUM' ? 'med' : 'low'));
                }
            });

            // Phone Notification Banner (Real-time alert for disengagement)
            const notifBanner = document.getElementById('phoneNotificationBanner');
            const notifTitle = document.getElementById('phoneNotifTitle');
            const notifBody = document.getElementById('phoneNotifBody');
            const phoneRecent = document.getElementById('phoneRecentText');

            const alertedStudent = this.students.find(s => s.targetBehavior === 'using_device' || s.targetBehavior === 'sleep');
            if (alertedStudent && notifBanner) {
                notifBanner.classList.remove('hidden');
                if (notifTitle) notifTitle.textContent = `Alert: ${alertedStudent.name}`;
                if (notifBody) notifBody.textContent = (alertedStudent.targetBehavior === 'using_device') ? 'Unauthorized device detected' : 'Dozing/Sleeping on desk';
                if (phoneRecent) phoneRecent.textContent = `${alertedStudent.name} is ${alertedStudent.targetBehavior.toUpperCase()}`;
            } else if (notifBanner) {
                notifBanner.classList.add('hidden');
                if (phoneRecent) phoneRecent.textContent = 'All students attentive and engaged.';
            }

            // --- 3. Live Update TM4C123GH6PM Hardware Board ---
            const boardUart = document.getElementById('boardUartStr');
            const lcdLine1 = document.getElementById('lcdLine1');
            const lcdLine2 = document.getElementById('lcdLine2');
            const ledWhite = document.getElementById('ledWhite');
            const ledYellow = document.getElementById('ledYellow');
            const ledBlue = document.getElementById('ledBlue');
            const ledRed = document.getElementById('ledRed');
            const uartTxDot = document.getElementById('uartTxDot');

            if (boardUart) boardUart.textContent = uartPacket;
            if (uartTxDot) {
                // Subtle pulse to simulate UART serial stream
                uartTxDot.style.opacity = (Math.sin(this.simTime * 8) > 0) ? '1' : '0.4';
            }

            // Physical 16x2 Character LCD
            if (lcdLine1) lcdLine1.textContent = `CLASS: ${Math.round(avgScore)} ${level}`;
            if (lcdLine2) lcdLine2.textContent = `F:${focused}  D:${device}  S:${sleep}`;

            // Physical 4-LED Status (White = HIGH, Yellow = MEDIUM, Blue = LOW, Red = ALERT)
            if (ledWhite) ledWhite.classList.toggle('active', level === 'HIGH');
            if (ledYellow) ledYellow.classList.toggle('active', level === 'MEDIUM');
            if (ledBlue) ledBlue.classList.toggle('active', level === 'LOW');
            if (ledRed) ledRed.classList.toggle('active', device > 0 || sleep > 0);
        }

        // =========================================================================
        // DRAWING PIPELINE (Canvas 2.5D Room, Characters, Detection Boxes)
        // =========================================================================
        draw() {
            const ctx = this.ctx;
            const w = this.viewportWidth;
            const h = this.viewportHeight;

            ctx.clearRect(0, 0, w, h);

            // 1. Draw Classroom Environment
            this.drawClassroom(ctx, w, h);

            // 2. Draw Simulated Students (Anatomical Poses & Movements)
            this.students.forEach(student => {
                this.drawStudent(ctx, student, w, h);
            });

            // 3. Draw AI Detection Overlays (Bounding Boxes, Track IDs, Behaviors, Confidence)
            if (this.showDetectionBoxes) {
                this.students.forEach(student => {
                    this.drawDetectionOverlay(ctx, student);
                });
            }

            // 4. Draw Camera HUD (Timestamp, FPS, Rec pill)
            if (this.showCamHUD && !this.presentationMode) {
                this.drawCameraHUD(ctx, w, h);
            }

            // 5. Draw Floating Event Toasts
            this.drawToasts(ctx, w, h);
        }

        drawClassroom(ctx, w, h) {
            const horizonY = h * 0.44;

            // Wall (Soft institutional slate-warm grey)
            const wallGrad = ctx.createLinearGradient(0, 0, 0, horizonY);
            wallGrad.addColorStop(0, '#1e293b');
            wallGrad.addColorStop(1, '#334155');
            ctx.fillStyle = wallGrad;
            ctx.fillRect(0, 0, w, horizonY);

            // Blackboard / Smartboard in center
            const boardW = w * 0.62;
            const boardH = horizonY * 0.68;
            const boardX = (w - boardW) / 2;
            const boardY = horizonY * 0.16;

            // Frame
            ctx.fillStyle = '#475569';
            ctx.fillRect(boardX - 6, boardY - 6, boardW + 12, boardH + 12);
            // Board surface
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(boardX, boardY, boardW, boardH);

            // Content on Smartboard
            ctx.fillStyle = '#38bdf8';
            ctx.font = 'bold 13px Inter, sans-serif';
            ctx.fillText('AI COMPUTER VISION & BEHAVIORAL ENGAGEMENT MONITORING', boardX + 24, boardY + 28);
            ctx.fillStyle = '#94a3b8';
            ctx.font = '11px monospace';
            ctx.fillText('• YOLO11s 8-Class Taxonomy: [handrise, look_forward, read, write, stand, turn_head, using_device, sleep]', boardX + 24, boardY + 52);
            ctx.fillText('• ByteTrack Persistent Identifiers (Track #1, Track #2, Track #3)', boardX + 24, boardY + 70);
            ctx.fillText('• TM4C123G Microcontroller UART Telemetry: 115200 8-N-1', boardX + 24, boardY + 88);

            // Classroom Wall Clock
            const clockX = boardX + boardW + (w - (boardX + boardW)) / 2;
            const clockY = boardY + 30;
            ctx.beginPath();
            ctx.arc(clockX, clockY, 20, 0, Math.PI * 2);
            ctx.fillStyle = '#f8fafc';
            ctx.fill();
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#0284c7';
            ctx.stroke();
            // Clock hands
            const clockSecAngle = (this.simTime * 6) * (Math.PI / 180);
            ctx.beginPath();
            ctx.moveTo(clockX, clockY);
            ctx.lineTo(clockX + Math.cos(clockSecAngle) * 14, clockY + Math.sin(clockSecAngle) * 14);
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Classroom Left Window (Natural exterior lighting)
            const winW = boardX * 0.65;
            const winH = horizonY * 0.75;
            const winX = 20;
            const winY = boardY;
            ctx.fillStyle = '#1e293b';
            ctx.fillRect(winX - 4, winY - 4, winW + 8, winH + 8);
            const winGrad = ctx.createLinearGradient(winX, winY, winX + winW, winY + winH);
            winGrad.addColorStop(0, '#7dd3fc');
            winGrad.addColorStop(1, '#bae6fd');
            ctx.fillStyle = winGrad;
            ctx.fillRect(winX, winY, winW, winH);
            // Window panes
            ctx.strokeStyle = '#0f172a';
            ctx.lineWidth = 3;
            ctx.strokeRect(winX, winY, winW, winH);
            ctx.beginPath();
            ctx.moveTo(winX + winW / 2, winY); ctx.lineTo(winX + winW / 2, winY + winH);
            ctx.moveTo(winX, winY + winH / 2); ctx.lineTo(winX + winW, winY + winH / 2);
            ctx.stroke();

            // Classroom Floor (Perspective hardwood tiles)
            const floorGrad = ctx.createLinearGradient(0, horizonY, 0, h);
            floorGrad.addColorStop(0, '#0f172a');
            floorGrad.addColorStop(1, '#020617');
            ctx.fillStyle = floorGrad;
            ctx.fillRect(0, horizonY, w, h - horizonY);

            // Perspective tile grid lines
            ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
            ctx.lineWidth = 1;
            for (let x = 0; x <= w; x += w / 12) {
                ctx.beginPath();
                ctx.moveTo(w * 0.5 + (x - w * 0.5) * 0.25, horizonY);
                ctx.lineTo(x, h);
                ctx.stroke();
            }
            for (let y = horizonY + 20; y < h; y += (y - horizonY) * 0.55 + 24) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }
        }

        // =========================================================================
        // DRAW INDIVIDUAL ARTICULATED STUDENT
        // =========================================================================
        drawStudent(ctx, student, w, h) {
            const centerX = w * student.deskX;
            // Base seated desk coordinates
            const deskW = Math.min(260, w * 0.26);
            const deskH = 65;
            const deskY = h * 0.65;
            const deskX = centerX - deskW / 2;

            // Student body reference anchor (sitting behind the desk)
            // Stand progress pushes hips/torso upward
            const standOffset = student.standProgress * -85;
            const seatY = deskY + 25 + standOffset;

            // Sleep progress slumps torso down onto desk
            const sleepSlump = student.sleepSlumpProgress;
            // Reading leans forward
            const readLean = student.readLeanProgress;

            // Breathing oscillation (subtle natural life)
            const breathY = Math.sin(student.animTime * 2.5) * 1.8;

            ctx.save();

            // --- 1. Chair Backrest (Visible behind student) ---
            if (student.standProgress < 0.95) {
                ctx.fillStyle = '#334155';
                ctx.fillRect(centerX - 42, deskY - 60, 84, 80);
                ctx.fillStyle = '#1e293b';
                ctx.fillRect(centerX - 36, deskY - 54, 72, 68);
            }

            // --- 2. Legs / Standing Body (when standing up) ---
            if (student.standProgress > 0.05) {
                ctx.fillStyle = student.colorTheme.pants;
                // Left leg
                ctx.fillRect(centerX - 24, seatY + 50, 18, 90);
                // Right leg
                ctx.fillRect(centerX + 6, seatY + 50, 18, 90);
                // Shoes
                ctx.fillStyle = '#0f172a';
                ctx.fillRect(centerX - 28, seatY + 135, 24, 12);
                ctx.fillRect(centerX + 4, seatY + 135, 24, 12);
            }

            // --- 3. Torso / Clothes ---
            const torsoY = seatY - 50 + breathY + (sleepSlump * 45) + (readLean * 15);
            const torsoW = 68;
            const torsoH = 75;

            ctx.fillStyle = student.colorTheme.shirt;
            // Rounded torso
            ctx.beginPath();
            if (sleepSlump > 0.2) {
                // Curved forward slouch onto desk
                ctx.ellipse(centerX, torsoY + 20, torsoW / 2 + 6, torsoH / 2 - 5, 0, 0, Math.PI * 2);
            } else {
                ctx.roundRect(centerX - torsoW / 2, torsoY - torsoH / 2, torsoW, torsoH, [18, 18, 8, 8]);
            }
            ctx.fill();

            // Collar / Neck
            ctx.fillStyle = student.colorTheme.skin;
            ctx.fillRect(centerX - 9, torsoY - torsoH / 2 - 8, 18, 12);

            // --- 4. Head & Face Articulation ---
            let headX = centerX;
            let headY = torsoY - torsoH / 2 - 30;

            if (sleepSlump > 0.1) {
                // Head resting flat on the desk
                headX = centerX + 12 * sleepSlump;
                headY = deskY - 8;
            } else if (readLean > 0.1) {
                headY += 16 * readLean;
            }

            ctx.save();
            ctx.translate(headX, headY);

            // Head rotation for TURN_HEAD
            if (student.headTurnProgress > 0.05) {
                const turnSign = (student.id === 3) ? -1 : 1;
                ctx.rotate(turnSign * student.headTurnProgress * 0.45);
            }

            // Head base (Skin)
            ctx.fillStyle = student.colorTheme.skin;
            ctx.beginPath();
            ctx.arc(0, 0, 22, 0, Math.PI * 2);
            ctx.fill();

            // Hair (Stylized hair sitting naturally atop head above eyebrows)
            ctx.fillStyle = student.colorTheme.hair;
            ctx.beginPath();
            ctx.arc(0, -6, 23, Math.PI * 0.85, Math.PI * 2.15);
            // Front fringe / bangs curve
            ctx.quadraticCurveTo(12, -8, 0, -10);
            ctx.quadraticCurveTo(-12, -8, -20, -3);
            ctx.fill();

            // Eyebrows
            ctx.strokeStyle = student.colorTheme.hair;
            ctx.lineWidth = 2.0;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(-12, -7); ctx.lineTo(-4, -8);
            ctx.moveTo(4, -8); ctx.lineTo(12, -7);
            ctx.stroke();

            // Facial features (Eyes, gaze orientation)
            if (student.targetBehavior === 'sleep' || student.sleepSlumpProgress > 0.6) {
                // Sleeping: Closed curved resting eyelids
                ctx.strokeStyle = '#475569';
                ctx.lineWidth = 2.2;
                ctx.beginPath();
                ctx.arc(-8, 3, 5, 0, Math.PI);
                ctx.arc(8, 3, 5, 0, Math.PI);
                ctx.stroke();
            } else {
                let eyeShiftX = 0;
                let eyeShiftY = 0;

                if (student.targetBehavior === 'read' || student.targetBehavior === 'write') {
                    eyeShiftY = 4.0;
                } else if (student.targetBehavior === 'turn_head') {
                    eyeShiftX = (student.id === 3) ? -6 : 6;
                } else if (student.targetBehavior === 'using_device') {
                    eyeShiftY = 5.0;
                }

                // Left & right eye whites
                ctx.fillStyle = '#ffffff';
                ctx.beginPath();
                ctx.ellipse(-8 + eyeShiftX * 0.3, 1 + eyeShiftY * 0.2, 4.5, 3.8, 0, 0, Math.PI * 2);
                ctx.ellipse(8 + eyeShiftX * 0.3, 1 + eyeShiftY * 0.2, 4.5, 3.8, 0, 0, Math.PI * 2);
                ctx.fill();

                // Pupils
                ctx.fillStyle = '#0f172a';
                ctx.beginPath();
                ctx.arc(-8 + eyeShiftX, 1 + eyeShiftY, 2.3, 0, Math.PI * 2);
                ctx.arc(8 + eyeShiftX, 1 + eyeShiftY, 2.3, 0, Math.PI * 2);
                ctx.fill();

                // Specular reflection glint in eye
                ctx.fillStyle = '#ffffff';
                ctx.beginPath();
                ctx.arc(-9 + eyeShiftX, 0 + eyeShiftY, 0.9, 0, Math.PI * 2);
                ctx.arc(7 + eyeShiftX, 0 + eyeShiftY, 0.9, 0, Math.PI * 2);
                ctx.fill();

                // Gentle natural mouth line
                ctx.strokeStyle = '#a16207';
                ctx.lineWidth = 1.6;
                ctx.beginPath();
                ctx.arc(0, 9, 5, 0.2, Math.PI - 0.2);
                ctx.stroke();
            }

            ctx.restore();

            // --- 5. Arms & Hands Articulation ---
            this.drawStudentArms(ctx, student, centerX, torsoY, deskX, deskY, deskW);

            // --- 6. Classroom Desk & Props (Notebook, Book, Phone) ---
            this.drawDeskAndProps(ctx, student, deskX, deskY, deskW, deskH, centerX);

            ctx.restore();

            // Calculate active bounding box for AI Vision detection overlay
            let boxTop = headY - 30;
            let boxBottom = (student.standProgress > 0.5) ? (seatY + 145) : (deskY + deskH + 10);
            let boxLeft = centerX - deskW * 0.42;
            let boxRight = centerX + deskW * 0.42;

            if (student.armRaiseProgress > 0.4) {
                boxTop -= 45 * student.armRaiseProgress;
            }

            student.bbox = {
                x: boxLeft,
                y: boxTop,
                w: boxRight - boxLeft,
                h: boxBottom - boxTop
            };
        }

        drawStudentArms(ctx, student, centerX, torsoY, deskX, deskY, deskW) {
            const skinColor = student.colorTheme.skin;
            const shirtColor = student.colorTheme.shirt;

            // SLEEP: Arms folded horizontally on the desk
            if (student.sleepSlumpProgress > 0.4) {
                ctx.strokeStyle = shirtColor;
                ctx.lineWidth = 14;
                ctx.lineCap = 'round';
                ctx.beginPath();
                ctx.moveTo(centerX - 35, deskY - 5);
                ctx.lineTo(centerX + 35, deskY - 5);
                ctx.stroke();

                ctx.fillStyle = skinColor;
                ctx.beginPath();
                ctx.arc(centerX + 30, deskY - 5, 8, 0, Math.PI * 2);
                ctx.fill();
                return;
            }

            // HANDRISE: Right arm reaching up high above head
            if (student.armRaiseProgress > 0.1) {
                const p = student.armRaiseProgress;
                // Left arm stays resting on desk
                ctx.strokeStyle = shirtColor;
                ctx.lineWidth = 12;
                ctx.beginPath();
                ctx.moveTo(centerX - 28, torsoY - 20);
                ctx.lineTo(centerX - 35, deskY - 4);
                ctx.stroke();

                // Right arm raised high
                const handY = torsoY - 80 - Math.sin(student.animTime * 3) * 4;
                ctx.strokeStyle = shirtColor;
                ctx.lineWidth = 13;
                ctx.beginPath();
                ctx.moveTo(centerX + 26, torsoY - 15);
                ctx.lineTo(centerX + 34, handY + 30);
                ctx.lineTo(centerX + 30, handY);
                ctx.stroke();

                // Open hand palm
                ctx.fillStyle = skinColor;
                ctx.beginPath();
                ctx.arc(centerX + 30, handY, 9, 0, Math.PI * 2);
                ctx.fill();
                return;
            }

            // USING_DEVICE: Both hands holding phone up close to chest/desk
            if (student.deviceProgress > 0.1) {
                ctx.strokeStyle = shirtColor;
                ctx.lineWidth = 12;
                ctx.beginPath();
                ctx.moveTo(centerX - 28, torsoY - 15);
                ctx.lineTo(centerX - 14, deskY - 18);
                ctx.moveTo(centerX + 28, torsoY - 15);
                ctx.lineTo(centerX + 14, deskY - 18);
                ctx.stroke();

                // Hands holding phone
                ctx.fillStyle = skinColor;
                ctx.beginPath();
                ctx.arc(centerX - 12, deskY - 18, 7, 0, Math.PI * 2);
                ctx.arc(centerX + 12, deskY - 18, 7, 0, Math.PI * 2);
                ctx.fill();

                // Illuminated Smartphone Screen with glowing screen
                ctx.fillStyle = '#0f172a';
                ctx.roundRect(centerX - 10, deskY - 32, 20, 32, 4);
                ctx.fill();
                // Screen glow
                const phoneGlow = ctx.createRadialGradient(centerX, deskY - 16, 2, centerX, deskY - 16, 24);
                phoneGlow.addColorStop(0, 'rgba(56, 189, 248, 0.9)');
                phoneGlow.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
                ctx.fillStyle = phoneGlow;
                ctx.fillRect(centerX - 18, deskY - 36, 36, 40);

                // Tapping thumb animation
                const thumbY = deskY - 20 + Math.sin(student.animTime * 12) * 2;
                ctx.fillStyle = skinColor;
                ctx.beginPath();
                ctx.arc(centerX, thumbY, 3.5, 0, Math.PI * 2);
                ctx.fill();
                return;
            }

            // WRITE: Right hand holds pen and actively scribbles on notebook
            if (student.writeMoveProgress > 0.1) {
                // Left arm holds paper
                ctx.strokeStyle = shirtColor;
                ctx.lineWidth = 12;
                ctx.beginPath();
                ctx.moveTo(centerX - 28, torsoY - 15);
                ctx.lineTo(centerX - 24, deskY - 4);
                ctx.stroke();
                ctx.fillStyle = skinColor;
                ctx.beginPath();
                ctx.arc(centerX - 24, deskY - 4, 7, 0, Math.PI * 2);
                ctx.fill();

                // Right arm writes
                const scribbleX = Math.sin(student.animTime * 14) * 8;
                ctx.strokeStyle = shirtColor;
                ctx.lineWidth = 12;
                ctx.beginPath();
                ctx.moveTo(centerX + 28, torsoY - 15);
                ctx.lineTo(centerX + 14 + scribbleX, deskY - 4);
                ctx.stroke();

                // Hand holding pen
                ctx.fillStyle = skinColor;
                ctx.beginPath();
                ctx.arc(centerX + 14 + scribbleX, deskY - 4, 7, 0, Math.PI * 2);
                ctx.fill();

                // Pen
                ctx.strokeStyle = '#0284c7';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(centerX + 14 + scribbleX, deskY - 4);
                ctx.lineTo(centerX + 18 + scribbleX, deskY + 4);
                ctx.stroke();
                return;
            }

            // Default / READ / LOOK_FORWARD: Natural arms resting on desk
            ctx.strokeStyle = shirtColor;
            ctx.lineWidth = 12;
            ctx.beginPath();
            ctx.moveTo(centerX - 28, torsoY - 15);
            ctx.lineTo(centerX - 28, deskY - 4);
            ctx.moveTo(centerX + 28, torsoY - 15);
            ctx.lineTo(centerX + 28, deskY - 4);
            ctx.stroke();

            ctx.fillStyle = skinColor;
            ctx.beginPath();
            ctx.arc(centerX - 28, deskY - 4, 7, 0, Math.PI * 2);
            ctx.arc(centerX + 28, deskY - 4, 7, 0, Math.PI * 2);
            ctx.fill();
        }

        drawDeskAndProps(ctx, student, deskX, deskY, deskW, deskH, centerX) {
            // Wooden Desk Top Surface
            const deskGrad = ctx.createLinearGradient(deskX, deskY, deskX, deskY + deskH);
            deskGrad.addColorStop(0, '#d97706'); // Warm maple wood
            deskGrad.addColorStop(1, '#92400e');
            ctx.fillStyle = deskGrad;
            ctx.roundRect(deskX, deskY, deskW, deskH, [6, 6, 2, 2]);
            ctx.fill();

            // Desk edge highlight
            ctx.fillStyle = '#b45309';
            ctx.fillRect(deskX, deskY + deskH - 8, deskW, 8);

            // Metal Desk Legs
            ctx.fillStyle = '#475569';
            ctx.fillRect(deskX + 16, deskY + deskH, 8, 80);
            ctx.fillRect(deskX + deskW - 24, deskY + deskH, 8, 80);

            // Props on Desk according to active state:
            if (student.targetBehavior === 'read' || student.readLeanProgress > 0.3) {
                // Open Textbook with page spread
                ctx.fillStyle = '#f8fafc';
                ctx.fillRect(centerX - 28, deskY + 6, 56, 32);
                ctx.fillStyle = '#cbd5e1';
                ctx.fillRect(centerX - 1, deskY + 6, 2, 32); // Spine
                // Faux text lines
                ctx.fillStyle = '#94a3b8';
                for (let l = 0; l < 4; l++) {
                    ctx.fillRect(centerX - 24, deskY + 12 + l * 6, 18, 2);
                    ctx.fillRect(centerX + 4, deskY + 12 + l * 6, 18, 2);
                }
            } else if (student.targetBehavior === 'write' || student.writeMoveProgress > 0.3) {
                // Lined Spiral Notebook
                ctx.fillStyle = '#fef08a';
                ctx.fillRect(centerX - 4, deskY + 8, 38, 32);
                ctx.fillStyle = '#475569';
                for (let r = 0; r < 5; r++) {
                    ctx.fillRect(centerX - 5, deskY + 10 + r * 6, 3, 2);
                }
            } else {
                // Closed binder / school supplies
                ctx.fillStyle = '#0284c7';
                ctx.fillRect(centerX - 24, deskY + 14, 30, 24);
                ctx.fillStyle = '#e2e8f0';
                ctx.fillRect(centerX + 14, deskY + 20, 18, 3); // Pencil
            }
        }

        // =========================================================================
        // AI DETECTION OVERLAY (YOLO11s Bounding Box, Track ID, Score, Confidence)
        // =========================================================================
        drawDetectionOverlay(ctx, student) {
            const b = student.bbox;
            if (!b || b.w <= 0 || b.h <= 0) return;

            const level = getEngagementLevel(student.score);
            const strokeColor = getLevelColor(level);

            ctx.save();

            // 1. Thin detection bounding box
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = strokeColor;
            ctx.strokeRect(b.x, b.y, b.w, b.h);

            // 2. Corner Reticles (High-tech computer vision feel)
            const reticleLen = 12;
            ctx.lineWidth = 3.0;
            ctx.strokeStyle = strokeColor;
            // Top-Left
            ctx.beginPath();
            ctx.moveTo(b.x, b.y + reticleLen); ctx.lineTo(b.x, b.y); ctx.lineTo(b.x + reticleLen, b.y);
            // Top-Right
            ctx.moveTo(b.x + b.w - reticleLen, b.y); ctx.lineTo(b.x + b.w, b.y); ctx.lineTo(b.x + b.w, b.y + reticleLen);
            // Bottom-Left
            ctx.moveTo(b.x, b.y + b.h - reticleLen); ctx.lineTo(b.x, b.y + b.h); ctx.lineTo(b.x + reticleLen, b.y + b.h);
            // Bottom-Right
            ctx.moveTo(b.x + b.w - reticleLen, b.y + b.h); ctx.lineTo(b.x + b.w, b.y + b.h); ctx.lineTo(b.x + b.w, b.y + b.h - reticleLen);
            ctx.stroke();

            // 3. Floating Label Badge above bounding box: "ID 1 WRITE 0.94 | ENG 80"
            const labelText = `ID: ${student.id}  ${student.targetBehavior.toUpperCase()}  ${student.confidence.toFixed(2)}`;
            const scoreText = `ENG: ${Math.round(student.score)}`;
            
            ctx.font = 'bold 11px monospace';
            const labelW = ctx.measureText(labelText + '  |  ' + scoreText).width + 16;
            const labelH = 20;
            const labelX = b.x;
            const labelY = b.y - labelH - 3;

            // Background badge
            ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
            ctx.fillRect(labelX, labelY, labelW, labelH);
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 1;
            ctx.strokeRect(labelX, labelY, labelW, labelH);

            // Text
            ctx.fillStyle = '#f8fafc';
            ctx.fillText(labelText, labelX + 8, labelY + 14);
            ctx.fillStyle = strokeColor;
            ctx.fillText(` |  ${scoreText}`, labelX + 8 + ctx.measureText(labelText).width, labelY + 14);

            // 4. Subtle Seat Tag at the bottom of the bounding box
            if (!this.presentationMode) {
                ctx.font = '10px Inter, sans-serif';
                ctx.fillStyle = 'rgba(241, 245, 249, 0.7)';
                ctx.fillText(`${student.name} • ${student.seatLabel}`, b.x + 4, b.y + b.h + 14);
            }

            ctx.restore();
        }

        // =========================================================================
        // CAMERA HUD & NOTIFICATIONS
        // =========================================================================
        drawCameraHUD(ctx, w, h) {
            // Minimal camera recording indicator at top-left
            ctx.save();
            ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
            ctx.roundRect(16, 16, 280, 48, 6);
            ctx.fill();
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
            ctx.stroke();

            // Pulsing Red Rec Dot
            const recAlpha = 0.5 + Math.sin(this.simTime * 5) * 0.5;
            ctx.fillStyle = `rgba(239, 68, 68, ${recAlpha})`;
            ctx.beginPath();
            ctx.arc(32, 40, 6, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = '#f8fafc';
            ctx.font = 'bold 12px monospace';
            ctx.fillText('CAM-01 [OVERHEAD WIDE]', 46, 36);
            ctx.font = '10px monospace';
            ctx.fillStyle = '#94a3b8';
            ctx.fillText('AI DETECTION ACTIVE • 30 FPS • SIMULATION', 46, 50);

            // Top-right Telemetry Summary Pill
            const avgScore = Math.round((this.students[0].score + this.students[1].score + this.students[2].score) / 3.0);
            const level = getEngagementLevel(avgScore);
            const levelCol = getLevelColor(level);

            const pillW = 260;
            const pillX = w - pillW - 16;
            ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
            ctx.roundRect(pillX, 16, pillW, 48, 6);
            ctx.fill();
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
            ctx.stroke();

            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px monospace';
            ctx.fillText('CLASSROOM ENGAGEMENT', pillX + 14, 33);
            ctx.fillStyle = levelCol;
            ctx.font = 'bold 15px Inter, sans-serif';
            ctx.fillText(`${avgScore}% [${level}]`, pillX + 14, 52);

            ctx.fillStyle = '#64748b';
            ctx.font = '10px monospace';
            ctx.fillText(`TRACKS: 3 ACTIVE`, pillX + 160, 42);

            ctx.restore();
        }

        drawToasts(ctx, w, h) {
            const now = performance.now();
            this.toasts = this.toasts.filter(t => t.expiry > now);

            ctx.save();
            let toastY = 80;
            this.toasts.forEach(toast => {
                const alpha = Math.min(1.0, (toast.expiry - now) / 800);
                ctx.font = 'bold 12px Inter, sans-serif';
                const toastW = ctx.measureText(toast.text).width + 36;
                const toastX = (w - toastW) / 2;

                ctx.fillStyle = `rgba(15, 23, 42, ${0.9 * alpha})`;
                ctx.roundRect(toastX, toastY, toastW, 32, 16);
                ctx.fill();
                ctx.strokeStyle = `rgba(245, 158, 11, ${0.8 * alpha})`;
                ctx.lineWidth = 1;
                ctx.stroke();

                ctx.fillStyle = `rgba(254, 243, 199, ${alpha})`;
                ctx.fillText(toast.text, toastX + 18, toastY + 20);

                toastY += 40;
            });
            ctx.restore();
        }
    }

    // --- Boot the Simulation once DOM is ready ---
    window.addEventListener('DOMContentLoaded', () => {
        window.classroomSim = new ClassroomSimulation();
    });

})();
