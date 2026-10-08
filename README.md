---
title: Real-Time Classroom Engagement Monitor
emoji: 🎓
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: true
---

# Real-Time Classroom Engagement Monitor

**Multi-Behavior Detection & Operational Engagement Scoring**

## Architecture
- **Model B V2**: YOLO11s-cls — Single-student 8-class behavior classifier
- **8 Classes**: handrise, look_forward, read, sleep, stand, turn_head, using_device, write
- **UART Telemetry**: Embedded behavioral engagement packets for TM4C123GXL
- **Real-time**: 40+ FPS on GPU, full webcam inference in-browser

## Features
- 🎥 Live Webcam Inference Dashboard
- 📊 Teacher Engagement Portal
- 🏫 2D Classroom Simulation
- 📡 Embedded UART Telemetry (TM4C123GXL)
- 🧪 Benchmark Sample Testing

## Usage
Open the app and navigate to the dashboard. Start the webcam for live inference.
