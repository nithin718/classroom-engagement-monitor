#!/usr/bin/env python3
"""
High-Speed Real-Webcam Dataset Collector (CLI & Automated Burst)
Captures live frames directly from the webcam (camera 0),
crops using Model B student framing, and saves samples for canonical classes.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from dataset_manager import dataset_manager, CANONICAL_CLASSES

CLASS_INSTRUCTIONS = {
    "using_device": "Hold a smartphone in your hands. Scroll, type, hold it at chest/desk height, tilt it naturally.",
    "write": "Hold a pen or pencil and pretend or actively write on paper, notebook, or desk surface.",
    "handrise": "Raise your hand (try right hand, left hand, high raise, moderate raise) as if answering in class.",
    "read": "Look down at a book, notepad, or printed page with your head slightly tilted down.",
    "turn_head": "Turn your head to the left, to the right, look toward neighbors or window.",
    "sleep": "Rest your head down on the desk or fold your arms and rest your head, simulating sleeping/drowsing.",
    "stand": "Stand up in view of the webcam, or show upper-body standing posture.",
    "look_forward": "Sit normally, looking forward attentively toward the camera/screen."
}

def capture_burst_for_class(
    cap: cv2.VideoCapture,
    class_name: str,
    target_count: int = 150,
    interval_sec: float = 0.1,
    crop_mode: str = "mode_b",
    session_id: str = "webcam_session_1",
    person_id: str = "student_1"
):
    print(f"\n" + "=" * 60)
    print(f"COLLECTING CLASS: {class_name.upper()} ({target_count} samples)")
    print("=" * 60)
    print(f"Instruction: {CLASS_INSTRUCTIONS.get(class_name, 'Perform the behavior.')}")
    print(f"Crop mode: {crop_mode}")
    print("\nGet in position! Starting in 3 seconds...")
    for s in range(3, 0, -1):
        print(f"  {s}...")
        time.sleep(1)
    print("🔴 RECORDING NOW! Move naturally (shift angles, hands, head)...")

    saved = 0
    start_time = time.time()
    
    while saved < target_count:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame from webcam")
            time.sleep(0.05)
            continue

        res = dataset_manager.save_sample(
            img_bgr=frame,
            label=class_name,
            crop_mode=crop_mode,
            session_id=session_id,
            person_id=person_id,
            notes=f"auto_burst_{saved+1}"
        )
        saved += 1
        
        # Progress bar
        bar_len = 30
        filled = int(round(bar_len * saved / float(target_count)))
        bar = '█' * filled + '-' * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {saved}/{target_count} ({saved*100//target_count}%)")
        sys.stdout.flush()

        time.sleep(interval_sec)

    elapsed = time.time() - start_time
    print(f"\n✅ Completed {saved} frames for '{class_name}' in {elapsed:.1f}s!")

def main():
    parser = argparse.ArgumentParser(description="Webcam Burst Collector for Model B V2")
    parser.add_argument("--class", dest="cls", type=str, default=None, choices=CANONICAL_CLASSES,
                        help="Specific class to capture")
    parser.add_argument("--count", type=int, default=150, help="Number of frames to capture per class (default: 150)")
    parser.add_argument("--interval", type=float, default=0.1, help="Interval between frames in seconds (default: 0.1)")
    parser.add_argument("--crop", type=str, default="mode_b", choices=["mode_b", "mode_c", "mode_a"],
                        help="Crop mode: mode_b (head+torso+hands), mode_c (wider), mode_a (full)")
    parser.add_argument("--session", type=str, default="session_webcam_v2", help="Session identifier")
    parser.add_argument("--person", type=str, default="person_1", help="Person identifier")
    parser.add_argument("--interactive", action="store_true", help="Interactive step-by-step collection for all 8 classes")
    parser.add_argument("--package", action="store_true", help="Run packaging & audit after collection")
    args = parser.parse_args()

    print("Opening webcam (device 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam device 0. Please check camera permissions or use the web collector at http://localhost:8000/rapid-collect")
        sys.exit(1)

    # Set preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        if args.cls:
            capture_burst_for_class(
                cap=cap,
                class_name=args.cls,
                target_count=args.count,
                interval_sec=args.interval,
                crop_mode=args.crop,
                session_id=args.session,
                person_id=args.person
            )
        elif args.interactive:
            print("============================================================")
            print("INTERACTIVE MODEL B V2 DATASET COLLECTION")
            print("============================================================")
            print("This tool will guide you through collecting 150+ frames for each of the 8 classes.")
            print("Canonical Classes:")
            for i, c in enumerate(CANONICAL_CLASSES):
                print(f"  {i}: {c}")
            print("============================================================\n")

            # Prioritize hardest classes first: using_device, write, handrise
            ordered_classes = ["using_device", "write", "handrise", "read", "turn_head", "look_forward", "sleep", "stand"]
            
            for i, c in enumerate(ordered_classes, 1):
                input(f"\n[Step {i}/8] Ready for '{c.upper()}'? Press ENTER to begin countdown...")
                capture_burst_for_class(
                    cap=cap,
                    class_name=c,
                    target_count=args.count,
                    interval_sec=args.interval,
                    crop_mode=args.crop,
                    session_id=args.session,
                    person_id=args.person
                )
        else:
            print("Please specify --class <classname> or --interactive")
            print("Example: python collect_webcam_stream.py --class using_device --count 150")
            print("Or use the Web Collector at: http://localhost:8000/rapid-collect")
            return
    finally:
        cap.release()

    if args.package or args.interactive:
        print("\nTriggering package & audit pipeline...")
        from package_dataset import run_pipeline
        run_pipeline()

if __name__ == "__main__":
    main()
