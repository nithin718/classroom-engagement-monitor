import os
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    YOLO = None
    torch = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'model_b_webcam_v2'
RAW_DIR = DATA_DIR / 'raw'
MANIFEST_PATH = DATA_DIR / 'dataset_manifest.json'
MODELS_DIR = BASE_DIR / 'models'
V1_MODEL_PATH = MODELS_DIR / 'best.pt'

CANONICAL_CLASSES = [
    'handrise',      # 0
    'look_forward',  # 1
    'read',          # 2
    'sleep',         # 3
    'stand',         # 4
    'turn_head',     # 5
    'using_device',  # 6
    'write'          # 7
]

CLASS_TO_ID = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
ID_TO_CLASS = {i: c for i, c in enumerate(CANONICAL_CLASSES)}

class DatasetManager:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.raw_dir = self.data_dir / 'raw'
        self.manifest_path = self.data_dir / 'dataset_manifest.json'
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            for c in CANONICAL_CLASSES:
                (self.data_dir / split / c).mkdir(parents=True, exist_ok=True)
                (self.raw_dir / c).mkdir(parents=True, exist_ok=True)
                
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f'[DatasetManager] Error reading manifest: {e}, initializing fresh')
        
        return {
            'version': '2.0',
            'created_at': datetime.now().isoformat(),
            'samples': [],
            'classes': CANONICAL_CLASSES,
            'session_counts': {},
            'split_info': None
        }

    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def extract_crop(self, img_bgr: np.ndarray, crop_mode: str = 'mode_b') -> Tuple[np.ndarray, Dict[str, int]]:
        h, w = img_bgr.shape[:2]
        crop_mode = str(crop_mode).lower().strip()

        if crop_mode in ('mode_b', 'head_torso_hands', 'b'):
            x1 = int(w * 0.15)
            x2 = int(w * 0.85)
            y1 = int(h * 0.08)
            y2 = int(h * 0.96)
        elif crop_mode in ('mode_c', 'wider_upper_body', 'c'):
            x1 = int(w * 0.08)
            x2 = int(w * 0.92)
            y1 = int(h * 0.04)
            y2 = int(h * 0.96)
        else:
            return img_bgr.copy(), {'x': 0, 'y': 0, 'width': w, 'height': h, 'mode': 'mode_a'}

        x1 = max(0, min(x1, w - 10))
        x2 = max(x1 + 10, min(x2, w))
        y1 = max(0, min(y1, h - 10))
        y2 = max(y1 + 10, min(y2, h))

        cropped = img_bgr[y1:y2, x1:x2].copy()
        return cropped, {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1, 'mode': crop_mode}

    def save_sample(
        self,
        img_bgr: np.ndarray,
        label: str,
        crop_mode: str = 'mode_b',
        session_id: str = 'session_1',
        person_id: str = 'person_1',
        notes: str = ''
    ) -> Dict[str, Any]:
        label = label.lower().strip()
        if label not in CANONICAL_CLASSES:
            raise ValueError(f'Invalid class label {label}. Must be one of: {CANONICAL_CLASSES}')

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
        sample_id = f'sample_{timestamp_str}_{label}'

        crop_bgr, crop_box = self.extract_crop(img_bgr, crop_mode=crop_mode)
        crop_224 = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_AREA)

        class_raw_dir = self.raw_dir / label
        class_raw_dir.mkdir(parents=True, exist_ok=True)

        crop_filename = f'{sample_id}_crop.jpg'
        orig_filename = f'{sample_id}_orig.jpg'
        meta_filename = f'{sample_id}_meta.json'

        crop_path = class_raw_dir / crop_filename
        orig_path = class_raw_dir / orig_filename
        meta_path = class_raw_dir / meta_filename

        cv2.imwrite(str(crop_path), crop_224, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(orig_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        sample_record = {
            'sample_id': sample_id,
            'label': label,
            'class_id': CLASS_TO_ID[label],
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'person_id': person_id,
            'crop_mode': crop_mode,
            'crop_box': crop_box,
            'crop_path': str(crop_path.relative_to(BASE_DIR)),
            'orig_path': str(orig_path.relative_to(BASE_DIR)),
            'split': 'unassigned',
            'notes': notes
        }

        with open(meta_path, 'w') as f:
            json.dump(sample_record, f, indent=2)

        self.manifest['samples'].append(sample_record)
        self.manifest['session_counts'][session_id] = self.manifest['session_counts'].get(session_id, 0) + 1
        self._save_manifest()

        return {
            'status': 'success',
            'sample_id': sample_id,
            'label': label,
            'class_id': CLASS_TO_ID[label],
            'crop_path': sample_record['crop_path'],
            'class_counts': self.get_class_counts(),
            'total_samples': len(self.manifest['samples'])
        }

    def delete_last_sample(self) -> Dict[str, Any]:
        if not self.manifest['samples']:
            return {'status': 'empty', 'message': 'No samples to delete'}

        last_sample = self.manifest['samples'].pop()
        sample_id = last_sample['sample_id']
        label = last_sample['label']
        session_id = last_sample.get('session_id', 'session_1')

        if session_id in self.manifest['session_counts']:
            self.manifest['session_counts'][session_id] = max(0, self.manifest['session_counts'][session_id] - 1)

        for p_key in ['crop_path', 'orig_path']:
            rel_p = last_sample.get(p_key)
            if rel_p:
                full_p = BASE_DIR / rel_p
                if full_p.exists():
                    full_p.unlink()

        meta_p = self.raw_dir / label / f'{sample_id}_meta.json'
        if meta_p.exists():
            meta_p.unlink()

        for split in ['train', 'val', 'test']:
            split_p = self.data_dir / split / label / f'{sample_id}_crop.jpg'
            if split_p.exists():
                split_p.unlink()

        self._save_manifest()
        return {
            'status': 'deleted',
            'sample_id': sample_id,
            'label': label,
            'remaining_total': len(self.manifest['samples']),
            'class_counts': self.get_class_counts()
        }

    def delete_sample_by_id(self, sample_id: str) -> Dict[str, Any]:
        idx = -1
        target = None
        for i, s in enumerate(self.manifest['samples']):
            if s['sample_id'] == sample_id:
                idx = i
                target = s
                break

        if idx == -1 or not target:
            return {'status': 'not_found', 'message': f'Sample {sample_id} not found'}

        self.manifest['samples'].pop(idx)
        label = target['label']
        session_id = target.get('session_id', 'session_1')

        if session_id in self.manifest['session_counts']:
            self.manifest['session_counts'][session_id] = max(0, self.manifest['session_counts'][session_id] - 1)

        for p_key in ['crop_path', 'orig_path']:
            rel_p = target.get(p_key)
            if rel_p:
                full_p = BASE_DIR / rel_p
                if full_p.exists():
                    full_p.unlink()

        meta_p = self.raw_dir / label / f'{sample_id}_meta.json'
        if meta_p.exists():
            meta_p.unlink()

        for split in ['train', 'val', 'test']:
            split_p = self.data_dir / split / label / f'{sample_id}_crop.jpg'
            if split_p.exists():
                split_p.unlink()

        self._save_manifest()
        return {
            'status': 'deleted',
            'sample_id': sample_id,
            'class_counts': self.get_class_counts()
        }

    def get_class_counts(self) -> Dict[str, int]:
        counts = {c: 0 for c in CANONICAL_CLASSES}
        for s in self.manifest['samples']:
            lbl = s.get('label')
            if lbl in counts:
                counts[lbl] += 1
        return counts

    def get_split_counts(self) -> Dict[str, Dict[str, int]]:
        res = {
            'train': {c: 0 for c in CANONICAL_CLASSES},
            'val': {c: 0 for c in CANONICAL_CLASSES},
            'test': {c: 0 for c in CANONICAL_CLASSES},
            'unassigned': {c: 0 for c in CANONICAL_CLASSES}
        }
        for s in self.manifest['samples']:
            split = s.get('split', 'unassigned')
            lbl = s.get('label')
            if split in res and lbl in res[split]:
                res[split][lbl] += 1
        return res

    def get_stats(self) -> Dict[str, Any]:
        class_counts = self.get_class_counts()
        split_counts = self.get_split_counts()
        total = len(self.manifest['samples'])

        recent = []
        for s in reversed(self.manifest['samples'][-12:]):
            recent.append({
                'sample_id': s['sample_id'],
                'label': s['label'],
                'timestamp': s['timestamp'],
                'session_id': s.get('session_id', 'session_1'),
                'crop_path': s.get('crop_path', '')
            })

        return {
            'total_samples': total,
            'target_per_class': '150–250',
            'class_counts': class_counts,
            'split_counts': split_counts,
            'sessions': self.manifest.get('session_counts', {}),
            'recent_samples': recent,
            'split_info': self.manifest.get('split_info')
        }

    def create_leakage_safe_split(
        self,
        strategy: str = 'session_aware',
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, Any]:
        if len(self.manifest['samples']) == 0:
            return {'status': 'error', 'message': 'Cannot split an empty dataset'}

        np.random.seed(seed)

        for split in ['train', 'val', 'test']:
            for c in CANONICAL_CLASSES:
                folder = self.data_dir / split / c
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir(parents=True, exist_ok=True)

        samples_by_class = {c: [] for c in CANONICAL_CLASSES}
        for s in self.manifest['samples']:
            samples_by_class[s['label']].append(s)

        split_assignment = {}

        for c, class_samples in samples_by_class.items():
            if not class_samples:
                continue

            session_groups: Dict[str, List[Dict[str, Any]]] = {}
            for s in class_samples:
                sess = s.get('session_id', 'default')
                session_groups.setdefault(sess, []).append(s)

            if len(session_groups) >= 3:
                sess_keys = list(session_groups.keys())
                np.random.shuffle(sess_keys)
                n_sess = len(sess_keys)
                n_test = max(1, int(round(n_sess * test_ratio)))
                n_val = max(1, int(round(n_sess * val_ratio)))

                test_sessions = set(sess_keys[:n_test])
                val_sessions = set(sess_keys[n_test:n_test + n_val])
                train_sessions = set(sess_keys[n_test + n_val:])

                for sess, items in session_groups.items():
                    target_split = 'train'
                    if sess in test_sessions:
                        target_split = 'test'
                    elif sess in val_sessions:
                        target_split = 'val'
                    for it in items:
                        split_assignment[it['sample_id']] = target_split
            else:
                sorted_items = sorted(class_samples, key=lambda x: x['timestamp'])
                n = len(sorted_items)
                num_blocks = 10
                block_size = max(1, n // num_blocks)
                for i, it in enumerate(sorted_items):
                    block_idx = min(num_blocks - 1, i // block_size)
                    if block_idx in [7]:
                        sp = 'val'
                    elif block_idx in [8, 9]:
                        sp = 'test'
                    else:
                        sp = 'train'
                    split_assignment[it['sample_id']] = sp

        counts = {'train': 0, 'val': 0, 'test': 0}
        for s in self.manifest['samples']:
            sid = s['sample_id']
            sp = split_assignment.get(sid, 'train')
            s['split'] = sp
            counts[sp] += 1

            crop_src = BASE_DIR / s['crop_path']
            if crop_src.exists():
                dst = self.data_dir / sp / s['label'] / f'{sid}_crop.jpg'
                shutil.copy2(crop_src, dst)

        split_info = {
            'strategy': strategy,
            'train_count': counts['train'],
            'val_count': counts['val'],
            'test_count': counts['test'],
            'split_timestamp': datetime.now().isoformat()
        }

        self.manifest['split_info'] = split_info
        self._save_manifest()

        return {
            'status': 'success',
            'split_info': split_info,
            'split_counts': self.get_split_counts()
        }

    def evaluate_model_on_split(
        self,
        weights_path: Path,
        split: str = 'test',
        output_prefix: str = 'v1_baseline'
    ) -> Dict[str, Any]:
        if YOLO is None:
            raise RuntimeError('Ultralytics YOLO is not available in environment')

        if not weights_path.exists():
            raise FileNotFoundError(f'Model weights not found at {weights_path}')

        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f'Split directory {split_dir} does not exist')

        print(f'[DatasetManager] Loading model from {weights_path}...')
        model = YOLO(str(weights_path))

        y_true = []
        y_pred = []
        confidences = []
        sample_results = []

        total_files = 0
        for class_idx, class_name in enumerate(CANONICAL_CLASSES):
            class_folder = split_dir / class_name
            if not class_folder.exists():
                continue

            images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
            total_files += len(images)

            for img_p in images:
                img_bgr = cv2.imread(str(img_p))
                if img_bgr is None:
                    continue

                res = model(img_bgr, verbose=False, imgsz=224)[0]
                pred_id = int(res.probs.top1)
                conf = float(res.probs.top1conf.cpu().numpy())

                y_true.append(class_idx)
                y_pred.append(pred_id)
                confidences.append(conf)

                sample_results.append({
                    'image': img_p.name,
                    'true_class': class_name,
                    'true_id': class_idx,
                    'pred_class': CANONICAL_CLASSES[pred_id],
                    'pred_id': pred_id,
                    'confidence': round(conf * 100, 2),
                    'correct': bool(pred_id == class_idx)
                })

        if len(y_true) == 0:
            return {
                'status': 'error',
                'message': f'No evaluation images found in {split_dir}. Make sure data is collected and split.'
            }

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        cm = np.zeros((8, 8), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        total_eval = len(y_true)
        correct_eval = int(np.sum(y_true_arr == y_pred_arr))
        accuracy = round(float(correct_eval / total_eval), 4)

        per_class = {}
        precisions = []
        recalls = []
        f1s = []

        for i, c_name in enumerate(CANONICAL_CLASSES):
            tp = int(cm[i, i])
            fp = int(np.sum(cm[:, i]) - tp)
            fn = int(np.sum(cm[i, :]) - tp)
            support = int(np.sum(cm[i, :]))

            prec = round(float(tp / (tp + fp)), 4) if (tp + fp) > 0 else 0.0
            rec = round(float(tp / (tp + fn)), 4) if (tp + fn) > 0 else 0.0
            f1 = round(float(2 * prec * rec / (prec + rec)), 4) if (prec + rec) > 0 else 0.0

            per_class[c_name] = {
                'class_id': i,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'support': support
            }

            if support > 0:
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)

        macro_precision = round(float(np.mean(precisions)), 4) if precisions else 0.0
        macro_recall = round(float(np.mean(recalls)), 4) if recalls else 0.0
        macro_f1 = round(float(np.mean(f1s)), 4) if f1s else 0.0

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(8),
            yticks=np.arange(8),
            xticklabels=CANONICAL_CLASSES,
            yticklabels=CANONICAL_CLASSES,
            title=f'Confusion Matrix: {output_prefix.upper()} on {split.upper()} (Acc: {accuracy*100:.1f}%, F1: {macro_f1:.4f})',
            ylabel='True Class (Ground Truth)',
            xlabel='Predicted Class (Raw Model)'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        thresh = cm.max() / 2. if cm.max() > 0 else 1
        for i in range(8):
            for j in range(8):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
        fig.tight_layout()

        cm_plot_path = self.data_dir / f'{output_prefix}_confusion_matrix.png'
        plt.savefig(str(cm_plot_path), dpi=200)
        plt.close(fig)

        report = {
            'model_path': str(weights_path),
            'split': split,
            'total_samples': total_eval,
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class': per_class,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_plot': str(cm_plot_path.relative_to(BASE_DIR)),
            'timestamp': datetime.now().isoformat()
        }

        report_json_path = self.data_dir / f'{output_prefix}_report.json'
        with open(report_json_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report

dataset_manager = DatasetManager()
