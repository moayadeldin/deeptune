"""Exercise video ingestion, training, evaluation, and embeddings end to end.

Uses generated moving-shape clips to check plumbing, not model quality.
Run from the repository root: python scripts/smoke_video.py --model resnet18
Native model example: python scripts/smoke_video.py --model r3d_18
"""
import argparse
import json
from pathlib import Path
import queue
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from desktop.config import RunConfig, list_runs, read_run_results
from desktop.runner import start_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--out', type=Path, default=Path('.artifacts/video-smoke'))
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--peft', action='store_true')
    parser.add_argument('--pooling', choices=['mean', 'attention'], default='mean')
    parser.add_argument('--manifest', action='store_true', help='Train from a labeled MP4 file list, as the desktop clip builder does.')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='deeptune-video-') as folder:
        root = Path(folder)
        for category in range(2):
            destination = root / f'class-{category}'
            destination.mkdir()
            for sample in range(20):
                path = destination / f'{sample:03}.mp4'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 8, (64, 64))
                if not writer.isOpened():
                    raise RuntimeError('OpenCV could not create the smoke-test clips.')
                try:
                    for frame_index in range(8):
                        frame = np.full((64, 64, 3), 20 + sample, dtype=np.uint8)
                        x = 4 + (frame_index * 5 + sample) % 40
                        color = (20, 40, 220) if category else (220, 100, 20)
                        cv2.rectangle(frame, (x, 12 + category * 20), (x + 12, 24 + category * 20), color, -1)
                        writer.write(frame)
                finally:
                    writer.release()
        before = set(list_runs(args.out))
        source = root
        if args.manifest:
            source = root / 'clips.csv'
            pd.DataFrame([(str(path.resolve()), path.parent.name) for path in sorted(root.glob('*/*.mp4'))],
                         columns=['videos', 'labels']).to_csv(source, index=False)
        cfg = RunConfig(modality='video', df=source, out=args.out, model_version=args.model,
                        num_epochs=1, batch_size=4, num_frames=16 if args.model == 'mvit_v2_s' else 2,
                        added_layers=2, embed_size=16, freeze_backbone=True, use_peft=args.peft,
                        pooling=args.pooling)
        handle = start_run(cfg)
        deadline = time.monotonic() + args.timeout
        while handle.thread.is_alive() and time.monotonic() < deadline:
            try:
                print(handle.log_queue.get(timeout=0.25), end='', flush=True)
            except queue.Empty:
                pass
        if handle.thread.is_alive():
            handle.close()
            raise TimeoutError(f'Video smoke test exceeded {args.timeout} seconds.')
        while not handle.log_queue.empty():
            print(handle.log_queue.get_nowait(), end='')
        status, message = handle.done_queue.get_nowait()
        if status != 'success':
            raise RuntimeError(f'{status}: {message}; full log: {handle.session_dir / "run.log"}')
        created = set(list_runs(args.out)) - before
        if len(created) != 1:
            raise AssertionError(f'Expected exactly one new experiment, found {len(created)}.')
        result = read_run_results(created.pop())
        assert result.checkpoint_path and result.checkpoint_path.is_file()
        assert result.metrics and 'accuracy' in result.metrics
        assert result.training_log is not None and len(result.training_log) == 1
        assert result.embeddings_shape == (8, 17), result.embeddings_shape
        assert result.run_status['status'] == 'success'
        print(json.dumps({'passed': True, 'model': args.model, 'peft': args.peft,
                          'source': 'mp4_file_list' if args.manifest else 'class_folders',
                          'pooling': args.pooling, 'embeddings_shape': result.embeddings_shape,
                          'run': str(result.run_dir)}, indent=2))


if __name__ == '__main__':
    main()
