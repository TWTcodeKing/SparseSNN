#!/usr/bin/env python3
"""Preprocess Prophesee Gen1 raw event data into training-ready .npy frames.

Converts raw .dat event files + .npy bbox annotations into:
- Frame stacks: (T, H, W, 3) uint8 .npy files (event count images)
- YOLO labels: .txt files with normalized (class cx cy w h)

Event encoding per frame:
- 127 (gray): no events
- 255 (white): positive polarity events
- 0 (black): negative polarity events

Usage:
    python datasets/gen1_preprocess.py \
        --raw-dir /data/gen1_raw \
        --out-dir /data/twt/datasets/gen1 \
        --T 5 --sample-size 250000

Raw directory structure expected:
    {raw_dir}/
        train/      (*.dat + *_bbox.npy pairs)
        val/
        test/

Output structure:
    {out_dir}/
        train/images/*.npy  train/labels/*.txt
        val/images/*.npy    val/labels/*.txt
        test/images/*.npy   test/labels/*.txt

Install: uv pip install prophesee-automotive-dataset-toolbox
  (or use PSEELoader from prophesee_utils)
"""

import argparse
import os
import sys

import numpy as np


def load_events_dat(dat_path, sample_size):
    """Load events from a .dat file using PSEELoader.

    Returns list of event arrays, one per temporal sample.
    Each sample contains `sample_size` events.
    """
    try:
        from prophesee_utils.io.psee_loader import PSEELoader
    except ImportError:
        raise ImportError(
            "prophesee_utils not found. Install with:\n"
            "  pip install prophesee-automotive-dataset-toolbox\n"
            "Or clone: https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox"
        )

    video = PSEELoader(dat_path)
    total_events = video.event_count()
    n_samples = total_events // sample_size

    samples = []
    for _ in range(n_samples):
        events = video.load_n_events(sample_size)
        if len(events) < sample_size:
            break
        samples.append(events)

    return samples


def events_to_frames(events, T, height=240, width=304):
    """Convert a batch of events into T frames.

    Args:
        events: structured numpy array with fields (x, y, p, t)
        T: number of temporal frames
        height, width: frame dimensions

    Returns:
        (T, height, width, 3) uint8 array
    """
    n_per_frame = len(events) // T
    frames = np.full((T, height, width, 3), 127, dtype=np.uint8)

    for t in range(T):
        start = t * n_per_frame
        end = (t + 1) * n_per_frame if t < T - 1 else len(events)
        ev = events[start:end]

        x = ev['x'].astype(np.int32)
        y = ev['y'].astype(np.int32)
        p = ev['p']  # polarity: 0 or 1

        # Clamp to valid range
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y, p = x[valid], y[valid], p[valid]

        # Positive polarity → 255, negative → 0
        frames[t, y[p == 1], x[p == 1], :] = 255
        frames[t, y[p == 0], x[p == 0], :] = 0

    return frames


def load_bbox_annotations(bbox_path, event_times, T, height=240, width=304):
    """Load and align bounding box annotations to frame timestamps.

    Args:
        bbox_path: path to _bbox.npy file
        event_times: (T+1,) array of frame boundary timestamps
        T: number of frames
        height, width: for normalization

    Returns:
        list of T label arrays, each (N, 5) with [class, cx, cy, w, h] normalized
    """
    bboxes = np.load(bbox_path)
    # Fields: t, x, y, w, h, class_id, track_id (or similar)

    labels_per_frame = []
    for t in range(T):
        t_start = event_times[t]
        t_end = event_times[t + 1]

        # Find boxes in this time range
        mask = (bboxes['t'] >= t_start) & (bboxes['t'] < t_end)
        frame_bboxes = bboxes[mask]

        if len(frame_bboxes) == 0:
            labels_per_frame.append(np.zeros((0, 5)))
            continue

        # Deduplicate by track_id (take latest per track)
        seen_tracks = {}
        for bb in frame_bboxes:
            tid = bb['track_id'] if 'track_id' in bb.dtype.names else 0
            seen_tracks[tid] = bb
        unique_bboxes = list(seen_tracks.values())

        labels = []
        for bb in unique_bboxes:
            cls_id = int(bb['class_id'])
            x, y, w, h = float(bb['x']), float(bb['y']), float(bb['w']), float(bb['h'])
            # Convert (x, y, w, h) top-left to (cx, cy, w, h) normalized
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            nw = w / width
            nh = h / height
            # Clamp
            cx = min(max(cx, 0), 1)
            cy = min(max(cy, 0), 1)
            nw = min(max(nw, 0), 1)
            nh = min(max(nh, 0), 1)
            if nw > 0.001 and nh > 0.001:
                labels.append([cls_id, cx, cy, nw, nh])

        labels_per_frame.append(np.array(labels) if labels else np.zeros((0, 5)))

    return labels_per_frame


def process_sequence(dat_path, bbox_path, out_img_dir, out_label_dir,
                     seq_name, T=5, sample_size=250000):
    """Process one event sequence into multiple frame samples."""
    height, width = 240, 304
    samples = load_events_dat(dat_path, sample_size)

    count = 0
    for i, events in enumerate(samples):
        # Generate frames
        frames = events_to_frames(events, T, height, width)

        # Get time boundaries for label alignment
        n_per_frame = len(events) // T
        event_times = np.array([events[j * n_per_frame]['t'] for j in range(T)]
                               + [events[-1]['t']])

        # Get labels
        labels_per_frame = load_bbox_annotations(
            bbox_path, event_times, T, height, width)

        # Use labels from last frame (most complete view)
        labels = labels_per_frame[-1] if labels_per_frame else np.zeros((0, 5))

        # Save
        img_name = f"{seq_name}_{i:05d}"
        np.save(os.path.join(out_img_dir, img_name + '.npy'), frames)

        label_path = os.path.join(out_label_dir, img_name + '.txt')
        if len(labels) > 0:
            np.savetxt(label_path, labels, fmt='%d %.6f %.6f %.6f %.6f')
        else:
            open(label_path, 'w').close()  # empty file

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Gen1 raw events to .npy frames + YOLO labels")
    parser.add_argument('--raw-dir', type=str, required=True,
                        help='Raw Gen1 directory (with train/val/test subdirs)')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Output directory for preprocessed data')
    parser.add_argument('--T', type=int, default=5,
                        help='Number of temporal frames per sample (default 5)')
    parser.add_argument('--sample-size', type=int, default=250000,
                        help='Events per sample (default 250000)')
    args = parser.parse_args()

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.raw_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split} (not found: {split_dir})")
            continue

        out_img_dir = os.path.join(args.out_dir, split, 'images')
        out_label_dir = os.path.join(args.out_dir, split, 'labels')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)

        # Find .dat files
        dat_files = sorted([f for f in os.listdir(split_dir) if f.endswith('_td.dat')])
        print(f"\n[{split}] Found {len(dat_files)} sequences")

        total = 0
        for dat_file in dat_files:
            seq_name = dat_file.replace('_td.dat', '')
            dat_path = os.path.join(split_dir, dat_file)
            bbox_path = os.path.join(split_dir, seq_name + '_bbox.npy')

            if not os.path.exists(bbox_path):
                print(f"  Skipping {seq_name} (no bbox file)")
                continue

            n = process_sequence(dat_path, bbox_path, out_img_dir, out_label_dir,
                                seq_name, T=args.T, sample_size=args.sample_size)
            total += n
            print(f"  {seq_name}: {n} samples")

        print(f"[{split}] Total: {total} samples")


if __name__ == '__main__':
    main()
