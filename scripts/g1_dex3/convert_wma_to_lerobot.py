"""WMA-format G1 Dex3 dataset -> LeRobot v2.1, for FastWAM.

The AllMerged maincam set (2851 episodes, 12 instructions, cam_left_high
only) is stored in the UnifoLM-WMA training format:

    <wma_root>/<name>.csv                       one row per episode
    <wma_root>/videos/<name>/<view>/<id>.mp4    H.264 640x480 @ 30 fps
    <wma_root>/transitions/<name>/<id>.h5       action + observation.state (T, 28)

The CSV carries the corrected per-episode instruction ("... red, yellow, and
green"), which is the label set every WMA run trained on. FastWAM reads
LeRobot v2.1/v3.0, so this script writes a v2.1 view of exactly that data:

    data/chunk-XXX/episode_XXXXXX.parquet  state, action, timestamp, indices
    videos/chunk-XXX/observation.images.cam_left_high/episode_XXXXXX.mp4
    meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl}

Videos are hardlinked, not copied (source and target must be on the same
filesystem and, inside a container, under the same bind mount), so the
conversion costs only the parquet files.
Episode index i == CSV row i == WMA videoid i; the mapping is written to
meta/source_map.jsonl for provenance.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

FPS = 30
CHUNK_SIZE = 1000
VIEW = "observation.images.cam_left_high"
JOINT_NAMES = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
]


def video_frame_count(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def feature_stats(x: np.ndarray) -> dict:
    x = x.reshape(len(x), -1).astype(np.float64)
    return {
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "count": [len(x)],
    }


def main(args):
    wma_root = Path(args.wma_root)
    out = Path(args.output_dir)
    csv = pd.read_csv(wma_root / f"{args.name}.csv")
    if args.num_episodes:
        csv = csv.iloc[: args.num_episodes]
    (out / "meta").mkdir(parents=True, exist_ok=True)

    tasks = list(dict.fromkeys(csv["instruction"].tolist()))  # first-seen order
    task_index = {t: i for i, t in enumerate(tasks)}

    episodes, ep_stats, source_map = [], [], []
    global_index = 0
    for ep, row in tqdm(enumerate(csv.itertuples(index=False)), total=len(csv)):
        vid = int(row.videoid)
        chunk = ep // CHUNK_SIZE
        with h5py.File(wma_root / "transitions" / args.name / f"{vid}.h5", "r") as f:
            state = f["observation.state"][()].astype(np.float32)
            action = f["action"][()].astype(np.float32)
        n = len(action)
        assert state.shape == (n, 28) and action.shape == (n, 28), (vid, state.shape, action.shape)

        src_video = wma_root / "videos" / args.name / VIEW / f"{vid}.mp4"
        if args.check_videos:
            nv = video_frame_count(src_video)
            assert nv == n, f"episode {vid}: video has {nv} frames, h5 has {n}"
        dst_video = out / "videos" / f"chunk-{chunk:03d}" / VIEW / f"episode_{ep:06d}.mp4"
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        if not dst_video.exists():
            os.link(src_video, dst_video)

        t = row.instruction
        frame_index = np.arange(n, dtype=np.int64)
        cols = {
            "observation.state": pa.array(list(state), type=pa.list_(pa.float32())),
            "action": pa.array(list(action), type=pa.list_(pa.float32())),
            "timestamp": pa.array((frame_index / FPS).astype(np.float32)),
            "frame_index": pa.array(frame_index),
            "episode_index": pa.array(np.full(n, ep, dtype=np.int64)),
            "index": pa.array(np.arange(global_index, global_index + n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, task_index[t], dtype=np.int64)),
        }
        dst_parquet = out / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), dst_parquet)

        episodes.append({"episode_index": ep, "tasks": [t], "length": n})
        ep_stats.append({
            "episode_index": ep,
            "stats": {
                "observation.state": feature_stats(state),
                "action": feature_stats(action),
                "timestamp": feature_stats(frame_index / FPS),
                "frame_index": feature_stats(frame_index),
                "episode_index": feature_stats(np.full(n, ep)),
                "index": feature_stats(np.arange(global_index, global_index + n)),
                "task_index": feature_stats(np.full(n, task_index[t])),
            },
        })
        source_map.append({"episode_index": ep, "wma_videoid": vid, "instruction": t})
        global_index += n

    vec = lambda: {"dtype": "float32", "shape": [28], "names": [JOINT_NAMES]}
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}
    info = {
        "codebase_version": "v2.1",
        "robot_type": "Unitree_G1_Dex3",
        "total_episodes": len(episodes),
        "total_frames": global_index,
        "total_tasks": len(tasks),
        "total_videos": len(episodes),
        "total_chunks": (len(episodes) - 1) // CHUNK_SIZE + 1,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": vec(),
            "action": vec(),
            VIEW: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": float(FPS), "video.height": 480, "video.width": 640,
                    "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False, "has_audio": False,
                },
            },
            "timestamp": scalar("float32"),
            "frame_index": scalar("int64"),
            "episode_index": scalar("int64"),
            "index": scalar("int64"),
            "task_index": scalar("int64"),
        },
    }

    def write_jsonl(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    write_jsonl(out / "meta" / "tasks.jsonl", [{"task_index": i, "task": t} for i, t in enumerate(tasks)])
    write_jsonl(out / "meta" / "episodes.jsonl", episodes)
    write_jsonl(out / "meta" / "episodes_stats.jsonl", ep_stats)
    write_jsonl(out / "meta" / "source_map.jsonl", source_map)
    print(f"wrote {len(episodes)} episodes, {global_index} frames, {len(tasks)} tasks -> {out}")
    for i, t in enumerate(tasks):
        print(f"  task {i}: {sum(e['tasks'][0] == t for e in episodes):4d} eps  {t}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wma_root", default="/wma_data")
    p.add_argument("--name", default="g1_dex3_allmerged_maincam")
    p.add_argument("--output_dir", default="/datasets/G1_Dex3_AllMerged_MainCam_LeRobot21")
    p.add_argument("--num_episodes", type=int, default=0, help="0 = all")
    p.add_argument("--check_videos", action="store_true",
                   help="ffprobe every mp4 and assert frame count == h5 length")
    main(p.parse_args())
