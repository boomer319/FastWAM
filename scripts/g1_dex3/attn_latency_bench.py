#!/usr/bin/env python3
"""Latency of the server's --attn_map option: same policy, map on vs off, alternating calls.

Runs FastWamPolicy.infer (the server's own code path) on one held-out frame, N calls per setting,
interleaved, with --fixed_noise so both settings produce the same draw; checks that the actions
are identical with the map on and off (the map must not change the output).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchcodec.decoders as dec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zmq_fastwam_server import FastWamPolicy  # noqa: E402
from ablation_input import heldout_episodes  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--stats", default="/datasets/G1_Dex3_AllMerged_MainCam_LeRobot21/meta/fastwam_dataset_stats.json")
p.add_argument("--dataset", default="/datasets/G1_Dex3_AllMerged_MainCam_LeRobot21")
p.add_argument("--task", default="g1_dex3_maincam_uncond_384x512_1e-4")
p.add_argument("--config_dir", default="/workspace/configs")
p.add_argument("--num_inference_steps", type=int, default=10)
p.add_argument("--client_stride", type=int, default=2)
p.add_argument("--seed", type=int, default=201)
p.add_argument("--n", type=int, default=30)
p.add_argument("--log_dir", default="/tmp/attn_bench_log")
a = p.parse_args()
a.fixed_noise, a.attn_map, a.attn_layers, a.image_warp = True, True, 10, ""

D = Path(a.dataset)
ep = heldout_episodes(sum(1 for _ in open(D / "meta/episodes.jsonl")))[0]
t = pq.read_table(D / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet", columns=["observation.state"]).to_pydict()
img = dec.VideoDecoder(str(D / f"videos/chunk-{ep // 1000:03d}/observation.images.cam_left_high/episode_{ep:06d}.mp4"))[100]
frame = img.permute(1, 2, 0).numpy()
state = np.asarray(t["observation.state"][100], np.float32)
payload = {"observation.images.top": np.stack([frame, frame]), "observation.state": np.stack([state, state]),
           "language_instruction": ["Stack the three cubic blocks from bottom to top in order of red, yellow, and green"]}

pol = FastWamPolicy(a)
for _ in range(3):                                   # warm-up
    pol.infer(payload)
lat = {True: [], False: []}
acts = {True: None, False: None}
for i in range(2 * a.n):
    on = i % 2 == 0
    pol.attn.enabled = on
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = pol.infer(payload)
    torch.cuda.synchronize()
    lat[on].append(time.perf_counter() - t0)
    acts[on] = r["action_30hz"]
    assert ("attn_map" in r) == on
    if on:
        amap = r["attn_map"]
for on in (False, True):
    x = np.array(lat[on]) * 1000
    print(f"attn_map {'on ' if on else 'off'}: median {np.median(x):.1f} ms  mean {x.mean():.1f}  p90 {np.percentile(x, 90):.1f}  (n={len(x)})")
print(f"overhead median {np.median(lat[True]) * 1000 - np.median(lat[False]) * 1000:+.1f} ms; "
      f"actions identical on/off: {np.array_equal(acts[True], acts[False])} (max |diff| {np.abs(acts[True] - acts[False]).max():.2e})")
print("attn_map shape", amap.shape, "sum", round(float(amap.sum()), 4), "peak cell", np.unravel_index(amap.argmax(), amap.shape))
