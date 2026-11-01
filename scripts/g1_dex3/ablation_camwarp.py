#!/usr/bin/env python3
"""Camera-warp sensitivity of first-frame FastWAM (open loop, held-out data).

Question: does the policy rely on one fixed camera geometry? Same held-out samples as
ablation_input.py (29 episodes, `per_episode` start frames each, identical per-sample noise seeds),
but the input frame is re-rendered as seen by a slightly different camera:

  true      unchanged frame (must reproduce ablation_input.py's `true` condition)
  zoom      focal length x1.10
  pitch+    camera pitched 4 deg (down), pitch- 4 deg (up), yaw+ 4 deg, roll+ 4 deg
  shift     principal point shifted 4 % to the right

Each warp is a camera rotation about its centre / intrinsics change, H = K_src R K_out^-1 (focal
384 px at 640 px width, as the sim camera), applied to the raw 480x640 frame before the server's
own preprocessing. Rotations get the smallest zoom-in (steps of 0.01) that keeps all four output
corners inside the source frame, so no content is invented beyond the image borders.

Metrics per condition: mae_gt (|pred - GT|) and the paired increase over `true`.
Output: summary.json + samples.jsonl.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchcodec.decoders as dec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zmq_fastwam_server import DEFAULT_PROMPT, FastWamPolicy  # noqa: E402  (same preprocessing as the server)
from ablation_input import H, heldout_episodes  # noqa: E402  (same held-out split)

W_, H_ = 640, 480
F_PX = 384.0
# name: (pitch, yaw, roll) in degrees, base zoom, shift (fraction of width)
WARPS = {"zoom": (0, 0, 0, 1.10, 0.0), "pitch+": (4, 0, 0, 1.0, 0.0), "pitch-": (-4, 0, 0, 1.0, 0.0),
         "yaw+": (0, 4, 0, 1.0, 0.0), "roll+": (0, 0, 4, 1.0, 0.0), "shift": (0, 0, 0, 1.0, 0.04)}


def rot(p, y, r):
    p, y, r = map(math.radians, (p, y, r))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def homography(pitch, yaw, roll, zoom, shift):
    k_src = np.array([[F_PX, 0, W_ / 2], [0, F_PX, H_ / 2], [0, 0, 1.0]])
    R = rot(pitch, yaw, roll)
    corners = np.array([[0, 0, 1], [W_, 0, 1], [0, H_, 1], [W_, H_, 1]], float)
    z = zoom
    while True:
        # a shift needs zoom too: cx moves by shift*W, so the frame must shrink to stay inside
        k_out = np.array([[F_PX * z, 0, W_ / 2 + shift * W_], [0, F_PX * z, H_ / 2], [0, 0, 1.0]])
        hom = k_src @ R @ np.linalg.inv(k_out)
        c = corners @ hom.T
        c = c[:, :2] / c[:, 2:3]
        if (c[:, 0] >= 0).all() and (c[:, 0] <= W_).all() and (c[:, 1] >= 0).all() and (c[:, 1] <= H_).all():
            return hom, round(z, 2)
        z += 0.01


def grid_for(hom):
    ys, xs = np.meshgrid(np.arange(H_) + 0.5, np.arange(W_) + 0.5, indexing="ij")
    pts = np.stack([xs, ys, np.ones_like(xs)], -1) @ hom.T
    src = pts[..., :2] / pts[..., 2:3]
    g = np.stack([src[..., 0] / W_ * 2 - 1, src[..., 1] / H_ * 2 - 1], -1)
    return torch.from_numpy(g.astype(np.float32)).unsqueeze(0)


def warp_frame(img, grid):
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)
    y = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return y[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


def main(a):
    D = Path(a.dataset)
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(D / "meta/tasks.jsonl")}
    n_ep = sum(1 for _ in open(D / "meta/episodes.jsonl"))
    val = heldout_episodes(n_ep)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grids, used = {}, {}
    for name, (p, y, r, z, s) in WARPS.items():
        hom, zz = homography(p, y, r, z, s)
        grids[name], used[name] = grid_for(hom), {"pitch": p, "yaw": y, "roll": r, "zoom": zz, "shift": s}
    print("warps (zoom = smallest that stays inside the frame):", json.dumps(used), flush=True)

    eps = {}
    for ep in val:
        t = pq.read_table(D / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet",
                          columns=["observation.state", "action", "task_index"]).to_pydict()
        eps[ep] = {"state": np.asarray(t["observation.state"], np.float32),
                   "action": np.asarray(t["action"], np.float32), "task": tasks[t["task_index"][0]],
                   "video": dec.VideoDecoder(str(D / f"videos/chunk-{ep // 1000:03d}/observation.images.cam_left_high/episode_{ep:06d}.mp4"))}

    policy = FastWamPolicy(a)
    norm = policy.processor.normalizer.normalizers["action"][policy.action_key]
    conds = ["true"] + list(WARPS)
    rows, t_start = [], time.time()
    for k_ep, ep in enumerate(val):
        e = eps[ep]
        n = len(e["action"])
        starts = np.linspace(15, n - H - 1, a.per_episode).astype(int)     # as ablation_input.py
        for i_s, s in enumerate(starts):
            img = e["video"][int(s)].permute(1, 2, 0).numpy()
            gt, state = e["action"][s:s + H], e["state"][s]
            seed = int(a.seed * 100000 + ep * 100 + i_s)                     # as ablation_input.py
            proprio = policy._norm_state(state[None])
            row = {"episode": ep, "task": e["task"], "start": int(s), "hold_mae": float(np.abs(state[None] - gt).mean())}
            for c in conds:
                frame = img if c == "true" else warp_frame(img, grids[c])
                with torch.no_grad():
                    p = policy.model.infer_action(prompt=DEFAULT_PROMPT.format(task=e["task"]),
                                                  input_image=policy._image(frame), action_horizon=H, proprio=proprio,
                                                  num_inference_steps=a.num_inference_steps, text_cfg_scale=1.0, seed=seed)
                act = p["action"]
                act = act.unsqueeze(0) if act.ndim == 2 else act
                pred = norm.backward(act.float().cpu())[0].numpy()
                row[f"{c}_mae_gt"] = float(np.abs(pred - gt).mean())
                row[f"{c}_mae_gt_arms"] = float(np.abs(pred[:, :14] - gt[:, :14]).mean())
            rows.append(row)
            with open(out / "samples.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"[{k_ep + 1}/{len(val)}] ep {ep} done, {len(rows)} samples, {time.time() - t_start:.0f} s", flush=True)

    def paired(c):
        per_ep = {}
        for r in rows:
            per_ep.setdefault(r["episode"], []).append(r[f"{c}_mae_gt"] - r["true_mae_gt"])
        d = np.array([np.mean(v) for v in per_ep.values()])
        return {"mean_increase": float(d.mean()), "ci95": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                "eps_worse": int((d > 0).sum()), "n_eps": int(len(d))}

    summary = {"ckpt": a.ckpt, "ckpt_sha256": policy.ckpt_sha256, "n_samples": len(rows), "warps": used,
               "mae_gt": {c: float(np.mean([r[f"{c}_mae_gt"] for r in rows])) for c in conds},
               "increase_vs_true": {c: paired(c) for c in WARPS}, "elapsed_s": round(time.time() - t_start, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\ntrue MAE {summary['mae_gt']['true']:.4f}")
    for c in WARPS:
        s = summary["increase_vs_true"][c]
        print(f"{c:7s} MAE {summary['mae_gt'][c]:.4f}  increase {s['mean_increase']:+.4f} ± {s['ci95']:.4f}  worse in {s['eps_worse']}/{s['n_eps']} eps")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats", default="/datasets/G1_Dex3_AllMerged_MainCam_LeRobot21/meta/fastwam_dataset_stats.json")
    p.add_argument("--dataset", default="/datasets/G1_Dex3_AllMerged_MainCam_LeRobot21")
    p.add_argument("--task", default="g1_dex3_maincam_uncond_384x512_1e-4")
    p.add_argument("--config_dir", default="/workspace/configs")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--client_stride", type=int, default=2)
    p.add_argument("--per_episode", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--log_dir", default="/tmp/ablation_server_log")
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
