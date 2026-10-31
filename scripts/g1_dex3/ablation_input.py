#!/usr/bin/env python3
"""Input-ablation test for first-frame FastWAM (the FastWAM analogue of the
DreamZero / UnifoLM-WMA teacher-forcing test).

Question: does the action expert use the image (and the prompt, the state),
or does it act from state + prior? The uncond FastWAM action tokens attend
only to the first frame + text/proprio, so we perturb exactly those inputs
and measure how far the 32-step @ 30 Hz prediction moves.

Data: the run's own held-out split (val_set_proportion 0.01, seed 42 ->
29 episodes, the same rng as BaseLerobotDataset). K start frames per
episode, evenly spaced, each with a full 32-step ground-truth future.

Conditions (one fixed noise seed per sample, shared by all conditions, so
differences come from the inputs only):
  true          true frame, true state, true prompt
  frame_time    frame from the same episode >= 3 s away, true state/prompt
  frame_other   frame from another held-out episode of another task
  frame_blank   mid-grey frame (0 after [-1, 1] scaling)
  state_mean    true frame, state = dataset mean (0 after z-scoring)
  state_other   true frame, state from the other episode (same index as frame_other)
  prompt_other  true frame + state, instruction of the other task
  prompt_empty  true frame + state, empty instruction

Metrics per condition (rad, absolute joints, over 32 x 28):
  mae_gt     |pred - GT|                       (task error)
  mae_hold   |state_t - GT| baseline           (do-nothing reference)
  shift      |pred_cond - pred_true|           (how much the input matters)
split into arms (0:14) and hands (14:28). Output: summary.json + samples.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchcodec.decoders as dec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zmq_fastwam_server import DEFAULT_PROMPT, FastWamPolicy  # noqa: E402  (same preprocessing as the server)

H = 32
FPS = 30


def heldout_episodes(n_episodes: int, proportion: float = 0.01, seed: int = 42):
    idx = list(range(n_episodes))
    np.random.default_rng(seed).shuffle(idx)
    return idx[int(n_episodes * (1 - proportion)):]


def main(a):
    D = Path(a.dataset)
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(D / "meta/tasks.jsonl")}
    n_ep = sum(1 for _ in open(D / "meta/episodes.jsonl"))
    val = heldout_episodes(n_ep)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    eps = {}
    for ep in val:
        t = pq.read_table(D / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet",
                          columns=["observation.state", "action", "task_index"]).to_pydict()
        eps[ep] = {
            "state": np.asarray(t["observation.state"], np.float32),
            "action": np.asarray(t["action"], np.float32),
            "task": tasks[t["task_index"][0]],
            "video": dec.VideoDecoder(str(D / f"videos/chunk-{ep // 1000:03d}/observation.images.cam_left_high/episode_{ep:06d}.mp4")),
        }
    print(f"held-out episodes: {len(val)}; tasks: {sorted(set(e['task'][:25] for e in eps.values()))}", flush=True)

    def frame(ep, i):
        return eps[ep]["video"][int(i)].permute(1, 2, 0).numpy()

    policy = FastWamPolicy(a)
    rng = np.random.default_rng(a.seed)
    conds = ["true", "frame_time", "frame_other", "frame_blank", "state_mean",
             "state_other", "prompt_other", "prompt_empty"]
    rows = []
    t_start = time.time()
    for k_ep, ep in enumerate(val):
        e = eps[ep]
        n = len(e["action"])
        starts = np.linspace(15, n - H - 1, a.per_episode).astype(int)
        others = [o for o in val if eps[o]["task"] != e["task"]]
        for i_s, s in enumerate(starts):
            o = int(rng.choice(others))
            so = int(rng.integers(0, len(eps[o]["action"]) - H))
            far = [j for j in range(n) if abs(j - s) >= 3 * FPS]
            st = int(rng.choice(far)) if far else int(rng.integers(0, n))
            gt = e["action"][s:s + H]
            state = e["state"][s]
            inputs = {
                "true": (frame(ep, s), state, e["task"]),
                "frame_time": (frame(ep, st), state, e["task"]),
                "frame_other": (frame(o, so), state, e["task"]),
                "frame_blank": (np.full_like(frame(ep, s), 128), state, e["task"]),
                "state_mean": (frame(ep, s), None, e["task"]),
                "state_other": (frame(ep, s), eps[o]["state"][so], e["task"]),
                "prompt_other": (frame(ep, s), state, eps[o]["task"]),
                "prompt_empty": (frame(ep, s), state, ""),
            }
            seed = int(a.seed * 100000 + ep * 100 + i_s)
            preds = {}
            for c in conds:
                img, stt, prompt = inputs[c]
                x = policy._image(img)
                if stt is None:
                    proprio = torch.zeros(1, 28)                    # z-score mean
                else:
                    proprio = policy._norm_state(stt[None])
                with torch.no_grad():
                    p = policy.model.infer_action(
                        prompt=DEFAULT_PROMPT.format(task=prompt),
                        input_image=x, action_horizon=H, proprio=proprio,
                        num_inference_steps=a.num_inference_steps, text_cfg_scale=1.0, seed=seed)
                act = p["action"]
                act = act.unsqueeze(0) if act.ndim == 2 else act
                norm = policy.processor.normalizer.normalizers["action"][policy.action_key]
                preds[c] = norm.backward(act.float().cpu())[0].numpy()
            row = {"episode": ep, "task": e["task"], "start": int(s), "other_episode": o,
                   "hold_mae": float(np.abs(state[None] - gt).mean())}
            for c in conds:
                row[f"{c}_mae_gt"] = float(np.abs(preds[c] - gt).mean())
                row[f"{c}_mae_gt_arms"] = float(np.abs(preds[c][:, :14] - gt[:, :14]).mean())
                row[f"{c}_mae_gt_hands"] = float(np.abs(preds[c][:, 14:] - gt[:, 14:]).mean())
                row[f"{c}_shift"] = float(np.abs(preds[c] - preds["true"]).mean())
            rows.append(row)
            with open(out / "samples.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"[{k_ep + 1}/{len(val)}] ep {ep} done, {len(rows)} samples, {time.time() - t_start:.0f} s", flush=True)

    def stat(key):
        v = np.array([r[key] for r in rows])
        return {"mean": float(v.mean()), "ci95": float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))}

    summary = {
        "ckpt": a.ckpt, "ckpt_sha256": policy.ckpt_sha256, "n_samples": len(rows),
        "n_episodes": len(val), "per_episode": a.per_episode, "horizon": H,
        "num_inference_steps": a.num_inference_steps, "hold_mae": stat("hold_mae"),
        "conditions": {c: {m: stat(f"{c}_{m}") for m in ["mae_gt", "mae_gt_arms", "mae_gt_hands", "shift"]}
                       for c in conds},
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\nhold-state baseline MAE {summary['hold_mae']['mean']:.4f}")
    print(f"{'condition':13s} {'MAE vs GT':>16s} {'arms':>8s} {'hands':>8s} {'shift vs true':>16s}")
    for c in conds:
        s = summary["conditions"][c]
        print(f"{c:13s} {s['mae_gt']['mean']:.4f} ±{s['mae_gt']['ci95']:.4f} "
              f"{s['mae_gt_arms']['mean']:8.4f} {s['mae_gt_hands']['mean']:8.4f} "
              f"{s['shift']['mean']:.4f} ±{s['shift']['ci95']:.4f}")


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
