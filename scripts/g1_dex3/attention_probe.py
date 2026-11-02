#!/usr/bin/env python3
"""Where does first-frame FastWAM look? Attention + occlusion maps on held-out frames (small probe).

Two maps per sample, same input frame, same noise seed as ablation_input.py:

  attention  In every layer and denoising step the 32 action tokens attend to the cached
             first-frame video tokens (12 x 16 grid for a 384 x 512 input, one token ~ 32 x 32 px)
             plus themselves. The fused attention call does not return its weights, so the
             module-level `flash_attention` used by the MoT core is wrapped: for the action-query
             call it recomputes softmax(QK^T / sqrt(d)) (same mask) next to the normal output,
             which is returned unchanged. Recorded: weight on each video token, averaged over
             action queries and heads, per layer and step; plus the total mass on video vs action.
  occlusion  The model input (384 x 512, [-1, 1]) with one 64 x 64 patch set to 0 (mid grey),
             6 x 8 patches; map = mean |action - action_unoccluded| (radians, de-normalised,
             all 32 steps x 28 joints). Causal: what the plan actually depends on.

Attention is not attribution; the occlusion map is the check. Output per sample: PNG panel
(frame | attention | occlusion) and maps.npz; summary.json.
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

import fastwam.models.wan22.mot as mot_mod  # noqa: E402

GRID = (12, 16)          # video tokens of the first frame (384/32, 512/32)
OCC = (6, 8)             # occlusion patches (64 x 64 px)


class AttnRecorder:
    """Wraps mot.flash_attention; records action->video attention for the action-query calls."""

    def __init__(self, n_video, n_action):
        self.n_video, self.n_action = n_video, n_action
        self.maps, self.video_mass = [], []      # per call (= layer x step, in call order)
        self.on = False
        self._orig = mot_mod.flash_attention

    def __call__(self, q, k, v, num_heads, ctx_mask=None, compatibility_mode=True):
        out = self._orig(q, k, v, num_heads, ctx_mask=ctx_mask, compatibility_mode=compatibility_mode)
        if self.on and q.shape[1] == self.n_action and k.shape[1] == self.n_video + self.n_action:
            b, s, hd = q.shape
            d = hd // num_heads
            qh = q.view(b, s, num_heads, d).transpose(1, 2).float()
            kh = k.view(b, k.shape[1], num_heads, d).transpose(1, 2).float()
            logits = qh @ kh.transpose(-1, -2) / math.sqrt(d)
            if ctx_mask is not None:
                m = ctx_mask.to(logits.device)
                logits = logits.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else logits + m
            w = logits.softmax(-1)                                   # [b, heads, Sa, Sv+Sa]
            wv = w[..., :self.n_video]
            self.maps.append(wv.mean(dim=(0, 2)).cpu().numpy())      # [heads, Sv]
            self.video_mass.append(float(wv.sum(-1).mean()))
        return out


def jet(h):
    """h in [0, 1] -> RGB uint8 (classic jet), no OpenCV/matplotlib in the image."""
    r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def overlay(rgb, heat, alpha=0.55):
    t = torch.from_numpy(heat.astype(np.float32))[None, None]
    h = torch.nn.functional.interpolate(t, size=rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0].numpy()
    h = (h - h.min()) / max(h.max() - h.min(), 1e-12)
    return (rgb * (1 - alpha) + jet(h) * alpha).astype(np.uint8)


def label(im, txt):
    from PIL import Image, ImageDraw
    pil = Image.fromarray(im)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, 8 + 7 * len(txt), 20], fill=(0, 0, 0))
    d.text((5, 4), txt, fill=(255, 255, 255))
    return np.asarray(pil)


def main(a):
    from PIL import Image
    D = Path(a.dataset)
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(D / "meta/tasks.jsonl")}
    n_ep = sum(1 for _ in open(D / "meta/episodes.jsonl"))
    val = heldout_episodes(n_ep)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    policy = FastWamPolicy(a)
    norm = policy.processor.normalizer.normalizers["action"][policy.action_key]
    rec = AttnRecorder(GRID[0] * GRID[1], H)
    mot_mod.flash_attention = rec

    # samples: held-out episodes of the requested task, `per_episode` start frames each (as ablation_input.py)
    picked = []
    for ep in val:
        t = pq.read_table(D / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet",
                          columns=["observation.state", "action", "task_index"]).to_pydict()
        if a.task_index >= 0 and t["task_index"][0] != a.task_index:
            continue
        picked.append((ep, t))
        if len(picked) == a.episodes:
            break

    def infer(img_t, proprio, task, seed):
        with torch.no_grad():
            p = policy.model.infer_action(prompt=DEFAULT_PROMPT.format(task=task), input_image=img_t, action_horizon=H,
                                          proprio=proprio, num_inference_steps=a.num_inference_steps,
                                          text_cfg_scale=1.0, seed=seed)
        act = p["action"]
        act = act.unsqueeze(0) if act.ndim == 2 else act
        return norm.backward(act.float().cpu())[0].numpy()

    summary, t0 = [], time.time()
    for ep, t in picked:
        state, action = np.asarray(t["observation.state"], np.float32), np.asarray(t["action"], np.float32)
        task = tasks[t["task_index"][0]]
        video = dec.VideoDecoder(str(D / f"videos/chunk-{ep // 1000:03d}/observation.images.cam_left_high/episode_{ep:06d}.mp4"))
        starts = np.linspace(15, len(action) - H - 1, a.per_episode).astype(int)
        for i_s, s in enumerate(starts):
            img = video[int(s)].permute(1, 2, 0).numpy()
            img_t = policy._image(img)
            proprio = policy._norm_state(state[s][None])
            seed = int(a.seed * 100000 + ep * 100 + i_s)
            rec.maps, rec.video_mass, rec.on = [], [], True
            base = infer(img_t, proprio, task, seed)
            rec.on = False
            maps = np.stack(rec.maps)                                 # [layers*steps, heads, Sv]
            n_calls = maps.shape[0]
            n_layers = n_calls // a.num_inference_steps
            maps = maps.reshape(a.num_inference_steps, n_layers, maps.shape[1], -1)
            attn = maps.mean(axis=(0, 1, 2)).reshape(GRID)
            third = max(n_layers // 3, 1)
            attn_groups = {g: maps[:, sl].mean(axis=(0, 1, 2)).reshape(GRID) for g, sl in
                           (("early", slice(0, third)), ("mid", slice(third, 2 * third)), ("late", slice(2 * third, n_layers)))}
            occ = np.zeros(OCC, np.float32)
            ph, pw = img_t.shape[-2] // OCC[0], img_t.shape[-1] // OCC[1]
            for r in range(OCC[0]):
                for c in range(OCC[1]):
                    x = img_t.clone()
                    x[..., r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = 0.0
                    occ[r, c] = float(np.abs(infer(x, proprio, task, seed) - base).mean())
            rgb = ((img_t[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            # occlusion vs attention agreement: rank correlation on the 6 x 8 grid
            attn_c = attn.reshape(OCC[0], 2, OCC[1], 2).mean(axis=(1, 3))
            rk = lambda z: np.argsort(np.argsort(z.ravel()))
            rho = float(np.corrcoef(rk(attn_c), rk(occ))[0, 1])
            tag = f"ep{ep:04d}_f{s:04d}"
            np.savez(out / f"{tag}.npz", attn=attn, attn_full=maps, occ=occ, frame=rgb, base_action=base,
                     gt=action[s:s + H], video_mass=np.array(rec.video_mass))
            row1 = np.concatenate([label(rgb, f"ep {ep} frame {s}"), label(overlay(rgb, attn), "attention (all layers)"),
                                   label(overlay(rgb, occ), f"occlusion (max {occ.max():.4f} rad)")], 1)
            row2 = np.concatenate([label(overlay(rgb, attn_groups[g]), f"attention {g} layers") for g in ("early", "mid", "late")], 1)
            Image.fromarray(np.concatenate([row1, row2], 0)).save(out / f"{tag}.png")
            row = {"tag": tag, "episode": ep, "frame": int(s), "task": task, "n_layers": n_layers,
                   "video_attention_mass": float(np.mean(rec.video_mass)),
                   "attn_peak_token": [int(v) for v in np.unravel_index(attn.argmax(), GRID)],
                   "attn_top10_share": float(np.sort(attn.ravel())[-10:].sum() / attn.sum()),
                   "occ_max": float(occ.max()), "occ_mean": float(occ.mean()), "rank_corr_attn_occ": rho,
                   "mae_gt": float(np.abs(base - action[s:s + H]).mean())}
            summary.append(row)
            print(json.dumps(row), f"{time.time() - t0:.0f} s", flush=True)
    (out / "summary.json").write_text(json.dumps({"ckpt": a.ckpt, "samples": summary}, indent=1) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats", default="/datasets/G1_Dex3_AllMerged_MainCam_LeRobot21/meta/fastwam_dataset_stats.json")
    p.add_argument("--dataset", default="/datasets/G1_Dex3_AllMerged_MainCam_LeRobot21")
    p.add_argument("--task", default="g1_dex3_maincam_uncond_384x512_1e-4")
    p.add_argument("--config_dir", default="/workspace/configs")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--client_stride", type=int, default=2)
    p.add_argument("--task_index", type=int, default=0, help="dataset task to probe (0 = stacking; -1 = any)")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--per_episode", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--log_dir", default="/tmp/attention_probe_log")
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
