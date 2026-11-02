#!/usr/bin/env python3
"""Attention overlay of a closed-loop run: server log dir (--attn_map --attn_frames) -> mp4 + contact sheet.

Per chunk: the model input frame (384 x 512, as the model saw it) with the live action->video
attention map (early layers, 12 x 16 tokens) upsampled bicubically and coloured with jet.
One colour scale for the whole run (0 .. 99th percentile of all cells), so frames are comparable
over time; the attention-sink token (row 1, col 0) is drawn but excluded from that scale.
Needs numpy, PIL and ffmpeg (for the mp4).

  render_attn_sim.py <server_log_dir> <out_dir> [--fps 2] [--sheet 12]
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SINK = (1, 0)


def jet(h):
    r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def overlay(rgb, attn, vmax, alpha=0.5):
    h = np.asarray(Image.fromarray(attn.astype(np.float32), mode="F").resize(rgb.shape[1::-1], Image.BICUBIC))
    h = np.clip(h / vmax, 0, 1)
    return (rgb * (1 - alpha) + jet(h) * alpha).astype(np.uint8)


def label(im, txt):
    pil = Image.fromarray(im)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, 8 + 6 * len(txt), 18], fill=(0, 0, 0))
    d.text((4, 3), txt, fill=(255, 255, 255))
    return np.asarray(pil)


def main(a):
    src, out = Path(a.log_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(src / "infer_log.jsonl")]
    rows = [r for r in rows if "attn_map" in r and (src / "attn_frames" / f"{r['n']:05d}.jpg").exists()]
    assert rows, "no chunks with attn_map + frame"
    maps = np.stack([np.asarray(r["attn_map"], np.float32) for r in rows])
    scale = maps.copy()
    scale[:, SINK[0], SINK[1]] = 0
    vmax = float(np.percentile(scale, 99))
    t0 = rows[0]["t"]
    fdir = out / "frames"
    fdir.mkdir(exist_ok=True)
    panels, peaks = [], []
    for i, (r, m) in enumerate(zip(rows, maps)):
        rgb = np.asarray(Image.open(src / "attn_frames" / f"{r['n']:05d}.jpg").convert("RGB"))
        mm = m.copy()
        mm[SINK] = 0
        pk = [int(v) for v in np.unravel_index(mm.argmax(), mm.shape)]
        peaks.append({"n": r["n"], "t_s": round(r["t"] - t0, 2), "peak_cell": pk, "sink_share": float(m[SINK])})
        pan = np.concatenate([label(rgb, f"chunk {r['n']}  t={r['t'] - t0:5.1f} s"),
                              label(overlay(rgb, m, vmax), f"attention early layers (peak {pk[0]},{pk[1]})")], 1)
        Image.fromarray(pan).save(fdir / f"{i:05d}.png")
        panels.append(pan)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(fdir / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    str(out / "attention_overlay.mp4")], check=True)
    idx = np.linspace(0, len(panels) - 1, min(a.sheet, len(panels))).astype(int)
    tiles = [np.asarray(Image.fromarray(label(overlay(
        np.asarray(Image.open(src / "attn_frames" / f"{rows[i]['n']:05d}.jpg").convert("RGB")), maps[i], vmax),
        f"t={rows[i]['t'] - t0:.0f} s"))) for i in idx]
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[k:k + cols], 1) for k in range(0, len(tiles), cols)], 0)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2), Image.LANCZOS).save(out / "contact_sheet.png")
    mean = maps.mean(0)
    (out / "summary.json").write_text(json.dumps({"log_dir": str(src), "chunks": len(rows), "vmax": vmax,
                                                  "mean_map": np.round(mean, 5).tolist(), "per_chunk": peaks}, indent=1) + "\n")
    print(f"{len(rows)} chunks -> {out}/attention_overlay.mp4, contact_sheet.png (vmax {vmax:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log_dir")
    p.add_argument("out_dir")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--sheet", type=int, default=12)
    main(p.parse_args())
