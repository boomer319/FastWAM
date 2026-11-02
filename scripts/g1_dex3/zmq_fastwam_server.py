#!/usr/bin/env python3
"""FastWAM policy server for G1 Dex3 over a ZMQ action protocol.

The wire format is compatible with the UnifoLM-WMA ZMQ server
(scripts/evaluation/zmq_wma_server.py), so these clients work unchanged:
  * the Isaac Lab sim action provider for UnifoLM-WMA
  * the real-robot UnifoLM-WMA client
Both send `observation.images.top` (2, H, W, 3) uint8 + `observation.state`
(2, 28) and expect a (16, 28) chunk executed at 15 Hz.

Transport: zmq.REP, msgpack + `__ndarray__` (identical packer/unpacker),
single client, one request at a time. Endpoints: metadata / infer /
save_video (no-op) / reset (no-op; the policy is stateless).

Model side (first-frame "uncond" FastWAM, as trained):
  * image: the NEWEST frame only -> [0,1] -> resize smallest side + center
    crop to 384x512 (for 640x480 this is exactly the training Resize) ->
    [-1, 1]. Same order as RobotVideoDataset + FastWAMProcessor.
  * state: the newest (28,) row, z-scored with the frozen training stats.
  * prompt: DEFAULT_PROMPT.format(task=instruction), T5-encoded on the fly
    (load_text_encoder=True, so any instruction works on the robot).
  * output: 32 actions @ 30 Hz (1.07 s), de-normalised; the client gets
    every 2nd step -> 16 actions @ 15 Hz covering the same 1.07 s
    (step k of the 15 Hz client happens at frame 2k). The full 30 Hz chunk
    is returned too as `action_30hz` for future clients.

Every infer is appended to <log_dir>/infer_log.jsonl (latency, prompt,
state, chunk) for offline analysis; metadata reports the checkpoint sha256.

Optional --attn_map: the reply also carries `attn_map` (12, 16) float32, the
action tokens' attention to the first-frame video tokens (one cell ~ 32x32 px
of the 384x512 model input), mean over the early layers (0..attn_layers-1),
heads, queries and denoising steps, normalised to sum 1. Recomputed next to
the fused attention call (output unchanged), accumulated on the GPU, one copy
per chunk. Early layers only: later layers park their attention on a
content-free sink token (doc: attention_probe.py). Clients ignore the key.
"""
import argparse
import functools
import hashlib
import json
import logging
import time
import traceback
from pathlib import Path

import msgpack
import numpy as np
import torch
import torchvision.transforms.functional as TF
import zmq
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
import fastwam.models.wan22.mot as mot_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [zmq_fastwam_server] %(message)s")


# --- msgpack + __ndarray__ (verbatim from zmq_wma_server.py) -------------------

def _pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class LiveAttention:
    """Wraps mot.flash_attention; sums early-layer action->video attention on the GPU."""

    def __init__(self, n_layers: int, early: int, n_video: int, n_action: int):
        self.n_layers, self.early, self.n_video, self.n_action = n_layers, early, n_video, n_action
        self.enabled = True
        self._orig = mot_mod.flash_attention
        self.reset()

    def reset(self):
        self.calls, self.acc = 0, None

    def __call__(self, q, k, v, num_heads, ctx_mask=None, compatibility_mode=True):
        out = self._orig(q, k, v, num_heads, ctx_mask=ctx_mask, compatibility_mode=compatibility_mode)
        if self.enabled and q.shape[1] == self.n_action and k.shape[1] == self.n_video + self.n_action:
            layer = self.calls % self.n_layers                  # calls run step-major, layer-minor
            self.calls += 1
            if layer < self.early:
                b, sq, hd = q.shape
                d = hd // num_heads
                qh = q.view(b, sq, num_heads, d).transpose(1, 2)
                kh = k.view(b, k.shape[1], num_heads, d).transpose(1, 2)
                logits = (qh @ kh.transpose(-1, -2)).float() / d ** 0.5
                if ctx_mask is not None and ctx_mask.dtype == torch.bool:
                    logits = logits.masked_fill(~ctx_mask.to(logits.device), float("-inf"))
                w = logits.softmax(-1)[..., : self.n_video].sum(dim=(0, 1, 2))   # [Sv]
                self.acc = w if self.acc is None else self.acc + w
        return out

    def take(self, grid) -> np.ndarray:
        m = self.acc.cpu().numpy().astype(np.float32)          # the one device->host copy per chunk
        self.reset()
        return (m / m.sum()).reshape(grid)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


class FastWamPolicy:
    def __init__(self, args):
        self.args = args
        with initialize_config_dir(config_dir=str(Path(args.config_dir).resolve()), version_base="1.3"):
            cfg = compose("train", overrides=[f"task={args.task}", "model.redirect_common_files=false"])
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        t0 = time.time()
        self.model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cuda")
        self.model.load_checkpoint(args.ckpt)
        self.model = self.model.to("cuda").eval()
        logging.info("model + %s loaded in %.1f s", args.ckpt, time.time() - t0)

        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(args.stats))
        self.state_key = self.processor.shape_meta["state"][0]["key"]
        self.action_key = self.processor.shape_meta["action"][0]["key"]
        self.height, self.width = (int(x) for x in cfg.data.train.video_size)
        self.horizon = int(cfg.data.train.num_frames) - 1          # 32 @ 30 Hz
        self.stride = int(args.client_stride)                      # 2 -> 15 Hz
        self.ckpt_sha256 = _sha256(args.ckpt)
        self.log_path = Path(args.log_dir) / "infer_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.n_infer = 0
        # optional camera remap: grid (H, W, 2) in [-1, 1] for grid_sample, applied to the raw frame
        self.warp = None
        if getattr(args, "image_warp", ""):          # absent when reused by ablation_input.py
            g = np.load(args.image_warp)["grid"].astype(np.float32)
            self.warp = torch.from_numpy(g).unsqueeze(0)
            logging.info("image warp %s, grid %s", args.image_warp, tuple(g.shape))
        self.attn = None
        self.attn_grid = (self.height // 32, self.width // 32)  # VAE 16x, patch 2x2
        if getattr(args, "attn_map", False):
            self.attn = LiveAttention(self.model.mot.num_layers, args.attn_layers,
                                      self.attn_grid[0] * self.attn_grid[1], self.horizon)
            mot_mod.flash_attention = self.attn
            logging.info("attn_map on: layers 0..%d, grid %s", args.attn_layers - 1, self.attn_grid)

    def metadata(self) -> dict:
        return {
            "result": "ok",
            "model": "fastwam_uncond",
            "task_config": self.args.task,
            "ckpt": self.args.ckpt,
            "ckpt_sha256": self.ckpt_sha256,
            "stats": self.args.stats,
            "action_dim": 28,
            "n_obs": 2,                       # accepted; only the newest is used
            "video_length": self.horizon // self.stride,   # client chunk length
            "native_horizon": self.horizon,
            "native_hz": 30,
            "client_hz": 30 // self.stride,
            "num_inference_steps": self.args.num_inference_steps,
            "noise": "fixed" if self.args.fixed_noise else "per_call(seed+n)",
            "seed": self.args.seed,
            "height": self.height,
            "width": self.width,
            "image_shape": [2, 480, 640, 3],
            "image_warp": getattr(self.args, "image_warp", "") or None,
            "attn_map": None if self.attn is None else {"grid": list(self.attn_grid), "layers": self.args.attn_layers},
            "note": "reply action = (video_length, 28) at client_hz; action_30hz = full chunk",
        }

    def _image(self, frames: np.ndarray) -> torch.Tensor:
        img = frames[-1] if frames.ndim == 4 else frames
        if img.shape[0] == 3 and img.shape[-1] != 3:      # (3, H, W) -> (H, W, 3)
            img = np.transpose(img, (1, 2, 0))
        x = torch.from_numpy(np.ascontiguousarray(img[..., :3])).permute(2, 0, 1).float() / 255.0
        if self.warp is not None and tuple(self.warp.shape[1:3]) == tuple(x.shape[-2:]):
            x = torch.nn.functional.grid_sample(x.unsqueeze(0), self.warp, mode="bilinear",
                                                padding_mode="border", align_corners=False)[0]
        h, w = x.shape[-2:]
        scale = max(self.height / h, self.width / w)
        x = TF.resize(x, [round(h * scale), round(w * scale)], antialias=True)
        x = TF.center_crop(x, [self.height, self.width])
        return (x * 2.0 - 1.0).unsqueeze(0)                # [1, 3, H, W] in [-1, 1]

    def _norm_state(self, state: np.ndarray) -> torch.Tensor:
        s = torch.as_tensor(np.asarray(state, dtype=np.float32)[-1].reshape(1, -1))
        batch = self.processor.normalizer.forward({"state": {self.state_key: s}})
        return batch["state"][self.state_key]              # [1, 28]

    def infer(self, payload: dict) -> dict:
        t0 = time.time()
        instr = payload.get("language_instruction", [""])
        instr = instr[0] if isinstance(instr, (list, tuple)) else str(instr)
        if isinstance(instr, bytes):
            instr = instr.decode()
        image = self._image(np.asarray(payload["observation.images.top"]))
        proprio = self._norm_state(payload["observation.state"])
        with torch.no_grad():
            pred = self.model.infer_action(
                prompt=DEFAULT_PROMPT.format(task=instr),
                input_image=image,
                action_horizon=self.horizon,
                proprio=proprio,
                num_inference_steps=self.args.num_inference_steps,
                text_cfg_scale=1.0,
                # per-call noise: base seed + call counter (reproducible, but a
                # different draw every inference, as in training). With one fixed
                # draw a bias in that draw repeats every chunk and compounds in
                # closed loop; --fixed_noise restores the old behaviour.
                seed=self.args.seed if self.args.fixed_noise else self.args.seed + self.n_infer,
            )
        a = pred["action"]
        if a.ndim == 2:
            a = a.unsqueeze(0)
        norm = self.processor.normalizer.normalizers["action"][self.action_key]
        chunk30 = norm.backward(a.to(dtype=torch.float32, device="cpu"))[0].numpy().astype(np.float32)
        chunk = np.ascontiguousarray(chunk30[:: self.stride])
        attn = self.attn.take(self.attn_grid) if self.attn is not None and self.attn.acc is not None else None
        latency = round(time.time() - t0, 3)
        self.n_infer += 1
        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "t": time.time(), "n": self.n_infer, "latency_s": latency, "prompt": instr,
                "state": np.asarray(payload["observation.state"], dtype=np.float32)[-1].tolist(),
                "action_30hz": chunk30.tolist(),
                **({"attn_map": np.round(attn, 5).tolist()} if attn is not None else {}),
            }) + "\n")
        reply = {"result": "ok", "action": chunk, "action_30hz": chunk30,
                 "desc": "success", "latency_s": latency}
        if attn is not None:
            reply["attn_map"] = attn
        return reply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats", default="/datasets/G1_Dex3_AllMerged_MainCam_LeRobot21/meta/fastwam_dataset_stats.json")
    p.add_argument("--task", default="g1_dex3_maincam_uncond_384x512_1e-4")
    p.add_argument("--config_dir", default="/workspace/configs")
    p.add_argument("--num_inference_steps", type=int, default=10)   # paper / upstream deploy default
    p.add_argument("--client_stride", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fixed_noise", action="store_true",
                   help="use the same noise seed for every inference (old behaviour)")
    p.add_argument("--image_warp", default="",
                   help="npz with 'grid' (H, W, 2) in [-1, 1]: remap the raw frame before preprocessing "
                        "(e.g. to emulate a different camera focal length / distortion / rotation)")
    p.add_argument("--attn_map", action="store_true",
                   help="return the early-layer action->image attention map (12x16) with every chunk")
    p.add_argument("--attn_layers", type=int, default=10, help="layers 0..N-1 averaged for --attn_map")
    p.add_argument("--log_dir", default="logs/fastwam_server")
    p.add_argument("--zmq_host", default="0.0.0.0")
    p.add_argument("--zmq_port", type=int, default=5580)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    policy = FastWamPolicy(args)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://{args.zmq_host}:{args.zmq_port}")
    logging.info("FastWAM ZMQ server ready on tcp://%s:%d (ZMQ action protocol, single client)",
                 args.zmq_host, args.zmq_port)
    while True:
        req = _unpackb(sock.recv())
        ep = req.get("endpoint")
        t0 = time.time()
        try:
            if ep == "metadata":
                reply = policy.metadata()
            elif ep == "infer":
                reply = policy.infer(req)
            elif ep in ("save_video", "reset"):
                reply = {"result": "ok", "desc": "no-op"}
            else:
                reply = {"result": "error", "desc": f"unknown endpoint {ep!r}"}
        except Exception:  # noqa: BLE001
            reply = {"result": "error", "desc": traceback.format_exc()}
            logging.error(reply["desc"])
        sock.send(_packer().pack(reply))
        logging.info("endpoint=%s -> %s in %.2f s", ep, reply.get("result", "?"), time.time() - t0)


if __name__ == "__main__":
    main()
