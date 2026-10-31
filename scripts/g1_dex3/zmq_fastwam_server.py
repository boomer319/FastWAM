#!/usr/bin/env python3
"""FastWAM policy server speaking the UnifoLM-WMA ZMQ protocol (G1 Dex3).

Drop-in for unifolm_wma scripts/evaluation/zmq_wma_server.py, so the existing
clients work unchanged:
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
            "height": self.height,
            "width": self.width,
            "image_shape": [2, 480, 640, 3],
            "note": "reply action = (video_length, 28) at client_hz; action_30hz = full chunk",
        }

    def _image(self, frames: np.ndarray) -> torch.Tensor:
        img = frames[-1] if frames.ndim == 4 else frames
        if img.shape[0] == 3 and img.shape[-1] != 3:      # (3, H, W) -> (H, W, 3)
            img = np.transpose(img, (1, 2, 0))
        x = torch.from_numpy(np.ascontiguousarray(img[..., :3])).permute(2, 0, 1).float() / 255.0
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
                seed=self.args.seed,
            )
        a = pred["action"]
        if a.ndim == 2:
            a = a.unsqueeze(0)
        norm = self.processor.normalizer.normalizers["action"][self.action_key]
        chunk30 = norm.backward(a.to(dtype=torch.float32, device="cpu"))[0].numpy().astype(np.float32)
        chunk = np.ascontiguousarray(chunk30[:: self.stride])
        latency = round(time.time() - t0, 3)
        self.n_infer += 1
        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "t": time.time(), "n": self.n_infer, "latency_s": latency, "prompt": instr,
                "state": np.asarray(payload["observation.state"], dtype=np.float32)[-1].tolist(),
                "action_30hz": chunk30.tolist(),
            }) + "\n")
        return {"result": "ok", "action": chunk, "action_30hz": chunk30,
                "desc": "success", "latency_s": latency}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats", default="/datasets/G1_Dex3_AllMerged_MainCam_LeRobot21/meta/fastwam_dataset_stats.json")
    p.add_argument("--task", default="g1_dex3_maincam_uncond_384x512_1e-4")
    p.add_argument("--config_dir", default="/workspace/configs")
    p.add_argument("--num_inference_steps", type=int, default=10)   # paper / upstream deploy default
    p.add_argument("--client_stride", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
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
    logging.info("FastWAM ZMQ server ready on tcp://%s:%d (WMA protocol, single client)",
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
