import torch
import torch.nn as nn
import torchvision.transforms as TF


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        x = x.to(torch.float32) / 255.0
        return x

class Pad(nn.Module):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)
    
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Can only pad tensor of 4 dims."
        return self.pad(x)


class ViewpointColorAug(nn.Module):
    """Clip-consistent camera-viewpoint + color augmentation for [T, C, H, W] float images in [0, 1].

    One random draw per call, applied to every frame of the clip (a clip = one camera pose):
      * camera rotation as a homography H = K R K^-1 (pitch, yaw, roll), the exact image change of a
        camera rotating about its centre (translation would need depth and is not modelled),
      * focal-length zoom and principal-point shift (an intrinsics change),
      * color jitter (brightness / contrast / saturation / hue), same factors for all frames.
    Nothing is invented beyond the image borders: a draw is accepted only if all four output corners
    map inside the source frame (re-drawn up to `tries` times, else the clip is left unwarped). So the
    view can tilt, rotate, shift and zoom in, but never shows more than the camera saw. With out_size the warp samples the
    output directly at that resolution (folds the following resize into the same pass); clips that are
    not warped are resized bilinearly so the output size is always out_size.
    """

    def __init__(self, p: float = 0.9, max_pitch_deg: float = 6.0, max_yaw_deg: float = 6.0,
                 max_roll_deg: float = 5.0, zoom=(1.0, 1.3), max_shift: float = 0.04, tries: int = 20,
                 focal_px: float = 384.0, color_p: float = 0.8, brightness: float = 0.2,
                 contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.0, out_size=None):
        super().__init__()
        self.p, self.max_pitch, self.max_yaw, self.max_roll = p, max_pitch_deg, max_yaw_deg, max_roll_deg
        self.zoom, self.max_shift, self.focal_px, self.color_p = tuple(zoom), max_shift, focal_px, color_p
        self.jitter = TF.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.out_size = tuple(out_size) if out_size is not None else None   # sample the output directly at this size
        self.tries = tries

    @staticmethod
    def _rot(pitch, yaw, roll):
        cx, sx, cy, sy, cz, sz = (torch.cos(pitch), torch.sin(pitch), torch.cos(yaw), torch.sin(yaw),
                                  torch.cos(roll), torch.sin(roll))
        one, zero = torch.ones(()), torch.zeros(())
        rx = torch.stack([torch.stack([one, zero, zero]), torch.stack([zero, cx, -sx]), torch.stack([zero, sx, cx])])
        ry = torch.stack([torch.stack([cy, zero, sy]), torch.stack([zero, one, zero]), torch.stack([-sy, zero, cy])])
        rz = torch.stack([torch.stack([cz, -sz, zero]), torch.stack([sz, cz, zero]), torch.stack([zero, zero, one])])
        return rz @ ry @ rx

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "expects [T, C, H, W]"
        _, _, h, w = x.shape
        oh, ow = self.out_size if self.out_size is not None else (h, w)
        hom = None
        if torch.rand(()) < self.p:
            u = lambda m: (torch.rand(()) * 2 - 1) * m
            f = self.focal_px * w / 640.0
            k_src = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
            sx, sy = ow / w, oh / h        # output pixels per source pixel
            corners = torch.tensor([[0.0, 0.0, 1.0], [ow, 0.0, 1.0], [0.0, oh, 1.0], [ow, oh, 1.0]])
            for _ in range(self.tries):
                R = self._rot(torch.deg2rad(u(self.max_pitch)), torch.deg2rad(u(self.max_yaw)), torch.deg2rad(u(self.max_roll)))
                zoom = self.zoom[0] + torch.rand(()) * (self.zoom[1] - self.zoom[0])
                k_out = torch.tensor([[f * zoom * sx, 0, (w / 2 + u(self.max_shift) * w) * sx],
                                      [0, f * zoom * sy, (h / 2 + u(self.max_shift) * h) * sy], [0, 0, 1.0]])
                # output pixel -> ray in the new camera -> rotate into the source camera -> source pixel
                cand = k_src @ R @ torch.linalg.inv(k_out)
                c = corners @ cand.T
                c = c[:, :2] / c[:, 2:3]
                if bool(((c[:, 0] >= 0) & (c[:, 0] <= w) & (c[:, 1] >= 0) & (c[:, 1] <= h)).all()):
                    hom = cand
                    break
        if hom is not None:
            ys, xs = torch.meshgrid(torch.arange(oh, dtype=torch.float32), torch.arange(ow, dtype=torch.float32), indexing="ij")
            pts = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], -1) @ hom.T
            src = pts[..., :2] / pts[..., 2:3]
            grid = torch.stack([src[..., 0] / w * 2 - 1, src[..., 1] / h * 2 - 1], -1).unsqueeze(0)
            x = torch.nn.functional.grid_sample(x, grid.expand(x.shape[0], -1, -1, -1).to(x.dtype),
                                                mode="bilinear", padding_mode="border", align_corners=False)
        elif (oh, ow) != (h, w):
            x = torch.nn.functional.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=False, antialias=True)
        if torch.rand(()) < self.color_p:
            x = self.jitter(x).clamp(0, 1)
        return x
