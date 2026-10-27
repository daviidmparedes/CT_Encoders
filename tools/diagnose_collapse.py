"""Where did a run collapse? Backbone features vs head, on real crops, CPU only (safe next to a GPU run).

Usage: [GLOBAL=1] nice -n 19 python tools/diagnose_collapse.py RUN_DIR|CHECKPOINT.pth [...]
For each run's checkpoint.pth, or a given checkpoint file (teacher weights): one 96^3 crop from each of N volumes of the training list
(GLOBAL=1: one 224^3 global crop, the view the teacher actually gets in training; slower), then across samples:
- backbone [CLS] feature: mean pairwise cosine (1 = every input maps to the same vector) and the std across
  samples relative to the feature norm
- head bottleneck (the 256-d vector before the prototypes): mean pairwise cosine
- teacher target after centering, at the epoch's teacher temperature: entropy (ln 8192 = 9.01 is uniform) and
  how many distinct argmax prototypes
Checkpoints from the original code (e.g. the lab's run) lack the newer flags: they get the original recipe
(min-max per volume, crops anywhere), and volumes come from the 1.25 mm cross-test list when their data_path
is not a list on this machine.
"""
import json, math, os, sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ibot"))
from gpu_transforms import GPUAugmentationCTiBOT  # noqa: E402
from loader import list_volumes, load_volume  # noqa: E402
from models.head import iBOTHead  # noqa: E402
from models.swin3D import swin_3D  # noqa: E402

torch.set_num_threads(8)
N = 8
GLOBAL = os.environ.get("GLOBAL") == "1"
CACHE_LIST = "/home/local_data/ct-v2/v2.0_1p25mm/ct-encoders.txt"


def pairwise_cos(x):
    x = F.normalize(x.float(), dim=-1)
    c = x @ x.T
    return c[~torch.eye(len(c), dtype=bool)].mean().item()


for run in sys.argv[1:]:
    path = run if run.endswith(".pth") else os.path.join(run, "checkpoint.pth")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    for k, v in dict(intensity_norm="minmax", hu_window=(-1000.0, 1000.0), crop_inside=False).items():
        if not hasattr(a, k):
            setattr(a, k, v)  # original code: these flags didn't exist
    data = a.data_path if a.data_path.endswith(".txt") and os.path.exists(a.data_path) else CACHE_LIST
    backbone = swin_3D(return_all_tokens=True, fused_attn=True)
    head = iBOTHead(backbone.num_features, a.out_dim, patch_out_dim=a.patch_out_dim, norm=a.norm_in_head,
                    act=a.act_in_head, shared_head=a.shared_head_teacher)
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in ck["teacher"].items()}
    backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
    head.load_state_dict({k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")})
    backbone.eval(), head.eval()

    files = list_volumes(data)
    pick = np.random.default_rng(0).choice(len(files), N, replace=False)
    aug = GPUAugmentationCTiBOT(a.global_crops_scale, a.local_crops_scale, int(GLOBAL), int(not GLOBAL),
                                intensity_norm=a.intensity_norm, hu_window=tuple(a.hu_window),
                                crop_inside=a.crop_inside, seed=0)
    vols = [torch.from_numpy(load_volume(files[i]).astype(np.int16)).unsqueeze(0) for i in pick]
    crops = aug(vols)[0]  # (N, 1, 96, 96, 96), or 224^3 with GLOBAL=1

    with torch.no_grad():
        tokens = backbone(crops)  # (N, 1 + patches, C)
        cls = tokens[:, 0]
        mlp = head.mlp(tokens[:, :1]).squeeze(1)  # bottleneck before l2-normalization
        logits = head(tokens)[0]  # [CLS] logits (N, out_dim)
    ep = ck["epoch"]
    w = a.warmup_teacher_temp_epochs
    temp = a.warmup_teacher_temp + (a.teacher_temp - a.warmup_teacher_temp) * min(ep, w - 1) / (w - 1)
    center = ck["ibot_loss"]["center"].squeeze(0).float()
    p = F.softmax((logits.float() - center) / temp, dim=-1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean().item()
    print(f"== {run} epoch {ep}, teacher temp {temp:.3f}, {N} volumes, {crops.shape[-1]}^3 crops, "
          f"{a.intensity_norm}, crop_inside={a.crop_inside}")
    print(f"   backbone CLS: mean pairwise cos {pairwise_cos(cls):.4f}, "
          f"std across samples / norm {(cls.std(0).norm() / cls.norm(dim=-1).mean()).item():.4f}")
    print(f"   head bottleneck: mean pairwise cos {pairwise_cos(mlp):.4f}, norm {mlp.norm(dim=-1).mean().item():.3f}")
    print(f"   logits: std across prototypes {logits.std(-1).mean().item():.4f}, "
          f"after centering {(logits - center).std(-1).mean().item():.4f}, center std {center.std().item():.4f}")
    print(f"   teacher target: entropy {ent:.3f} (uniform {math.log(a.out_dim):.3f}), "
          f"argmax prototypes {len(set(p.argmax(-1).tolist()))}/{N}")
