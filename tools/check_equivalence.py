"""Regression checks against the original code and transforms (david-ct/CODE.md, "Validation").

Usage: python tools/check_equivalence.py CHECK [options]
  model            optimized models/swin3D.py vs the original: outputs and gradients in fp32 (224^3 with a MIM
                   mask, 96^3 local crops), the multi-scale outputs, and the relative-position-bias gradients
                   under fp16 and bf16 autocast (an fp32-only test once missed a broken fp16 gradient). GPU.
  local-crop-loss  iBOTLoss vs the original: 2+0 crops give the identical loss; with 4 local crops the original
                   gives them zero gradient (bug from commit eb28c32) and the fixed loss uses them. CPU.
  crop-geometry    GPU crop (gpu_transforms) vs SpatialTransformRatioCrop fed the same sampled parameters. GPU.
  aug-stats        GPUAugmentationCTiBOT vs DataAugmentationCTiBOT2: crop statistics over 40 volumes (native files
                   read from the NAS, their cache entries from the pool), plus PNG
                   grids of the central slice (original recipe, and cache + HU window + crops inside). GPU.
The original code is extracted with `git archive REF ibot` (REF defaults to aa5393e, the last upstream commit
this work starts from); --original DIR uses an ibot/ directory instead.
Expected results (2026-10): model <=1e-5 relative in fp32 and fp16 bias grads identical; local-crop-loss prints
OK; crop-geometry correlation 1.0000 (crop only) and >=0.999 (rotation+scaling); aug-stats means within ~0.01.
"""
import argparse, importlib.util, os, subprocess, sys, tempfile

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IBOT = os.path.join(REPO, "ibot")
CACHE_LIST = "/home/local_data/ct-v2/v2.0_1p25mm/ct-encoders.txt"
LOCAL_MODULES = {"utils", "models", "loader", "transforms", "gpu_transforms", "SpatialTransformRatioCrop"}


def original_ibot(args):
    """Path to the original ibot/ tree: --original, or extracted from git at --ref."""
    if args.original:
        return args.original
    tmp = tempfile.mkdtemp(prefix="ibot_original_")
    archive = subprocess.run(["git", "-C", REPO, "archive", args.ref, "ibot"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
    return os.path.join(tmp, "ibot")


def load_module(path, rel, name):
    """Import ibot/<rel> from a given tree; each tree imports its own helpers."""
    for m in [m for m in sys.modules if m.split(".")[0] in LOCAL_MODULES]:
        del sys.modules[m]
    sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(name, os.path.join(path, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


def check_model(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    old_mod = load_module(original_ibot(args), "models/swin3D.py", "swin_old")
    new_mod = load_module(IBOT, "models/swin3D.py", "swin_new")
    torch.manual_seed(0)
    old = old_mod.swin_3D(return_all_tokens=True, masked_im_modeling=True).cuda()
    new = new_mod.swin_3D(return_all_tokens=True, masked_im_modeling=True).cuda()
    new.load_state_dict(old.state_dict())
    with torch.no_grad():
        old.masked_embed.normal_()
        new.masked_embed.copy_(old.masked_embed)

    for size, use_mask in [(224, True), (96, False)]:
        x = torch.randn(2, 1, size, size, size, device="cuda")
        m = (torch.rand(2, size // 32, size // 32, size // 32, device="cuda") < 0.3) if use_mask else None
        for model in (old, new):
            model.zero_grad()
        o1, o2 = old(x.clone(), mask=m), new(x.clone(), mask=m)
        (o1.square().mean()).backward()
        (o2.square().mean()).backward()
        gd = max((a.grad - b.grad).abs().max().item() / (a.grad.abs().max().item() + 1e-12)
                 for a, b in zip(old.parameters(), new.parameters()) if a.grad is not None)
        print(f"{size}^3 mask={use_mask}: max|out diff|={(o1 - o2).abs().max().item():.2e} "
              f"(out scale {o1.abs().max().item():.2e}), max relative grad diff={gd:.2e}")

    o1 = old(x, return_all_tokens=False)  # the multi-scale output path must match too
    o2 = new(x, return_all_tokens=False)
    print("multi-scale outputs:", [f"{(a - b).abs().max().item():.1e}" for a, b in zip(o1, o2)])

    # gradients under the real autocast dtypes (fp16 must fall back to the explicit attention)
    x = torch.rand(4, 1, 224, 224, 224, device="cuda")
    m = torch.rand(4, 7, 7, 7, device="cuda") < 0.3
    for dt in (torch.float16, torch.bfloat16):
        grads = []
        for model in (old, new):
            model.zero_grad()
            with torch.autocast("cuda", dtype=dt):
                o = model(x, mask=m)
            (o.float().square().mean() * 1024).backward()  # x1024 ~ a loss scale
            grads.append({k: p.grad.float() for k, p in model.named_parameters() if p.grad is not None})
        worst = max(((grads[0][k] - grads[1][k]).norm() / grads[0][k].norm().clamp(min=1e-12)).item()
                    for k in grads[0] if "relative_position_bias_table" in k)
        print(f"{dt}: worst relative gradient diff over the relative-position bias tables = {worst:.2e}")


def check_local_crop_loss(args):
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29655")
    dist.init_process_group("gloo", rank=0, world_size=1)
    Old = load_module(original_ibot(args), "main_ibot.py", "main_old").iBOTLoss
    New = load_module(IBOT, "main_ibot.py", "main_new").iBOTLoss
    K, B, N = 512, 3, 27
    for nl in (0, 4):
        torch.manual_seed(0)
        s_cls = torch.randn(2 * B, K, requires_grad=True)
        s_patch = torch.randn(2 * B, N, K, requires_grad=True)
        t_cls, t_patch = torch.randn(2 * B, K), torch.randn(2 * B, N, K)
        s_local = torch.randn(nl * B, K, requires_grad=True) if nl else None
        masks = [torch.rand(B, 3, 3, 3) < 0.3 for _ in range(2)]
        out = {}
        for name, cls in (("original", Old), ("fixed", New)):
            loss_fn = cls(K, K, 2, nl, 0.04, 0.07, 0.04, 0.07, 30, 200, mim_start_epoch=50)
            for t in (s_cls, s_patch) + ((s_local,) if nl else ()):
                t.grad = None
            r = loss_fn((s_cls, s_patch), (t_cls, t_patch), s_local, masks, epoch=0)
            r["loss"].backward()
            g_local = s_local.grad.abs().sum().item() if nl else float("nan")
            out[name] = (r["cls"].item(), r["patch"].item(), g_local)
            print(f"2+{nl} {name:8s}: cls={out[name][0]:.6f} patch={out[name][1]:.6f} |grad local crops|={g_local:.3e}")
        if nl == 0:
            assert out["original"][:2] == out["fixed"][:2], "2+0 must be unchanged"
        else:
            assert out["original"][2] == 0 and out["fixed"][2] > 0 and out["fixed"][1] == out["original"][1]
    dist.destroy_process_group()
    print("OK: 2+0 unchanged; with local crops only the fixed loss uses them (patch loss unchanged)")


def check_crop_geometry(args):
    import torch.nn.functional as F
    sys.path.insert(0, IBOT)
    from SpatialTransformRatioCrop import SpatialTransformRatioCrop
    from gpu_transforms import GPUAugmentationCTiBOT
    from loader import list_volumes
    path = args.volume or list_volumes(CACHE_LIST)[0]
    vol = torch.from_numpy(np.load(path).astype(np.float32))[None]
    vol = (vol - vol.min()) / (vol.max() - vol.min())
    np.random.seed(0)
    for p_rot, label in [(0.0, "crop only"), (1.0, "rotation+scaling")]:
        t = SpatialTransformRatioCrop((224,) * 3, 0, True, crop_scale=(0.4, 1.0), p_rotation=p_rot,
                                      rotation=(-np.pi / 6, np.pi / 6), p_scaling=p_rot, scaling=(0.7, 1.4),
                                      p_synchronize_scaling_across_axes=1, bg_style_seg_sampling=False)
        errs, corrs = [], []
        for _ in range(args.n):
            prm = t.get_parameters(image=vol)
            cpu = t._apply_to_image(vol.clone(), **prm)
            theta = GPUAugmentationCTiBOT.crop_theta(prm["intermediate_crop_size"], prm["center_location_in_pixels"],
                                                     prm["affine"], vol.shape[1:])
            th = torch.tensor(theta, dtype=torch.float32, device="cuda")[None]
            grid = F.affine_grid(th, (1, 1, 224, 224, 224), align_corners=False)
            gpu = F.grid_sample(vol.cuda()[None], grid, mode="bilinear", padding_mode="zeros",
                                align_corners=False)[0].cpu()
            a, b = cpu.flatten(), gpu.flatten()
            errs.append(((a - b).abs().mean() / a.abs().mean().clamp(min=1e-6)).item())
            corrs.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
        print(f"{label:17s}: relative MAE median {np.median(errs):.4f} (max {np.max(errs):.4f}), "
              f"correlation median {np.median(corrs):.4f} (min {np.min(corrs):.4f})")


def check_aug_stats(args):
    sys.path.insert(0, IBOT)
    from loader import load_volume
    from transforms import DataAugmentationCTiBOT2
    from gpu_transforms import GPUAugmentationCTiBOT
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    names = [line.strip() for line in open(args.names) if line.strip()]
    paths = [os.path.join(args.raw_root, n) for n in rng.choice(names, 40, replace=False)]
    np.random.seed(0)
    torch.manual_seed(0)
    cpu = DataAugmentationCTiBOT2((0.4, 1.0), (0.05, 0.4), 2, 0)
    gpu = GPUAugmentationCTiBOT((0.4, 1.0), (0.05, 0.4), 2, 0, seed=0)
    gpu_new = GPUAugmentationCTiBOT((0.4, 1.0), (0.05, 0.4), 2, 0, intensity_norm="hu_window", crop_inside=True,
                                    seed=0)

    def stats(c):
        c = c.float()
        return [c.mean().item(), c.std().item(), (c == 0).float().mean().item(), (c > 0.95).float().mean().item()]

    S = {"cpu": [], "gpu": [], "gpu_new": []}
    mids = {"cpu": [], "gpu": [], "gpu_new": []}
    for p in paths:
        vol = load_volume(p)
        v = torch.tensor(vol, dtype=torch.float32)[None]
        for crop in cpu(v):
            S["cpu"].append(stats(crop)); mids["cpu"].append(crop[0, 112].numpy())
        for crop in gpu([torch.from_numpy(vol.astype(np.int16))[None].cuda()]):
            S["gpu"].append(stats(crop[0])); mids["gpu"].append(crop[0, 0, 112].cpu().numpy())
        cached = os.path.join(args.cache_dir, os.path.basename(p)[:-7] + ".npy")
        if os.path.exists(cached):
            for crop in gpu_new([torch.from_numpy(np.load(cached))[None].cuda()]):
                S["gpu_new"].append(stats(crop[0])); mids["gpu_new"].append(crop[0, 0, 112].cpu().numpy())

    print(f"{'pipeline':10s} {'n':>4s} {'mean':>14s} {'std':>14s} {'frac==0 (pad)':>16s} {'frac>0.95':>14s}")
    for k, v in S.items():
        a = np.array(v)
        print(f"{k:10s} {len(a):4d} " + " ".join(f"{a[:, j].mean():7.3f}±{a[:, j].std():5.3f}" for j in range(4)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for k, imgs in mids.items():
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        for ax, im in zip(axes.flat, imgs[:32]):
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        fig.suptitle(f"{k}: central axial slice of 224^3 global crops")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"qa_{k}.png"), dpi=60)
        plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="check", required=True)
    for name in ("model", "local-crop-loss"):
        p = sub.add_parser(name)
        p.add_argument("--ref", default="aa5393e", help="git ref of the original code")
        p.add_argument("--original", help="an ibot/ directory to use instead of --ref")
    p = sub.add_parser("crop-geometry")
    p.add_argument("--volume", help=".npy volume (default: the first one of the 1.25 mm cache)")
    p.add_argument("--n", type=int, default=20)
    p = sub.add_parser("aug-stats")
    p.add_argument("--raw-root", default="/home/shared_data/CT-Encoders-Dataset", help="native .nii.gz files")
    p.add_argument("--names", default=os.path.join(REPO, "david-ct", "metadata", "sample_1019.txt"))
    p.add_argument("--cache-dir", default="/home/local_data/ct-v2/v2.0_1p25mm/ct-encoders")
    p.add_argument("--out", default=os.path.join(REPO, "runs", "checks", "aug_stats"))
    args = ap.parse_args()
    {"model": check_model, "local-crop-loss": check_local_crop_loss, "crop-geometry": check_crop_geometry,
     "aug-stats": check_aug_stats}[args.check](args)
