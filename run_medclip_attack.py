"""
Reproduce the MedCLIP explanation-robustness experiment on a chest X-ray.

Runs a gradient-based adversarial attack on MedCLIP (Swin-T vision encoder)
and compares patch-text similarity maps with and without "surgery"
(MedCLIP-Surgery, see medclip/medclip_surgery_model.py) on clean vs.
adversarial inputs. All figures are saved to --out (default results/medclip_xray).

Usage:
    python run_medclip_attack.py --image path/to/chest_xray.jpg \
        --target "chronic rib deformity" --attack-class "no acute cardiopulmonary process"
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import AutoTokenizer

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip.build_model import build_model

DEFAULT_TEXTS = [
    "focal consolidation",
    "pleural effusion",
    "pneumothorax",
    "bilateral nodular opacities",
    "nipple shadows",
    "normal cardiomediastinal silhouette",
    "surgical clips in left lung",
    "normal upper abdomen",
    "chronic rib deformity",
    "posterior rib fracture",
    "no acute cardiopulmonary process",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Chest X-ray image (jpg/png)")
    p.add_argument("--target", default="chronic rib deformity",
                   help="Finding whose similarity map is visualised")
    p.add_argument("--attack-class", default="no acute cardiopulmonary process",
                   help="Finding whose image-text similarity the attack suppresses")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.01, help="SGD step size")
    p.add_argument("--out", default="results/medclip_xray")
    return p.parse_args()


def generate_adversarial_image(model, image, text_features, idx, steps, epsilon):
    adv = image.clone().detach().requires_grad_(True)
    opt = torch.optim.SGD([adv], lr=epsilon)
    for _ in range(steps):
        opt.zero_grad()
        feat = model.vision_model(adv, project=True, return_patch_tokens=False)
        feat = F.normalize(feat, dim=-1)
        loss = -(feat @ text_features.T)[0, idx]
        loss.backward()
        opt.step()
        with torch.no_grad():
            adv.clamp_(0, 1)
    return adv.detach()


def patch_similarity(encoder, image, text_features, centered):
    with torch.no_grad():
        p = encoder(image, project=True, return_patch_tokens=True)[:, 1:, :]
        p = F.normalize(p, dim=-1)
        if centered:
            p = p - p.mean(dim=1, keepdim=True)
        return p @ text_features.T  # [B, P, N]


def grid_for(n_patches):
    return {48: (6, 8), 49: (7, 7), 196: (14, 14)}[n_patches]


def to_heatmap(sim, img_hw):
    s = sim.cpu().numpy()
    s = (s - s.min()) / (s.max() - s.min() + 1e-5)
    s = s.reshape(grid_for(s.shape[0]))
    return cv2.resize(s, (img_hw[1], img_hw[0]))


def blend(sim_map, bgr):
    vis = cv2.applyColorMap((sim_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(cv2.addWeighted(bgr, 0.4, vis, 0.6, 0), cv2.COLOR_BGR2RGB)


def save_pair(m1, m2, l1, l2, bgr, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=14)
    for a, m, l in zip(ax, (m1, m2), (l1, l2)):
        a.imshow(blend(m, bgr)); a.set_title(l); a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = DEFAULT_TEXTS
    t_idx, a_idx = texts.index(args.target), texts.index(args.attack_class)

    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    preprocess = Compose([Resize((224, 224)), ToTensor(), Normalize([0.5] * 3, [0.5] * 3)])
    pil = Image.open(args.image).convert("RGB")
    x = preprocess(pil).unsqueeze(0).to(device)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    hw = bgr.shape[:2]

    tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    text_features = F.normalize(model.encode_text(tok["input_ids"], tok["attention_mask"]), dim=-1).detach()

    x_adv = generate_adversarial_image(model, x, text_features, a_idx, args.steps, args.epsilon)

    surgery = build_model("MCS-MedCLIP", model.state_dict()).to(device).eval()
    # build_model loads with strict=False and MedCLIP's keys are prefixed "vision_model.",
    # while MedCLIPSurgery expects "visual." -- so the vision weights silently fail to load.
    # Load them explicitly so the surgery branch uses the same pretrained MedCLIP encoder.
    surgery.visual.load_state_dict(model.vision_model.state_dict())

    maps = {
        "clean": patch_similarity(model.vision_model, x, text_features, False),
        "adv": patch_similarity(model.vision_model, x_adv, text_features, False),
        "clean_surg": patch_similarity(surgery.visual, x, text_features, True),
        "adv_surg": patch_similarity(surgery.visual, x_adv, text_features, True),
    }
    m = {k: to_heatmap(v[0, :, t_idx], hw) for k, v in maps.items()}

    tag = args.target.replace(" ", "_")
    save_pair(m["clean"], m["adv"], "Clean", "Adversarial", bgr,
              f"MedCLIP (no surgery) - {args.target}", f"{args.out}/1_nosurgery_clean_vs_adv_{tag}.png")
    save_pair(m["clean_surg"], m["adv_surg"], "Clean + Surgery", "Adversarial + Surgery", bgr,
              f"MedCLIP Surgery - {args.target}", f"{args.out}/2_surgery_clean_vs_adv_{tag}.png")
    save_pair(m["clean"], m["clean_surg"], "No Surgery", "With Surgery", bgr,
              f"Clean image - {args.target}", f"{args.out}/3_clean_nosurgery_vs_surgery_{tag}.png")
    save_pair(m["adv"], m["adv_surg"], "No Surgery", "With Surgery", bgr,
              f"Adversarial image - {args.target}", f"{args.out}/4_adv_nosurgery_vs_surgery_{tag}.png")

    # Original / adversarial / amplified perturbation
    delta = (x_adv - x).squeeze().cpu()
    panels = [np.asarray(pil) / 255.0,
              x_adv.squeeze().cpu().permute(1, 2, 0).numpy().clip(0, 1),
              (delta.permute(1, 2, 0) * 50 + 0.5).clamp(0, 1).numpy()]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, img, t in zip(ax, panels, ["Original", "Adversarial", "Perturbation x50"]):
        a.imshow(img); a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig(f"{args.out}/5_perturbation.png", dpi=150); plt.close(fig)

    # Zero-shot scores before/after the attack
    with torch.no_grad():
        s_clean = (F.normalize(model.vision_model(x, project=True), dim=-1) @ text_features.T)[0]
        s_adv = (F.normalize(model.vision_model(x_adv, project=True), dim=-1) @ text_features.T)[0]
    with open(f"{args.out}/scores.csv", "w") as f:
        f.write("finding,clean_similarity,adversarial_similarity\n")
        for t, c, a in zip(texts, s_clean.tolist(), s_adv.tolist()):
            f.write(f"{t},{c:.4f},{a:.4f}\n")
    print(f"Saved figures and scores.csv to {args.out}")


if __name__ == "__main__":
    main()
