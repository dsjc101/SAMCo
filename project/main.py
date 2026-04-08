"""
main.py
-------
SAMCo quick-run CLI script.

Usage:
  python main.py \
    --images img1.jpg img2.jpg img3.jpg \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --output_dir ./output \
    --visualize \
    --metrics
"""

import argparse
import os
import numpy as np
from PIL import Image


from models.cosegmentation  import SAMCo
from utils.visualization   import make_results_grid, save_masks, overlay_mask_on_image
from utils.metrics         import (mask_statistics, evaluate_group,
                              generate_all_plots, plot_confidence_scores)


def parse_args():
    p = argparse.ArgumentParser(
        description="SAMCo: Automatic Co-segmentation via Semantic Consensus Prompting"
    )
    p.add_argument("--images",         nargs="+", required=True)
    p.add_argument("--sam_checkpoint", type=str,  required=True)
    p.add_argument("--sam_model_type", type=str,  default="vit_h")
    p.add_argument("--n_fg_points",    type=int,  default=5)
    p.add_argument("--n_bg_points",    type=int,  default=3)
    p.add_argument("--top_k_ratio",    type=float,default=0.30)
    p.add_argument("--n_refine_iter",  type=int,  default=2)
    p.add_argument("--output_dir",     type=str,  default="./output")
    p.add_argument("--visualize",      action="store_true", help="Save results grid")
    p.add_argument("--metrics",        action="store_true", help="Generate metric plots")
    p.add_argument("--device",         type=str,  default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load images ─────────────────────────────────────────────────
    print(f"[SAMCo] Loading {len(args.images)} images ...")
    images = []
    for path in args.images:
        img = Image.open(path).convert("RGB")
        images.append(img)
        print(f"  {os.path.basename(path):30s}  {img.width}×{img.height}")

    # ── Run SAMCo ───────────────────────────────────────────────────
    model = SAMCo(
        sam_model_type=args.sam_model_type,
        sam_checkpoint=args.sam_checkpoint,
        n_fg_points=args.n_fg_points,
        n_bg_points=args.n_bg_points,
        top_k_ratio=args.top_k_ratio,
        n_refine_iter=args.n_refine_iter,
        device=args.device,
    )

    # Run pipeline step-by-step to capture intermediate outputs
    orig_sizes    = [(img.height, img.width) for img in images]
    all_feats     = model.feat_extractor.extract_batch(images)
    saliency_maps = model.prompter.compute_consensus_saliency(all_feats)
    prompts       = model.prompter.generate_prompts(saliency_maps, model.feat_extractor, orig_sizes)
    results       = model.sam.predict_batch(images, prompts)
    masks         = [m for m, _ in results]
    scores        = [s for _, s in results]

    for _ in range(args.n_refine_iter):
        prompts = model.prompter.refine_prompts_with_masks(
            prompts, masks, saliency_maps, model.feat_extractor, orig_sizes
        )
        results = model.sam.predict_batch(images, prompts)
        masks   = [m for m, _ in results]
        scores  = [s for _, s in results]

    print(f"[SAMCo] Done. Mean SAM confidence: {np.mean(scores):.3f}")

    # ── Save masks & overlays ───────────────────────────────────────
    save_masks(masks, os.path.join(args.output_dir, "masks"))

    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    for i, (img, mask) in enumerate(zip(images, masks)):
        fname = os.path.splitext(os.path.basename(args.images[i]))[0]
        overlay_mask_on_image(img, mask).save(
            os.path.join(overlay_dir, f"{fname}_overlay.png"))
    print(f"[SAMCo] Overlays saved to {overlay_dir}")

    # ── Visualisation grid ──────────────────────────────────────────
    if args.visualize:
        grid = make_results_grid(images, masks, saliency_maps, prompts,
                                  title="SAMCo Co-segmentation")
        grid.save(os.path.join(args.output_dir, "results_grid.png"))
        print(f"[SAMCo] Grid saved to {args.output_dir}/results_grid.png")

    # ── Metric plots ────────────────────────────────────────────────
    if args.metrics:
        plot_dir = os.path.join(args.output_dir, "plots")
        saved = generate_all_plots(masks, saliency_maps, scores, output_dir=plot_dir)
        print(f"[SAMCo] {len(saved)} metric plots saved to {plot_dir}/")

    # ── Print mask stats ────────────────────────────────────────────
    stats = mask_statistics(masks)
    print(f"\n{'='*50}")
    print(f"  N images          : {stats['n_masks']}")
    print(f"  Mean FG coverage  : {stats['mean_coverage']*100:.1f}%")
    print(f"  Std FG coverage   : {stats['std_coverage']*100:.1f}%")
    print(f"  Mean pairwise Dice: {stats['mean_pairwise_dice']:.4f}")
    print(f"  Mean SAM conf.    : {np.mean(scores):.4f}")
    print(f"{'='*50}\n")

    print("[SAMCo] Pipeline complete!")


if __name__ == "__main__":
    main()
