"""
feature_extractor.py
--------------------
DINOv2-based patch-level feature extractor.
We use the Vision Transformer (ViT-S/14) variant of DINOv2 to obtain
dense, semantically-rich patch embeddings for each input image.

These features are the foundation of our Semantic Consensus Prompting (SCP) module.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DINO_PATCH_SIZE = 14          # DINOv2 ViT-S/14 patch size in pixels
DINO_IMG_SIZE   = 518         # Canonical input size (37×37 patches → 1369 tokens)
DINO_FEAT_DIM   = 384         # ViT-S/14 embedding dimension


# ──────────────────────────────────────────────────────────────────────────────
# Transform
# ──────────────────────────────────────────────────────────────────────────────
dino_transform = T.Compose([
    T.Resize((DINO_IMG_SIZE, DINO_IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class DINOv2FeatureExtractor:
    """
    Wraps Meta's DINOv2 ViT-S/14 model to extract patch tokens.

    For an image of size (DINO_IMG_SIZE × DINO_IMG_SIZE), the model produces
    (DINO_IMG_SIZE / DINO_PATCH_SIZE)^2 = 37×37 = 1369 patch tokens,
    each of dimension DINO_FEAT_DIM = 384.

    We return these as a spatial feature map of shape (H_p, W_p, D)
    where H_p = W_p = DINO_IMG_SIZE // DINO_PATCH_SIZE = 37.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"[DINOv2] Loading DINOv2 ViT-S/14 on {device} ...")
        # Load from torch.hub (downloads ~80 MB on first run)
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=True,
        ).to(device).eval()
        print("[DINOv2] Model loaded.")

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Extract patch features from a single PIL image.

        Args:
            image: PIL.Image.Image (RGB)

        Returns:
            feats: np.ndarray of shape (H_p, W_p, D)
                   e.g. (37, 37, 384) for 518×518 input
        """
        tensor = dino_transform(image.convert("RGB")).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Forward pass — get_intermediate_layers returns a list of layer outputs
        # We take the last layer's patch tokens (excluding the [CLS] token)
        outputs = self.model.get_intermediate_layers(tensor, n=1, return_class_token=False)
        # outputs[0]: (1, N_patches, D)  where N_patches = H_p * W_p
        patch_tokens = outputs[0].squeeze(0)  # (N_patches, D)

        H_p = W_p = DINO_IMG_SIZE // DINO_PATCH_SIZE
        feats = patch_tokens.reshape(H_p, W_p, -1).cpu().numpy()  # (37, 37, 384)
        return feats

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_batch(self, images: list) -> list:
        """
        Extract features for a list of PIL images.

        Args:
            images: list of PIL.Image.Image

        Returns:
            list of np.ndarray, each of shape (H_p, W_p, D)
        """
        return [self.extract(img) for img in images]

    # ------------------------------------------------------------------ #
    def get_patch_grid_size(self) -> tuple:
        """Returns (H_p, W_p) = (37, 37) for the default settings."""
        H_p = W_p = DINO_IMG_SIZE // DINO_PATCH_SIZE
        return H_p, W_p

    def patch_to_pixel_coords(self, patch_row: int, patch_col: int,
                               orig_h: int, orig_w: int) -> tuple:
        """
        Convert (patch_row, patch_col) indices back to pixel coordinates
        in the *original* image space.

        Returns:
            (px, py): pixel (x, y) in the original image
        """
        H_p, W_p = self.get_patch_grid_size()
        # Centre of the patch in the resized (DINO_IMG_SIZE) space
        cx_resized = (patch_col + 0.5) * DINO_PATCH_SIZE
        cy_resized = (patch_row + 0.5) * DINO_PATCH_SIZE
        # Scale back to original image size
        px = int(cx_resized / DINO_IMG_SIZE * orig_w)
        py = int(cy_resized / DINO_IMG_SIZE * orig_h)
        return px, py
