"""
CLIP-based 3D asset retrieval from Objaverse dataset.

This module provides two modes:
1. Pre-computed features mode: Fast retrieval using pre-computed CLIP embeddings (no CLIP model needed)
2. Live encoding mode: Encode new images with CLIP model (requires CLIP installation)
"""

from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


@dataclass
class AssetMatch:
    """A matched 3D asset from the database."""

    asset_id: str
    similarity: float
    asset_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CLIPRetrieval:
    """
    CLIP-based retrieval system for 3D assets from Objaverse.

    Supports two modes:
    - use_precomputed=True: Use pre-computed features (fast, no CLIP model needed)
    - use_precomputed=False: Encode images with CLIP model (requires CLIP installation)
    """

    def __init__(
        self,
        features_path: Optional[str] = None,
        annotations_path: Optional[str] = None,
        assets_base_path: Optional[str] = None,
        device: str = "cuda",
        use_precomputed: bool = True,
    ) -> None:
        """
        Initialize CLIP retrieval system.

        Args:
            features_path: Path to pre-computed CLIP features pickle file
            annotations_path: Path to annotations JSON file
            assets_base_path: Base path to 3D asset files
            device: Device for CLIP model ('cuda' or 'cpu')
            use_precomputed: If True, use pre-computed features only (no CLIP model needed)
        """
        self.device = device
        self.use_precomputed = use_precomputed

        # Set default paths if not provided
        if features_path is None:
            features_path = str(
                Path.home() / ".objathor-assets" / "2023_09_23" / "features" / "clip_features.pkl"
            )
        if annotations_path is None:
            annotations_path = str(
                Path.home() / ".objathor-assets" / "2023_09_23" / "annotations.json.gz"
            )
        if assets_base_path is None:
            assets_base_path = str(
                Path.home() / ".objathor-assets" / "2023_09_23" / "assets"
            )

        self.features_path = Path(features_path)
        self.annotations_path = Path(annotations_path)
        self.assets_base_path = Path(assets_base_path)

        # Lazy loading
        self._clip_model = None
        self._clip_preprocess = None
        self._features_db = None  # Dict mapping asset_id -> feature vector
        self._annotations = None

    def _load_clip_model(self):
        """Lazy load CLIP model (only needed if use_precomputed=False)."""
        if self._clip_model is None:
            if self.use_precomputed:
                raise RuntimeError(
                    "CLIP model not needed when use_precomputed=True. "
                    "Set use_precomputed=False to enable live encoding."
                )

            try:
                import clip
                import torch

                self._clip_model, self._clip_preprocess = clip.load(
                    "ViT-L/14", device=self.device
                )
                self._clip_model.eval()
            except ImportError:
                raise RuntimeError(
                    "CLIP not installed. Please run: pip install git+https://github.com/openai/CLIP.git"
                )

    def _load_features_db(self):
        """
        Lazy load pre-computed CLIP features database.

        The features file contains:
        - 'uids': List of asset IDs
        - 'img_features': numpy array of shape (N, 3, 768) where N is number of assets
        """
        if self._features_db is None:
            if not self.features_path.exists():
                raise FileNotFoundError(
                    f"CLIP features not found at {self.features_path}. "
                    "Please download the features using objathor."
                )

            print(f"  加载CLIP特征: {self.features_path}")
            with open(self.features_path, "rb") as f:
                features_data = pickle.load(f)

            # Extract UIDs and features
            uids = features_data["uids"]
            img_features = features_data["img_features"]  # Shape: (N, 3, 768)

            # Average over the 3 views to get a single feature vector per asset
            # Shape: (N, 768)
            avg_features = np.mean(img_features, axis=1)

            # Normalize features
            norms = np.linalg.norm(avg_features, axis=1, keepdims=True)
            avg_features = avg_features / (norms + 1e-8)

            # Build dictionary mapping asset_id -> feature vector
            self._features_db = {
                uid: avg_features[i] for i, uid in enumerate(uids)
            }

            print(f"  ✓ 加载了 {len(self._features_db)} 个资产的CLIP特征")

    def _load_annotations(self):
        """Lazy load annotations database."""
        if self._annotations is None:
            if not self.annotations_path.exists():
                raise FileNotFoundError(
                    f"Annotations not found at {self.annotations_path}. "
                    "Please download annotations using objathor."
                )

            print(f"  加载标注数据: {self.annotations_path}")
            with gzip.open(self.annotations_path, "rt", encoding="utf-8") as f:
                self._annotations = json.load(f)

            print(f"  ✓ 加载了 {len(self._annotations)} 个资产的标注")

    def _encode_image_with_clip(self, image: Image.Image) -> np.ndarray:
        """
        Encode an image using CLIP model.

        Args:
            image: PIL Image

        Returns:
            Normalized feature vector (768-dim)
        """
        self._load_clip_model()

        import torch

        image_input = self._clip_preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self._clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().squeeze()

    def retrieve_from_image(
        self,
        image: Image.Image | bytes | str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[AssetMatch]:
        """
        Retrieve top-k matching 3D assets for a given image.

        When use_precomputed=True, this method cannot encode new images and will raise an error.
        Use retrieve_by_asset_id() instead to find similar assets based on existing assets.

        Args:
            image: PIL Image, image bytes, or path to image file
            top_k: Number of top matches to return
            category_filter: Optional category to filter results (e.g., "chair", "table")

        Returns:
            List of AssetMatch objects sorted by similarity (highest first)
        """
        if self.use_precomputed:
            raise NotImplementedError(
                "Image encoding not supported in pre-computed mode. "
                "Either set use_precomputed=False and install CLIP, "
                "or use retrieve_by_asset_id() to find similar assets."
            )

        # Load image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            from io import BytesIO
            image = Image.open(BytesIO(image)).convert("RGB")

        # Encode image
        query_embedding = self._encode_image_with_clip(image)

        # Retrieve similar assets
        return self._retrieve_by_embedding(query_embedding, top_k, category_filter)

    def retrieve_by_asset_id(
        self,
        asset_id: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[AssetMatch]:
        """
        Retrieve top-k similar assets to a given asset ID.

        This method works in both pre-computed and live encoding modes.

        Args:
            asset_id: ID of the reference asset
            top_k: Number of top matches to return (excluding the reference asset itself)
            category_filter: Optional category to filter results

        Returns:
            List of AssetMatch objects sorted by similarity (highest first)
        """
        self._load_features_db()

        if asset_id not in self._features_db:
            raise ValueError(f"Asset ID not found in database: {asset_id}")

        query_embedding = self._features_db[asset_id]

        return self._retrieve_by_embedding(
            query_embedding, top_k + 1, category_filter, exclude_id=asset_id
        )

    def retrieve_by_text_label(
        self,
        label: str,
        top_k: int = 5,
    ) -> List[AssetMatch]:
        """
        Retrieve top-k assets matching a text label using category filtering.

        This is a simple text-based retrieval that filters by category name.
        For semantic text-to-image retrieval, use retrieve_from_text() with CLIP model.

        Args:
            label: Object label (e.g., "chair", "table", "washing machine")
            top_k: Number of top matches to return

        Returns:
            List of AssetMatch objects
        """
        self._load_features_db()
        self._load_annotations()

        # Find all assets matching the label
        matching_assets = []

        for asset_id in self._features_db.keys():
            asset_info = self._annotations.get(asset_id, {})
            asset_category = asset_info.get("category", "").lower()

            # Simple substring matching
            if label.lower() in asset_category or asset_category in label.lower():
                # Get asset path - try .glb in subdirectory first, then root .glb, then .pkl.gz
                asset_path = self.assets_base_path / asset_id / f"{asset_id}.glb"
                if not asset_path.exists():
                    asset_path = self.assets_base_path / f"{asset_id}.glb"
                    if not asset_path.exists():
                        # Try objathor format: asset_id/asset_id.pkl.gz
                        asset_path = self.assets_base_path / asset_id / f"{asset_id}.pkl.gz"
                        if not asset_path.exists():
                            asset_path = None

                matching_assets.append(
                    AssetMatch(
                        asset_id=asset_id,
                        similarity=1.0,  # No similarity score for text-based retrieval
                        asset_path=str(asset_path) if asset_path else None,
                        metadata=asset_info,
                    )
                )

        # Return top-k matches
        return matching_assets[:top_k]

    def _retrieve_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        category_filter: Optional[str] = None,
        exclude_id: Optional[str] = None,
    ) -> List[AssetMatch]:
        """
        Internal method to retrieve assets by embedding vector.

        Args:
            query_embedding: Query feature vector (normalized)
            top_k: Number of top matches to return
            category_filter: Optional category to filter results
            exclude_id: Optional asset ID to exclude from results

        Returns:
            List of AssetMatch objects sorted by similarity
        """
        self._load_features_db()
        self._load_annotations()

        # Compute similarities with all assets in database
        similarities = []
        asset_ids = []

        for asset_id, asset_embedding in self._features_db.items():
            # Skip excluded asset
            if exclude_id and asset_id == exclude_id:
                continue

            # Apply category filter if specified
            if category_filter:
                asset_info = self._annotations.get(asset_id, {})
                asset_category = asset_info.get("category", "").lower()
                if category_filter.lower() not in asset_category:
                    continue

            # Compute cosine similarity
            similarity = np.dot(query_embedding, asset_embedding)

            similarities.append(similarity)
            asset_ids.append(asset_id)

        if not similarities:
            return []

        # Get top-k matches
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build result list
        matches = []
        for idx in top_indices:
            asset_id = asset_ids[idx]
            similarity = float(similarities[idx])

            # Get asset path - try .glb in subdirectory first, then root .glb, then .pkl.gz
            asset_path = self.assets_base_path / asset_id / f"{asset_id}.glb"
            if not asset_path.exists():
                asset_path = self.assets_base_path / f"{asset_id}.glb"
                if not asset_path.exists():
                    # Try objathor format: asset_id/asset_id.pkl.gz
                    asset_path = self.assets_base_path / asset_id / f"{asset_id}.pkl.gz"
                    if not asset_path.exists():
                        asset_path = None

            # Get metadata
            metadata = self._annotations.get(asset_id, {})

            matches.append(
                AssetMatch(
                    asset_id=asset_id,
                    similarity=similarity,
                    asset_path=str(asset_path) if asset_path else None,
                    metadata=metadata,
                )
            )

        return matches
