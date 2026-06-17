"""
Pre-trained AI enhancement models using OpenCV DNN Super Resolution.
Downloads lightweight models automatically on first use.
"""

import os
import requests
import logging
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import cv2

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_URLS = {
    'espcn_x2': {
        'url': 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb',
        'path': MODELS_DIR / 'ESPCN_x2.pb',
        'scale': 2,
        'description': 'ESPCN x2 - Fast, lightweight super-resolution (~86KB)',
    },
    'espcn_x3': {
        'url': 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb',
        'path': MODELS_DIR / 'ESPCN_x3.pb',
        'scale': 3,
        'description': 'ESPCN x3 - Fast upscaling by 3x (~92KB)',
    },
    'espcn_x4': {
        'url': 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb',
        'path': MODELS_DIR / 'ESPCN_x4.pb',
        'scale': 4,
        'description': 'ESPCN x4 - Fast upscaling by 4x (~100KB)',
    },
    'fsrcnn_x2': {
        'url': 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/refs/heads/master/models/FSRCNN_x2.pb',
        'path': MODELS_DIR / 'FSRCNN_x2.pb',
        'scale': 2,
        'description': 'FSRCNN x2 - Smallest model (~39KB)',
    },
    'lapsrn_x2': {
        'url': 'https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x2.pb',
        'path': MODELS_DIR / 'LapSRN_x2.pb',
        'scale': 2,
        'description': 'LapSRN x2 - Better quality (~1.3MB)',
    },
    'lapsrn_x4': {
        'url': 'https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb',
        'path': MODELS_DIR / 'LapSRN_x4.pb',
        'scale': 4,
        'description': 'LapSRN x4 - Highest quality upscaling (~2.7MB)',
    },
}

AVAILABLE_MODELS = list(MODEL_URLS.keys())


def download_model(model_key: str, force: bool = False) -> Optional[Path]:
    """Download a pre-trained super-resolution model."""
    if model_key not in MODEL_URLS:
        logger.error(f"Unknown model: {model_key}. Available: {AVAILABLE_MODELS}")
        return None

    info = MODEL_URLS[model_key]
    model_path = info['path']

    if model_path.exists() and not force:
        return model_path

    url = info['url']
    logger.info(f"Downloading {model_key} model from {url}...")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))

        with open(model_path, 'wb') as f:
            if total > 0:
                from tqdm import tqdm
                pbar = tqdm(total=total, unit='B', unit_scale=True, desc=model_key)
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
                pbar.close()
            else:
                f.write(response.content)

        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Downloaded {model_key} ({size_mb:.2f} MB) to {model_path}")
        return model_path

    except Exception as e:
        logger.warning(f"Failed to download {model_key}: {e}")
        return None


class PretrainedEnhancer:
    """AI enhancer using OpenCV DNN pre-trained super-resolution models."""

    def __init__(self, model_key: str = 'espcn_x2'):
        self.model_key = model_key
        self.model_path = None
        self.sr = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        """Load the OpenCV DNN super-resolution model."""
        model_path = download_model(self.model_key)
        if model_path is None:
            logger.warning(f"Could not load model {self.model_key}, falling back to cv2.resize")
            return

        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            algorithm = self.model_key.rsplit('_', 1)[0]
            scale = MODEL_URLS[self.model_key]['scale']
            self.sr.readModel(str(model_path))
            self.sr.setModel(algorithm, scale)
            self._loaded = True
            logger.info(f"Loaded {self.model_key} model (scale={scale})")
        except Exception as e:
            logger.warning(f"Failed to initialize DNN model: {e}")
            self.sr = None

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply AI super-resolution enhancement."""
        if not self._loaded or self.sr is None:
            scale = MODEL_URLS[self.model_key]['scale']
            h, w = image.shape[:2]
            return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        try:
            result = self.sr.upsample(image)
            return result
        except Exception as e:
            logger.warning(f"DNN upsampling failed: {e}")
            scale = MODEL_URLS[self.model_key]['scale']
            h, w = image.shape[:2]
            return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    @property
    def scale(self) -> int:
        return MODEL_URLS.get(self.model_key, {}).get('scale', 2)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def list_models() -> Dict[str, str]:
    """List available pre-trained models."""
    return {k: v['description'] for k, v in MODEL_URLS.items()}


def get_default_model() -> str:
    """Get the recommended default model key."""
    return 'espcn_x2'