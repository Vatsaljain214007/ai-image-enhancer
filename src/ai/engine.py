"""
AI Enhancement Engine - handles training, inference, and model management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import cv2
from PIL import Image
from .model import create_model, ImageEnhancerAI, count_parameters
import logging

logger = logging.getLogger(__name__)


class ImagePairDataset(Dataset):
    """Dataset for training (low/high quality pairs)."""
    
    def __init__(self, lq_paths: List[str], hq_paths: List[str], 
                 patch_size: int = 128, scale: int = 2):
        self.lq_paths = lq_paths
        self.hq_paths = hq_paths
        self.patch_size = patch_size
        self.scale = scale
        self._verify_pairs()
    
    def _verify_pairs(self):
        assert len(self.lq_paths) == len(self.hq_paths), "LQ and HQ paths must match"
    
    def __len__(self):
        return len(self.lq_paths)
    
    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _extract_patch(self, lq_img: np.ndarray, hq_img: np.ndarray) -> tuple:
        h, w = lq_img.shape[:2]
        ph, pw = self.patch_size, self.patch_size
        
        if h < ph or w < pw:
            lq_img = cv2.resize(lq_img, (max(w, pw), max(h, ph)))
            hq_img = cv2.resize(hq_img, (max(w * self.scale, pw * self.scale), 
                                          max(h * self.scale, ph * self.scale)))
            h, w = lq_img.shape[:2]
        
        y = np.random.randint(0, h - ph + 1)
        x = np.random.randint(0, w - pw + 1)
        lq_patch = lq_img[y:y+ph, x:x+pw]
        hq_patch = hq_img[y*self.scale:(y+ph)*self.scale, 
                          x*self.scale:(x+pw)*self.scale]
        
        return lq_patch, hq_patch
    
    def __getitem__(self, idx):
        lq_img = self._load_image(self.lq_paths[idx])
        hq_img = self._load_image(self.hq_paths[idx])
        
        if self.patch_size:
            lq_img, hq_img = self._extract_patch(lq_img, hq_img)
        
        lq_tensor = torch.from_numpy(lq_img.transpose(2, 0, 1)).float() / 255.0
        hq_tensor = torch.from_numpy(hq_img.transpose(2, 0, 1)).float() / 255.0
        
        return lq_tensor, hq_tensor


class AIEnhancerEngine:
    """AI enhancement engine with training and inference capabilities."""
    
    def __init__(self, model_type: str = 'standard', device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model = create_model(model_type, self.device)
        self.model_type = model_type
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.L1Loss()
        self.trained = False
        logger.info(f"Model has {count_parameters(self.model):,} parameters")
    
    def train(self, train_lq: List[str], train_hq: List[str],
              val_lq: Optional[List[str]] = None, val_hq: Optional[List[str]] = None,
              epochs: int = 100, batch_size: int = 8, lr: float = 1e-4,
              patch_size: int = 128, save_path: Optional[str] = None,
              log_interval: int = 10) -> Dict[str, List[float]]:
        """Train the AI enhancement model."""
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        train_dataset = ImagePairDataset(train_lq, train_hq, patch_size=patch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        val_loader = None
        if val_lq and val_hq:
            val_dataset = ImagePairDataset(val_lq, val_hq, patch_size=None)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (lq, hq) in enumerate(train_loader):
                lq, hq = lq.to(self.device), hq.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(lq)
                if output.shape != hq.shape:
                    output = torch.nn.functional.interpolate(output, size=hq.shape[2:])
                loss = self.criterion(output, hq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            
            if val_loader:
                val_loss = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                logger.info(f"Epoch {epoch+1}/{epochs} - Train: {avg_loss:.6f}, Val: {val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train: {avg_loss:.6f}")
            
            self.scheduler.step()
        
        self.trained = True
        
        if save_path:
            self.save(save_path)
        
        return history
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validation step."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for lq, hq in val_loader:
                lq, hq = lq.to(self.device), hq.to(self.device)
                output = self.model(lq)
                if output.shape != hq.shape:
                    output = torch.nn.functional.interpolate(output, size=hq.shape[2:])
                total_loss += self.criterion(output, hq).item()
        self.model.train()
        return total_loss / len(val_loader)
    
    @torch.no_grad()
    def enhance(self, image: np.ndarray, scale: Optional[int] = None) -> np.ndarray:
        """Enhance an image using the AI model."""
        self.model.eval()
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        h, w = image.shape[:2]
        
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        
        output = self.model(tensor)
        
        if scale and output.shape[-1] != w * scale:
            output = torch.nn.functional.interpolate(
                output, size=(h * scale, w * scale), mode='bicubic', align_corners=False
            )
        
        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
        
        if pad_h or pad_w:
            output_np = output_np[:h + (pad_h if not isinstance(scale, int) else 0), 
                                  :w + (pad_w if not isinstance(scale, int) else 0)]
        
        return output_np
    
    @torch.no_grad()
    def enhance_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Enhance multiple images in batch."""
        self.model.eval()
        batch_tensors = []
        original_shapes = []
        
        for image in images:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            h, w = image.shape[:2]
            original_shapes.append((h, w))
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            batch_tensors.append(tensor)
        
        batch = torch.stack(batch_tensors).to(self.device)
        outputs = self.model(batch)
        
        results = []
        for i, out in enumerate(outputs):
            out_np = out.cpu().numpy().transpose(1, 2, 0)
            out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
            results.append(out_np)
        
        return results
    
    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'trained': self.trained,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model = create_model(checkpoint.get('model_type', self.model_type), self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trained = checkpoint.get('trained', True)
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")
    
    def fine_tune(self, lq_paths: List[str], hq_paths: List[str],
                  epochs: int = 10, lr: float = 5e-5) -> Dict[str, List[float]]:
        """Fine-tune pre-trained model on custom data."""
        if not self.trained:
            logger.warning("Fine-tuning an untrained model. Training from scratch instead.")
        return self.train(lq_paths, hq_paths, epochs=epochs, lr=lr)