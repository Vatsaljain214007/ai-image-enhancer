"""
Traditional Image Enhancer using OpenCV and scikit-image.
Provides classical image processing techniques for enhancement.
"""

import cv2
import numpy as np
from skimage import exposure, restoration, filters, morphology
from skimage.util import img_as_float, img_as_ubyte
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TraditionalEnhancer:
    """Classical image enhancement using OpenCV and scikit-image."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        defaults = self._default_config()
        if config is None:
            self.config = defaults
        else:
            self.config = {}
            for key, default_val in defaults.items():
                if key in config:
                    if isinstance(default_val, dict):
                        merged = {**default_val, **config[key]}
                        self.config[key] = merged
                    else:
                        self.config[key] = config[key]
                else:
                    self.config[key] = default_val
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'denoise': {'enabled': True, 'strength': 10, 'template_window': 7, 'search_window': 21},
            'sharpen': {'enabled': True, 'amount': 1.5, 'radius': 1.0, 'threshold': 0},
            'contrast': {'enabled': True, 'method': 'clahe', 'clip_limit': 2.0, 'grid_size': (8, 8)},
            'color_balance': {'enabled': True, 'method': 'gray_world'},
            'exposure': {'enabled': True, 'gamma': 1.0, 'gain': 1.0},
            'deblur': {'enabled': False, 'method': 'wiener', 'psf_size': 5, 'noise': 0.01},
            'upscale': {'enabled': False, 'scale': 2, 'method': 'lanczos'},
        }
    
    def _strip_enabled(self, cfg: dict) -> dict:
        return {k: v for k, v in cfg.items() if k != 'enabled'}
    
    def enhance(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply traditional enhancement pipeline."""
        config = {**self.config, **kwargs}
        result = image.copy()
        
        if config['denoise']['enabled']:
            result = self.denoise(result, **self._strip_enabled(config['denoise']))
        
        if config['deblur']['enabled']:
            result = self.deblur(result, **self._strip_enabled(config['deblur']))
        
        if config['sharpen']['enabled']:
            result = self.sharpen(result, **self._strip_enabled(config['sharpen']))
        
        if config['contrast']['enabled']:
            result = self.enhance_contrast(result, **self._strip_enabled(config['contrast']))
        
        if config['color_balance']['enabled']:
            result = self.balance_color(result, **self._strip_enabled(config['color_balance']))
        
        if config['exposure']['enabled']:
            result = self.adjust_exposure(result, **self._strip_enabled(config['exposure']))
        
        if config['upscale']['enabled']:
            result = self.upscale_image(result, **self._strip_enabled(config['upscale']))
        
        return result
    
    def denoise(self, image: np.ndarray, strength: float = 10, 
                template_window: int = 7, search_window: int = 21) -> np.ndarray:
        """Non-local means denoising."""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, strength, strength, template_window, search_window
            )
        return cv2.fastNlMeansDenoising(image, None, strength, template_window, search_window)
    
    def sharpen(self, image: np.ndarray, amount: float = 1.5, 
                radius: float = 1.0, threshold: int = 0) -> np.ndarray:
        """Unsharp mask sharpening."""
        if len(image.shape) == 3:
            blurred = cv2.GaussianBlur(image, (0, 0), radius)
            sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
            if threshold > 0:
                diff = cv2.absdiff(image, blurred)
                mask = diff > threshold
                sharpened = np.where(mask, sharpened, image)
            return sharpened
        
        blurred = cv2.GaussianBlur(image, (0, 0), radius)
        return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe',
                         clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Contrast enhancement using CLAHE or histogram equalization."""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            if method == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
                l = clahe.apply(l)
            elif method == 'he':
                l = cv2.equalizeHist(l)
            elif method == 'adaptive':
                l = exposure.equalize_adapthist(img_as_float(l), clip_limit=clip_limit/255.0)
                l = img_as_ubyte(l)
            
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            return clahe.apply(image)
        elif method == 'he':
            return cv2.equalizeHist(image)
        elif method == 'adaptive':
            return img_as_ubyte(exposure.equalize_adapthist(img_as_float(image), clip_limit=clip_limit/255.0))
        
        return image
    
    def balance_color(self, image: np.ndarray, method: str = 'gray_world') -> np.ndarray:
        """Color balance correction."""
        if len(image.shape) != 3:
            return image
        
        if method == 'gray_world':
            result = image.astype(np.float32)
            avg_b, avg_g, avg_r = cv2.mean(result)[:3]
            avg = (avg_b + avg_g + avg_r) / 3
            scale_b = avg / (avg_b + 1e-6)
            scale_g = avg / (avg_g + 1e-6)
            scale_r = avg / (avg_r + 1e-6)
            result[:, :, 0] *= scale_b
            result[:, :, 1] *= scale_g
            result[:, :, 2] *= scale_r
            return np.clip(result, 0, 255).astype(np.uint8)
        
        elif method == 'white_patch':
            result = image.astype(np.float32)
            max_vals = np.max(result.reshape(-1, 3), axis=0)
            scale = 255.0 / (max_vals + 1e-6)
            result *= scale
            return np.clip(result, 0, 255).astype(np.uint8)
        
        return image
    
    def adjust_exposure(self, image: np.ndarray, gamma: float = 1.0, gain: float = 1.0) -> np.ndarray:
        """Gamma correction for exposure adjustment."""
        if gamma == 1.0 and gain == 1.0:
            return image
        
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 * gain for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
    def deblur(self, image: np.ndarray, method: str = 'wiener', 
               psf_size: int = 5, noise: float = 0.01) -> np.ndarray:
        """Image deblurring using Wiener or Richardson-Lucy deconvolution."""
        if len(image.shape) == 3:
            channels = cv2.split(image)
            deblurred = []
            for ch in channels:
                deblurred.append(self._deblur_channel(ch, method, psf_size, noise))
            return cv2.merge(deblurred)
        return self._deblur_channel(image, method, psf_size, noise)
    
    def _deblur_channel(self, image: np.ndarray, method: str, psf_size: int, noise: float) -> np.ndarray:
        psf = np.ones((psf_size, psf_size), dtype=np.float32) / (psf_size ** 2)
        
        if method == 'wiener':
            from scipy.signal import wiener
            return wiener(image.astype(np.float32), mysize=psf_size, noise=noise).astype(np.uint8)
        elif method == 'richardson_lucy':
            from skimage.restoration import richardson_lucy
            return img_as_ubyte(richardson_lucy(img_as_float(image), psf, num_iter=30))
        
        return image
    
    def upscale_image(self, image: np.ndarray, scale: int = 2, method: str = 'lanczos') -> np.ndarray:
        """Upscale image using various interpolation methods."""
        h, w = image.shape[:2]
        new_size = (w * scale, h * scale)
        
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
        }
        interpolation = interp_map.get(method, cv2.INTER_LANCZOS4)
        return cv2.resize(image, new_size, interpolation=interpolation)
    
    def auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """Automatic enhancement based on image analysis."""
        analysis = self.analyze_image(image)
        
        config = self._default_config()
        
        if analysis['is_noisy']:
            config['denoise']['strength'] = 15
        if analysis['is_blurry']:
            config['sharpen']['amount'] = 2.0
            config['deblur']['enabled'] = True
        if analysis['low_contrast']:
            config['contrast']['clip_limit'] = 3.0
        if analysis['color_cast']:
            config['color_balance']['enabled'] = True
        if analysis['under_exposed']:
            config['exposure']['gamma'] = 0.7
        elif analysis['over_exposed']:
            config['exposure']['gamma'] = 1.3
        
        return self.enhance(image, **config)
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image quality metrics."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < 100
        
        mean_brightness = np.mean(gray)
        under_exposed = mean_brightness < 80
        over_exposed = mean_brightness > 180
        
        contrast = np.std(gray)
        low_contrast = contrast < 40
        
        if len(image.shape) == 3:
            b, g, r = cv2.split(image)
            color_std = np.std([np.mean(b), np.mean(g), np.mean(r)])
            color_cast = color_std > 15
        else:
            color_cast = False
        
        noise_estimate = self._estimate_noise(gray)
        is_noisy = noise_estimate > 15
        
        return {
            'laplacian_variance': laplacian_var,
            'is_blurry': is_blurry,
            'mean_brightness': mean_brightness,
            'under_exposed': under_exposed,
            'over_exposed': over_exposed,
            'contrast': contrast,
            'low_contrast': low_contrast,
            'color_cast': color_cast,
            'noise_estimate': noise_estimate,
            'is_noisy': is_noisy,
        }
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level using high-pass filter."""
        h, w = image.shape
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        filtered = cv2.filter2D(image.astype(np.float32), -1, kernel)
        return np.std(filtered)


def create_traditional_enhancer(config: Optional[Dict] = None) -> TraditionalEnhancer:
    """Factory function to create traditional enhancer."""
    return TraditionalEnhancer(config)