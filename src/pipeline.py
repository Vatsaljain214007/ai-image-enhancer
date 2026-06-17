"""
Integrated Enhancement Pipeline.
Combines traditional and AI enhancement for maximum quality.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .enhancers.traditional_enhancer import TraditionalEnhancer
from .ai.pretrained import PretrainedEnhancer

logger = logging.getLogger(__name__)


@dataclass
class EnhancementConfig:
    """Configuration for the full enhancement pipeline."""
    traditional: Dict[str, Any] = field(default_factory=lambda: {
        'denoise': {'enabled': True, 'strength': 10},
        'sharpen': {'enabled': True, 'amount': 1.5},
        'contrast': {'enabled': True, 'method': 'clahe'},
        'color_balance': {'enabled': True},
        'exposure': {'enabled': True, 'gamma': 1.0},
    })
    ai: Dict[str, Any] = field(default_factory=lambda: {
        'model_type': 'standard',
        'model_key': 'espcn_x2',
        'device': None,
        'model_path': None,
    })
    pipeline_order: List[str] = field(default_factory=lambda: ['traditional', 'ai'])
    auto_analyze: bool = True
    save_intermediate: bool = False
    output_quality: int = 95
    max_size: Optional[Tuple[int, int]] = None


class EnhancementPipeline:
    """
    Full image enhancement pipeline:
    1. Analyzes image for quality issues
    2. Applies traditional enhancement (denoise, sharpen, contrast, color)
    3. Applies AI-based enhancement (super-resolution, quality boost)
    """
    
    def __init__(self, config: Optional[EnhancementConfig] = None, 
                 ai_engine: Optional[AIEnhancerEngine] = None):
        self.config = config or EnhancementConfig()
        self.traditional = TraditionalEnhancer(self.config.traditional)
        self.ai = ai_engine
        if self.config.pipeline_order is None or 'ai' in self.config.pipeline_order:
            self._init_ai()
    
    def _init_ai(self):
        """Initialize AI engine (load model if path provided)."""
        if self.ai is None:
            ai_config = self.config.ai
            model_type = ai_config.get('model_type', 'standard')
            if model_type == 'pretrained':
                model_key = ai_config.get('model_key', 'espcn_x2')
                self.ai = PretrainedEnhancer(model_key)
            else:
                from .ai.engine import AIEnhancerEngine
                self.ai = AIEnhancerEngine(
                    model_type=model_type,
                    device=ai_config.get('device')
                )
                model_path = ai_config.get('model_path')
                if model_path and Path(model_path).exists():
                    self.ai.load(model_path)
    
    def enhance(self, image: np.ndarray, **overrides) -> np.ndarray:
        """Run full enhancement pipeline on an image."""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(f"Cannot load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result = image.copy()
        results = {'original': result}
        
        analysis = {}
        if self.config.auto_analyze:
            analysis = self.traditional.analyze_image(result)
            logger.info(f"Image analysis: {analysis}")
        
        for stage in self.config.pipeline_order:
            if stage == 'traditional':
                result = self.traditional.enhance(result)
                logger.info("Traditional enhancement applied")
            elif stage == 'ai':
                if self.ai is not None:
                    result = self.ai.enhance(result)
                    logger.info("AI enhancement applied")
                else:
                    logger.warning("AI engine not initialized, skipping AI stage")
            
            if self.config.save_intermediate:
                results[f'after_{stage}'] = result.copy()
        
        if self.config.max_size:
            max_h, max_w = self.config.max_size
            h, w = result.shape[:2]
            if h > max_h or w > max_w:
                scale = min(max_h / h, max_w / w)
                new_size = (int(w * scale), int(h * scale))
                result = cv2.resize(result, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        return result
    
    def enhance_from_path(self, input_path: str, output_path: str) -> str:
        """Enhance image from file path and save result."""
        result = self.enhance(input_path)
        return self._save_result(result, output_path)
    
    def enhance_batch(self, input_paths: List[str], output_dir: str,
                      max_workers: int = 4) -> List[str]:
        """Batch enhance multiple images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for inp_path in input_paths:
                inp = Path(inp_path)
                out_path = output_dir / f"enhanced_{inp.name}"
                futures[executor.submit(self.enhance_from_path, str(inp), str(out_path))] = out_path
            
            for future in as_completed(futures):
                out_path = futures[future]
                try:
                    result = future.result()
                    output_paths.append(result)
                    logger.info(f"Enhanced: {out_path}")
                except Exception as e:
                    logger.error(f"Failed to enhance {out_path}: {e}")
        
        return output_paths
    
    def _save_result(self, result: np.ndarray, output_path: str) -> str:
        """Save enhanced image."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        ext = output_path.suffix.lower()
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(str(output_path), result_bgr, 
                       [cv2.IMWRITE_JPEG_QUALITY, self.config.output_quality])
        elif ext == '.png':
            cv2.imwrite(str(output_path), result_bgr)
        else:
            cv2.imwrite(str(output_path), result_bgr)
        
        return str(output_path)
    
    def compare_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Compare original vs enhanced quality metrics."""
        original = image.copy()
        enhanced = self.enhance(image)
        
        original_analysis = self.traditional.analyze_image(original)
        enhanced_analysis = self.traditional.analyze_image(enhanced)
        
        return {
            'original': original_analysis,
            'enhanced': enhanced_analysis,
            'improvement': {
                'laplacian_variance': (
                    enhanced_analysis['laplacian_variance'] - original_analysis['laplacian_variance']
                ),
                'contrast': enhanced_analysis['contrast'] - original_analysis['contrast'],
                'noise_reduction': original_analysis['noise_estimate'] - enhanced_analysis['noise_estimate'],
            }
        }


def create_pipeline(config: Optional[EnhancementConfig] = None) -> EnhancementPipeline:
    """Factory function for enhancement pipeline."""
    return EnhancementPipeline(config)