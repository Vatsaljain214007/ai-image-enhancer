"""
Basic usage examples for the Image Enhancing Tool.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


def example_1_basic_enhancement():
    """Basic single image enhancement."""
    from src.enhancers.traditional_enhancer import TraditionalEnhancer
    
    print("Example 1: Basic Enhancement")
    image = np.ones((300, 400, 3), dtype=np.uint8) * 100
    cv2.rectangle(image, (50, 50), (150, 150), (200, 80, 50), -1)
    noise = np.random.normal(0, 20, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    enhancer = TraditionalEnhancer()
    enhanced = enhancer.enhance(image)
    print(f"  Original: {image.shape}, Enhanced: {enhanced.shape}")
    print(f"  Quality improved: {enhancer.analyze_image(image)}")
    print()


def example_2_pipeline():
    """Full pipeline with traditional + AI enhancement."""
    from src.pipeline import EnhancementPipeline, EnhancementConfig
    
    print("Example 2: Full Pipeline")
    image = np.ones((256, 256, 3), dtype=np.uint8) * 128
    cv2.circle(image, (128, 128), 80, (30, 150, 220), -1)
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    config = EnhancementConfig()
    config.pipeline_order = ['traditional']
    config.auto_analyze = True
    
    pipeline = EnhancementPipeline(config)
    enhanced = pipeline.enhance(image)
    print(f"  Input: {image.shape}, Output: {enhanced.shape}")
    
    comparison = pipeline.compare_quality(image)
    imp = comparison['improvement']
    print(f"  Sharpness change: {imp['laplacian_variance']:.2f}")
    print(f"  Contrast change: {imp['contrast']:.2f}")
    print()


def example_3_batch_processing():
    """Batch processing multiple images."""
    from src.pipeline import EnhancementPipeline
    from pathlib import Path
    import tempfile
    import os
    
    print("Example 3: Batch Processing")
    
    temp_dir = Path(tempfile.mkdtemp())
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(3):
        img = np.ones((100, 100, 3), dtype=np.uint8) * (50 + i * 50)
        cv2.putText(img, f"Img{i}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        cv2.imwrite(str(input_dir / f"test_{i}.jpg"), img)
    
    pipeline = EnhancementPipeline()
    input_files = [str(f) for f in input_dir.glob("*.jpg")]
    results = pipeline.enhance_batch(input_files, str(output_dir))
    print(f"  Processed {len(results)} images")
    print(f"  Output directory: {output_dir}")
    
    for f in output_dir.glob("*"):
        f.unlink()
    input_dir.rmdir()
    output_dir.rmdir()
    temp_dir.rmdir()
    print()


def example_4_ai_training_demo():
    """Demonstrate AI model creation and inference."""
    from src.ai.model import create_model, count_parameters
    from src.ai.engine import AIEnhancerEngine
    import torch
    
    print("Example 4: AI Model Demo")
    
    engine = AIEnhancerEngine(model_type='lightweight')
    params = count_parameters(engine.model)
    print(f"  Model parameters: {params:,}")
    print(f"  Device: {engine.device}")
    
    dummy = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    result = engine.enhance(dummy)
    print(f"  Inference test: {dummy.shape} -> {result.shape}")
    print()


def example_5_gui_launch():
    """Launch the GUI interface."""
    print("Example 5: Launch GUI")
    print("  Run: python main.py gui")
    print()


if __name__ == '__main__':
    print("=" * 50)
    print("Image Enhancement Tool - Examples")
    print("=" * 50)
    print()
    example_1_basic_enhancement()
    example_2_pipeline()
    example_3_batch_processing()
    example_4_ai_training_demo()
    example_5_gui_launch()