"""Tests for the image enhancement pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2


def create_test_image(width=640, height=480):
    """Create a test image with known properties."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    cv2.rectangle(image, (50, 50), (200, 200), (60, 80, 200), -1)
    cv2.rectangle(image, (300, 100), (500, 300), (200, 100, 60), -1)
    cv2.circle(image, (400, 350), 80, (50, 200, 100), -1)
    cv2.line(image, (0, 0), (width, height), (255, 255, 255), 2)
    
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def test_traditional_enhancer():
    """Test the traditional enhancer."""
    from src.enhancers.traditional_enhancer import TraditionalEnhancer
    
    enhancer = TraditionalEnhancer()
    image = create_test_image()
    
    assert image.shape == (480, 640, 3)
    
    enhanced = enhancer.enhance(image)
    assert enhanced.shape == (480, 640, 3)
    assert enhanced.dtype == np.uint8
    
    analysis = enhancer.analyze_image(image)
    assert 'is_blurry' in analysis
    assert 'is_noisy' in analysis
    assert 'contrast' in analysis
    assert 'mean_brightness' in analysis
    
    denoised = enhancer.denoise(image)
    assert denoised.shape == image.shape
    
    sharpened = enhancer.sharpen(image)
    assert sharpened.shape == image.shape
    
    contrasted = enhancer.enhance_contrast(image)
    assert contrasted.shape == image.shape
    
    print("[OK] Traditional enhancer tests passed")


def test_ai_model():
    """Test the AI model creation and forward pass."""
    from src.ai.model import (
        create_model, ImageEnhancerAI, LightweightEnhancer,
        count_parameters
    )
    import torch
    
    model = create_model('standard', 'cpu')
    assert isinstance(model, ImageEnhancerAI)
    params = count_parameters(model)
    assert params > 0
    print(f"  Standard model parameters: {params:,}")
    
    model_light = create_model('lightweight', 'cpu')
    assert isinstance(model_light, LightweightEnhancer)
    params_light = count_parameters(model_light)
    print(f"  Lightweight model parameters: {params_light:,}")
    
    model_hq = create_model('high_quality', 'cpu')
    assert isinstance(model_hq, ImageEnhancerAI)
    params_hq = count_parameters(model_hq)
    print(f"  High-quality model parameters: {params_hq:,}")
    
    dummy_input = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)
        assert output.shape == (1, 3, 256, 256)
        print(f"  Forward pass: {dummy_input.shape} -> {output.shape}")
    
    print("[OK] AI model tests passed")


def test_pipeline():
    """Test the full enhancement pipeline."""
    from src.pipeline import EnhancementPipeline, EnhancementConfig
    
    config = EnhancementConfig()
    config.pipeline_order = ['traditional']
    config.ai['model_path'] = None
    
    pipeline = EnhancementPipeline(config)
    image = create_test_image()
    
    enhanced = pipeline.enhance(image)
    assert enhanced.shape[:2] == image.shape[:2]
    assert enhanced.dtype == np.uint8
    
    comparison = pipeline.compare_quality(image)
    assert 'original' in comparison
    assert 'enhanced' in comparison
    assert 'improvement' in comparison
    
    print("[OK] Pipeline tests passed")


def test_ai_engine():
    """Test the AI engine inference."""
    from src.ai.engine import AIEnhancerEngine
    
    engine = AIEnhancerEngine(model_type='lightweight')
    
    image = create_test_image(256, 256)
    enhanced = engine.enhance(image)
    assert enhanced.shape[0] >= image.shape[0]
    assert enhanced.dtype == np.uint8
    print(f"  AI inference: {image.shape} -> {enhanced.shape}")
    
    images = [create_test_image(128, 128) for _ in range(2)]
    results = engine.enhance_batch(images)
    assert len(results) == 2
    
    print("[OK] AI engine tests passed")


def test_auto_enhance():
    """Test auto enhancement mode."""
    from src.enhancers.traditional_enhancer import TraditionalEnhancer
    
    enhancer = TraditionalEnhancer()
    image = create_test_image()
    
    auto_enhanced = enhancer.auto_enhance(image)
    assert auto_enhanced.shape == image.shape
    assert auto_enhanced.dtype == np.uint8
    
    print("[OK] Auto enhance tests passed")


if __name__ == '__main__':
    print("\n=== Image Enhancement Tool Tests ===\n")
    test_traditional_enhancer()
    test_ai_model()
    test_ai_engine()
    test_pipeline()
    test_auto_enhance()
    print("\n=== All tests passed! ===\n")