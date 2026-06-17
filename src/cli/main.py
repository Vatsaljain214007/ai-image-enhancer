"""
Command-line interface for Image Enhancing Tool.
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import numpy as np
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline import EnhancementPipeline, EnhancementConfig, create_pipeline
from src.enhancers.traditional_enhancer import create_traditional_enhancer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def enhance_command(args):
    """Enhance a single image."""
    config = EnhancementConfig()
    if args.no_ai:
        config.pipeline_order = ['traditional']
    if args.auto_analyze is False:
        config.auto_analyze = False
    
    pipeline = create_pipeline(config)
    
    if args.ai_model and Path(args.ai_model).exists():
        config.ai['model_path'] = args.ai_model
    
    output = args.output or f"enhanced_{Path(args.input).name}"
    result = pipeline.enhance_from_path(args.input, output)
    logger.info(f"Enhanced image saved to: {result}")


def batch_command(args):
    """Enhance multiple images."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    input_files = []
    for ext in extensions:
        input_files.extend(input_dir.glob(f'*{ext}'))
        input_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if not input_files:
        logger.error(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(input_files)} images to enhance")
    pipeline = create_pipeline()
    output_paths = pipeline.enhance_batch(
        [str(f) for f in input_files],
        str(output_dir),
        max_workers=args.workers
    )
    logger.info(f"Enhanced {len(output_paths)} images saved to {output_dir}")


def analyze_command(args):
    """Analyze image quality."""
    import cv2
    import json
    
    image = cv2.imread(args.input)
    if image is None:
        logger.error(f"Cannot load image: {args.input}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhancer = create_traditional_enhancer()
    analysis = enhancer.analyze_image(image_rgb)
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)
    
    print(json.dumps(analysis, indent=2, cls=NumpyEncoder))
    
    quality_score = sum([
        not analysis['is_blurry'],
        not analysis['low_contrast'],
        not analysis['color_cast'],
        not analysis['is_noisy'],
        analysis['laplacian_variance'] / 500 if analysis['laplacian_variance'] < 500 else 1,
        1 - abs(analysis['mean_brightness'] - 128) / 128,
        analysis['contrast'] / 100 if analysis['contrast'] < 100 else 1,
    ]) / 7 * 100
    
    print(f"\nOverall Quality Score: {quality_score:.1f}/100")


def train_command(args):
    """Train or fine-tune the AI model."""
    low_quality_dir = Path(args.lq_dir)
    high_quality_dir = Path(args.hq_dir)
    
    lq_files = sorted([str(f) for f in low_quality_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
    hq_files = sorted([str(f) for f in high_quality_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
    
    if not lq_files or not hq_files:
        logger.error("Both LQ and HQ directories must contain images")
        return
    
    from src.ai.engine import AIEnhancerEngine
    engine = AIEnhancerEngine(model_type=args.model_type)
    
    history = engine.train(
        lq_files, hq_files,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        save_path=args.save_model or 'model_checkpoint.pth',
    )
    
    logger.info("Training completed!")
    if history['val_loss']:
        logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"Final val loss: {history['val_loss'][-1]:.6f}")


def compare_command(args):
    """Compare original vs enhanced image quality."""
    import cv2
    from src.enhancers.traditional_enhancer import TraditionalEnhancer
    
    image = cv2.imread(args.input)
    if image is None:
        logger.error(f"Cannot load image: {args.input}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pipeline = create_pipeline()
    comparison = pipeline.compare_quality(image_rgb)
    
    print("=== Quality Comparison ===")
    print(f"{'Metric':<25} {'Original':<12} {'Enhanced':<12} {'Change':<12}")
    print("-" * 61)
    
    for metric in ['laplacian_variance', 'contrast', 'noise_estimate']:
        orig = comparison['original'].get(metric, 0)
        enh = comparison['enhanced'].get(metric, 0)
        change = comparison['improvement'].get(metric, 0)
        arrow = "▲" if change > 0 else "▼" if change < 0 else "─"
        print(f"{metric:<25} {orig:<12.2f} {enh:<12.2f} {arrow} {abs(change):.2f}")


def gui_command(args):
    """Launch the GUI application."""
    try:
        from src.gui.app import run_gui
        run_gui()
    except ImportError as e:
        logger.error(f"Cannot launch GUI: {e}")
        logger.info("Make sure tkinter is available on your system")


def main():
    parser = argparse.ArgumentParser(
        description='AI-Powered Image Enhancement Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s enhance input.jpg -o output.jpg
  %(prog)s batch ./images/ ./enhanced/
  %(prog)s analyze input.jpg
  %(prog)s compare input.jpg
  %(prog)s train --lq-dir ./low_quality/ --hq-dir ./high_quality/
  %(prog)s gui
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    enhance_parser = subparsers.add_parser('enhance', help='Enhance a single image')
    enhance_parser.add_argument('input', help='Input image path')
    enhance_parser.add_argument('-o', '--output', help='Output image path')
    enhance_parser.add_argument('--no-ai', action='store_true', help='Skip AI enhancement')
    enhance_parser.add_argument('--ai-model', help='Path to pre-trained AI model')
    enhance_parser.add_argument('--no-analyze', dest='auto_analyze', action='store_false', 
                              help='Skip auto-analysis')
    enhance_parser.set_defaults(func=enhance_command)
    
    batch_parser = subparsers.add_parser('batch', help='Enhance multiple images')
    batch_parser.add_argument('input_dir', help='Input directory')
    batch_parser.add_argument('output_dir', help='Output directory')
    batch_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    batch_parser.set_defaults(func=batch_command)
    
    analyze_parser = subparsers.add_parser('analyze', help='Analyze image quality')
    analyze_parser.add_argument('input', help='Input image path')
    analyze_parser.set_defaults(func=analyze_command)
    
    train_parser = subparsers.add_parser('train', help='Train AI enhancement model')
    train_parser.add_argument('--lq-dir', required=True, help='Low quality images directory')
    train_parser.add_argument('--hq-dir', required=True, help='High quality images directory')
    train_parser.add_argument('--model-type', default='standard', 
                            choices=['standard', 'lightweight', 'high_quality'],
                            help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch-size', type=int, default=8)
    train_parser.add_argument('--learning-rate', type=float, default=1e-4)
    train_parser.add_argument('--save-model', help='Path to save trained model')
    train_parser.set_defaults(func=train_command)
    
    compare_parser = subparsers.add_parser('compare', help='Compare quality metrics')
    compare_parser.add_argument('input', help='Input image path')
    compare_parser.set_defaults(func=compare_command)
    
    gui_parser = subparsers.add_parser('gui', help='Launch GUI application')
    gui_parser.set_defaults(func=gui_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()