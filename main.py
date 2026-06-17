"""
AI-Powered Image Enhancement Tool
==================================
Combines traditional computer vision techniques with deep learning
for state-of-the-art image enhancement.

Usage:
    python main.py enhance input.jpg -o output.jpg
    python main.py batch ./images/ ./enhanced/
    python main.py analyze input.jpg
    python main.py compare input.jpg
    python main.py train --lq-dir ./lq/ --hq-dir ./hq/
    python main.py gui
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli.main import main

if __name__ == '__main__':
    main()