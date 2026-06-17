"""
AI Image Enhancer - Web Application
Run with: streamlit run web_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time

from src.pipeline import EnhancementPipeline, EnhancementConfig
from src.enhancers.traditional_enhancer import TraditionalEnhancer
from src.ai.pretrained import list_models

st.set_page_config(
    page_title="AI Image Enhancer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding: 0 2rem; }
    .st-emotion-cache-16txtl3 h1 { font-size: 2rem; }
    div[data-testid="stImage"] img { border-radius: 8px; }
    .quality-good { color: #27ae60; }
    .quality-medium { color: #f39c12; }
    .quality-bad { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_traditional_enhancer():
    return TraditionalEnhancer()


def convert_img_for_display(img: np.ndarray) -> Image.Image:
    if len(img.shape) == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img)


def analyze_image_quality(image: np.ndarray) -> dict:
    enhancer = get_traditional_enhancer()
    return enhancer.analyze_image(image)


def enhance_image(
    image: np.ndarray,
    denoise_strength: int,
    sharpen_amount: float,
    contrast_method: str,
    color_balance: bool,
    enable_ai: bool,
    ai_model: str,
    gamma: float,
) -> np.ndarray:
    config = EnhancementConfig()
    config.traditional['denoise'] = {'enabled': denoise_strength > 0, 'strength': denoise_strength}
    config.traditional['sharpen'] = {'enabled': sharpen_amount > 0, 'amount': sharpen_amount}
    config.traditional['contrast'] = {'enabled': contrast_method != 'None', 'method': contrast_method.lower()}
    config.traditional['color_balance'] = {'enabled': color_balance}
    config.traditional['exposure'] = {'enabled': gamma != 1.0, 'gamma': gamma}

    if enable_ai:
        config.pipeline_order = ['traditional', 'ai']
        config.ai['model_type'] = 'pretrained'
        config.ai['model_key'] = ai_model
    else:
        config.pipeline_order = ['traditional']

    pipeline = EnhancementPipeline(config)
    return pipeline.enhance(image)


def quality_bar(analysis: dict):
    metrics = {
        'Sharpness': min(analysis['laplacian_variance'] / 500, 1.0),
        'Brightness': 1 - abs(analysis['mean_brightness'] - 128) / 128,
        'Contrast': min(analysis['contrast'] / 80, 1.0),
        'Noise Level': 1 - min(analysis['noise_estimate'] / 100, 1.0),
    }

    col1, col2 = st.columns(2)
    for i, (label, score) in enumerate(metrics.items()):
        with col1 if i % 2 == 0 else col2:
            color = "quality-good" if score > 0.6 else "quality-medium" if score > 0.3 else "quality-bad"
            st.markdown(
                f"<div style='margin-bottom:8px'>{label}: "
                f"<span class='{color}'><b>{score*100:.0f}%</b></span></div>",
                unsafe_allow_html=True
            )
            st.progress(float(score))


def main():
    st.title("✨ AI Image Enhancer")
    st.markdown("Upload an image and enhance it with traditional filters + AI super-resolution.")

    with st.sidebar:
        st.header("Settings")

        uploaded_file = st.file_uploader(
            "Upload Image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff'],
            label_visibility="collapsed"
        )

        st.subheader("Traditional Enhancement")
        denoise = st.slider("Denoise", 0, 30, 10, help="Non-local means denoising strength")
        sharpen = st.slider("Sharpen", 0.0, 3.0, 1.5, 0.1, help="Unsharp mask amount")
        contrast = st.selectbox(
            "Contrast", ['None', 'CLAHE', 'HE', 'Adaptive'],
            index=1, help="Contrast enhancement method"
        )
        color_balance = st.checkbox("Color Balance", True, help="Auto white balance")
        gamma = st.slider("Gamma", 0.3, 2.5, 1.0, 0.1, help="<1 brightens, >1 darkens")

        st.subheader("AI Enhancement")
        enable_ai = st.checkbox("Enable AI Super-Resolution", True, help="Use pre-trained model")

        ai_models = list_models()
        ai_model_keys = list(ai_models.keys())
        ai_model_key = st.selectbox(
            "AI Model",
            ai_model_keys,
            format_func=lambda k: ai_models.get(k, k),
            disabled=not enable_ai,
        )

        enhance_btn = st.button("✨ Enhance", type="primary", use_container_width=True, disabled=(uploaded_file is None))

        with st.expander("About"):
            st.markdown("""
            **Traditional**: OpenCV denoising, sharpening, CLAHE contrast, white balance, gamma correction.

            **AI**: Pre-trained CNN models for super-resolution. Models auto-downloaded on first use.

            Deploy your own: [GitHub](https://github.com)
            """)

    col_left, col_right = st.columns(2)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with col_left:
            st.subheader("Original")
            st.image(image_rgb, use_container_width=True)
            st.caption(f"Size: {image.shape[1]}x{image.shape[0]}")

        with col_right:
            st.subheader("Enhanced")
            if enhance_btn:
                with st.spinner("Enhancing..."):
                    progress_bar = st.progress(0, text="Starting...")
                    for pct, msg in [(10, "Analyzing..."), (30, "Denoising..."), (50, "Sharpening..."),
                                     (70, "Color correction..."), (85, "AI upscaling..."), (100, "Done!")]:
                        time.sleep(0.15)
                        progress_bar.progress(pct, text=msg)

                    result = enhance_image(
                        image, denoise, sharpen, contrast,
                        color_balance, enable_ai, ai_model_key, gamma
                    )
                    progress_bar.empty()

                st.image(result, use_container_width=True)

                buf = io.BytesIO()
                result_pil = Image.fromarray(result)
                result_pil.save(buf, format="PNG")
                st.download_button(
                    "💾 Download Enhanced",
                    data=buf.getvalue(),
                    file_name=f"enhanced_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                    mime="image/png",
                    use_container_width=True,
                )

                st.subheader("Quality Analysis")
                orig_analysis = analyze_image_quality(image_rgb)
                enh_analysis = analyze_image_quality(result)

                tab1, tab2, tab3 = st.tabs(["Original", "Enhanced", "Comparison"])
                with tab1:
                    quality_bar(orig_analysis)
                with tab2:
                    quality_bar(enh_analysis)
                with tab3:
                    comparisons = {
                        'Sharpness': {
                            'orig': min(orig_analysis['laplacian_variance'] / 500, 1.0),
                            'enh': min(enh_analysis['laplacian_variance'] / 500, 1.0),
                        },
                        'Noise': {
                            'orig': 1 - min(orig_analysis['noise_estimate'] / 100, 1.0),
                            'enh': 1 - min(enh_analysis['noise_estimate'] / 100, 1.0),
                        },
                    }
                    for metric, vals in comparisons.items():
                        c1, c2, c3 = st.columns([1, 1, 1])
                        c1.metric(metric, f"{vals['orig']*100:.0f}%")
                        delta = f"+{(vals['enh'] - vals['orig'])*100:.0f}%" if vals['enh'] > vals['orig'] else f"{(vals['enh'] - vals['orig'])*100:.0f}%"
                        c2.metric("Enhanced", f"{vals['enh']*100:.0f}%", delta=delta)
            else:
                st.info("Click Enhance to process")

    else:
        with col_left:
            st.info("Upload an image to begin")
        with col_right:
            st.info("Enhanced image will appear here")


if __name__ == "__main__":
    main()
