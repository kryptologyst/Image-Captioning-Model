"""
Streamlit web interface for the Image Captioning project.

This module provides a user-friendly web interface for testing
and demonstrating the image captioning capabilities.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import logging
from typing import List, Dict, Any
import json

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from image_captioner import ImageCaptioner
from config import Config
from data_generator import SyntheticImageGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Image Captioning Demo",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .caption-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_captioner():
    """Load the image captioner model (cached)."""
    try:
        config = Config()
        captioner = ImageCaptioner(**config.get_model_kwargs())
        return captioner, None
    except Exception as e:
        logger.error(f"Failed to load captioner: {e}")
        return None, str(e)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ Image Captioning Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "BLIP Base": "Salesforce/blip-image-captioning-base",
            "BLIP Large": "Salesforce/blip-image-captioning-large"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        
        # Generation parameters
        st.subheader("🎛️ Generation Parameters")
        
        max_length = st.slider("Max Length", 10, 100, 50)
        num_beams = st.slider("Number of Beams", 1, 10, 4)
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        do_sample = st.checkbox("Use Sampling", False)
        
        # Load model
        if st.button("🔄 Load Model", type="primary"):
            st.cache_resource.clear()
            st.rerun()
    
    # Load captioner
    captioner, error = load_captioner()
    
    if error:
        st.error(f"Failed to load model: {error}")
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Upload Image", "🎨 Generate Samples", "📊 Batch Processing", "ℹ️ About"])
    
    with tab1:
        st.header("Upload and Caption an Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload an image to generate a caption"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
                # Save uploaded file temporarily
                temp_path = Path("temp_upload.jpg")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
        with col2:
            if uploaded_file is not None:
                if st.button("🎯 Generate Caption", type="primary"):
                    with st.spinner("Generating caption..."):
                        try:
                            # Generate caption
                            caption = captioner.generate_caption(
                                temp_path,
                                max_length=max_length,
                                num_beams=num_beams,
                                temperature=temperature,
                                do_sample=do_sample
                            )
                            
                            # Display caption
                            st.markdown('<div class="caption-box">', unsafe_allow_html=True)
                            st.markdown(f"**Generated Caption:**")
                            st.markdown(f"_{caption}_")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Generate multiple captions
                            if st.checkbox("Generate Multiple Captions"):
                                with st.spinner("Generating multiple captions..."):
                                    captions = captioner.generate_multiple_captions(
                                        temp_path,
                                        num_captions=3,
                                        max_length=max_length,
                                        num_beams=num_beams,
                                        temperature=temperature,
                                        do_sample=True
                                    )
                                
                                st.subheader("Multiple Captions:")
                                for i, cap in enumerate(captions, 1):
                                    st.markdown(f"**{i}.** {cap}")
                            
                        except Exception as e:
                            st.error(f"Error generating caption: {e}")
                        finally:
                            # Clean up temp file
                            if temp_path.exists():
                                temp_path.unlink()
    
    with tab2:
        st.header("Generate Sample Images")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎨 Create Synthetic Images")
            
            num_images = st.number_input("Number of Images", 1, 50, 10)
            categories = st.multiselect(
                "Categories",
                ["animals", "vehicles", "nature", "objects"],
                default=["animals", "objects"]
            )
            
            if st.button("🎨 Generate Sample Images"):
                with st.spinner("Generating sample images..."):
                    try:
                        generator = SyntheticImageGenerator()
                        dataset = generator.create_dataset(
                            num_images=num_images,
                            categories=categories
                        )
                        
                        st.success(f"Generated {num_images} sample images!")
                        
                        # Display sample images
                        st.subheader("Sample Images:")
                        
                        for category, image_paths in dataset.items():
                            if image_paths:
                                st.write(f"**{category.title()}:**")
                                
                                # Show first few images
                                cols = st.columns(min(3, len(image_paths)))
                                for i, (col, path) in enumerate(zip(cols, image_paths[:3])):
                                    with col:
                                        st.image(path, caption=f"{category} {i+1}")
                                
                                if len(image_paths) > 3:
                                    st.write(f"... and {len(image_paths) - 3} more")
                    
                    except Exception as e:
                        st.error(f"Error generating images: {e}")
        
        with col2:
            st.subheader("🎯 Caption Sample Images")
            
            # List available sample images
            sample_dir = Path("data/synthetic")
            if sample_dir.exists():
                image_files = list(sample_dir.glob("*.jpg"))
                
                if image_files:
                    selected_image = st.selectbox(
                        "Select Sample Image",
                        image_files,
                        format_func=lambda x: x.name
                    )
                    
                    if selected_image:
                        st.image(selected_image, caption="Selected Image", use_column_width=True)
                        
                        if st.button("🎯 Caption Selected Image"):
                            with st.spinner("Generating caption..."):
                                try:
                                    caption = captioner.generate_caption(
                                        selected_image,
                                        max_length=max_length,
                                        num_beams=num_beams,
                                        temperature=temperature,
                                        do_sample=do_sample
                                    )
                                    
                                    st.markdown('<div class="caption-box">', unsafe_allow_html=True)
                                    st.markdown(f"**Caption:** {caption}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                except Exception as e:
                                    st.error(f"Error generating caption: {e}")
                else:
                    st.info("No sample images found. Generate some images first!")
            else:
                st.info("No sample directory found. Generate some images first!")
    
    with tab3:
        st.header("Batch Processing")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📁 Process Multiple Images")
            
            # File uploader for multiple files
            uploaded_files = st.file_uploader(
                "Choose multiple image files",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                accept_multiple_files=True,
                help="Upload multiple images for batch processing"
            )
            
            if uploaded_files:
                st.write(f"Uploaded {len(uploaded_files)} images")
                
                if st.button("🚀 Process All Images", type="primary"):
                    with st.spinner("Processing images..."):
                        results = []
                        
                        for uploaded_file in uploaded_files:
                            try:
                                # Save temp file
                                temp_path = Path(f"temp_{uploaded_file.name}")
                                with open(temp_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                # Generate caption
                                caption = captioner.generate_caption(
                                    temp_path,
                                    max_length=max_length,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    do_sample=do_sample
                                )
                                
                                results.append({
                                    "Image": uploaded_file.name,
                                    "Caption": caption
                                })
                                
                                # Clean up
                                temp_path.unlink()
                            
                            except Exception as e:
                                results.append({
                                    "Image": uploaded_file.name,
                                    "Caption": f"Error: {e}"
                                })
                        
                        # Display results
                        if results:
                            df = pd.DataFrame(results)
                            st.dataframe(df, use_container_width=True)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="caption_results.csv",
                                mime="text/csv"
                            )
        
        with col2:
            st.subheader("📊 Processing Statistics")
            
            if 'results' in locals() and results:
                st.metric("Total Images", len(results))
                
                successful = len([r for r in results if not r["Caption"].startswith("Error")])
                st.metric("Successful Captions", successful)
                
                if len(results) > 0:
                    success_rate = successful / len(results) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with tab4:
        st.header("About This Demo")
        
        st.markdown("""
        ### 🖼️ Image Captioning Demo
        
        This application demonstrates state-of-the-art image captioning using BLIP models
        from Hugging Face Transformers.
        
        #### 🚀 Features:
        - **Single Image Captioning**: Upload an image and get a descriptive caption
        - **Multiple Captions**: Generate diverse captions for the same image
        - **Synthetic Data Generation**: Create sample images for testing
        - **Batch Processing**: Process multiple images at once
        - **Configurable Parameters**: Adjust generation settings
        
        #### 🛠️ Technical Details:
        - **Model**: BLIP (Bootstrapped Language Image Pretraining)
        - **Framework**: Hugging Face Transformers
        - **Interface**: Streamlit
        - **Backend**: PyTorch
        
        #### 📚 Use Cases:
        - Accessibility tools for visually impaired users
        - Automatic alt-text generation for websites
        - Image retrieval and search systems
        - Content moderation and analysis
        
        #### 🔧 Configuration:
        Use the sidebar to adjust:
        - Model selection (Base vs Large)
        - Generation parameters (length, beams, temperature)
        - Sampling strategies
        """)
        
        # Model info
        st.subheader("🤖 Model Information")
        
        model_info = {
            "Model Name": "BLIP (Bootstrapped Language Image Pretraining)",
            "Provider": "Salesforce",
            "Task": "Image-to-Text Generation",
            "Architecture": "Vision-Language Transformer",
            "Parameters": "~248M (Base), ~1.2B (Large)",
            "Training Data": "COCO, Conceptual Captions, SBU Captions"
        }
        
        for key, value in model_info.items():
            st.write(f"**{key}:** {value}")


if __name__ == "__main__":
    main()
