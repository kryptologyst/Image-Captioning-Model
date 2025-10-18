# Image Captioning Model

A production-ready image captioning system using state-of-the-art BLIP models from Hugging Face Transformers. This project provides both a Python API and a user-friendly web interface for generating natural language descriptions of images.

## Features

- **State-of-the-art Models**: Uses BLIP (Bootstrapped Language Image Pretraining) models
- **Web Interface**: Beautiful Streamlit-based web application
- **Batch Processing**: Process multiple images simultaneously
- **Synthetic Data Generation**: Create sample images for testing
- **Configurable**: YAML-based configuration system
- **Modern Code**: Type hints, docstrings, and PEP8 compliance
- **Multiple Captions**: Generate diverse captions for the same image
- **Production Ready**: Logging, error handling, and testing

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Image-Captioning-Model.git
   cd Image-Captioning-Model
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Interface

Launch the Streamlit web application:

```bash
streamlit run web_app/app.py
```

Open your browser to `http://localhost:8501` to access the interface.

### Using the Python API

```python
from src.image_captioner import ImageCaptioner

# Initialize the captioner
captioner = ImageCaptioner()

# Generate a caption for an image
caption = captioner.generate_caption("path/to/your/image.jpg")
print(f"Caption: {caption}")

# Generate multiple diverse captions
captions = captioner.generate_multiple_captions("path/to/your/image.jpg", num_captions=3)
for i, cap in enumerate(captions, 1):
    print(f"Caption {i}: {cap}")
```

## 📁 Project Structure

```
image-captioning-model/
├── src/                    # Source code
│   ├── image_captioner.py  # Main captioning class
│   ├── config.py           # Configuration management
│   └── data_generator.py   # Synthetic data generation
├── web_app/               # Web interface
│   └── app.py             # Streamlit application
├── config/                # Configuration files
│   └── config.yaml        # Main configuration
├── data/                  # Data directory
│   └── synthetic/         # Generated sample images
├── models/                # Model cache directory
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🛠️ Configuration

The project uses YAML configuration files for easy customization. Edit `config/config.yaml` to modify:

- **Model settings**: Choose between BLIP base/large models
- **Generation parameters**: Control caption length, diversity, etc.
- **Application settings**: Debug mode, logging level, directories

### Environment Variables

You can also override configuration using environment variables:

```bash
export MODEL_NAME="Salesforce/blip-image-captioning-large"
export DEVICE="cuda"
export DEBUG="true"
```

## Usage Examples

### Single Image Captioning

```python
from src.image_captioner import ImageCaptioner

captioner = ImageCaptioner()

# Basic captioning
caption = captioner.generate_caption("dog_in_park.jpg")
print(caption)  # "a dog running in a park"

# With custom parameters
caption = captioner.generate_caption(
    "dog_in_park.jpg",
    max_length=30,
    temperature=0.8,
    do_sample=True
)
```

### Batch Processing

```python
# Process multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = captioner.batch_process(image_paths)

for path, caption in results.items():
    print(f"{path}: {caption}")
```

### Synthetic Data Generation

```python
from src.data_generator import SyntheticImageGenerator

# Create sample images
generator = SyntheticImageGenerator()
dataset = generator.create_dataset(num_images=20, categories=["animals", "objects"])

# Generate scene images
scene_paths = generator.create_scene_dataset(num_scenes=10)
```

### Visualization

```python
# Visualize image with caption
captioner.visualize_result("image.jpg", save_path="result.png")
```

## Web Interface Features

The Streamlit web interface provides:

1. **Upload Image**: Upload and caption single images
2. **Generate Samples**: Create synthetic images for testing
3. **Batch Processing**: Process multiple images at once
4. **Configuration**: Adjust model parameters in real-time
5. **Export Results**: Download results as CSV

### Web Interface Tabs

- **Upload Image**: Single image captioning with multiple caption generation
- **Generate Samples**: Create and caption synthetic images
- **Batch Processing**: Process multiple uploaded images
- **About**: Project information and model details

## 🔧 Advanced Usage

### Custom Model Configuration

```python
from src.image_captioner import ImageCaptioner
from src.config import Config

# Load custom configuration
config = Config("path/to/custom_config.yaml")
captioner = ImageCaptioner(**config.get_model_kwargs())
```

### Programmatic Configuration

```python
# Initialize with custom settings
captioner = ImageCaptioner(
    model_name="Salesforce/blip-image-captioning-large",
    device="cuda",
    use_pipeline=True
)
```

### Error Handling

```python
try:
    caption = captioner.generate_caption("image.jpg")
except FileNotFoundError:
    print("Image file not found")
except Exception as e:
    print(f"Caption generation failed: {e}")
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Generate sample data for testing:

```bash
python src/data_generator.py
```

## Performance

### Model Comparison

| Model | Parameters | Speed | Quality |
|-------|------------|-------|---------|
| BLIP Base | ~248M | Fast | Good |
| BLIP Large | ~1.2B | Slower | Better |

### Hardware Requirements

- **CPU**: Modern multi-core processor
- **RAM**: 8GB+ recommended
- **GPU**: Optional but recommended for faster inference
- **Storage**: 2GB+ for model cache

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "web_app/app.py", "--server.address", "0.0.0.0"]
```

### Production Considerations

- Use GPU acceleration for better performance
- Implement caching for frequently accessed models
- Add authentication for web interface
- Monitor memory usage for batch processing
- Implement rate limiting for API endpoints

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Salesforce](https://www.salesforce.com/) for the BLIP models
- [Streamlit](https://streamlit.io/) for the web framework
- [PyTorch](https://pytorch.org/) for the deep learning framework

## References

- [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model download fails**: Check internet connection and disk space
3. **Import errors**: Ensure all dependencies are installed
4. **Slow inference**: Use GPU acceleration or smaller model


# Image-Captioning-Model
