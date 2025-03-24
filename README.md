# TANGO_Medical
TANGO_Medical is a comprehensive platform designed for medical AI applications, providing an intuitive interface for training, fine-tuning, and inference using state-of-the-art deep learning models in the medical domain.

## Features
- **Quick Inference Dashboard**: Perform AI-based medical inference via a user-friendly Streamlit interface
- **Training & Fine-Tuning**: Train and fine-tune models for specialized medical AI applications
- **Learning Materials**: Educational resources for healthcare professionals to understand medical AI
- **Model Repository**: Access pre-trained models optimized for various medical imaging tasks
- **Data Preprocessing Tools**: Specialized tools for medical data preparation
- **Performance Evaluation**: Comprehensive metrics to assess model performance on medical datasets

## Setup
```bash
# Clone the repository
git clone https://github.com/your-org/tango_medical.git
cd tango_medical

# Create and activate virtual environment
python3 -m venv tango_medical_env
source tango_medical_env/bin/activate  # On Windows: tango_medical_env\Scripts\activate

# Install dependencies
(tango_medical_env) pip install -r requirements.txt

# Download pre-trained models (optional)
(tango_medical_env) python download_models.py
```

## System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 10GB+ free disk space

## Usage

### Quick Inference
To run inference using the Streamlit dashboard:
```bash
streamlit run dashboard.py
```
The dashboard allows you to:
- Upload medical images (X-rays, CT scans, MRIs, etc.)
- Select appropriate pre-trained models
- Visualize results with explanatory heatmaps
- Export findings in structured reports

### Training & Fine-Tuning
For model training and fine-tuning:
```bash
streamlit run medical_ai_app.py
```
This interface provides:
- Dataset management tools
- Model architecture selection
- Hyperparameter optimization
- Training progress visualization
- Performance metrics tracking

### Supported Medical Tasks
- Disease detection in chest X-rays
- Tumor segmentation in MRI scans
- Retinal disease classification
- Medical report generation
- Patient outcome prediction

## Configuration
Customize TANGO_Medical by editing the `config.yaml` file:
```yaml
# Example configuration
models:
  chest_xray:
    path: "models/chest_xray_v2.pt"
    input_size: [224, 224]
  brain_mri:
    path: "models/brain_tumor_seg_v1.pt"
    input_size: [256, 256]

inference:
  batch_size: 8
  device: "cuda:0"
  
training:
  learning_rate: 0.0001
  epochs: 50
  early_stopping: true
```

## For Developers
Extend TANGO_Medical with custom modules:
```python
# Example: Adding a custom model
from tango_medical.nets import BaseModel

class CustomMedicalModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Custom initialization
        
    def forward(self, x):
        # Custom forward pass
        return x
```

## Documentation
For complete documentation, visit the [TANGO_Medical Wiki](https://github.com/HyunwooCho/tango_medical/wiki).

## License
TANGO_Medical is released under the MIT License. See LICENSE file for details.
