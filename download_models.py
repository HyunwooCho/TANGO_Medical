"""
download_models.py

This script downloads pre-trained medical neural network models from MONAI.
It creates a structured directory for storing the models and provides a
command-line interface for selecting which models to download.

1. Model Types: The script supports downloading three main categories of models:
 - Classification models (DenseNet121, EfficientNet, SENet, Vision Transformer)
 - Segmentation models (UNet, SegResNet, VNet, DynUNet)
 - Detection models (RetinaNet, Faster R-CNN, Lesion detector, MULAN)

2. MONAI Bundles: It also supports downloading complete pre-trained model bundles 
for various medical tasks:
 - Spleen CT segmentation
 - Lung nodule detection
 - Brain tumor MRI segmentation
 - Pathology nuclei segmentation & classification
 - Pathology tumor detection
 - Breast density classification

3. Features:
 - Command-line interface with multiple options
 - Progress bars for downloads using tqdm
 - Automatic extraction of zip files
 - Creation of model skeletons with placeholders
 - Comprehensive metadata and README files
 - Organized directory structure

4. Usage Examples:
 - List all available models and bundles
   python download_models.py --list

 - Download all model skeletons
   python download_models.py --all

 - Download specific models
   python download_models.py --models unet densenet121

 - Download specific bundles
   python download_models.py --bundles spleen_ct_segmentation brain_tumor_mri_segmentation

 - Download all models from a specific category
   python download_models.py --category detection

 - Download all bundles
   python download_models.py --all_bundles

 - Specify custom directory
   python download_models.py --all --model_dir ./custom_models_dir
"""
import os
import argparse
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import requests
import json
import shutil
import zipfile
from monai.networks.nets import (
    DenseNet121, UNet, SegResNet, VNet, DynUNet,
    SEResNet50, EfficientNetBN, ViT
)
# from monai.bundle import download

# Try to import detection models
try:
    from monai.networks.nets.detection import RetinaNet
    HAS_DETECTION = True
except ImportError:
    HAS_DETECTION = False
    # Create a placeholder class for RetinaNet
    class RetinaNet:
        def __init__(self, *args, **kwargs):
            pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TANGO_Medical")

# Define model directory
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Define available models
AVAILABLE_MODELS = {
    "classification": {
        "densenet121": {
            "description": "DenseNet121 for medical image classification",
            "class": DenseNet121,
            "source": "monai",
            "bundle_name": "densenet121_classification"
        },
        "efficient_net": {
            "description": "EfficientNet for medical image classification",
            "class": EfficientNetBN,
            "source": "monai",
            "bundle_name": "efficientnet_b0_classification"
        },
        "senet": {
            "description": "SE-ResNet50 for medical image classification",
            "class": SEResNet50,
            "source": "monai",
            "bundle_name": "senet_classification" 
        },
        "vision_transformer": {
            "description": "Vision Transformer for medical image classification",
            "class": ViT,
            "source": "monai",
            "bundle_name": "vit_classification"
        }
    },
    "segmentation": {
        "unet": {
            "description": "UNet for medical image segmentation",
            "class": UNet,
            "source": "monai",
            "bundle_name": "unet_segmentation"
        },
        "segresnet": {
            "description": "SegResNet for 3D medical image segmentation",
            "class": SegResNet,
            "source": "monai",
            "bundle_name": "segresnet_btcv_segmentation"
        },
        "vnet": {
            "description": "VNet for 3D medical image segmentation",
            "class": VNet,
            "source": "monai",
            "bundle_name": "vnet_segmentation"
        },
        "dynunet": {
            "description": "Dynamic UNet for various medical segmentation tasks",
            "class": DynUNet,
            "source": "monai",
            "bundle_name": "dynunet_segmentation"
        }
    },
    "detection": {
        "retinanet": {
            "description": "RetinaNet for lesion/nodule detection in medical images",
            "class": RetinaNet,
            "source": "monai",
            "bundle_name": "retinanet_detection"
        },
        "faster_rcnn": {
            "description": "Faster R-CNN adapted for medical object detection",
            "class": None,  # Will be implemented with PyTorch's implementation
            "source": "pytorch_custom",
            "bundle_name": "faster_rcnn_detection"
        },
        "lesion_detector": {
            "description": "Specialized detector for various types of lesions",
            "class": None,  # Custom architecture
            "source": "custom",
            "bundle_name": "lesion_detection"
        },
        "mulan": {
            "description": "Multi-task Universal Lesion Analysis Network",
            "class": None,  # Custom architecture
            "source": "custom",
            "bundle_name": "mulan_detection"
        }
    }
}

# Available MONAI bundles to download
MONAI_BUNDLES = {
    "spleen_ct_segmentation": {
        "description": "Spleen CT segmentation model",
        "url": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/spleen_ct_segmentation_v0.5.3.zip"
    },
    "lung_nodule_ct_detection": {
        "description": "Lung nodule detection from CT scans",
        "url": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/lung_nodule_ct_detection_v0.5.9.zip"
    },
    "brain_tumor_mri_segmentation": {
        "description": "Brain tumor segmentation from MRI",
        "url": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/brats_mri_segmentation_v0.4.8.zip"
    },
    "pathology_nuclei_segmentation_classification": {
        "description": "Nuclei segmentation and classification in pathology images",
        "url": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/pathology_nuclei_segmentation_classification_v0.2.1.zip"
    },
    "breast_density_classification": {
        "description": "Breast density classification from mammograms",
        "url": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/breast_density_classification_v0.1.5.zip"
    },
    "pathology_tumor_detection": {
        "description": "Tumor detection model, based on ResNet18",
        "url": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/pathology_tumor_detection_v0.5.7.zip"
    },
}

def download_file(url, destination):
    """Download a file from URL to destination with progress bar"""
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        logger.warning(f"Download failed! Status code: {response.status_code}")

    total_size = response.headers.get('content-length')
    total_size = int(total_size) if total_size else None
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    return destination

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory"""
    if not zipfile.is_zipfile(zip_path):
        logger.warning(f"{zip_path} is not a valid ZIP file.")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info(f"Extracted {zip_path} to {extract_to}")
    
    # Clean up zip file after extraction
    os.remove(zip_path)

def download_monai_bundle(bundle_name, model_dir):
    """Download a MONAI bundle with error handling"""
    try:
        if bundle_name not in MONAI_BUNDLES:
            logger.error(f"Bundle {bundle_name} not found in available bundles.")
            return False
        
        bundle_dir = os.path.join(model_dir, "bundles", bundle_name)
        os.makedirs(bundle_dir, exist_ok=True)
        
        # Download bundle from the specified source or url
        zip_path = os.path.join(bundle_dir, f"{bundle_name}.zip")
        logger.info("")
        logger.info(f"Downloading bundle {bundle_name}...")
        download_file(MONAI_BUNDLES[bundle_name]["url"], zip_path)
        # download(name=bundle_name, bundle_dir=bundle_dir)
        
        # Extract the zip file
        logger.info(f"Extracting bundle {bundle_name}...")
        extract_zip(zip_path, bundle_dir)
        
        # Create a metadata file
        metadata = {
            "name": bundle_name,
            "description": MONAI_BUNDLES[bundle_name]["description"],
            "source": "MONAI Model Zoo",
            "download_date": str(datetime.now().date()),
        }
        
        with open(os.path.join(bundle_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully downloaded and extracted bundle {bundle_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading bundle {bundle_name}: {str(e)}")
        return False

def create_model_skeleton(model_name, model_info, model_dir):
    """Create a model skeleton directory with info"""
    try:
        # Create model directory
        model_category = next(cat for cat, models in AVAILABLE_MODELS.items() if model_name in models)
        model_path = os.path.join(model_dir, model_category, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Create a placeholder model
        if model_info["class"] is not None:
            model_class = model_info["class"]
            if model_name == "densenet121":
                model = model_class(spatial_dims=2, in_channels=3, out_channels=2)
            elif model_name == "efficient_net":
                model = model_class("efficientnet-b0", spatial_dims=2, in_channels=3, num_classes=2)
            elif model_name == "senet":
                model = model_class(spatial_dims=2, in_channels=3, num_classes=2)
            elif model_name == "vision_transformer":
                model = model_class(in_channels=3, img_size=(224, 224), patch_size=(16, 16), 
                                    hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12, 
                                    proj_type='conv', pos_embed_type='learnable', 
                                    classification=True, num_classes=2, spatial_dims=2)
            elif model_name == "unet":
                model = model_class(spatial_dims=2, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), 
                                    strides=(2, 2, 2, 2))
            elif model_name == "segresnet":
                model = model_class(spatial_dims=3, in_channels=1, out_channels=2, init_filters=8)
            elif model_name == "vnet":
                model = model_class(spatial_dims=3, in_channels=1, out_channels=2)
            elif model_name == "dynunet":
                model = model_class(spatial_dims=3, in_channels=1, out_channels=2, 
                                    kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 
                                    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                                    upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2]])
            elif model_name == "retinanet" and HAS_DETECTION:
                model = model_class(
                    spatial_dims=2,
                    in_channels=3,
                    num_classes=2,
                    backbone_name="resnet50"
                )
            else:
                logger.info(f"Model {model_name} doesn't have an initialized skeleton, creating metadata only")
                model = None
                
            # Save model placeholder if model was created
            if model is not None:
                torch.save({"model": model.state_dict()}, os.path.join(model_path, "model_placeholder.pt"))
                model_architecture = model_info["class"].__name__
                architecture_details = str(model)
            else:
                model_architecture = "Custom"
                architecture_details = "Custom architecture"
        else:
            # For models without classes (custom implementations)
            logger.info(f"Model {model_name} doesn't have a class implementation, creating metadata only")
            model = None
            model_architecture = "Custom"
            architecture_details = "Custom architecture"
        
        # Create metadata file
        metadata = {
            "name": model_name,
            "description": model_info["description"],
            "architecture": model_architecture,
            "created_date": str(datetime.now().date()),
            "source": model_info["source"],
            "status": "placeholder",
            "config": {
                "architecture_details": architecture_details
            }
        }
        
        with open(os.path.join(model_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create a README file
        with open(os.path.join(model_path, "README.md"), 'w') as f:
            f.write(f"# {model_name.capitalize()}\n\n")
            f.write(f"{model_info['description']}\n\n")
            f.write("## Model Details\n\n")
            f.write(f"- Architecture: {model_architecture}\n")
            f.write(f"- Source: {model_info['source']}\n")
            f.write("- Status: Placeholder (replace with trained model)\n\n")
            f.write("## Usage\n\n")
            
            if model_info["source"] == "monai":
                f.write("```python\n")
                f.write(f"from monai.networks.nets import {model_architecture}\n")
                f.write(f"model = {model_architecture}(...)\n")
                f.write("model.load_state_dict(torch.load('model.pt'))\n")
                f.write("```\n")
            else:
                f.write("```python\n")
                f.write("# Custom implementation - see documentation\n")
                f.write("model = YourModelImplementation(...)\n")
                f.write("model.load_state_dict(torch.load('model.pt'))\n")
                f.write("```\n")
            
            # Add detector-specific information for detection models
            if model_category == "detection":
                f.write("\n## Detection-Specific Information\n\n")
                f.write("### Input Format\n")
                f.write("- Input: Medical images (CT, MRI, X-ray, etc.)\n")
                f.write("- Preprocessing: Normalization, resizing to required dimensions\n\n")
                
                f.write("### Output Format\n")
                f.write("- Bounding boxes: [x1, y1, x2, y2]\n")
                f.write("- Class probabilities for each detected object\n")
                f.write("- Confidence scores\n\n")
                
                f.write("### Detection Metrics\n")
                f.write("- mAP (mean Average Precision)\n")
                f.write("- Precision, Recall, F1-score\n")
                f.write("- FROC (Free-Response Receiver Operating Characteristic)\n")
        
        logger.info(f"Created model skeleton for {model_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating model skeleton for {model_name}: {str(e)}")
        return False

def main():
    """Main function to parse arguments and download models"""
    parser = argparse.ArgumentParser(description="Download pre-trained medical models from MONAI")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Directory to store models (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--list", action="store_true",
                        help="List available models")
    parser.add_argument("--all", action="store_true",
                        help="Download all available models")
    parser.add_argument("--models", type=str, nargs="+",
                        help="Specific models to download (e.g., unet segresnet)")
    parser.add_argument("--bundles", type=str, nargs="+",
                        help="Specific MONAI bundles to download")
    parser.add_argument("--all_bundles", action="store_true",
                        help="Download all available MONAI bundles")
    parser.add_argument("--category", type=str, choices=["classification", "segmentation", "detection"],
                        help="Download all models from a specific category")
    
    args = parser.parse_args()
    
    # Create model directory
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir, exist_ok=True)
        logger.info(f"Created model directory: {args.model_dir}")
    
    # Just list available models
    if args.list:
        print("\n=== Available Models ===")
        for category, models in AVAILABLE_MODELS.items():
            print(f"\n{category.upper()}:")
            for model_name, model_info in models.items():
                print(f"  - {model_name}: {model_info['description']}")
        
        print("\n=== Available MONAI Bundles ===")
        for bundle_name, bundle_info in MONAI_BUNDLES.items():
            print(f"  - {bundle_name}: {bundle_info['description']}")
        return
    
    # Create category directories
    for category in AVAILABLE_MODELS.keys():
        os.makedirs(os.path.join(args.model_dir, category), exist_ok=True)
    
    # Create bundle directory
    os.makedirs(os.path.join(args.model_dir, "bundles"), exist_ok=True)
    
    # Download models from a specific category
    if args.category:
        logger.info(f"Downloading all models from category: {args.category}")
        if args.category in AVAILABLE_MODELS:
            for model_name, model_info in AVAILABLE_MODELS[args.category].items():
                create_model_skeleton(model_name, model_info, args.model_dir)
        else:
            logger.error(f"Category {args.category} not found")
    
    # Download specified models
    if args.all:
        logger.info("Downloading all model skeletons...")
        for category, models in AVAILABLE_MODELS.items():
            for model_name, model_info in models.items():
                create_model_skeleton(model_name, model_info, args.model_dir)
    
    elif args.models:
        for model_name in args.models:
            # Find the model in the available models
            found = False
            for category, models in AVAILABLE_MODELS.items():
                if model_name in models:
                    logger.info(f"Creating skeleton for {model_name}...")
                    create_model_skeleton(model_name, models[model_name], args.model_dir)
                    found = True
                    break
            
            if not found:
                logger.error(f"Model {model_name} not found in available models")
    
    # Download specified bundles
    if args.all_bundles:
        logger.info("Downloading all MONAI bundles...")
        for bundle_name in MONAI_BUNDLES.keys():
            download_monai_bundle(bundle_name, args.model_dir)
    
    elif args.bundles:
        for bundle_name in args.bundles:
            if bundle_name in MONAI_BUNDLES:
                logger.info(f"Downloading bundle {bundle_name}...")
                download_monai_bundle(bundle_name, args.model_dir)
            else:
                logger.error(f"Bundle {bundle_name} not found in available bundles")
    
    # If no specific action was requested, show help
    if not (args.list or args.all or args.models or args.all_bundles or args.bundles or args.category):
        parser.print_help()
        print("\nUse --list to see available models and bundles")

if __name__ == "__main__":
    main()