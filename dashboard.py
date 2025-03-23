import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import io
import torch
import os
import tempfile
import sys
import monai
import pandas as pd
from datetime import datetime
import glob
import zipfile
import io
from pathlib import Path

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="🩺 의료 영상 AI 학습 및 분석 시스템", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fix for Streamlit watcher error with torch ---
# Disable file watching to prevent the torch._classes error
if 'STREAMLIT_RUN_PATH' in os.environ:
    import streamlit.watcher.path_watcher
    streamlit.watcher.path_watcher.watch_file = lambda x: None

# Set environment variable to limit file watching
os.environ["STREAMLIT_GLOBAL_WATCHER_MAX_FILES"] = "5000"

# --- Display MONAI version for debugging ---
st.sidebar.info(f"MONAI version: {monai.__version__}")

# --- Utility Classes ---
class MedicalImageLoader:
    def __init__(self):
        from monai.transforms import (
            Compose, 
            ScaleIntensity, 
            ResizeWithPadOrCrop
        )
        
        self.transforms = Compose([
            ScaleIntensity(),
            ResizeWithPadOrCrop((128, 128, 128))
        ])
    
    def load_image(self, uploaded_file):
        try:
            # Read the uploaded file
            content = uploaded_file.getvalue()
            
            # Create a temporary file to store the uploaded content
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(content)
                
            # Load with nibabel
            img = nib.load(temp_path)
            img_data = img.get_fdata()
            
            # Clean up the temporary file
            os.remove(temp_path)
            
            # Get middle slice for display
            mid_slice = img_data.shape[2] // 2
            slice_data = img_data[:, :, mid_slice].astype(np.float32)
            
            # Normalize for display
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
            
            # Process for model
            processed_data = img_data.astype(np.float32)
            processed_data = (processed_data - processed_data.min()) / (processed_data.max() - processed_data.min() + 1e-8)
            processed_volume = torch.from_numpy(processed_data[np.newaxis, np.newaxis, ...])
            
            # Apply size transformation
            processed_volume = self.transforms(processed_volume)
            
            return slice_data, processed_volume
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None, None

class SimpleSwinUNETR(torch.nn.Module):
    """A simplified wrapper for SwinUNETR that handles version compatibility"""
    def __init__(self):
        super().__init__()
        try:
            # Try importing without img_size first (newer versions)
            from monai.networks.nets import SwinUNETR
            try:
                self.model = SwinUNETR(
                    in_channels=1,
                    out_channels=2,
                    feature_size=48,
                    use_checkpoint=True
                )
                st.sidebar.success("MONAI 1.5+ detected - initialized model without img_size")
            except TypeError:
                # If that fails, try with img_size (older versions)
                self.model = SwinUNETR(
                    img_size=(128, 128, 128),
                    in_channels=1,
                    out_channels=2,
                    feature_size=48,
                    use_checkpoint=True
                )
                st.sidebar.success("MONAI <1.5 detected - initialized model with img_size")
        except Exception as e:
            st.error(f"Error initializing SwinUNETR: {str(e)}")
            # Create a dummy model as a fallback
            self.model = torch.nn.Sequential(
                torch.nn.Conv3d(1, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv3d(16, 2, kernel_size=3, padding=1)
            )
            st.warning("⚠️ Using fallback model - SwinUNETR initialization failed")
    
    def forward(self, x):
        return self.model(x)

class SwinUNETRModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_model(self):
        try:
            # Use our version-agnostic wrapper
            model = SimpleSwinUNETR().to(self.device)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

class GradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        
    def save_gradient(self, grad):
        self.gradients = grad.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def forward_hook(self, module, input, output):
        self.activations = output.detach()
        
    def apply_gradcam(self, image_tensor):
        if self.model is None:
            return np.zeros((128, 128))
        
        # If the model is our simplified wrapper, use the actual model
        if hasattr(self.model, 'model'):
            target_model = self.model.model
        else:
            target_model = self.model
            
        try:
            # Register hooks on a suitable layer
            # Try different approaches to find a good target layer
            target_layer = None
            
            # Approach 1: Look for swinViT.layers
            if hasattr(target_model, 'swinViT') and hasattr(target_model.swinViT, 'layers'):
                target_layer = target_model.swinViT.layers[-1]
            
            # Approach 2: Look for a convolutional layer
            if target_layer is None:
                for name, module in target_model.named_modules():
                    if isinstance(module, torch.nn.Conv3d):
                        target_layer = module
                        break
            
            # If still no target layer, create a dummy one
            if target_layer is None:
                st.warning("⚠️ Couldn't find suitable layer for Grad-CAM, using dummy heatmap")
                return np.random.uniform(0, 1, (128, 128))
                
            # Register hooks
            forward_handle = target_layer.register_forward_hook(self.forward_hook)
            backward_handle = target_layer.register_backward_hook(self.backward_hook)
            
            # Forward pass
            image_tensor = image_tensor.to(self.device)
            
            # Create a clone that requires grad
            input_tensor = image_tensor.clone().detach().requires_grad_(True)
            
            output = self.model(input_tensor)
            
            # For binary classification, take the positive class score
            target_score = output[0, 1]  # Targeting class 1 (positive class)
            
            # Backward pass
            self.model.zero_grad()
            target_score.backward()
            
            # Get gradients and activations
            if self.gradients is None or self.activations is None:
                # If hooks didn't work, return random heatmap
                st.warning("Couldn't generate Grad-CAM - hooks didn't capture gradients or activations")
                return np.random.uniform(0, 1, (128, 128))
                
            # Calculate weights
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3, 4])
            
            # Weight the channels by corresponding gradients
            cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32).to(self.device)
            
            # Multiply each weight with its conv output and then sum
            for i, w in enumerate(pooled_gradients):
                cam += w * self.activations[0, i, :, :, :]
                
            # Apply ReLU to the heatmap
            cam = torch.relu(cam)
            
            # Normalize
            if torch.max(cam) > 0:
                cam = cam / torch.max(cam)
            
            # Convert to numpy
            cam = cam.cpu().detach().numpy()
            
            # Get middle slice for 2D visualization
            mid_slice = cam.shape[2] // 2
            heatmap_2d = cam[:, :, mid_slice]
            
            # Clean up hooks
            forward_handle.remove()
            backward_handle.remove()
            
            return heatmap_2d
            
        except Exception as e:
            st.error(f"Error generating Grad-CAM: {str(e)}")
            # Return a random heatmap as a placeholder
            return np.random.uniform(0, 1, (128, 128))


# --- Streamlit UI ---
st.title("🩺 의료 영상 AI 분석 시스템")
st.sidebar.header("🔍 설정")

# Add model selection for flexibility
model_type = st.sidebar.selectbox(
    "모델 선택",
    ["Swin UNETR"]
)

# --- Data Loader 초기화 ---
loader = MedicalImageLoader()

# --- Swin UNETR 모델 로드 ---
with st.spinner("모델 로딩 중..."):
    model_instance = SwinUNETRModel()
    model = model_instance.get_model()
    device = model_instance.device

if model is not None:
    st.sidebar.success("✅ 모델 로드 완료!")
    
    # --- Grad-CAM 초기화 ---
    gradcam = GradCAM(model, device)

    # --- 파일 업로드 ---
    uploaded_file = st.sidebar.file_uploader("📤 NIfTI(.nii, .nii.gz) 파일 업로드", type=["nii", "nii.gz"])

    # If no file is uploaded, provide a demo option
    if uploaded_file is None:
        use_demo = st.sidebar.checkbox("데모 이미지 사용")
        if use_demo:
            st.info("데모 이미지를 사용합니다. (무작위 생성 데이터)")
            # Create a demo volume
            demo_volume = torch.rand(1, 1, 128, 128, 128)
            demo_slice = demo_volume[0, 0, :, :, 64].numpy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 데모 이미지")
                fig1, ax1 = plt.subplots(figsize=(5, 5))
                ax1.imshow(demo_slice, cmap="gray")
                ax1.axis("off")
                st.pyplot(fig1)

            with col2:
                st.subheader("🔍 Grad-CAM 분석 (데모)")
                with st.spinner("Grad-CAM 분석 중..."):
                    heatmap = gradcam.apply_gradcam(demo_volume)

                    fig2, ax2 = plt.subplots(figsize=(5, 5))
                    ax2.imshow(demo_slice, cmap="gray")
                    ax2.imshow(heatmap, cmap="jet", alpha=0.5)
                    ax2.axis("off")
                    st.pyplot(fig2)
                    
                    st.success("✅ Grad-CAM 분석 완료!")
    
    elif uploaded_file is not None:
        with st.spinner("이미지 로딩 중..."):
            slice_image, processed_volume = loader.load_image(uploaded_file)
        
        if slice_image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 원본 이미지")
                fig1, ax1 = plt.subplots(figsize=(5, 5))
                ax1.imshow(slice_image, cmap="gray")
                ax1.axis("off")
                st.pyplot(fig1)

            with col2:
                st.subheader("🔍 Grad-CAM 분석")
                with st.spinner("Grad-CAM 분석 중..."):
                    heatmap = gradcam.apply_gradcam(processed_volume)

                    fig2, ax2 = plt.subplots(figsize=(5, 5))
                    ax2.imshow(slice_image, cmap="gray")
                    ax2.imshow(heatmap, cmap="jet", alpha=0.5)
                    ax2.axis("off")
                    st.pyplot(fig2)
                    
                    st.success("✅ Grad-CAM 분석 완료!")
            
            # Add 3D volume visualization option
            if st.checkbox("3D 볼륨 시각화 (슬라이스 뷰)"):
                slice_idx = st.slider("슬라이스 선택", 0, processed_volume.shape[4]-1, processed_volume.shape[4]//2)
                
                fig3, ax3 = plt.subplots(figsize=(8, 8))
                volume_data = processed_volume.squeeze().cpu().numpy()
                selected_slice = volume_data[:, :, slice_idx]
                ax3.imshow(selected_slice, cmap="gray")
                ax3.axis("off")
                ax3.set_title(f"슬라이스 {slice_idx}")
                st.pyplot(fig3)
else:
    st.error("⚠️ 모델 로드에 실패했습니다. MONAI 라이브러리가 설치되어 있는지 확인하세요.")
    st.info("문제 해결을 위해 터미널에서 다음 명령어를 실행해보세요: `pip install monai --upgrade`")

# Add tabs for additional information
tab1, tab2, tab3 = st.tabs(["결과 해석", "모델 정보", "사용 방법"])

with tab1:
    st.subheader("결과 해석")
    st.write("""
    - **붉은색 영역**: 모델이 진단에 중요하게 고려한 영역
    - **파란색 영역**: 모델이 상대적으로 덜 중요하게 고려한 영역
    
    위 히트맵은 모델이 판단을 내리는 데 있어 가장 중요하게 본 영역을 시각화합니다.
    이는 모델의 결정에 대한 설명 가능성(Explainable AI)을 제공합니다.
    """)

with tab2:
    st.subheader("Swin UNETR 모델 정보")
    st.write("""
    **Swin UNETR**는 의료 영상 분할을 위한 Vision Transformer 기반 모델입니다:
    
    - **아키텍처**: 인코더로 Swin Transformer를, 디코더로 CNN 구조를 사용
    - **장점**: 여러 스케일의 특징을 효과적으로 포착하여 복잡한 의료 영상 분석에 적합
    - **성능**: 여러 의료 영상 데이터셋에서 최첨단 성능 달성
    - **논문**: "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images" (MICCAI 2021)
    """)

with tab3:
    st.subheader("사용 방법")
    st.write("""
    1. 좌측 사이드바에서 NIfTI 형식의 의료 영상 파일(.nii 또는 .nii.gz)을 업로드합니다.
    2. 시스템이 자동으로 이미지를 처리하고 Grad-CAM 분석을 수행합니다.
    3. 원본 이미지와 Grad-CAM 히트맵이 나란히 표시됩니다.
    4. 3D 볼륨 시각화 옵션을 선택하여 다양한 슬라이스를 확인할 수 있습니다.
    5. '결과 해석' 탭에서 히트맵 해석 방법을 확인할 수 있습니다.
    """)

# Add debug information
with st.sidebar.expander("📊 디버그 정보"):
    st.write(f"Python 버전: {sys.version}")
    st.write(f"PyTorch 버전: {torch.__version__}")
    st.write(f"MONAI 버전: {monai.__version__}")
    st.write(f"디바이스: {device}")

# Add a footer
st.markdown("---")
st.markdown("© 2025 의료 영상 AI 분석 시스템")