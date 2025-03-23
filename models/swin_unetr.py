import torch
from monai.networks.nets import SwinUNETR

class SwinUNETRModel:
    """ViT 기반 Swin UNETR 모델을 로드하는 클래스"""

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=1, feature_size=48).to(self.device)
        self.model.eval()

    def get_model(self):
        """모델 반환"""
        return self.model
