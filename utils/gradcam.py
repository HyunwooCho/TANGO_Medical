import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    """Grad-CAM을 활용하여 중요 영역을 시각화하는 클래스"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def apply_gradcam(self, image):
        """입력 이미지에 Grad-CAM 적용"""
        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        self.model.eval()
        output = self.model(image)
        output.backward(torch.ones_like(output))

        # 마지막 레이어에서 Gradients 추출
        gradients = self.model.swinViT.layers[-1].register_forward_hook(lambda mod, inp, out: out)
        weights = torch.mean(gradients, dim=[2, 3, 4])

        # Grad-CAM 적용
        cam = torch.sum(weights * image.squeeze(), dim=0).cpu().detach().numpy()
        cam = cv2.resize(cam, (128, 128))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize

        return cam

# # 예제 데이터에 적용
# image_sample = train_ds[0]["image"]
# grad_cam_map = apply_gradcam(model, image_sample)

# # 결과 시각화
# plt.imshow(grad_cam_map, cmap="jet", alpha=0.5)
# plt.colorbar()
# plt.show()
