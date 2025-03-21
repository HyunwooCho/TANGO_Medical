import numpy as np
import cv2
import matplotlib.pyplot as plt

# Grad-CAM 적용 함수
def apply_gradcam(model, image):
    image = image.unsqueeze(0).to(device)
    model.eval()

    # Forward pass
    output = model(image)
    
    # Gradients 가져오기
    output.backward(torch.ones_like(output))
    gradients = model.swinViT.layers[-1].register_forward_hook(lambda mod, inp, out: out)
    
    # Grad-CAM 적용
    weights = torch.mean(gradients, dim=[2, 3, 4])
    cam = torch.sum(weights * image.squeeze(), dim=0).cpu().detach().numpy()
    cam = cv2.resize(cam, (128, 128))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize
    
    return cam

# 예제 데이터에 적용
image_sample = train_ds[0]["image"]
grad_cam_map = apply_gradcam(model, image_sample)

# 결과 시각화
plt.imshow(grad_cam_map, cmap="jet", alpha=0.5)
plt.colorbar()
plt.show()
