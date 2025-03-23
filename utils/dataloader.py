import numpy as np
import nibabel as nib
from PIL import Image
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

class MedicalImageLoader:
    """NIfTI(.nii, .nii.gz) 형식의 의료 영상을 로드하는 클래스"""

    @staticmethod
    def load_image(file_path):
        """NIfTI 파일을 읽고, 2D 이미지로 변환"""
        try:
            img = nib.load(file_path)
            data = img.get_fdata()

            # 3D 데이터에서 가운데 단면을 2D로 변환
            slice_index = data.shape[2] // 2
            image_array = data[:, :, slice_index]

            # 픽셀 값을 0~255 범위로 정규화
            image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
            image_array = image_array.astype(np.uint8)

            return Image.fromarray(image_array)

        except Exception as e:
            print(f"❌ 오류: {e}")
            return None
