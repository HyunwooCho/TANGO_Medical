import monai
import torch
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Compose
)
from monai.data import CacheDataset, DataLoader
from monai.apps import download_and_extract

# 데이터 다운로드 (BraTS 같은 오픈 데이터셋 활용 가능)
data_dir = "./data"
resource = "https://drive.google.com/uc?id=1XwEY8OxxAHu8Hq8rX1h8Wrf1s7DAwLWV" # BraTS 샘플 데이터
download_and_extract(resource, data_dir)

# 데이터 리스트 정의
train_files = [{"image": f"{data_dir}/image.nii.gz", "label": f"{data_dir}/label.nii.gz"}]

# 변환 정의
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"])
])

# 데이터셋 및 데이터로더 생성
train_ds = CacheDataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
