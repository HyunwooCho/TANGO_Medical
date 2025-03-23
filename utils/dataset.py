import os
import monai
import torch
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Compose
)
from monai.data import CacheDataset, DataLoader
from monai.apps import download_and_extract

class MedicalDataset:
    """MONAI를 활용한 의료 영상 데이터셋 다운로드 및 로드"""

    def __init__(self, data_dir="./data", batch_size=2):
        self.data_dir = data_dir
        self.batch_size = batch_size

        # 데이터 다운로드 (BraTS 같은 오픈 데이터셋 활용 가능)
        self.resource = "https://drive.google.com/uc?id=1XwEY8OxxAHu8Hq8rX1h8Wrf1s7DAwLWV"  # BraTS 샘플 데이터
        self.download_dataset()

        # 데이터셋 생성
        self.train_loader = self.create_dataloader()

    def download_dataset(self):
        """BraTS 같은 공개 데이터셋 다운로드 및 압축 해제"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            download_and_extract(self.resource, self.data_dir)
            print(f"✅ 데이터 다운로드 완료: {self.data_dir}")
        else:
            print(f"✅ 기존 데이터 사용: {self.data_dir}")

    def create_dataloader(self):
        """데이터셋을 로드하고 DataLoader 생성"""
        train_files = [{"image": f"{self.data_dir}/image.nii.gz", "label": f"{self.data_dir}/label.nii.gz"}]

        # 변환 정의
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image", "label"])
        ])

        # 데이터셋 및 데이터로더 생성
        train_ds = CacheDataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        return train_loader

    def get_dataloader(self):
        """DataLoader 반환"""
        return self.train_loader
