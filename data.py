import os
import logging
import nibabel as nib
import numpy as np
from monai.data import Dataset, CacheDataset, partition_dataset
from monai.apps import download_and_extract, MedNISTDataset, DecathlonDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TANGO-MedicalDataset")

# MONAI 데이터셋 다운로드 및 처리
def download_monai_dataset(dataset_name, root_dir, cache_rate):
    """
    MONAI 데이터셋 다운로드 및 처리

    Args:
        dataset_name: MONAI 데이터셋(MedNIST: 분류, DecathlonDataset: 분할)
        root_dir: 저장할 루트 디렉토리
        cache_rate: 메모리 캐싱 비율

    Returns:
        data_list: 데이터 파일 리스트
        len(data_list): 총 데이터 개수
        num_classes: 클래스 개수
    """
    # 루트 디렉토리가 존재하는지 확인하고 없으면 생성
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
        logger.info(f"'{root_dir}' 디렉토리를 생성했습니다.")

    # 데이터셋 준비
    if dataset_name == "MedNIST":
        dataset = MedNISTDataset(
            root_dir=root_dir,
            transform=None,
            section="training",
            download=True,
            cache_rate=cache_rate
        )
        num_classes = 6  # MedNIST는 6개 클래스

        # MedNIST는 분류 데이터셋이므로 Segmentation용으로 변환 필요
        data_list = []
        for item in dataset:
            # MedNIST 이미지를 NIfTI 형식으로 가정하는 딕셔너리 구조로 변환
            # TODO: 변환 코드 필요함
            # data_item = {
            #     "image": item["image"],
            #     "label": item["label"],
            #     "task_type": "classification"
            # }
            data_list.append(data_item)

    else:
        # Medical Decathlon 데이터셋
        dataset = DecathlonDataset(
            root_dir=root_dir,
            task=dataset_name,
            transform=None,
            section="training",
            download=True,
            cache_rate=cache_rate
        )

        # 세그멘테이션 데이터셋 파일 정보 수집
        data_list = []
        for item in dataset.data:
            # Decathlon 데이터셋은 이미지 파일 경로와 라벨 파일 경로 제공
            data_item = {
                "image": item["image"],
                "label": item["label"],
                "task_type": "segmentation"
            }
            data_list.append(data_item)

        # 세그멘테이션 데이터셋의 클래스 수 설정
        if dataset_name == "Task01_BrainTumour":
            num_classes = 4
        elif dataset_name == "Task02_Heart":
            num_classes = 2
        elif dataset_name == "Task03_Liver":
            num_classes = 3
        elif dataset_name == "Task04_Hippocampus":
            num_classes = 3
        elif dataset_name == "Task05_Prostate":
            num_classes = 3
        elif dataset_name == "Task06_Lung":
            num_classes = 2
        elif dataset_name == "Task07_Pancreas":
            num_classes = 3
        elif dataset_name == "Task08_HepaticVessel":
            num_classes = 3
        elif dataset_name == "Task09_Spleen":
            num_classes = 2
        elif dataset_name == "Task10_Colon":
            num_classes = 2
    
    logger.info(f"{len(data_list)}개의 {dataset_name} 데이터를 {root_dir}에 생성했습니다.")
    return data_list, len(data_list), num_classes

# 데모용 샘플 데이터 생성
def generate_sample_data(num_samples=5, output_dir="./sample_data"):
    """
    샘플 데이터셋 생성 (데모 목적)
    
    Args:
        num_samples: 생성할 샘플 수
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 샘플 데이터셋 파일 정보 생성
    data_list = []
    
    for i in range(num_samples):
        # 랜덤 볼륨 생성 (이미지)
        img_data = np.random.rand(64, 64, 64) * 0.1  # 배경
        
        # 단순한 형태 추가 (구)
        center = np.array([32, 32, 32])
        radius = 16
        
        x, y, z = np.ogrid[:64, :64, :64]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        img_data[dist <= radius] = np.random.rand() * 0.5 + 0.5
        
        # 노이즈 추가
        img_data += np.random.randn(64, 64, 64) * 0.02
        
        # 세그멘테이션 마스크 생성
        seg_data = np.zeros((64, 64, 64), dtype=np.int32)
        seg_data[dist <= radius * 0.8] = 1  # 클래스 1 (전경)
        
        # NIfTI 파일 생성
        img_path = os.path.join(output_dir, f"sample_img_{i}.nii.gz")
        seg_path = os.path.join(output_dir, f"sample_seg_{i}.nii.gz")
        
        nib.save(nib.Nifti1Image(img_data, np.eye(4)), img_path)
        nib.save(nib.Nifti1Image(seg_data, np.eye(4)), seg_path)
        
        # 데이터 사전 추가
        data_list.append({"image": img_path, "label": seg_path})
    
    logger.info(f"{num_samples}개의 샘플 데이터를 {output_dir}에 생성했습니다.")
    return data_list

# 학습/검증을 위한 데이터셋 준비
def prepare_dataset(train_files, val_files=None, val_ratio=0.2, cache_data=False):
    """
    학습/검증 데이터 분할(필요시)과 데이터 전처리를 한 데이터셋 생성

    Args:
        train_files: (학습) 데이터 파일
        val_files: 검증 데이터 파일(None이면 train_files 중에 일부를 활용)
        val_ratio: train_files을 학습/검증으로 나눌 경우 검증 데이터 비율
        cache_data: 학습 속도 향상을 위해 데이터를 캐싱하는지 여부

    Returns:
        trains_ds: 학습 데이터셋
        val_ds: 검증 데이터셋
    """
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
        RandRotated, RandZoomd, RandGaussianNoised,
        ResizeWithPadOrCropd, ToTensord
    )
    # 검증 데이터 분할
    if val_files is None and train_files:
        train_files, val_files = partition_dataset(
            train_files, 
            ratios=[1 - val_ratio, val_ratio], 
            shuffle=True
        )

    # 데이터 변환
    def prepare_transforms(is_train=True):
        if is_train:
            transforms = Compose([
                LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                # EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys=["image"]),
                ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(128, 128, 128)),
                RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.2),
                RandZoomd(keys=["image", "label"], min_zoom=0.7, max_zoom=1.2, prob=0.2),
                RandGaussianNoised(keys=["image"], prob=0.2),
                ToTensord(keys=["image", "label"]),
            ])
        else:
            transforms = Compose([
                LoadImaged(keys=["image", "label"],  ensure_channel_first=True),
                # EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys=["image"]),
                ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(128, 128, 128)),
                ToTensord(keys=["image", "label"]),
            ])
        
        return transforms
    train_transforms = prepare_transforms(is_train=True)
    val_transforms = prepare_transforms(is_train=False)

    # 데이터셋 클래스 선택
    dataset_class = CacheDataset if cache_data else Dataset
    
    # 데이터셋 생성
    train_ds = dataset_class(data=train_files, transform=train_transforms) if train_files else None
    val_ds = dataset_class(data=val_files, transform=val_transforms) if val_files else None
    
    return train_ds, val_ds

# 학습/검증을 위한 데이터 로더 준비
def prepare_datalodaer(train_ds, val_ds):
    """
    주어진 데이터셋으로 학습/검증 중에 로딩하기 위한 데이터 로더 생성

    Args:
        train_ds: 학습 데이터셋
        val_ds: 검증 데이터셋

    Returns:
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
    """
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds, 
        batch_size=2, 
        shuffle=True, 
        num_workers=4
    ) if train_ds else None
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4
    ) if val_ds else None
    
    return train_loader, val_loader

# 학습 데이터셋 미리보기
def prepare_data_preview(data_list, max_samples=3):
    """
    학습 데이터셋 미리보기
    
    Args:
        data_list: 데이터 파일 경로 정보가 담긴 딕셔너리 리스트
        max_samples: 표시할 최대 샘플 수
        
    Returns:
        dict: 미리보기 정보
    """
    import matplotlib.pyplot as plt
    from monai.transforms import LoadImage
    import io
    from PIL import Image
    
    preview_info = {
        "total_samples": len(data_list),
        "samples": []
    }
    
    # 로더 준비
    """
    monai.transforms.LoadImage: MONAI에서 제공하는 이미지 로드 변환(transform)

    사용법:
        loader = LoadImage()
        img = loader("path/to/img.nii.gz")

        loader = LoadImage(image_only=True) # NumPy 배열로 로드
        img_np = loader("path/to/img.nii.gz")

        loader = LoadImage(reade"PydicomReader") # DICOM 이미지 로드
        img = loader("path/to/img.dcm")
    """
    loader = LoadImage()
    
    # 최대 샘플 수 제한
    samples_to_preview = min(max_samples, len(data_list))
    
    for i in range(samples_to_preview):
        sample = data_list[i]
        
        # 영상 로드
        try:
            img_data = loader(sample["image"])
            seg_data = loader(sample["label"])

            # 채널 우선 포맷으로 변경
            if img_data.ndim == 4:  # (H, W, D, C) -> (C, H, W, D)
                img_data = np.transpose(img_data, (3, 0, 1, 2))
            if seg_data.ndim == 3:  # (H, W, D) -> (1, H, W, D)
                seg_data = np.expand_dims(seg_data, axis=0)

            # 중앙 슬라이스 추출
            slice_idx = img_data.shape[-1] // 2
            img_slice = img_data[0, :, :, slice_idx]
            seg_slice = seg_data[0, :, :, slice_idx]
            
            # 미리보기 이미지 생성
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            base_dir = os.getcwd()
            img_path = os.path.relpath(sample['image'], base_dir)
            seg_path = os.path.relpath(sample['label'], base_dir)
            
            # 원본 이미지
            axes[0].imshow(img_slice, cmap='gray')
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # 세그멘테이션 마스크
            axes[1].imshow(seg_slice, cmap='viridis')
            axes[1].set_title("Segementation Mask")
            axes[1].axis('off')

            fig.text(0.25, 0.05, img_path, ha='center', va='center', 
                     fontsize=10, color='white')
            fig.text(0.75, 0.05, seg_path, ha='center', va='center', 
                     fontsize=10, color='white')
            
            # 이미지를 바이트로 변환
            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            
            # 샘플 정보 추가
            preview_info["samples"].append({
                "index": i,
                "image_path": sample["image"],
                "label_path": sample["label"],
                "preview_img": buf
            })
            
        except Exception as e:
            logger.error(str(e))
            preview_info["samples"].append({
                "index": i,
                "image_path": sample["image"],
                "label_path": sample["label"],
                "error": str(e)
            })
    
    return preview_info

