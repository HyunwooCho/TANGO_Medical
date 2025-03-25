import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import monai
from monai.data import Dataset, CacheDataset, partition_dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandRotated, RandZoomd, RandGaussianNoised,
    ResizeWithPadOrCropd, ToTensord
)
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import nibabel as nib
from tqdm.notebook import tqdm
import logging
import time
from pathlib import Path
import asyncio
import concurrent.futures

class MedicalImageTrainer:
    def __init__(self, model_type="SwinUNETR", num_classes=2, learning_rate=1e-4, device=None):
        """
        의료 영상 모델 학습 및 파인튜닝을 위한 클래스
        
        Args:
            model_type: 모델 유형 ("SwinUNETR", "UNET", "SegResNet")
            num_classes: 분할할 클래스 수
            learning_rate: 학습률
            device: 학습 디바이스 (None이면 자동 감지)
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.loss_function = None
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.best_metric = -1
        self.best_metric_epoch = -1
        self.epoch_loss_values = []
        self.metric_values = []
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TANGO-MedicalTrainer")
        
    def create_model(self, pretrained_path=None):
        """
        모델 초기화
        
        Args:
            pretrained_path: 사전 학습된 모델 경로 (None이면 새로 생성)
        """
        if self.model_type == "SwinUNETR":
            try:
                from monai.networks.nets import SwinUNETR
                try:
                    # 최신 버전 MONAI (1.5+)
                    self.model = SwinUNETR(
                        in_channels=1,
                        out_channels=self.num_classes,
                        feature_size=48,
                        use_checkpoint=True
                    )
                except TypeError:
                    # 이전 버전 MONAI
                    self.model = SwinUNETR(
                        img_size=(128, 128, 128),
                        in_channels=1,
                        out_channels=self.num_classes,
                        feature_size=48,
                        use_checkpoint=True
                    )
            except Exception as e:
                self.logger.error(f"SwinUNETR 초기화 오류: {str(e)}")
                # 대체 모델
                self.model = nn.Sequential(
                    nn.Conv3d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(16, self.num_classes, kernel_size=3, padding=1)
                )
                self.logger.warning("⚠️ 대체 모델 사용 - SwinUNETR 초기화 실패")
        
        elif self.model_type == "UNET":
            from monai.networks.nets import UNet
            self.model = UNet(
                dimensions=3,
                in_channels=1,
                out_channels=self.num_classes,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
            
        elif self.model_type == "SegResNet":
            from monai.networks.nets import SegResNet
            self.model = SegResNet(
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=16,
                in_channels=1,
                out_channels=self.num_classes,
            )
        
        # 사전 학습 모델 로드
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
                self.logger.info(f"✅ 사전 학습 모델 로드: {pretrained_path}")
            except Exception as e:
                self.logger.error(f"사전 학습 모델 로드 실패: {str(e)}")
        
        self.model = self.model.to(self.device)
        
        # 손실 함수 및 옵티마이저 설정
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        return self.model
    
    def prepare_transforms(self, is_train=True):
        """
        데이터 전처리 변환 설정
        
        Args:
            is_train: 학습용 변환 여부 (검증 시 False)
        """

        """
        변환 과정 설명(모든 함수 끝의 'd'는 딕셔너리를 의미)
            1. LoadImaged(keys=["image", "label"], ensure_channel_first=True)
                image와 label을 파일에서 로드하여 딕셔너리 형태로 변환한다.
                예를 들어, NIfTI (.nii, .nii.gz) 같은 의료 영상 데이터를 불러올 때 사용된다.
                채널 우선 형식 자동 변환 (H, W, D, C) -> (C, H, W, D) / (H, W, D) -> (1, H, W, D)

            2. EnsureChannelFirstd(keys=["label"])
                label의 채널 차원을 가장 앞 축으로 설정한다.
                3D 의료 영상의 경우 (H, W, D) → (1, H, W, D)로 채널 차원을 추가한다.

            3. ScaleIntensityd(keys=["image"])
                image 데이터의 픽셀 값을 정규화한다.
                일반적으로 의료 영상은 다양한 범위의 픽셀 값을 가지므로, 정규화를 통해 학습을 더 안정적으로 만든다.

            4. ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(128, 128, 128))
                image와 label을 (128, 128, 128) 크기로 맞춘다.
                작은 이미지는 padding을 추가하여 키우고, 큰 이미지는 cropping하여 줄인다.
                image와 label이 동일한 크기가 되도록 유지하는 것이 중요하다.

            5. RandRotateD(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.2)
                image와 label을 랜덤으로 회전시킨다.
                range_x=0.3, range_y=0.3, range_z=0.3 → x, y, z축 기준 최대 0.3 라디안(약 17도) 회전.
                prob=0.2 → 20% 확률로 회전 적용.

            6. RandZoomd(keys=["image", "label"], min_zoom=0.7, max_zoom=1.2, prob=0.2)
                image와 label에 랜덤 확대/축소 적용.
                min_zoom=0.7, max_zoom=1.2 → 70%~120% 크기로 랜덤 변환.
                prob=0.2 → 20% 확률로 적용.

            7. RandGaussianNoised(keys=["image"], prob=0.2)
                image에 가우시안 노이즈 추가.
                prob=0.2 → 20% 확률로 노이즈 적용.
                데이터 다양성을 높이고 모델의 일반화 성능을 개선하는 데 도움.

            8. ToTensord(keys=["image", "label"])
                image와 label을 PyTorch 텐서로 변환.
                모델이 입력으로 받을 수 있는 형식으로 변환.        
        """
        if is_train:
            transforms = Compose([
                LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                # EnsureChannelFirstd(keys=["label"]),
                ScaleIntensityd(keys=["image"]),
                ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(128, 128, 128)),
                RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.2),
                RandZoomd(keys=["image", "label"], min_zoom=0.7, max_zoom=1.2, prob=0.2),
                RandGaussianNoised(keys=["image"], prob=0.2),
                ToTensord(keys=["image", "label"]),
            ])
        else:
            transforms = Compose([
                LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                # EnsureChannelFirstd(keys=["label"]),
                ScaleIntensityd(keys=["image"]),
                ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(128, 128, 128)),
                ToTensord(keys=["image", "label"]),
            ])
        
        return transforms
    
    def prepare_data(self, train_files, val_files=None, val_ratio=0.2, batch_size=16, cache_data=False):
        """
        nii.gz 파일로부터 변환하여 학습/검증 데이터셋 생성하고 그를 위한 데이터 로더 준비
        
        Args:
            train_files: 학습 데이터 파일 리스트
            val_files: 검증 데이터 파일 리스트 (None이면 train_files에서 분할)
            val_ratio: 검증 데이터 비율 (val_files가 None일 때 사용)
            cache_data: 데이터 캐싱 여부
        """
        # 검증 데이터 분할
        if val_files is None and train_files:
            train_files, val_files = partition_dataset(
                train_files, 
                ratios=[1 - val_ratio, val_ratio], 
                shuffle=True
            )
        
        self.logger.info(f"학습 데이터: {len(train_files)}, 검증 데이터: {len(val_files) if val_files else 0}")

        # 데이터 변환
        train_transforms = self.prepare_transforms(is_train=True)
        val_transforms = self.prepare_transforms(is_train=False)

        # 데이터셋 클래스 선택
        dataset_class = CacheDataset if cache_data else Dataset
        
        # 데이터셋 생성
        train_ds = dataset_class(data=train_files, transform=train_transforms) if train_files else None
        val_ds = dataset_class(data=val_files, transform=val_transforms) if val_files else None
        
        # 데이터셋 프리뷰
        # self.prepare_data_preview(train_ds)

        # 데이터 로더 생성
        train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        ) if train_ds else None
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        ) if val_ds else None
        
        self.train_loader = train_loader
        self.val_loader = val_loader

        # return train_loader, val_loader

    async def train(self, num_epochs=50, save_dir="./models", progress_callback=None):
        """
        비동기 모델 학습 및 검증 with UI 콜백
        
        Args:
            num_epochs: 학습 에포크 수
            save_dir: 모델 저장 경로
            progress_callback: UI 업데이트를 위한 콜백 함수
        """
        if self.model is None:
            self.create_model()
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 학습 시작
        total_start = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.logger.info("-" * 10)
            self.logger.info(f"에포크 {epoch + 1}/{num_epochs}")
            self.model.train()
            epoch_loss = 0
            step = 0
            
            for batch_data in self.train_loader:
                # 비동기 작업에 yield 포인트 추가
                await asyncio.sleep(0)
                
                step += 1
                inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                if step % 10 == 0:
                    self.logger.info(f"{step}/{len(self.train_loader)}, "
                                    f"학습 손실: {loss.item():.4f}")
            
            epoch_loss /= step
            self.epoch_loss_values.append(epoch_loss)
            self.logger.info(f"에포크 평균 손실: {epoch_loss:.4f}")
            
            # 검증
            metric = None
            if self.val_loader:
                self.model.eval()
                with torch.no_grad():
                    metric_sum = 0.0
                    metric_count = 0
                    step = 0
                    
                    for val_data in self.val_loader:
                        # 비동기 작업에 yield 포인트 추가
                        await asyncio.sleep(0)
                        
                        step += 1
                        val_inputs, val_labels = val_data["image"].to(self.device), val_data["label"].to(self.device)
                        val_outputs = self.model(val_inputs)
                        
                        # 다이스 점수 계산
                        val_outputs = torch.softmax(val_outputs, dim=1)
                        value = self.dice_metric(y_pred=val_outputs, y=val_labels).item()
                        metric_count += 1
                        metric_sum += value
                    
                    metric = metric_sum / metric_count
                    self.metric_values.append(metric)
                    
                    if metric > self.best_metric:
                        self.best_metric = metric
                        self.best_metric_epoch = epoch + 1
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(save_dir, f"best_model_{self.model_type}.pth")
                        )
                        self.logger.info("모델 저장!")
            
            # 학습률 가져오기 (옵티마이저에 따라 다를 수 있음)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # UI 콜백 호출
            if progress_callback:
                await progress_callback({
                    'epoch': epoch + 1,
                    'loss': epoch_loss,
                    'dice': metric if metric is not None else 0,
                    'learning_rate': current_lr
                })
            
            # 중간 모델 저장 (5 에포크마다)
            if (epoch + 1) % 5 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(save_dir, f"{self.model_type}_epoch{epoch+1}.pth")
                )
            
            epoch_end = time.time()
            self.logger.info(f"에포크 시간: {epoch_end - epoch_start:.4f} 초")
        
        total_end = time.time()
        self.logger.info(f"학습 완료! 총 시간: {total_end - total_start:.4f} 초")
        self.logger.info(f"최고 성능: {self.best_metric:.4f} (에포크: {self.best_metric_epoch})")
        
        # 최종 모델 저장
        torch.save(
            self.model.state_dict(),
            os.path.join(save_dir, f"final_model_{self.model_type}.pth")
        )
        
        return self.model

    # async def train(self, num_epochs=50, save_dir="./models"):
    #     """
    #     비동기 모델 학습 및 검증
        
    #     Args:
    #         num_epochs: 학습 에포크 수
    #         save_dir: 모델 저장 경로
    #     """
    #     if self.model is None:
    #         self.create_model()
        
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # 학습 시작
    #     total_start = time.time()
        
    #     for epoch in range(num_epochs):
    #         epoch_start = time.time()
    #         self.logger.info("-" * 10)
    #         self.logger.info(f"에포크 {epoch + 1}/{num_epochs}")
            
    #         # 학습 단계
    #         train_metrics = await self._train_epoch(epoch)
            
    #         # 검증 단계
    #         if self.val_loader:
    #             val_metrics = await self._validate_epoch(epoch)
            
    #         # 모델 저장
    #         await self._save_model(epoch, save_dir, train_metrics, val_metrics)
            
    #         epoch_end = time.time()
    #         self.logger.info(f"에포크 시간: {epoch_end - epoch_start:.4f} 초")
        
    #     total_end = time.time()
    #     self.logger.info(f"학습 완료! 총 시간: {total_end - total_start:.4f} 초")
        
    #     # 최종 모델 저장
    #     await self._save_final_model(save_dir)
        
    #     return self.model

    # async def _train_epoch(self, epoch):
    #     """
    #     단일 에포크 학습 비동기 메서드
    #     """
    #     self.model.train()
    #     epoch_loss = 0
    #     step = 0
        
    #     # 비동기 처리를 위한 대기 지점 추가
    #     for batch_data in self.train_loader:
    #         await asyncio.sleep(0)  # 이벤트 루프에 제어권 양도
            
    #         step += 1
    #         inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
            
    #         self.optimizer.zero_grad()
    #         outputs = self.model(inputs)
    #         loss = self.loss_function(outputs, labels)
    #         loss.backward()
    #         self.optimizer.step()
            
    #         epoch_loss += loss.item()
    #         if step % 10 == 0:
    #             self.logger.info(f"{step}/{len(self.train_loader)}, "
    #                             f"학습 손실: {loss.item():.4f}")
        
    #     epoch_loss /= step
    #     self.epoch_loss_values.append(epoch_loss)
    #     self.logger.info(f"에포크 평균 손실: {epoch_loss:.4f}")
        
    #     return {
    #         'epoch_loss': epoch_loss,
    #         'step': step
    #     }

    # async def _validate_epoch(self, epoch):
    #     """
    #     검증 에포크 비동기 메서드
    #     """
    #     self.model.eval()
    #     with torch.no_grad():
    #         metric_sum = 0.0
    #         metric_count = 0
    #         step = 0
            
    #         for val_data in self.val_loader:
    #             await asyncio.sleep(0)  # 이벤트 루프에 제어권 양도
                
    #             step += 1
    #             val_inputs, val_labels = val_data["image"].to(self.device), val_data["label"].to(self.device)
    #             val_outputs = self.model(val_inputs)
                
    #             # 다이스 점수 계산
    #             val_outputs = torch.softmax(val_outputs, dim=1)
    #             value = self.dice_metric(y_pred=val_outputs, y=val_labels).item()
    #             metric_count += 1
    #             metric_sum += value
            
    #         metric = metric_sum / metric_count
    #         self.metric_values.append(metric)
            
    #         # 최고 성능 모델 추적
    #         if metric > self.best_metric:
    #             self.best_metric = metric
    #             self.best_metric_epoch = epoch + 1
            
    #         self.logger.info(
    #             f"현재 에포크: {epoch + 1} 검증 평균 다이스: {metric:.4f}"
    #             f"\n최고 성능: {self.best_metric:.4f} (에포크: {self.best_metric_epoch})"
    #         )
            
    #         return {
    #             'metric': metric,
    #             'best_metric': self.best_metric,
    #             'best_metric_epoch': self.best_metric_epoch
    #         }

    # async def _save_model(self, epoch, save_dir, train_metrics, val_metrics=None):
    #     """
    #     중간 모델 저장 비동기 메서드
    #     """
    #     # 5 에포크마다 중간 모델 저장
    #     if (epoch + 1) % 5 == 0:
    #         model_path = os.path.join(save_dir, f"{self.model_type}_epoch{epoch+1}.pth")
    #         torch.save(self.model.state_dict(), model_path)
        
    #     # 최고 성능 모델 저장 (검증 시)
    #     if val_metrics and val_metrics['metric'] > self.best_metric:
    #         best_model_path = os.path.join(save_dir, f"best_model_{self.model_type}.pth")
    #         torch.save(self.model.state_dict(), best_model_path)
    #         self.logger.info("최고 성능 모델 저장!")

    # async def _save_final_model(self, save_dir):
    #     """
    #     최종 모델 저장 비동기 메서드
    #     """
    #     final_model_path = os.path.join(save_dir, f"final_model_{self.model_type}.pth")
    #     torch.save(self.model.state_dict(), final_model_path)
        
    #     self.logger.info(f"최종 모델 저장: {final_model_path}")
    #     self.logger.info(f"최고 성능: {self.best_metric:.4f} (에포크: {self.best_metric_epoch})")

    # def train(self, num_epochs=50, save_dir="./models"):
    #     """
    #     모델 학습 및 검증
        
    #     Args:
    #         train_loader: 학습 데이터 로더
    #         val_loader: 검증 데이터 로더
    #         num_epochs: 학습 에포크 수
    #         save_dir: 모델 저장 경로
    #     """
    #     if self.model is None:
    #         self.create_model()
        
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # 학습 시작
    #     total_start = time.time()
        
    #     for epoch in range(num_epochs):
    #         epoch_start = time.time()
    #         self.logger.info("-" * 10)
    #         self.logger.info(f"에포크 {epoch + 1}/{num_epochs}")
    #         self.model.train()
    #         epoch_loss = 0
    #         step = 0
            
    #         for batch_data in self.train_loader:
    #             step += 1
    #             inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
                
    #             self.optimizer.zero_grad()
    #             outputs = self.model(inputs)
    #             loss = self.loss_function(outputs, labels)
    #             loss.backward()
    #             self.optimizer.step()
                
    #             epoch_loss += loss.item()
    #             if step % 10 == 0:
    #                 self.logger.info(f"{step}/{len(self.train_loader)}, "
    #                                 f"학습 손실: {loss.item():.4f}")
            
    #         epoch_loss /= step
    #         self.epoch_loss_values.append(epoch_loss)
    #         self.logger.info(f"에포크 평균 손실: {epoch_loss:.4f}")
            
    #         # 검증
    #         if self.val_loader:
    #             self.model.eval()
    #             with torch.no_grad():
    #                 metric_sum = 0.0
    #                 metric_count = 0
    #                 step = 0
                    
    #                 for val_data in self.val_loader:
    #                     step += 1
    #                     val_inputs, val_labels = val_data["image"].to(self.device), val_data["label"].to(self.device)
    #                     val_outputs = self.model(val_inputs)
                        
    #                     # 다이스 점수 계산
    #                     val_outputs = torch.softmax(val_outputs, dim=1)
    #                     value = self.dice_metric(y_pred=val_outputs, y=val_labels).item()
    #                     metric_count += 1
    #                     metric_sum += value
                        
    #                 metric = metric_sum / metric_count
    #                 self.metric_values.append(metric)
                    
    #                 if metric > self.best_metric:
    #                     self.best_metric = metric
    #                     self.best_metric_epoch = epoch + 1
    #                     torch.save(
    #                         self.model.state_dict(),
    #                         os.path.join(save_dir, f"best_model_{self.model_type}.pth")
    #                     )
    #                     self.logger.info("모델 저장!")
                    
    #                 self.logger.info(
    #                     f"현재 에포크: {epoch + 1} 검증 평균 다이스: {metric:.4f}"
    #                     f"\n최고 성능: {self.best_metric:.4f} (에포크: {self.best_metric_epoch})"
    #                 )
            
    #         # 중간 모델 저장 (5 에포크마다)
    #         if (epoch + 1) % 5 == 0:
    #             torch.save(
    #                 self.model.state_dict(),
    #                 os.path.join(save_dir, f"{self.model_type}_epoch{epoch+1}.pth")
    #             )
                
    #         epoch_end = time.time()
    #         self.logger.info(f"에포크 시간: {epoch_end - epoch_start:.4f} 초")
        
    #     total_end = time.time()
    #     self.logger.info(f"학습 완료! 총 시간: {total_end - total_start:.4f} 초")
    #     self.logger.info(f"최고 성능: {self.best_metric:.4f} (에포크: {self.best_metric_epoch})")
        
    #     # 최종 모델 저장
    #     torch.save(
    #         self.model.state_dict(),
    #         os.path.join(save_dir, f"final_model_{self.model_type}.pth")
    #     )
        
    #     return self.model
    
    def fine_tune(self, train_loader, val_loader, pretrained_path, num_epochs=10, save_dir="./models", unfreeze_layers="all"):
        """
        사전 학습된 모델 파인튜닝
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            pretrained_path: 사전 학습된 모델 경로
            num_epochs: 학습 에포크 수
            save_dir: 모델 저장 경로
            unfreeze_layers: 학습할 레이어 유형 ("all", "decoder", "last")
        """
        # 모델 로드
        self.create_model(pretrained_path=pretrained_path)
        
        # 레이어 고정 (파인튜닝을 위한 설정)
        if unfreeze_layers != "all":
            # 모든 파라미터 고정
            for param in self.model.parameters():
                param.requires_grad = False
                
            if unfreeze_layers == "decoder" and self.model_type == "SwinUNETR":
                # 디코더 부분만 학습 가능하게 설정
                for name, param in self.model.named_parameters():
                    if "decoder" in name:
                        param.requires_grad = True
            elif unfreeze_layers == "last":
                # 마지막 레이어만 학습 가능하게 설정
                if self.model_type == "SwinUNETR":
                    try:
                        for param in self.model.decoder5.parameters():
                            param.requires_grad = True
                    except AttributeError:
                        self.logger.warning("마지막 레이어를 찾을 수 없어 모든 파라미터를 학습합니다.")
                        for param in self.model.parameters():
                            param.requires_grad = True
        
        # 훈련 가능한 파라미터만 최적화
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                    lr=self.learning_rate * 0.1)  # 학습률 감소
        
        self.logger.info(f"파인튜닝 모드: {unfreeze_layers}")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"학습 가능한 파라미터: {trainable_params:,} / 전체 파라미터: {total_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # 모델 학습 진행
        return self.train(train_loader, val_loader, num_epochs, save_dir)
    
    def plot_train_history(self):
        """학습 히스토리 시각화"""
        plt.figure("학습 결과", figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("학습 손실")
        x = range(1, len(self.epoch_loss_values) + 1)
        plt.plot(x, self.epoch_loss_values, label="학습 손실")
        plt.xlabel("에포크")
        plt.ylabel("손실")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.title("검증 성능 (Dice)")
        x = range(1, len(self.metric_values) + 1)
        plt.plot(x, self.metric_values, label="평균 다이스 점수")
        plt.xlabel("에포크")
        plt.ylabel("다이스 점수")
        plt.legend()
        
        plt.tight_layout()
        return plt.gcf()
    
    def prepare_sample_data(self, num_samples=5, output_dir="./sample_data"):
        """
        샘플 데이터셋 생성 (데모 목적)
        
        Args:
            num_samples: 생성할 샘플 수
            output_dir: 출력 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 샘플 데이터셋 파일 정보 생성
        data_dicts = []
        
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
            data_dicts.append({"image": img_path, "label": seg_path})
        
        self.logger.info(f"{num_samples}개의 샘플 데이터를 {output_dir}에 생성했습니다.")
        return data_dicts

def generate_sample_data(num_samples=5, num_classes=2, output_dir="./sample_data"):
    """
    샘플 의료 영상 데이터셋 생성
    
    Args:
        num_samples: 생성할 샘플 수
        num_classes: 분할 클래스 수
        output_dir: 출력 디렉토리
        
    Returns:
        list: 데이터 파일 경로 정보가 담긴 딕셔너리 리스트
    """
    # 임시 트레이너 객체 생성
    temp_trainer = MedicalImageTrainer(num_classes=num_classes)
    
    # 샘플 데이터 생성
    data_dicts = temp_trainer.prepare_sample_data(num_samples=num_samples, output_dir=output_dir)
    
    return data_dicts

def run_training_demo():
    """학습 및 파인튜닝 샘플 실행"""
    # 학습기 초기화
    trainer = MedicalImageTrainer(model_type="SwinUNETR", num_classes=2)
    
    # 샘플 데이터 생성
    data_dicts = trainer.prepare_sample_data(num_samples=10)
    
    # 데이터 준비
    train_files, val_files = partition_dataset(data_dicts, ratios=[0.7, 0.3], shuffle=True)
    train_loader, val_loader = trainer.prepare_data(train_files, val_files)
    
    # 모델 생성
    trainer.create_model()
    
    # 모델 학습 (미니 에포크)
    trainer.train(num_epochs=5, save_dir="./models")
    
    # 학습 결과 시각화
    train_plot = trainer.plot_train_history()
    
    # 파인튜닝 (원래는 사전학습 모델을 로드하지만, 여기서는 방금 학습한 모델 사용)
    trainer.fine_tune(
        train_loader, 
        val_loader,
        pretrained_path="./models/best_model_SwinUNETR.pth",
        num_epochs=3,
        unfreeze_layers="decoder"
    )
    
    return trainer, train_plot