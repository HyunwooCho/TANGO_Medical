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
import asyncio

# Import the training module
from trainer import MedicalImageTrainer
from data import (
    generate_sample_data, 
    download_monai_dataset, 
    prepare_data_preview,
)

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="🩺 의료 영상 AI 학습 시스템", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- setting & utilities
# 디렉토리에서 NIfTI 파일 찾기
def find_nifti_files(directory):
    nifti_files = []
    for ext in ['*.nii', '*.nii.gz']:
        nifti_files.extend(
            glob.glob(os.path.join(directory, '**', ext), recursive=True)
        )
    return nifti_files

# NIfTI 파일 쌍 찾기(이미지 & 세그멘테이션)
def find_paired_files(directory):
    all_files = find_nifti_files(directory)

    # 파일명 패턴 분석하여 쌍 찾기
    pairs = []
    image_pattern_keywords = ['image', 'img', 'volume', 'scan', 't1' 't2', 'flair']
    label_pattern_keywords = ['label', 'seg', 'mask', 'annotation', 'ground']

    # 방법 1. 기본 패턴 (파일 이름 기반으로 파일 쌍 찾기)
    remaining_files = all_files.copy()
    for img_file in all_files:
        img_name = os.path.basename(img_file).lower()
        img_dir = os.path.dirname(img_file)

        # 이미지 파일 확인
        if any(kw in img_name for kw in image_pattern_keywords) and img_file in remaining_files:
            # 가능한 레이블 파일 찾기
            for label_file in all_files:
                if label_file == img_file:
                    continue

                label_name = os.path.basename(label_file).lower()
                label_dir = os.path.dirname(label_file)

                if img_dir == label_dir and any(kw in label_name for kw in label_pattern_keywords):
                    # 파일 이름에서 공통 부분 찾기
                    common_parts = set(img_name.split('_')) & set(label_name.split('_'))
                    if common_parts:
                        pairs.append({"image": img_file, "label": label_file})
                        if img_file in remaining_files:
                            remaining_files.remove(img_file)
                        if label_file in remaining_files:
                            remaining_files.remove(label_file)
                        break

    # 방법 2. 순서쌍 (같은 이름 다른 디렉토리)
    img_dirs = set()
    label_dirs = set()

    # 가능한 이미지/레이블 디렉토리 식별
    for f in all_files:
        dir_name = os.path.basename(os.path.dirname(f)).lower()
        if any(kw in dir_name for kw in image_pattern_keywords):
            img_dirs.add(os.path.dirname(f))
        if any(kw in dir_name for kw in label_pattern_keywords):
            label_dirs.add(os.path.dirname(f))

    # 같은 파일명을 가진 이미지-레이블 쌍 찾기
    for img_dir in img_dirs:
        for label_dir in label_dirs:
            img_files = glob.glob(os.path.join(img_dir, '*.nii*'))
            label_files = glob.glob(os.path.join(label_dir, '*.nii*'))
            
            img_names = [os.path.basename(f) for f in img_files]
            label_names = [os.path.basename(f) for f in label_files]
            
            # 공통 파일명 찾기
            common_names = set(img_names) & set(label_names)
            for name in common_names:
                img_file = os.path.join(img_dir, name)
                label_file = os.path.join(label_dir, name)
                if img_file in remaining_files and label_file in remaining_files:
                    pairs.append({"image": img_file, "label": label_file})
                    remaining_files.remove(img_file)
                    remaining_files.remove(label_file)

    return pairs

# Trainer 초기화
def initialize_trainer(model_type, num_classes, learning_rate, data, batch_size=16, val_ratio=0.2):
    
    # 학습기 초기화
    trainer = MedicalImageTrainer(model_type, num_classes, learning_rate)

    # 데이터 준비
    trainer.prepare_data(data, val_ratio=val_ratio, batch_size=batch_size)

    # 모델 생성
    trainer.create_model()

    return trainer

# 비동기적으로 학습 수행(비동기 제너레이션 방식)
# async def run_training(trainer, total_epochs, progress_callback):
#     for epoch in range(1, total_epochs+1):
#         async for batch_metrics in trainer.train(epoch):
#             await progress_callback(batch_metrics)

# 

# --- Fix for Streamlit watcher error with torch ---
# Disable file watching to prevent the torch._classes error
if 'STREAMLIT_RUN_PATH' in os.environ:
    import streamlit.watcher.path_watcher
    streamlit.watcher.path_watcher.watch_file = lambda x: None

# Set environment variable to limit file watching
os.environ["STREAMLIT_GLOBAL_WATCHER_MAX_FILES"] = "5000"

# --- Display MONAI version for debugging ---
# st.sidebar.info(f"MONAI version: {monai.__version__}")

# 세션 상태 초기화
# if 'trainer' not in st.session_state:
#     st.session_state.trainer = None
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0
if 'total_epochs' not in st.session_state:
    st.session_state.total_epochs = 0

# --- UI 구성 ---
st.title("🩺 의료 영상 AI 학습 시스템")

# 탭 생성
tabs = st.tabs(["🔧 모델 학습 및 파인튜닝", "📊 추론 및 분석", "📚 학습 자료"])

with tabs[0]:
    st.header("모델 학습 및 파인튜닝")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("모델 설정")
        model_type = st.selectbox(
            "모델 아키텍처",
            ["SwinUNETR", "UNET", "SegResNet"],
            help="사용할 모델 아키텍처"
        )
        
        num_classes = st.number_input(
            "분할 클래스 수",
            min_value=2,
            max_value=10,
            value=2,
            help="배경 클래스 포함한 전체 클래스 수"
        )

        learning_rate = st.number_input(
            "학습률",
            min_value=0.00001,
            max_value=0.1,
            value=0.0001,
            format="%.5f",
            help="모델 학습률 (낮을수록 안정적이지만 느림)"
        )
        
        num_epochs = st.number_input(
            "에포크 수",
            min_value=1,
            max_value=10000,
            value=20,
            help="학습 반복 횟수"
        )
        
        batch_size = st.number_input(
            "배치 크기",
            min_value=1,
            max_value=1024,
            value=16,
            help="한 번에 처리할 데이터 수 (메모리에 따라 조정)"
        )
        
        # 모델 학습 시작 버튼
        if st.button("모델 다운로드 및 초기화"):
            if 'training_data' not in st.session_state:
                st.error("학습 데이터가 없습니다. 먼저 데이터를 로드해주세요.")
            else:
                if 'trainer' not in st.session_state:
                    if 'num_classes' in st.session_state:
                        num_classes = st.session_state.num_classes
                    with st.spinner("모델 학습을 초기화 중입니다..."):
                        trainer = initialize_trainer(
                            model_type=model_type,
                            num_classes=st.session_state.num_classes if st.session_state.num_classes else num_classes,
                            learning_rate=learning_rate,
                            data=st.session_state.training_data,
                            batch_size=batch_size,
                            val_ratio=st.session_state.val_ratio if st.session_state.val_ratio else None
                        )
                        st.session_state.trainer = trainer
                        st.session_state.training_in_progress = True
                        st.session_state.current_epoch = 0
                        st.session_state.total_epochs = num_epochs
                        # st.rerun()
                        st.success(f"{model_type} 모델을 성공적으로 초기화하였습니다.")

        # 데이터 소스 선택
        st.subheader("데이터셋 설정")
        data_source = st.radio(
            "데이터 소스",
            ["샘플 데이터 생성", "데이터셋 업로드", "디렉토리 선택", "MONAI 데이터셋 다운로드"],
            help="학습에 사용할 데이터 소스 선택"
        )
        
        if data_source == "샘플 데이터 생성":
            num_samples = st.number_input(
                "생성할 샘플 수",
                min_value=5,
                max_value=50,
                value=10,
                help="생성할 샘플 데이터 수 (학습용)"
            )
            
            if st.button("샘플 데이터 생성"):
                with st.spinner("샘플 데이터를 생성 중입니다..."):
                    generated_data = generate_sample_data(num_samples, num_classes)
                    st.session_state.training_data = generated_data
                    st.success(f"{num_samples}개의 샘플 데이터가 생성되었습니다!")
                    
        elif data_source == "데이터셋 업로드":
            uploaded_files = st.file_uploader(
                "학습 데이터 업로드 (이미지와 마스크)",
                type=["nii.gz", "nii", "png", "jpg", "zip"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("데이터 처리"):
                    with st.spinner("업로드된 데이터를 처리 중입니다..."):
                        processed_data = process_uploaded_files(uploaded_files, num_classes)
                        st.session_state.training_data = processed_data
                        st.success(f"{len(uploaded_files)}개의 파일이 처리되었습니다!")
                    
        elif data_source == "디렉토리 선택":
            data_dir = st.text_input("데이터 디렉토리 경로 입력")
            
            if data_dir and st.button("디렉토리 데이터 로드"):
                with st.spinner("디렉토리에서 데이터를 로드 중입니다..."):
                    loaded_data = load_data_from_directory(data_dir, num_classes)
                    st.session_state.training_data = loaded_data
                    st.success(f"디렉토리에서 데이터를 성공적으로 로드했습니다!")
    
        elif data_source == "MONAI 데이터셋 다운로드":
            monai_dataset_options = [
                # "MedNIST", # 분류(classification)용 데이터셋이므로 분할(segmentation)용으로 변환이 필요함
                "Task01_BrainTumour",
                "Task02_Heart",
                "Task03_Liver",
                "Task04_Hippocampus",
                "Task05_Prostate",
                "Task06_Lung",
                "Task07_Pancreas",
                "Task08_HepaticVessel",
                "Task09_Spleen",
                "Task10_Colon"
            ]
            
            selected_dataset = st.selectbox(
                "MONAI 데이터셋 선택",
                monai_dataset_options,
                help="다운로드할 MONAI 데이터셋 선택"
            )
            
            default_dir = os.path.join(os.getcwd(), "datasets")
            dataset_root_dir = st.text_input(
                "데이터셋 저장 경로",
                value=default_dir, #"./datasets",
                help="다운로드한 데이터셋을 저장할 절대 경로 (디렉토리가 없으면 자동 생성됩니다)"
            )
            
            cache_rate = st.slider(
                "캐시 비율",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                help="데이터 캐싱 비율 (0: 캐싱 없음, 1: 모든 데이터 캐싱)"
            )

            val_ratio = st.slider(
                "검증 데이터 비율",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="검증 데이터로 사용할 비율"
            )

            enable_cache = st.checkbox("데이터 캐싱 활성화", value=True, help="학습 속도 향상을 위한 데이터 캐싱")

            if st.button("데이터셋 다운로드 및 준비"):
                with st.spinner(f"{selected_dataset} 데이터셋을 다운로드하고 준비 중입니다..."):
                    try:
                        # 디렉토리 상태 확인 및 출력
                        if not os.path.exists(dataset_root_dir):
                            st.info(f"'{dataset_root_dir}' 디렉토리가 존재하지 않습니다. 자동으로 생성합니다.")
                        
                        # 데이터셋 다운로드 실행
                        data_list, data_count, num_classes = download_monai_dataset(
                            selected_dataset, 
                            dataset_root_dir, 
                            cache_rate
                        )

                        # 데이터 정보 저장
                        st.session_state.training_data = data_list
                        st.session_state.val_ratio = val_ratio
                        st.session_state.num_classes = num_classes

                        # 다운로드 완료 메시지
                        st.success(f"{selected_dataset} 데이터셋을 성공적으로 다운로드했습니다! ({len(data_list)}개 데이터)")
                        
                    except Exception as e:
                        st.error(f"데이터셋 다운로드 중 오류가 발생했습니다: {str(e)}")
                        st.info("문제 해결 팁:")
                        st.info("1. 저장 경로에 쓰기 권한이 있는지 확인하세요.")
                        st.info("2. 인터넷 연결 상태를 확인하세요.")
                        st.info("3. 일부 데이터셋은 크기가 큰 경우 다운로드에 시간이 오래 걸릴 수 있습니다.")
                        
                        # 더 자세한 오류 정보 표시 (디버깅용)
                        with st.expander("상세 오류 정보"):
                            import traceback
                            st.code(traceback.format_exc())

                    
    # 우측 컬럼 - 학습 진행 상황 및 데이터 시각화
    with col2:
        st.subheader("학습 진행 상황 및 데이터 시각화")
        
        # 학습 진행 중일 때 진행 상태 표시
        if st.session_state.training_in_progress:
            if st.button("모델 학습 시작"):
                # Streamlit UI 요소 준비
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_container = st.container()

                async def update_progress(metrics):
                    progress = metrics['epoch'] / st.session_state.total_epochs
                    progress_bar.progress(progress)
                    status_text.text(f"에포크 {st.session_state.current_epoch}/{st.session_state.total_epochs} 완료")
            
                    # 학습 지표 표시
                    with metrics_container:
                        col_loss, col_dice, col_lr = st.columns(3)
                        with col_loss:
                            st.metric("손실 (Loss)", f"{metrics['loss']:.4f}")
                        with col_dice:
                            st.metric("Dice 점수", f"{metrics['dice']:.4f}")
                        with col_lr:
                            st.metric("학습률", f"{metrics['learning_rate']:.6f}")
                
                # 비동기 학습 실행
                asyncio.run(st.session_state.trainer.train( 
                    num_epochs=st.session_state.total_epochs,
                    progress_callback=update_progress
                ))

            # 학습 진행 상태 및 지표 업데이트
            # if st.session_state.current_epoch < st.session_state.total_epochs:
            #     try:
            #         # 한 에포크 학습 진행
            #         metrics = st.session_state.trainer.train_epoch()
            #         st.session_state.current_epoch += 1
                    
            #         # 진행 상태 업데이트
            #         progress.progress(st.session_state.current_epoch / st.session_state.total_epochs)
            #         status_text.text(f"에포크 {st.session_state.current_epoch}/{st.session_state.total_epochs} 완료")
                    
            #         # 학습 지표 표시
            #         with metrics_container:
            #             col_loss, col_dice, col_lr = st.columns(3)
            #             with col_loss:
            #                 st.metric("손실 (Loss)", f"{metrics['loss']:.4f}")
            #             with col_dice:
            #                 st.metric("Dice 점수", f"{metrics['dice']:.4f}")
            #             with col_lr:
            #                 st.metric("학습률", f"{metrics['learning_rate']:.6f}")
                        
            #             # 샘플 이미지 시각화 (있는 경우)
            #             if 'sample_images' in metrics:
            #                 st.image(metrics['sample_images'], caption=["입력", "예측", "정답"], width=150)
                    
            #         # 자동으로 다음 에포크 진행을 위한 재실행
            #         if st.session_state.current_epoch < st.session_state.total_epochs:
            #             st.experimental_rerun()
            #         else:
            #             st.session_state.training_in_progress = False
            #             st.success("모델 학습이 완료되었습니다!")
                        
            #             # 모델 저장 버튼 표시
            #             if st.button("모델 저장"):
            #                 model_path = save_model(st.session_state.trainer)
            #                 st.success(f"모델이 저장되었습니다: {model_path}")
                
            #     except Exception as e:
            #         st.error(f"학습 중 오류가 발생했습니다: {str(e)}")
            #         st.session_state.training_in_progress = False
        
        else:
            # 학습 시작 전 데이터 미리보기
            if 'training_data' in st.session_state:
                st.subheader("데이터 미리보기")
                preview_data = prepare_data_preview(
                    st.session_state.training_data,
                    max_samples=2
                )
                
                # 이미지 미리보기 표시
                # st.image(preview_data['images'][:3], 
                #          caption=preview_data['captions'][:3], 
                #          width=200)

                # 데이터 통계 표시
                # st.write(f"총 데이터 수: {preview_data['total_samples']}")
                # st.write(f"클래스 분포: {preview_data['class_distribution']}")
                
                for sample in preview_data["samples"]:
                    st.subheader(f"샘플 #{sample['index'] + 1}")
                    if "error" in sample:
                        st.error(f"이미지 로드 오류: {sample['error']}")

                        # 더 자세한 오류 정보 표시 (디버깅용)
                        with st.expander("상세 오류 정보"):
                            import traceback
                            st.code(traceback.format_exc())
                    else:
                        st.image(sample["preview_img"], caption="중앙 슬라이스 미리보기")

            else:
                st.info("왼쪽에서 데이터를 로드한 후 학습을 시작하세요.")

with tabs[1]:
    st.header("추론 및 분석")
    
    inference_col1, inference_col2 = st.columns([1, 2])
    
    with inference_col1:
        st.subheader("추론 설정")
        
        # 모델 선택 (학습된 모델 또는 사전학습 모델)
        model_source = st.radio(
            "모델 소스",
            ["방금 학습한 모델", "저장된 모델 불러오기", "사전학습 모델"]
        )
        
        if model_source == "저장된 모델 불러오기":
            model_path = st.text_input("모델 파일 경로")
            
            if model_path and st.button("모델 불러오기"):
                with st.spinner("모델을 불러오는 중..."):
                    loaded_model = load_model(model_path)
                    st.session_state.inference_model = loaded_model
                    st.success("모델을 성공적으로 불러왔습니다!")
        
        elif model_source == "사전학습 모델":
            pretrained_model = st.selectbox(
                "사전학습 모델 선택",
                ["swin_unetr.segresnet_btcv", "unet_btcv", "segresnet_btcv"]
            )
            
            if st.button("사전학습 모델 불러오기"):
                with st.spinner("사전학습 모델을 불러오는 중..."):
                    pretrained = load_pretrained_model(pretrained_model)
                    st.session_state.inference_model = pretrained
                    st.success("사전학습 모델을 성공적으로 불러왔습니다!")
        
        # 추론할 이미지 업로드
        uploaded_image = st.file_uploader(
            "분석할 의료 영상 업로드",
            type=["nii.gz", "nii", "dcm", "png", "jpg"]
        )
        
        # 후처리 옵션
        st.subheader("후처리 옵션")
        post_process = st.checkbox("분할 결과 후처리", value=True)
        
        if post_process:
            smoothing = st.slider(
                "경계 스무딩 정도",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )
            
            min_region_size = st.number_input(
                "최소 영역 크기 (복셀)",
                min_value=10,
                max_value=1000,
                value=100
            )
        
        # 추론 실행 버튼
        if st.button("영상 분석 실행"):
            if uploaded_image is None:
                st.error("분석할 영상을 업로드해주세요.")
            elif 'inference_model' not in st.session_state and model_source != "방금 학습한 모델":
                st.error("먼저 모델을 불러와주세요.")
            else:
                with st.spinner("영상을 분석하는 중..."):
                    if model_source == "방금 학습한 모델":
                        if 'trainer' not in st.session_state:
                            st.error("학습된 모델이 없습니다. 먼저 모델을 학습해주세요.")
                        else:
                            model = st.session_state.trainer.get_model()
                    else:
                        model = st.session_state.inference_model
                    
                    # 영상 처리 및 추론
                    result = run_inference(
                        model=model,
                        image=uploaded_image,
                        post_process=post_process,
                        smoothing=smoothing if post_process else 0.0,
                        min_region_size=min_region_size if post_process else 0
                    )
                    
                    st.session_state.inference_result = result
                    st.success("영상 분석이 완료되었습니다!")
    
    with inference_col2:
        st.subheader("분석 결과")
        
        if 'inference_result' in st.session_state:
            result = st.session_state.inference_result
            
            # 결과 시각화 선택
            view_mode = st.radio(
                "결과 보기 모드",
                ["2D 슬라이스", "3D 렌더링", "오버레이"]
            )
            
            if view_mode == "2D 슬라이스":
                # 슬라이스 선택기
                if result['is_3d']:
                    axis = st.selectbox("표시 축", ["Axial", "Coronal", "Sagittal"])
                    slice_idx = st.slider(
                        "슬라이스 선택",
                        min_value=0,
                        max_value=result['dims'][{"Axial": 0, "Coronal": 1, "Sagittal": 2}[axis]],
                        value=result['dims'][{"Axial": 0, "Coronal": 1, "Sagittal": 2}[axis]]//2
                    )
                    
                    # 선택된 슬라이스 표시
                    slice_viewer = display_slice(
                        result['original'],
                        result['segmentation'],
                        axis={"Axial": 0, "Coronal": 1, "Sagittal": 2}[axis],
                        slice_idx=slice_idx
                    )
                    st.image(slice_viewer, use_column_width=True)
                else:
                    # 2D 이미지 결과 표시
                    st.image([result['original'], result['segmentation']], 
                             caption=["원본 이미지", "분할 결과"],
                             width=300)
            
            elif view_mode == "3D 렌더링":
                if result['is_3d']:
                    # 3D 렌더링 옵션
                    opacity = st.slider("투명도", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
                    
                    # 3D 렌더링 표시
                    render_3d(result['segmentation'], opacity=opacity)
                else:
                    st.warning("3D 렌더링은 3D 볼륨 데이터에만 사용할 수 있습니다.")
            
            elif view_mode == "오버레이":
                # 오버레이 옵션
                overlay_opacity = st.slider("오버레이 투명도", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
                
                if result['is_3d']:
                    axis = st.selectbox("오버레이 축", ["Axial", "Coronal", "Sagittal"])
                    overlay_slice_idx = st.slider(
                        "오버레이 슬라이스",
                        min_value=0,
                        max_value=result['dims'][{"Axial": 0, "Coronal": 1, "Sagittal": 2}[axis]],
                        value=result['dims'][{"Axial": 0, "Coronal": 1, "Sagittal": 2}[axis]]//2
                    )
                    
                    # 오버레이 표시
                    overlay_viewer = display_overlay(
                        result['original'],
                        result['segmentation'],
                        axis={"Axial": 0, "Coronal": 1, "Sagittal": 2}[axis],
                        slice_idx=overlay_slice_idx,
                        opacity=overlay_opacity
                    )
                    st.image(overlay_viewer, use_column_width=True)
                else:
                    # 2D 이미지 오버레이 표시
                    overlay_image = create_overlay(
                        result['original'], 
                        result['segmentation'],
                        opacity=overlay_opacity
                    )
                    st.image(overlay_image, caption="분할 오버레이", use_column_width=True)
            
            # 분석 지표
            st.subheader("분석 지표")
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                for idx, metric in enumerate(result['metrics'][:len(result['metrics'])//2]):
                    st.metric(metric['name'], metric['value'], delta=metric.get('delta'))
            
            with metrics_col2:
                for idx, metric in enumerate(result['metrics'][len(result['metrics'])//2:]):
                    st.metric(metric['name'], metric['value'], delta=metric.get('delta'))
            
            # 결과 내보내기
            st.subheader("결과 내보내기")
            export_format = st.selectbox(
                "내보내기 형식",
                ["NIfTI (.nii.gz)", "DICOM (.dcm)", "PNG 시리즈"]
            )
            
            if st.button("결과 내보내기"):
                with st.spinner("결과를 내보내는 중..."):
                    export_path = export_results(
                        result,
                        format=export_format
                    )
                    st.success(f"결과가 저장되었습니다: {export_path}")
                    # 다운로드 버튼
                    with open(export_path, "rb") as file:
                        st.download_button(
                            label="결과 다운로드",
                            data=file,
                            file_name=os.path.basename(export_path),
                            mime="application/octet-stream"
                        )
        
        else:
            st.info("영상을 업로드하고 분석을 실행하면 여기에 결과가 표시됩니다.")

with tabs[2]:
    st.header("학습 자료")
    
    material_type = st.radio(
        "자료 유형",
        ["튜토리얼", "이론 설명", "논문 및 참고자료"]
    )
    
    if material_type == "튜토리얼":
        st.subheader("의료 영상 분할 모델 학습 튜토리얼")
        
        tutorial_steps = [
            "1. 데이터 준비",
            "2. 모델 선택 및 설정",
            "3. 학습 과정",
            "4. 모델 평가",
            "5. 추론 및 활용"
        ]
        
        selected_step = st.selectbox("튜토리얼 단계", tutorial_steps)
        
        # 선택된 튜토리얼 단계에 따른 내용 표시
        if selected_step == "1. 데이터 준비":
            st.write("""
            ### 데이터 준비 단계
            
            의료 영상 분할을 위한 데이터 준비 과정에서 고려해야 할 사항:
            
            1. **데이터 포맷**: NIfTI, DICOM 등의 의료 영상 포맷을 적절히 불러오고 처리하는 방법
            2. **데이터 전처리**: 정규화, 리샘플링, 크롭 등의 전처리 기법
            3. **데이터 증강**: 회전, 플립, 명암 조정 등을 통한 데이터 증강 방법
            4. **데이터셋 분할**: 학습, 검증, 테스트 데이터셋의 적절한 분할 방법
            """)
            
            # 예제 코드
            st.code("""
            # 데이터 로드 및 전처리 예제
            import monai
            from monai.transforms import (
                Compose, LoadImaged, AddChanneld, ScaleIntensityd,
                RandRotated, RandZoomd, RandFlipd, ToTensord
            )
            
            # 데이터 변환 파이프라인 정의
            train_transforms = Compose([
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityd(keys=["image"]),
                RandRotated(keys=["image", "label"], prob=0.5, range_x=0.3),
                RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.8, max_zoom=1.2),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                ToTensord(keys=["image", "label"])
            ])
            
            # 데이터 로더 생성
            train_ds = monai.data.CacheDataset(
                data=train_files, transform=train_transforms, cache_rate=1.0
            )
            train_loader = monai.data.DataLoader(
                train_ds, batch_size=2, shuffle=True, num_workers=4
            )
            """, language="python")
            
        elif selected_step == "2. 모델 선택 및 설정":
            st.write("""
            ### 모델 선택 및 설정
            
            의료 영상 분할을 위한 딥러닝 모델 선택과 구성:
            
            1. **모델 아키텍처**: U-Net, SegResNet, SwinUNETR 등 의료 영상 분할에 적합한 아키텍처 선택
            2. **하이퍼파라미터**: 배치 크기, 학습률, 옵티마이저 등의 하이퍼파라미터 설정
            3. **손실 함수**: Dice Loss, Focal Loss 등 의료 영상 분할에 적합한 손실 함수 선택
            4. **평가 지표**: Dice 계수, Hausdorff 거리 등의 성능 평가 지표
            """)
            
            # 예제 코드
            st.code("""
            # 모델 초기화 예제
            from monai.networks.nets import UNet, SegResNet, SwinUNETR
            from monai.losses import DiceLoss
            import torch
            
            # 모델 선택 및 초기화
            model = UNet(
                dimensions=3,
                in_channels=1,
                out_channels=2,  # 배경 + 분할 클래스
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
            
            # 손실 함수 및 옵티마이저 설정
            loss_function = DiceLoss(to_onehot_y=True, softmax=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # 학습률 스케줄러 설정
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', factor=0.5, patience=5
            )
            """, language="python")
            
        elif selected_step == "3. 학습 과정":
            st.write("""
            ### 학습 과정
            
            의료 영상 분할 모델의 효과적인 학습을 위한 팁:
            
            1. **배치 정규화**: 학습 안정성을 위한 배치 정규화 사용
            2. **그래디언트 클리핑**: 그래디언트 폭발 방지를 위한 클리핑
            3. **조기 종료**: 과적합 방지를 위한 조기 종료 전략
            4. **학습률 스케줄링**: 효과적인 학습을 위한 학습률 조정
            5. **검증 모니터링**: 학습 중 검증 성능 모니터링
            """)
            
            # 예제 코드
            st.code("""
            # 학습 루프 예제
            from monai.engines import SupervisedTrainer
            from monai.handlers import (
                StatsHandler, TensorBoardStatsHandler, 
                CheckpointSaver, LrScheduleHandler
            )
            
            # 학습 엔진 설정
            trainer = SupervisedTrainer(
                device=device,
                max_epochs=50,
                train_data_loader=train_loader,
                network=model,
                optimizer=optimizer,
                loss_function=loss_function,
                inferer=SimpleInferer(),
                key_train_metric={"train_dice": MeanDice(include_background=False)},
                train_handlers=[
                    LrScheduleHandler(
                        lr_scheduler=scheduler,
                        print_lr=True
                    ),
                    StatsHandler(
                        tag_name="train_loss",
                        output_transform=lambda x: x["loss"]
                    ),
                    TensorBoardStatsHandler(
                        log_dir="./runs",
                        tag_name="train_loss",
                        output_transform=lambda x: x["loss"]
                    ),
                    CheckpointSaver(
                        save_dir="./checkpoints",
                        save_dict={"model": model, "optimizer": optimizer},
                        save_interval=5
                    )
                ]
            )
            
            # 학습 실행
            trainer.run()
            """, language="python")
            
        elif selected_step == "4. 모델 평가":
            st.write("""
            ### 모델 평가
            
            의료 영상 분할 모델의 성능 평가 방법:
            
            1. **교차 검증**: 모델 안정성 평가를 위한 교차 검증
            2. **평가 지표**: Dice, Jaccard, Hausdorff 거리 등의 평가 지표
            3. **혼동 행렬**: 클래스별 성능 분석을 위한 혼동 행렬
            4. **시각화**: 예측 결과와 실제 마스크의 시각적 비교
            5. **통계 분석**: 평가 결과의 통계적 분석
            """)
            
            # 예제 코드
            st.code("""
            # 모델 평가 예제
            from monai.engines import SupervisedEvaluator
            from monai.handlers import StatsHandler, CheckpointLoader
            from monai.metrics import DiceMetric, HausdorffDistanceMetric
            
            # 평가 지표 설정
            dice_metric = DiceMetric(include_background=False)
            hausdorff_metric = HausdorffDistanceMetric(include_background=False)
            
            # 평가 엔진 설정
            evaluator = SupervisedEvaluator(
                device=device,
                val_data_loader=val_loader,
                network=model,
                inferer=SimpleInferer(),
                key_val_metric={
                    "val_dice": dice_metric,
                    "val_hausdorff": hausdorff_metric
                },
                val_handlers=[
                    StatsHandler(
                        output_transform=lambda x: None
                    ),
                    CheckpointLoader(
                        load_path="./checkpoints/best_model.pt",
                        load_dict={"model": model}
                    )
                ]
            )
            
            # 평가 실행
            evaluator.run()
            
            # 결과 출력
            metrics = evaluator.state.metrics
            print(f"Dice: {metrics['val_dice']:.4f}")
            print(f"Hausdorff: {metrics['val_hausdorff']:.4f}")
            """, language="python")
            
        elif selected_step == "5. 추론 및 활용":
            st.write("""
            ### 추론 및 활용
            
            학습된 의료 영상 분할 모델의 활용 방법:
            
            1. **단일 영상 추론**: 새로운 영상에 대한 분할 예측
            2. **배치 처리**: 다수의 영상을 한번에 처리
            3. **임상 환경 활용**: 실제 진료 환경에서의 적용 방법
            4. **결과 시각화**: 분할 결과의 효과적인 시각화 방법
            """)
            
            st.markdown("#### 추론 코드 예시")
            
            code = '''
            # 저장된 모델 불러오기
            model = UNet(in_channels=1, out_channels=num_classes).to(device)
            model.load_state_dict(torch.load('best_model.pth'))
            model.eval()
            
            # 단일 이미지 추론
            with torch.no_grad():
                image = load_and_preprocess_image('new_image.png')
                image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
                prediction = model(image_tensor)
                segmentation_mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
                
            # 결과 시각화
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title('원본 영상')
            
            plt.subplot(1, 2, 2)
            plt.imshow(segmentation_mask, cmap='viridis')
            plt.title('분할 결과')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('segmentation_result.png')
            plt.show()
            '''
            
            st.code(code, language='python')
            
            st.markdown("#### 임상 활용 방법")
            st.write("""
            - 영상 분할 결과를 DICOM 포맷으로 저장하여 PACS 시스템과 통합
            - 분할된 영역의 체적 계산을 통한 정량적 분석 제공
            - 종단적 연구를 위한 이전 검사와의 비교 기능 구현
            - 다양한 장기/병변 유형에 대한 앙상블 모델 구축
            """)
            
            st.warning("주의: 임상 환경에서 사용하기 전에 반드시 의료기기 인증 및 임상 검증 과정을 거쳐야 합니다.")

    elif material_type == "이론 설명":
        st.subheader("의료 영상 분할 이론")
        
        theory_topics = [
            "의료 영상 분할 개요",
            "전통적 분할 기법",
            "딥러닝 기반 분할 기법",
            "평가 지표와 검증 방법"
        ]
        
        selected_theory = st.selectbox("이론 주제", theory_topics)
        
        if selected_theory == "의료 영상 분할 개요":
            st.write("""
            ### 의료 영상 분할 개요
            
            의료 영상 분할(Medical Image Segmentation)은 의료 영상에서 관심 영역(ROI)을 식별하고 분리하는 과정입니다. 
            이는 진단, 수술 계획, 치료 모니터링 등 다양한 임상 응용 분야에서 중요한 역할을 합니다.
            
            #### 의료 영상 분할의 중요성
            
            - **정확한 진단**: 정밀한 분할을 통해 병변의 크기, 위치, 형태에 대한 정확한 정보 제공
            - **수술 계획**: 중요 구조물과 병변의 3D 시각화를 통한 수술 전 계획 수립
            - **방사선 치료**: 정확한 타겟 영역 정의를 통한 방사선 치료 최적화
            - **정량적 분석**: 장기 부피, 병변 크기 변화 등의 정량적 측정 가능
            - **컴퓨터 보조 진단**: 자동화된 진단 지원 시스템 개발의 기반
            
            #### 의료 영상 분할의 도전 과제
            
            - **영상 품질**: 노이즈, 아티팩트, 저대비 등의 영상 품질 문제
            - **해부학적 변이**: 환자마다 다른 해부학적 구조 및 병리학적 변이
            - **경계 모호성**: 인접 조직과의 불명확한 경계
            - **데이터 희소성**: 레이블링된 의료 데이터의 부족
            - **전문 지식 요구**: 정확한 분할을 위한 의학적 전문 지식 필요
            """)
        
        elif selected_theory == "전통적 분할 기법":
            st.write("""
            ### 전통적 분할 기법
            
            딥러닝 이전에 사용되던 전통적인 의료 영상 분할 기법들은 여전히 중요한 역할을 합니다.
            
            #### 임계값 기반 기법
            
            - **전역 임계값**: 단일 임계값을 사용하여 전체 영상에 적용
            - **지역 임계값**: 영역별로 다른 임계값 적용
            - **Otsu 방법**: 클래스 내 분산을 최소화하는 최적의 임계값 자동 계산
            
            #### 영역 기반 기법
            
            - **영역 성장법(Region Growing)**: 시드 포인트로부터 유사한 특성을 가진 이웃 픽셀을 통합
            - **워터쉐드(Watershed)**: 지형학적 접근법을 사용하여 영역 경계 식별
            - **그래프 컷(Graph Cut)**: 에너지 함수를 최소화하는 최적 분할 계산
            
            #### 경계 기반 기법
            
            - **경계 검출기(Edge Detectors)**: Sobel, Canny 등의 필터를 사용한 경계 검출
            - **액티브 컨투어(Active Contours/Snakes)**: 에너지를 최소화하며 경계에 맞게 변형되는 곡선
            - **레벨 셋(Level Sets)**: 고차원 함수의 영점 집합으로 경계 표현
            
            #### 아틀라스 기반 기법
            
            - **단일 아틀라스**: 표준 템플릿을 대상 영상에 정합(registration)
            - **다중 아틀라스**: 여러 템플릿을 정합한 후 결과 융합
            - **확률적 아틀라스**: 해부학적 구조의 통계적 변이 모델링
            """)
            
            st.image("https://via.placeholder.com/600x300.png?text=Traditional+Segmentation+Methods", 
                    caption="전통적 분할 기법의 예시 (이미지는 실제 구현 시 관련 이미지로 대체)")
        
        elif selected_theory == "딥러닝 기반 분할 기법":
            st.write("""
            ### 딥러닝 기반 분할 기법
            
            최근 딥러닝 기술의 발전으로 의료 영상 분할 성능이 비약적으로 향상되었습니다.
            
            #### CNN 기반 분할 아키텍처
            
            - **U-Net**: 인코더-디코더 구조와 스킵 연결을 활용한 의료 영상 분할의 대표적 아키텍처
            - **V-Net**: 3D 볼륨 데이터를 직접 처리하기 위한 U-Net의 3D 확장 버전
            - **SegNet**: 인코더의 풀링 인덱스를 디코더에 전달하는 구조
            - **DeepLab**: 확장 합성곱(Dilated Convolution)과 공간 피라미드 풀링을 활용
            
            #### 트랜스포머 기반 분할 아키텍처
            
            - **UNETR**: U-Net과 트랜스포머를 결합한 구조
            - **SwinUNETR**: Swin 트랜스포머 블록을 활용한 U-Net 구조
            - **TransUNet**: CNN과 트랜스포머의 장점을 결합한 하이브리드 모델
            
            #### 주요 기술 요소
            
            - **어텐션 메커니즘**: 중요 특징에 가중치를 부여하는 어텐션 모듈
            - **멀티스케일 처리**: 다양한 크기의 객체를 효과적으로 처리
            - **약지도 학습**: 적은 레이블 데이터로도 학습 가능한 기법
            - **도메인 적응**: 서로 다른 스캐너나 프로토콜 간의 격차 해소
            
            #### 손실 함수
            
            - **Dice Loss**: 클래스 불균형에 강건한 성능을 보이는 손실 함수
            - **Focal Loss**: 어려운 샘플에 더 큰 가중치를 부여
            - **Boundary Loss**: 객체 경계에 집중하는 손실 함수
            - **Compound Loss**: 여러 손실 함수의 조합(예: Dice + Cross Entropy)
            """)
            
            st.image("https://via.placeholder.com/600x300.png?text=Deep+Learning+Segmentation+Architectures", 
                    caption="딥러닝 기반 분할 아키텍처 (이미지는 실제 구현 시 관련 이미지로 대체)")
        
        elif selected_theory == "평가 지표와 검증 방법":
            st.write("""
            ### 평가 지표와 검증 방법
            
            의료 영상 분할 모델의 성능을 정확하게 평가하기 위한 다양한 지표와 방법을 이해하는 것이 중요합니다.
            
            #### 분할 성능 평가 지표
            
            - **Dice 유사 계수(DSC)**: 예측과 실제 마스크 간의 중첩 정도를 측정
                - $DSC = \\frac{2|X \\cap Y|}{|X| + |Y|}$
            
            - **Jaccard 인덱스(IoU)**: 합집합 대비 교집합의 비율
                - $IoU = \\frac{|X \\cap Y|}{|X \\cup Y|}$
            
            - **Hausdorff 거리**: 두 점 집합 간의 최대 거리를 측정
                - $HD(X,Y) = \\max\\{\\sup_{x \\in X} \\inf_{y \\in Y} d(x,y), \\sup_{y \\in Y} \\inf_{x \\in X} d(y,x)\\}$
            
            - **민감도와 특이도**: 진양성률과 진음성률을 측정
                - 민감도(Sensitivity) = $\\frac{TP}{TP+FN}$
                - 특이도(Specificity) = $\\frac{TN}{TN+FP}$
            
            - **체적 유사도 지표**: 예측된 체적과 실제 체적의 차이를 측정
                - 상대적 체적 차이(RVD) = $\\frac{|V_{pred} - V_{true}|}{V_{true}}$
            
            #### 검증 방법
            
            - **교차 검증**: 데이터를 여러 폴드로 나누어 모델 성능의 일반화 능력 평가
                - k-fold 교차 검증
                - Leave-one-out 교차 검증
            
            - **홀드아웃 검증**: 데이터를 학습/검증/테스트 세트로 분할
            
            - **과적합 방지 전략**:
                - 데이터 증강
                - 정규화 기법 적용
                - 조기 종료(Early stopping)
            
            #### 임상적 유효성 검증
            
            - **관찰자 간 일치도**: 여러 전문가의 분할 결과와 알고리즘 결과 비교
            - **임상적 영향 평가**: 분할 결과가 임상적 의사결정에 미치는 영향
            - **시간 효율성**: 수동 분할 대비 시간 절약 효과
            """)
            
            formula_explanation = """
            #### 수식 설명
            - DSC와 IoU: X는 예측 마스크, Y는 실제 마스크를 의미
            - Hausdorff 거리: 두 집합 X, Y 간의 최대 거리를 측정
            - TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative
            """
            
            st.markdown(formula_explanation)
            
            # 시각화 예시
            st.code("""
            # 평가 지표 시각화 예제
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            # 혼동 행렬 시각화
            def plot_confusion_matrix(y_true, y_pred, classes):
                cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=classes, yticklabels=classes)
                plt.ylabel('실제 레이블')
                plt.xlabel('예측 레이블')
                plt.title('혼동 행렬')
                plt.show()
                
            # Dice 계수 계산 및 시각화
            def plot_dice_scores(dice_scores, classes):
                plt.figure(figsize=(10, 6))
                bar_colors = plt.cm.viridis(np.linspace(0, 0.8, len(classes)))
                plt.bar(classes, dice_scores, color=bar_colors)
                plt.axhline(y=np.mean(dice_scores), color='r', linestyle='--', 
                        label=f'평균: {np.mean(dice_scores):.3f}')
                plt.ylim([0, 1.0])
                plt.xlabel('클래스')
                plt.ylabel('Dice 계수')
                plt.title('클래스별 Dice 계수')
                plt.legend()
                plt.show()
            """, language="python")

    elif material_type == "논문 및 참고자료":
        st.subheader("의료 영상 분할 관련 논문 및 참고자료")
        
        reference_categories = [
            "기초 논문",
            "최신 연구 동향",
            "오픈 소스 도구",
            "데이터셋 및 챌린지"
        ]
        
        selected_category = st.selectbox("카테고리", reference_categories)
        
        if selected_category == "기초 논문":
            st.write("""
            ### 의료 영상 분할의 기초 논문
            
            #### U-Net 및 딥러닝 기초 아키텍처
            
            1. **U-Net: Convolutional Networks for Biomedical Image Segmentation**
            - 저자: Ronneberger, O., Fischer, P., & Brox, T.
            - 출판 연도: 2015
            - 주요 내용: 스킵 연결을 활용한 인코더-디코더 구조의 CNN 아키텍처 제안
            - 링크: [Paper](https://arxiv.org/abs/1505.04597)

            2. **V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation**
            - 저자: Milletari, F., Navab, N., & Ahmadi, S. A.
            - 출판 연도: 2016
            - 주요 내용: 3D 의료 영상을 위한 U-Net의 확장 버전, Dice 손실 함수 도입
            - 링크: [Paper](https://arxiv.org/abs/1606.04797)

            3. **3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation**
            - 저자: Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O.
            - 출판 연도: 2016
            - 주요 내용: 3D 볼륨 데이터를 위한 U-Net 확장, 희소 어노테이션 활용
            - 링크: [Paper](https://arxiv.org/abs/1606.06650)

            4. **Attention U-Net: Learning Where to Look for the Pancreas**
            - 저자: Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Glocker, B.
            - 출판 연도: 2018
            - 주요 내용: 어텐션 게이트를 U-Net에 통합하여 타겟 구조물에 집중
            - 링크: [Paper](https://arxiv.org/abs/1804.03999)
            """)
            
        elif selected_category == "최신 연구 동향":
            st.write("""
            ### 의료 영상 분할의 최신 연구 동향
            
            #### 트랜스포머 기반 모델
            
            1. **TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation**
            - 저자: Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., ... & Zhou, Y.
            - 출판 연도: 2021
            - 주요 내용: CNN과 트랜스포머를 결합한 하이브리드 아키텍처
            - 링크: [Paper](https://arxiv.org/abs/2102.04306)

            2. **UNETR: Transformers for 3D Medical Image Segmentation**
            - 저자: Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., ... & Xu, D.
            - 출판 연도: 2022
            - 주요 내용: 3D 의료 영상을 위한 트랜스포머 기반 U-Net 구조
            - 링크: [Paper](https://arxiv.org/abs/2103.10504)

            3. **Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images**
            - 저자: Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H., & Xu, D.
            - 출판 연도: 2022
            - 주요 내용: Swin 트랜스포머를 활용한 계층적 특징 추출 아키텍처
            - 링크: [Paper](https://arxiv.org/abs/2201.01266)

            #### 약지도 및 자가지도 학습
            
            1. **nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation**
            - 저자: Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H.
            - 출판 연도: 2021
            - 주요 내용: 데이터셋에 자동으로 적응하는 U-Net 프레임워크
            - 링크: [Paper](https://arxiv.org/abs/1809.10486)

            2. **SwinSupLoss: Supervised Pre-training and Contrastive Learning in one Loss for Medical Image Segmentation**
            - 저자: Azad, R., Aghdam, E. K., Cohen, J. P., Clifton, D., & Dagan, M.
            - 출판 연도: 2023
            - 주요 내용: 감독 학습과 대조 학습을 결합한 새로운 손실 함수
            - 링크: [Paper](https://arxiv.org/abs/2303.17051)

            #### 다중 모달리티 및 도메인 적응
            
            1. **Synergistic Image and Feature Adaptation: Towards Cross-Modality Domain Adaptation for Medical Image Segmentation**
            - 저자: Chen, C., Dou, Q., Chen, H., Qin, J., & Heng, P. A.
            - 출판 연도: 2019
            - 주요 내용: 다양한 영상 모달리티 간의 도메인 적응 기법
            - 링크: [Paper](https://arxiv.org/abs/1901.08211)
            """)
            
        elif selected_category == "오픈 소스 도구":
            st.write("""
            ### 의료 영상 분할을 위한 오픈 소스 도구
            
            #### 라이브러리 및 프레임워크
            
            1. **MONAI (Medical Open Network for AI)**
            - 설명: PyTorch 기반의 의료 영상 딥러닝 프레임워크
            - 특징: 의료 영상 특화 데이터 변환, 사전 학습 모델, 평가 지표 제공
            - 링크: [GitHub](https://github.com/Project-MONAI/MONAI)
            - 문서: [Documentation](https://docs.monai.io/)

            2. **NiftyNet**
            - 설명: TensorFlow 기반의 의료 영상 분석 플랫폼
            - 특징: 의료 영상 세분화, 분류, 생성 모델 지원
            - 링크: [GitHub](https://github.com/NifTK/NiftyNet)
            - 논문: [Gibson et al., 2018](https://www.sciencedirect.com/science/article/pii/S0169260717311823)

            3. **TorchIO**
            - 설명: PyTorch를 위한 의료 영상 처리 및 증강 라이브러리
            - 특징: 3D 의료 영상을 위한 효율적인 데이터 로딩 및 변환
            - 링크: [GitHub](https://github.com/fepegar/torchio)
            - 문서: [Documentation](https://torchio.readthedocs.io/)

            4. **nnU-Net**
            - 설명: 자기 구성 U-Net 기반 의료 영상 분할 프레임워크
            - 특징: 데이터셋에 자동으로 적응하는 아키텍처 및 전처리 파이프라인
            - 링크: [GitHub](https://github.com/MIC-DKFZ/nnUNet)
            - 논문: [Isensee et al., 2021](https://www.nature.com/articles/s41592-020-01008-z)

            #### 시각화 및 평가 도구
            
            1. **3D Slicer**
            - 설명: 의료 영상 시각화 및 분석을 위한 오픈 소스 소프트웨어
            - 특징: Python 및 C++ 확장 모듈, 딥러닝 모델 통합 가능
            - 링크: [Homepage](https://www.slicer.org/)
            - 확장: [AI 확장 모듈](https://github.com/NVIDIA/ai-assisted-annotation-client)

            2. **ITK-SNAP**
            - 설명: 의료 영상 분할을 위한 대화형 소프트웨어
            - 특징: 반자동 및 수동 분할 도구 제공
            - 링크: [Homepage](http://www.itksnap.org/)

            3. **MedicalZoo**
            - 설명: PyTorch 기반의 의료 영상 분할 모델 모음
            - 특징: 다양한 3D 분할 아키텍처 구현 및 비교
            - 링크: [GitHub](https://github.com/black0017/MedicalZoo-PyTorch)
            """)
            
        elif selected_category == "데이터셋 및 챌린지":
            st.write("""
            ### 의료 영상 분할 데이터셋 및 챌린지
            
            #### 공개 데이터셋
            
            1. **Medical Segmentation Decathlon**
            - 설명: 10개 다른 의료 영상 분할 작업을 포함한 대규모 데이터셋
            - 모달리티: CT, MRI
            - 해부학적 구조: 뇌, 심장, 간, 췌장, 전립선 등
            - 링크: [Homepage](http://medicaldecathlon.com/)

            2. **BRATS (Brain Tumor Segmentation Challenge)**
            - 설명: 다중 모달리티 MRI에서 뇌종양 분할을 위한 데이터셋
            - 모달리티: T1, T1ce, T2, FLAIR MRI
            - 해부학적 구조: 뇌, 뇌종양
            - 링크: [Homepage](https://www.med.upenn.edu/cbica/brats2021/)

            3. **PROMISE12 (Prostate MR Image Segmentation)**
            - 설명: 전립선 MRI 분할 데이터셋
            - 모달리티: T2 MRI
            - 해부학적 구조: 전립선
            - 링크: [Homepage](https://promise12.grand-challenge.org/)

            4. **ACDC (Automated Cardiac Diagnosis Challenge)**
            - 설명: 심장 MRI 분할 데이터셋
            - 모달리티: 심장 MRI
            - 해부학적 구조: 좌심실, 우심실, 심근
            - 링크: [Homepage](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

            #### 주요 챌린지
            
            1. **MICCAI Grand Challenges**
            - 설명: 의료 영상 분석 분야의 다양한 경쟁 대회 플랫폼
            - 특징: 매년 새로운 분할 작업 및 데이터셋 제공
            - 링크: [Homepage](https://grand-challenge.org/)

            2. **ISBI Cell Tracking Challenge**
            - 설명: 세포 추적 및 분할을 위한 데이터셋 및 챌린지
            - 모달리티: 현미경 영상
            - 링크: [Homepage](http://celltrackingchallenge.net/)

            3. **COVID-19 Lung CT Lesion Segmentation Challenge**
            - 설명: COVID-19 환자의 폐 CT 영상에서 병변 분할
            - 모달리티: 흉부 CT
            - 링크: [Homepage](https://covid-segmentation.grand-challenge.org/)

            #### 데이터 접근 및 활용 팁
            
            - **IRB 및 윤리적 고려사항**: 의료 데이터 사용 시 관련 규정 및 윤리적 고려사항 숙지
            - **데이터 전처리**: 다양한 스캐너와 프로토콜로 인한 이질성 해결을 위한 표준화 과정 필요
            - **데이터 불균형**: 클래스 불균형 문제 해결을 위한 샘플링 또는 가중치 기법 활용
            - **데이터 증강**: 제한된 데이터에서 모델 일반화 능력 향상을 위한 효과적인 증강 전략 수립
            """)
            
            st.warning("""
            참고: 각 데이터셋 및 챌린지는 사용 조건과 라이센스가 다를 수 있습니다. 
            연구 또는 개발에 활용하기 전 해당 조건을 반드시 확인하시기 바랍니다.
            """)

# Add a footer
st.markdown("---")
st.markdown("© 2025 ETRI 의료 영상 AI 학습 시스템 TANGO-MEDICAL")