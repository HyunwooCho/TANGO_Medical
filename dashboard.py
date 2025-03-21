import streamlit as st
from PIL import Image

st.title("🩺 의료 영상 AI 분석 시스템")
st.sidebar.header("🔍 설정")
uploaded_file = st.sidebar.file_uploader("이미지 업로드", type=["nii", "nii.gz"])

if uploaded_file is not None:
    image = load_image(uploaded_file)  # 업로드된 이미지 로드
    st.image(image, caption="원본 이미지", use_column_width=True)

    # Grad-CAM 적용
    heatmap = apply_gradcam(model, image)
    st.image(heatmap, caption="Grad-CAM 시각화", use_column_width=True)
