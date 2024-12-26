import streamlit as st
import PIL.Image as Image
import requests

HOST = "http://localhost:3000"

def main():
    st.title("이미지 업로드 및 분류 예시")
    st.write("이미지를 업로드하면, 사전 학습된 ResNet50 모델을 통해 분류한 뒤 예측 라벨을 역순으로 표시합니다.")

    # 4. 사이드바/메인 영역에 파일 업로더 추가
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 업로드된 이미지 열기
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="업로드된 이미지", use_column_width=True)
        if st.button("Predict"):
            header = {"accept": "image/*"}
            files = {"images": (uploaded_file.name,
                                uploaded_file.getvalue(),
                                "image/png")}
            response = requests.post(url=HOST + "/classify", files=files, headers=header)

            if response.status_code == 200:
                res_json = response.json()
                label = res_json["label"]
                prob = res_json["score"]
                st.success(f"예측 클래스 라벨: {label}, 확률: {prob:.4f}")
            else:
                st.error(f"API Error: {response.text}")


if __name__ == "__main__":
    main()