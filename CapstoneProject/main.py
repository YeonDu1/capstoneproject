import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


# 모델 클래스 정의 (이전에 정의한 CNNModel이 필요)
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 512)
        self.fc2 = torch.nn.Linear(512, 26)  # A~Z (총 26개 클래스)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 모델 불러오기 함수
@st.cache_resource
def load_model():
    model = CNNModel()
    model.load_state_dict(torch.load("sign_language_model.pth", map_location=torch.device('cpu')))
    model.eval()  # 평가 모드 설정
    return model


# 이미지 전처리 함수
def preprocess_image(image):
    # 이미지를 시계 방향으로 90도 회전
    #image = image.rotate(-90)

    # 전처리 파이프라인
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 흑백 변환
        transforms.Resize((28, 28)),  # 28x28 크기 변경
        transforms.ToTensor(),  # 텐서 변환
        #transforms.Normalize((0.5,), (0.5,))  # 정규화
    ])

    processed_image = transform(image)
    return processed_image.unsqueeze(0)  # 배치 차원 추가 (1, 1, 28, 28)


# 예측 함수
def predict(image, model):
    with torch.no_grad():
        output = model(image)
        predicted_label = torch.argmax(output, dim=1).item()
        return chr(predicted_label + 65)  # 0~25 → A~Z 변환


# Streamlit UI 실행 함수
def run_app():
    st.title("Sign Language Recognition")
    st.write("Drag and drop an image of a sign language letter to classify it.")

    # 이미지 업로드
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # 원본 이미지 열기
        image = Image.open(uploaded_file)

        # 전처리 전 이미지 표시
        st.image(image, caption="Original Image (Before Preprocessing)", use_container_width=True)

        # 모델 불러오기
        model = load_model()

        # 이미지 전처리
        processed_image = preprocess_image(image)

        # 전처리 후 이미지 시각화
        processed_image_np = processed_image.squeeze().numpy()  # 텐서를 NumPy 배열로 변환

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(processed_image_np, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

        # 예측 실행
        prediction = predict(processed_image, model)

        # 결과 출력
        st.success(f"Predicted Alphabet: {prediction}")


# 실행 진입점
if __name__ == "__main__":
    run_app()