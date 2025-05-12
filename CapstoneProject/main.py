import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


# Defining Model Class (need the previously defined CNNModel)
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 512)
        self.fc2 = torch.nn.Linear(512, 26)  # A~Z (26 classes in total)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import os
#Function to load the model
@st.cache_resource
def load_model():
    model = CNNModel()
    model_path = os.path.join(os.path.dirname(__file__), "sign_language_model.pth")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() #Set Evaluation Mode
    return model


#Image preprocessing function
def preprocess_image(image):
    #image = image.rotate(-90)

    # Preprocesing pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # 텐서 변환
        #transforms.Normalize((0.5,), (0.5,))  # Normalization
    ])

    processed_image = transform(image)
    return processed_image.unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)


# Prediction function
def predict(image, model):
    with torch.no_grad():
        output = model(image)
        predicted_label = torch.argmax(output, dim=1).item()
        return chr(predicted_label + 65)  # Convert 0~25 → A~Z


# Function to run Streamlit UI
def run_app():
    st.title("Sign Language Recognition")
    st.write("Drag and drop an image of a sign language letter to classify it.")

    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Open original image
        image = Image.open(uploaded_file)

        # Display image before preprocessing
        st.image(image, caption="Original Image (Before Preprocessing)", use_container_width=True)

        # Load model
        model = load_model()

        # Preprocess image
        processed_image = preprocess_image(image)

        # Visualize image after preprocessing
        processed_image_np = processed_image.squeeze().numpy()  # Convert tensor to NumPy array

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(processed_image_np, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

        # Run prediction
        prediction = predict(processed_image, model)

        # Print result
        st.success(f"Predicted Alphabet: {prediction}")


# Entry point
if __name__ == "__main__":
    run_app()
