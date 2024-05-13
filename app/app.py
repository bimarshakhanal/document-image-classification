"""
Streamlit application to predict document class
"""

import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
doc_classes = ['Citizenship', 'License', 'Passport']
transform = v2.Compose([
    v2.ToPILImage(),
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


st.set_page_config(page_title="Document Image Classification")

st.header('Document Image Classification')
st.info('Classify document image into citizenship, passport,license and other.')

with st.spinner(text='Loading classification model'):
    model = torch.load('resnet18.pth', map_location=torch.device('cpu'))
    model = model.to(device)


def predict(img):
    '''Predict document class'''
    img = v2.ToTensor()(img)[:3]
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    print("IM shape:", img.shape)
    probs = model(img)
    doc = torch.argmax(probs)
    return doc.item()


file_up = st.file_uploader(
    "Upload document image to classify", type=['jpeg', 'png'])

col1, col2 = st.columns(2)
IMAGE = None

with col1:
    if file_up:
        IMAGE = Image.open(file_up)
        st.image(IMAGE, caption='Document Image.', width=224)
with col2:
    try:
        if IMAGE:
            doc_class = predict(IMAGE)
            st.success(f"The document is classified as {
                       doc_classes[doc_class]}")
    except SystemError:
        st.error("Failed to make prediction")