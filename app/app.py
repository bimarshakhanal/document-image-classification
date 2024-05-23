"""
Streamlit application to predict document class
"""

import streamlit as st
from PIL import Image
import torch
from torch.nn import Softmax
from torchvision.transforms import v2

from gradcam import get_gradcam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
doc_classes = ['Citizenship', 'License', 'Other', 'Passport']
transform = v2.Compose([
    v2.ToPILImage(),
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


st.set_page_config(page_title="Document Image Classification")

st.header('Document Image Classification')
st.info('Classify document image into citizenship, passport,license\
        and other.')


with st.spinner(text='Loading classification model'):
    model = torch.load('resnet_all.pth', map_location=torch.device('cpu'))
    model = model.to(device)


def predict(img):
    '''Predict document class'''
    # Some png image can have an extra alpha channel
    # so extract first three channels
    img = v2.ToTensor()(img)[:3]
    # apply image transformation
    img = transform(img)
    # batch the image input
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    # get logit outputs and use softmax to get probabilities
    logits = model(img)
    probs = Softmax(dim=1)(logits).squeeze(0)

    # get document class index
    doc_idx = torch.argmax(probs).item()

    return doc_idx, doc_classes[doc_idx], probs[doc_idx].item()


file_up = st.file_uploader(
    "Upload document image to classify", type=['jpeg', 'png'])

col1, col2 = st.columns(2)
IMAGE = None

with col1:
    if file_up:
        IMAGE = Image.open(file_up)
        st.image(IMAGE, caption='Document Image.', width=224)

# global variable to store grad_cam image, document class and confidence
GRAD_CAM_IMAGE = None
doc_class, conf = None, None

try:
    if IMAGE:
        doc_idx, doc_class, conf = predict(IMAGE)
        GRAD_CAM_IMAGE = get_gradcam(model, IMAGE, doc_idx)
        st.success(f"Document Class: {doc_class}: {round(conf*100, 2)}%")
except SystemError:
    st.error("Failed to make prediction")

with col2:
    if GRAD_CAM_IMAGE is not None:
        st.image(GRAD_CAM_IMAGE, caption='Gradcam Image.', width=224)
