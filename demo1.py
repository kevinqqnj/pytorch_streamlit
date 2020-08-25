# ## A pretrained network that recognizes the subject of an image
import torch
from PIL import Image
from torchvision import models, transforms

import streamlit as st
import pandas as pd

st.title("RESNET image recognizer")

st.markdown(
    """A state-of-the-art deep neural network that was pretrained on an object-recognition task.  
    ImageNet dataset (http://imagenet.stanford.edu) is a very large dataset of over 14 million images maintained by Stanford University."""
)

resnet = models.resnet101(pretrained=True)
resnet.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

with open("./imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]


@st.cache
def process(img):
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    out = resnet(batch_t)
    return out


image = st.file_uploader("Upload an image to recognize:")  # image upload widget

# image = Image.open(image)
# st.image(image, width=400)
# btn = st.button("Recognize it")

if image is not None:
    image = Image.open(image)
    st.image(image, caption='Your uploaded image', use_column_width=True)
    st.write("")

    with st.spinner('Classifying...'):
        out = process(image)
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        top5 = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

    st.subheader("Result:")
    df = pd.DataFrame(top5, columns=('Subject', 'confidence %'))
    st.dataframe(df)
