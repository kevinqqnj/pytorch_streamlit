# ## A pretrained network that recognizes the subject of an image
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import requests
from io import BytesIO


import streamlit as st
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Cycle GAN - horse to zebra")

st.markdown(
    """A pretrained model that fakes it until it makes it.  
    GAN stands for generative adversarial network, where generative means something is being created (in this case, fake masterpieces), adversarial means the two networks are competing to outsmart the other, and well.  
    These networks are one of the most original outcomes of recent deep learning research"""
)


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
        ]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):  # <3>

        assert n_blocks >= 0
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):  # <3>
        return self.model(input)


netG = ResNetGenerator()
url = "https://github.com/deep-learning-with-pytorch/dlwpt-code/raw/master/data/p1ch2/horse2zebra_0.4.0.pth"
model_data = torch.load(BytesIO(requests.get(url).content))
netG.load_state_dict(model_data)
netG.eval()
preprocess = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])


@st.cache
def process(img):
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_out = netG(batch_t)
    out_t = (batch_out.data.squeeze() + 1.0) / 2.0
    out_img = transforms.ToPILImage()(out_t)
    return out_img


image = st.file_uploader("Upload a horse image to fake:")  # image upload widget

if image is not None:
    image = Image.open(image)

    with st.spinner("Generating..."):
        st.image(image, caption="Your uploaded image", use_column_width=True)
        out_img = process(image)

    st.subheader("Result:")
    st.image(out_img, caption="Fake image", use_column_width=True)
