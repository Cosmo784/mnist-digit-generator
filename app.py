import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# Define VAE model (same as training)
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28*28 + 10, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 28*28)

    def encode(self, x, labels):
        x = torch.cat([x, labels], dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        z = torch.cat([z, labels], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, labels):
        mu, logvar = self.encode(x.view(-1, 28*28), labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

# Load model
@st.cache_resource
def load_model(path="vae_mnist.pth"):
    device = torch.device("cpu")
    model = VAE()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def generate_images(digit, model, num_images=5):
    device = torch.device("cpu")
    z = torch.randn(num_images, 20)
    labels = torch.tensor([digit] * num_images)
    onehot = one_hot(labels).to(device)
    with torch.no_grad():
        samples = model.decode(z, onehot).view(-1, 1, 28, 28)
    return samples

def display_images(images):
    grid = make_grid(images, nrow=5, padding=5)
    np_img = grid.permute(1, 2, 0).numpy()
    return Image.fromarray((np_img * 255).astype(np.uint8).squeeze())

# Streamlit UI
st.set_page_config(page_title="Handwritten Digit Image Generator")
st.title("Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))
if st.button("Generate Images"):
    model = load_model()
    samples = generate_images(digit, model)
    img = display_images(samples)
    st.image(img, caption=f"Generated images of digit {digit}", use_column_width=True)
