import matplotlib
import torch
from torch import nn

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def image():
    tsfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img_path = "../dataset/Apple.jpg"
    image = Image.open(img_path)
    print(image.size)

    img_tensor = tsfm(image)
    print(img_tensor.shape)

    img_numpy = img_tensor.numpy()
    img_numpy = img_numpy.transpose((1, 2, 0))

    plt.imshow(img_numpy)
    plt.axis('off')
    plt.show()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        print(x.shape)
        return x


model = Autoencoder()

x = model.forward(torch.randn(1, 3, 256, 256))
