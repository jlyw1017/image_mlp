
import cv2
import numpy as np
import torch
import torchvision.ops
from torch import nn


class ImageMLP(nn.Module):
    def __init__(self):
        super(ImageMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class ImageMLPDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, img_path):
        """Initialization"""
        img = cv2.imread(img_path)
        img = cv2.resize(img, (200, 250))
        self.img = img

    def __len__(self):
        """Denotes the total number of samples"""
        return self.img.shape[0] * self.img.shape[1]

    def __getitem__(self, index):
        """Generates one sample of data"""
        y = int(index / self.img.shape[1])
        x = index - y * self.img.shape[1]
        label = self.img[y, x]
        return torch.tensor([x, y], dtype=torch.float), torch.tensor(label, dtype=torch.float)


def train_loop(dataloader, model, loss_fn, optimizer, use_gpu=True):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        if use_gpu:
            X = X.cuda()

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def predict_img(model, img_shape):
    model.eval()

    pre_img = np.zeros(img_shape)
    for y in range(0, img_shape[0]):
        for x in range(0, img_shape[1]):
            coord = torch.tensor([x, y], dtype=torch.float)
            pre_img[y, x] = model(coord).detach().numpy()

    model.train()
    return pre_img


def main():
    img_path = "/home/jlyw/elden_ring.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (200, 250))
    cv2.imwrite(f"/home/jlyw/t/epoch_origin.jpg", img)

    use_cuda = torch.cuda.is_available() and False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = ImageMLP()
    if use_cuda:
        model.to(device)

    dataset = ImageMLPDataset(img_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    loss_fn = nn.HuberLoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)

    for epoch in range(0, 100):
        print("Epoch ", epoch)
        train_loop(dataloader, model, loss_fn, optimizer, False)
        if epoch % 10 == 0:
            pre_img = predict_img(model, img.shape)
            cv2.imwrite(f"/home/jlyw/t/epoch_{epoch}.jpg", pre_img)

    model.eval()

    pre_img = predict_img(model, img.shape)
    cv2.imwrite("/home/jlyw/t/epoch_final.jpg", pre_img)


if __name__ == '__main__':
    main()

