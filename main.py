
import cv2
import numpy as np
import torch
import torchvision.ops
from torch import nn
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir="/home/jlyw/mlp/tensorboard",
                       comment='mlp_writer')


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
        img = cv2.resize(img, (250, 250))
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
            y = y.cuda()

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalar('loss', loss)



def predict_img(model, img_shape):
    model.eval()

    pre_img = np.zeros(img_shape)
    for y in range(0, img_shape[0]):
        for x in range(0, img_shape[1]):
            coord = torch.tensor([x, y], dtype=torch.float).cuda()
            result = model(coord)
            pre_img[y, x] = result.cpu().detach().numpy()

    model.train()
    return pre_img


def train(img_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = ImageMLP()
    if use_cuda:
        print("use cuda ", use_cuda)
        model = model.to(device)

    dataset = ImageMLPDataset(img_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=True)

    #loss_fn = nn.HuberLoss()
    #optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(0, 100):
        print("Epoch ", epoch)
        train_loop(dataloader, model, loss_fn, optimizer, True)
        if epoch % 10 == 0:
            pre_img = predict_img(model, img.shape)
            cv2.imwrite(f"/home/jlyw/mlp/epoch_{epoch}.jpg", pre_img)


    model.eval()
    pre_img = predict_img(model, img.shape)
    cv2.imwrite("/home/jlyw/t/epoch_final.jpg", pre_img)



def fourier_diff(img_1, img_2):
    diff_img = img_1 - img_2
    dft_image_1 = cv2.dft(img_1.astype(np.float32),
                          flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_image_2 = cv2.dft(img_2.astype(np.float32),
                          flags=cv2.DFT_COMPLEX_OUTPUT)
    magnitude_img_1 = cv2.magnitude(dft_image_1[:, :, 0], dft_image_1[:, :, 1])
    magnitude_img_2 = cv2.magnitude(dft_image_2[:, :, 0], dft_image_2[:, :, 1])
    diff = np.abs(magnitude_img_1 - magnitude_img_2)
    diff = diff - np.min(diff)
    diff = diff / np.max(diff) * 255
    cv2.imshow("img 1", img_1.astype(np.uint8))
    cv2.imshow("img 2", img_2.astype(np.uint8))
    cv2.namedWindow("fft diff", flags=cv2.WINDOW_NORMAL)
    cv2.imshow("fft diff", diff.astype(np.uint8))

    diff_img = cv2.dft(diff_img.astype(np.float32),
                          flags=cv2.DFT_COMPLEX_OUTPUT)
    diff_img = cv2.magnitude(diff_img[:, :, 0], diff_img[:, :, 1])
    cv2.imshow("fft diff diff_img", diff_img.astype(np.uint8))
    cv2.waitKey(0)

def main():
    img_path = "/home/jlyw/mlp/epoch_origin.jpg"
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)

    img_path = "/home/jlyw/mlp/epoch_90.jpg"
    img_2 = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    fourier_diff(img, img_2)




if __name__ == '__main__':
    main()

