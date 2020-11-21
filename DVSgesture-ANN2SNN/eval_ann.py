import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
import time, os
import argparse
from gesture import celexEvalOnce


dt = 30
step = 3

mapping = {
    0: 'arm roll',
    1: 'arm updown',
    2: 'hand clap',
    3: 'right hand clockwise ',
    4: 'right hand wave',
}

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(128 * 8 * 8 * 4, 256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, input):
        out = self.relu1(self.conv1(input))
        out = self.avgpool1(out)

        out = self.relu2(self.conv2(out))
        out = self.avgpool2(out)

        out = self.relu3(self.conv3(out))
        out = self.avgpool3(out)

        out = out.view(input.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

def test(model, device, eventflow):
    model.eval()
    with torch.no_grad():
        data = eventflow.float().to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        print(mapping[pred[0].cpu().numpy()[0]])


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ANN Eval')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    device = torch.device("cpu")
    
    ANNtest = ANN()
    model = ANNtest.to(device)

    checkpoint_path = './tmp/ann_channel.pt'
    if os.path.isdir(checkpoint_path) or True:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print('Model loaded.')

    eventflow = celexEvalOnce('./data/2.csv', step, dt)

    start = time.clock()
    test(model, device, eventflow)
    end = time.clock()
    print(end - start)

    #seeGesture('./DvsGesture', True, step, dt)


if __name__ == '__main__':
    main()