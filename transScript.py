import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from gesture import celexEvalOnce
import time, os


win = 30

mapping = {
    0: 'arm roll',
    1: 'arm updown',
    2: 'hand clap',
    3: 'right hand clockwise ',
    4: 'right hand wave',
}

# torch.jit.ScriptModule
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
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

        self.device = torch.device("cuda:2") # or cpu

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


def main():
    # Training settings
    torch.manual_seed(1)

    model = ANN()

    checkpoint_path = './tmp/ann.pt'
    if os.path.isdir(checkpoint_path) or True:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        print('Model loaded.')

    model.eval()

    #data = torch.rand(1, 2, 128, 128, 70)
    eventflow = celexEvalOnce('./data/1.csv', win)
    data = eventflow.float()

    script_module = torch.jit.trace(model,data)

    output = script_module(data)
    pred = output.argmax(dim=1, keepdim=True)
    print(mapping[pred[0].cpu().numpy()[0]])

    script_module.save('gesture_ann.pt')

if __name__ == '__main__':
    main()