import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import time, os
import argparse
from gesture import DVSGesture

dt = 30
step = 3

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(16 * 16 * 16, 256)
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



def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(device, non_blocking=True), target.float().to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('running loss', loss, epoch * len(train_loader) + batch_idx * args.batch_size)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))



def test(args, model, device, test_loader, epoch, name, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().to(device, non_blocking=True), target.float().to(device, non_blocking=True)
            output = model(data)
            #if name == 'test':
            #    print(output)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target_label = target.argmax(dim=1, keepdim=True)

            correct += pred.eq(target_label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    
    writer.add_scalar('Avg {} loss'.format(name),
                    test_loss,
                    epoch)
    writer.add_scalar('{} acc'.format(name),
                    100. * correct / len(test_loader.dataset),
                    epoch)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        name, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch DvsGesture ANN Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    writer = SummaryWriter('./summary/gestrue_cnn')

    device = torch.device("cuda:2" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    
    train_dataset = DVSGesture('./Gesture/train', train=True,  step=step, dt=dt)
    test_dataset  = DVSGesture('./Gesture/test',  train=False, step=step, dt=dt)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True, drop_last=True, **kwargs)
    


    ANNtest = ANN()
    model = ANNtest.to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = model.to(device)

    for epoch in range(1, args.epochs + 1):
        #adjust_learning_rate(args.lr, optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, train_loader, epoch, 'train', writer)
        test(args, model, device, test_loader, epoch, 'test', writer)

    if (args.save_model):
        torch.save(model.state_dict(), "./tmp/ann_channel.pt")


if __name__ == '__main__':
    main()