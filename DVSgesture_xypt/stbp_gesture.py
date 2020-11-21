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

step = 40
dt = 40
simwin = dt * step
a = 0.5
aa = 0.25 # a /2
Vth = 0.3#0.3
tau = 0.3#0.3



def adjust_learning_rate(lr, optimizer, epoch):
    new_lr = lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class SpikeAct(Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u - Vth > 0 then output = 1
        output = torch.gt(input, 0) 
        return output.float()

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors # input = u - Vth
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        #hu = abs(input) < a/2 
        #hu = hu.float() / a
        hu = abs(input) < aa
        hu = hu.float() / (2 * aa)
        #print(hu.sum())

        return grad_input * hu


spikeAct = SpikeAct.apply


# n1 means n+1
def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = spikeAct(u_t1_n1 - Vth)
    return u_t1_n1, o_t1_n1


class SpikeNN(nn.Module):
    def __init__(self):
        super(SpikeNN, self).__init__()
        #self.conv0 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        #self.fc1 = nn.Linear(8 * 8 * 64, 256)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 5)

        self.device = torch.device("cuda:2") # or cpu

    def forward(self, input):
        # temp variable define / initial state should be zero
        #conv0_u = conv0_out = torch.zeros(input.shape[0], 2, 128, 128, device=self.device)
        pool1_u = pool1_out = torch.zeros(input.shape[0], 2, 32, 32, device=self.device)
        conv1_u = conv1_out = torch.zeros(input.shape[0], 64, 32, 32, device=self.device)
        conv2_u = conv2_out = torch.zeros(input.shape[0], 128, 32, 32, device=self.device)
        pool2_u = pool2_out = torch.zeros(input.shape[0], 128, 16, 16, device=self.device)
        conv3_u = conv3_out = torch.zeros(input.shape[0], 128, 16, 16, device=self.device)
        pool3_u = pool3_out = torch.zeros(input.shape[0], 128, 8, 8, device=self.device)
        
        fc1_u = fc1_out = torch.zeros(input.shape[0], 256, device=self.device)
        fc2_u = fc2_out = spike_sum = torch.zeros(input.shape[0], 5, device=self.device)

        for t in range(step):
            in_t = input[:, :, :, :, t]
            # encoding layer
            pool1_u, pool1_out = state_update(pool1_u, pool1_out, F.max_pool2d(in_t, 4))
            conv1_u, conv1_out = state_update(conv1_u, conv1_out, self.conv1(pool1_out))
            conv2_u, conv2_out = state_update(conv2_u, conv2_out, self.conv2(conv1_out))
            pool2_u, pool2_out = state_update(pool2_u, pool2_out, F.avg_pool2d(conv2_out, 2))
            conv3_u, conv3_out = state_update(conv3_u, conv3_out, self.conv3(pool2_out))
            pool3_u, pool3_out = state_update(pool3_u, pool3_out, F.avg_pool2d(conv3_out, 2))

            # flatten
            fc_in = pool3_out.view(input.shape[0], -1)

            fc1_u, fc1_out = state_update(fc1_u, fc1_out, self.fc1(fc_in))
            fc2_u, fc2_out = state_update(fc2_u, fc2_out, self.fc2(fc1_out))
            spike_sum += fc2_out

        return spike_sum / step # rate coding




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
            writer.add_scalar('running loss',
                            loss,
                            epoch * len(train_loader) + batch_idx * args.batch_size)

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
    parser = argparse.ArgumentParser(description='PyTorch DvsGesture SNN Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
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
    


    SNN = SpikeNN()
    model = SNN.to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = model.to(device)

    for epoch in range(1, args.epochs + 1):
        #adjust_learning_rate(args.lr, optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, train_loader, epoch, 'train', writer)
        test(args, model, device, test_loader, epoch, 'test', writer)

    if (args.save_model):
        torch.save(model.state_dict(), "./tmp40/gesture_cnn.pt")


if __name__ == '__main__':
    main()