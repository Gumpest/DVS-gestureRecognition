import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
import time, os
import argparse
from gesture import celexEvalOnce

step = 5
dt = 12
a = 0.5
aa = 0.25 # a /2
Vth = 0.3
tau = 0.3

simwin = step * dt

mapping = {
    0: 'arm roll',
    1: 'arm updown',
    2: 'hand clap',
    3: 'right hand clockwise ',
    4: 'right hand wave',
}


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
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        #self.fc1 = nn.Linear(8 * 8 * 64, 256)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 5)

        self.device = torch.device("cpu") # or cpu

    def forward(self, input):
        # temp variable define / initial state should be zero
        #conv0_u = conv0_out = torch.zeros(input.shape[0], 2, 128, 128, device=self.device)
        pool1_u = pool1_out = torch.zeros(input.shape[0], 1, 32, 32, device=self.device)
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


def test(model, device, eventflow):
    model.eval()
    with torch.no_grad():
        data = eventflow.float().to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        print(mapping[pred[0].cpu().numpy()[0]])


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SNN Eval Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    device = torch.device("cpu")
    
    SNN = SpikeNN()
    model = SNN.to(device)

    checkpoint_path = './model/gesture_xyp_20.pt'
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