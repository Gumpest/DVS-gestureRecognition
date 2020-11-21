import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from gesture import celexEvalOnce
import time, os

step = 5
dt = 12
a = 0.5
aa = 0.25  # a /2
Vth = 0.3
tau = 0.3
tau_t = torch.tensor([tau])

simwin = step * dt

mapping = {
    0: 'arm roll',
    1: 'arm updown',
    2: 'hand clap',
    3: 'right hand clockwise ',
    4: 'right hand wave',
}

# n1 means n+1
@torch.jit.script
def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n):
    u_t1_n1 = 0.3 * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n

    o_t1_n1 = F.relu(torch.sign(torch.add(u_t1_n1, -0.3)))
    return u_t1_n1, o_t1_n1

# torch.jit.ScriptModule
class SpikeNN(nn.Module):

    def __init__(self):
        super(SpikeNN, self).__init__()
        # self.conv0 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(8 * 8 * 64, 256)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 5)


    def forward(self, input):
        # temp variable define / initial state should be zero
        # conv0_u = conv0_out = torch.zeros(input.shape[0], 2, 128, 128, device=self.device)
        pool1_out = torch.zeros(input.shape[0], 1, 32, 32)
        pool1_u = pool1_out
        conv1_out = torch.zeros(input.shape[0], 64, 32, 32)
        conv1_u = conv1_out
        conv2_out = torch.zeros(input.shape[0], 128, 32, 32)
        conv2_u = conv2_out
        pool2_out = torch.zeros(input.shape[0], 128, 16, 16)
        pool2_u = pool2_out
        conv3_out = torch.zeros(input.shape[0], 128, 16, 16)
        conv3_u = conv3_out
        pool3_out = torch.zeros(input.shape[0], 128, 8, 8)
        pool3_u = pool3_out

        fc1_out = torch.zeros(input.shape[0], 256)
        fc1_u = fc1_out
        spike_sum = torch.zeros(input.shape[0], 5)
        fc2_out = spike_sum
        fc2_u = fc2_out

        for t in range(5):
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

        return spike_sum / 5  # rate coding


def main():
    # Training settings
    torch.manual_seed(1)

    model = SpikeNN()

    checkpoint_path = './model/gesture_xyp_20.pt'
    if os.path.isdir(checkpoint_path) or True:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        print('Model loaded.')

    model.eval()

    #data = torch.rand(1, 2, 128, 128, 70)
    eventflow = celexEvalOnce('./Gesture/train/3/2.csv', step, dt)
    data = eventflow.float()

    script_module = torch.jit.trace(model,data)

    output = script_module(data)
    pred = output.argmax(dim=1, keepdim=True)
    print(mapping[pred[0].cpu().numpy()[0]])

    script_module.save('gesture_xyp_20_trans.pt')

if __name__ == '__main__':
    main()