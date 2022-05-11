import cv2
import torch
import torch.nn as nn

from model import Net
from visualize3D import show3DhairPlotByStrands

test_sample = [
    './data/strands00006_00008_00000_v0.png',
    './data/strands00164_00321_11001_v1.png',
    './data/strands00164_00321_10101_v1.png',
]

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

net = Net().cuda()
net.load_state_dict(torch.load('./weight/12:37:48/weight.pt'))
net.eval()
# net.apply(init_weights)
# torch.save(net.state_dict(), "./weight/model.pt")

for i in range(len(test_sample)):
    img = cv2.imread(test_sample[i])
    img = img.reshape(1, 3, 128, 128)
    img = torch.from_numpy(img).float().cuda()
    
    output = net(img)

    strands = output.squeeze().cpu().detach().numpy()
    show3DhairPlotByStrands(strands)