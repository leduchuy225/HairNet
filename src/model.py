"""
Copyright@ Qiao-Mu(Albert) Ren. 
All Rights Reserved.
This is the code to build a neural network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # encoder
        # self.conv1 = nn.Conv2d(3, 32, 8, 2, 3)
        self.conv2 = nn.Conv2d(3, 64, 8, 2, 3)
        self.conv3 = nn.Conv2d(64, 128, 6, 2, 2)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)
        # decoder
        self.fc1 = nn.Linear(512, 4096)
        # self.fc2 = nn.Linear(1024, 4096)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        # MLP
        # Position
        self.branch1_fc1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.branch1_fc2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.branch1_fc3 = nn.Conv2d(512, 300, 1, 1, 0)
        # Curvature
        self.branch2_fc1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.branch2_fc2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.branch2_fc3 = nn.Conv2d(512, 100, 1, 1, 0)

    def forward(self, x, interp_factor=1):
        # encoder
        # x = F.relu(self.conv1(x)) # (batch_size, 32, 128, 128)
        x = F.relu(self.conv2(x))  # (batch_size, 64, 64, 64)
        x = F.relu(self.conv3(x))  # (batch_size, 128, 32, 32)
        x = F.relu(self.conv4(x))  # (batch_size, 256, 16, 16)
        x = F.relu(self.conv5(x))  # (batch_size, 512, 8, 8)
        x = torch.tanh(F.max_pool2d(x, 8))  # (batch_size, 512, 1, 1)
        # decoder
        x = x.view(-1, 1 * 1 * 512)
        x = F.relu(self.fc1(x))
        # x = x.view(-1, 1*1*1024)
        # x = F.relu(self.fc2(x))
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.conv6(x))  # (batch_size, 256, 4, 4)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )  # (batch_size, 256, 8, 8)
        x = F.relu(self.conv7(x))  # (batch_size, 256, 8, 8)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )  # (batch_size, 256, 16, 16)
        x = F.relu(self.conv8(x))  # (batch_size, 256, 16, 16)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )  # (batch_size, 256, 32, 32)
        # interpolate feature map
        if interp_factor != 1:
            x = F.interpolate(
                x, scale_factor=interp_factor, mode="bilinear", align_corners=False
            )  # (batch_size, 256, 32, 32)
        # MLP
        # Position
        branch1_x = F.relu(self.branch1_fc1(x))
        branch1_x = F.relu(self.branch1_fc2(branch1_x))
        branch1_x = self.branch1_fc3(branch1_x)
        branch1_x = branch1_x.view(-1, 100, 3, 32 * interp_factor, 32 * interp_factor)
        # Curvature
        branch2_x = F.relu(self.branch2_fc1(x))
        branch2_x = F.relu(self.branch2_fc2(branch2_x))
        branch2_x = self.branch2_fc3(branch2_x)
        branch2_x = branch2_x.view(-1, 100, 1, 32 * interp_factor, 32 * interp_factor)
        x = torch.cat([branch1_x, branch2_x], 2)
        return x  # (batch_size, 100, 4, 32, 32)


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output, convdata, visweight):
        # removing nested for-loops (0.238449s -> 0.001860s)
        pos_loss = PosMSE().forward(output, convdata, visweight)
        cur_loss = CurMSE().forward(output, convdata, visweight)
        col_loss = CollisionLoss().forward(output, convdata)

        return pos_loss + cur_loss + col_loss


class CollisionLoss(nn.Module):
    def __init__(self):
        super(CollisionLoss, self).__init__()

    def forward(self, output, convdata):
        x, y, z = 0.005, 1.75, 0.01
        a, b, c = 0.08, 0.12, 0.1
        L1 = torch.add(
            torch.add(
                torch.abs(output[:, :, 0, :, 1:] - output[:, :, 0, :, :-1]),
                torch.abs(output[:, :, 1, :, 1:] - output[:, :, 1, :, :-1]),
            ),
            torch.abs(output[:, :, 2, :, 1:] - output[:, :, 2, :, :-1]),
        )
        D = 1 - torch.add(
            torch.add(
                torch.pow((output[:, :, 0, :, 1:] - x) / a, 2),
                torch.pow((output[:, :, 1, :, 1:] - y) / b, 2),
            ),
            torch.pow((output[:, :, 2, :, 1:] - z) / c, 2),
        )
        D[D < 0] = 0
        C = torch.sum(L1 * D)
        return C / 1024.0
        # loss = C / (convdata.shape[0] * convdata.shape[1] * 1024.0)
        # return loss


class PosMSE(nn.Module):
    def __init__(self):
        super(PosMSE, self).__init__()

    def forward(self, output, convdata, visweight):
        visweight = visweight[:, :, :, :].reshape(1, -1)
        e_squared = torch.pow(
            (convdata[:, :, 0:3, :, :] - output[:, :, 0:3, :, :]), 2
        ).reshape(-1, 3)
        loss = visweight.mm(e_squared).sum()

        # convdata.shape 8, 100, 4, 32, 32
        return loss / 1024

        # return loss / (convdata.shape[0] * convdata.shape[1] * 1024.0)


class CurMSE(nn.Module):
    def __init__(self):
        super(CurMSE, self).__init__()

    def forward(self, output, convdata, visweight):
        visweight = visweight[:, :, :, :].reshape(1, -1)
        e_squared = torch.pow(
            (convdata[:, :, 3, :, :] - output[:, :, 3, :, :]), 2
        ).reshape(-1, 1)
        loss = visweight.mm(e_squared).sum()

        return loss / 1024

        # return loss / (convdata.shape[0] * convdata.shape[1] * 1024.0)
