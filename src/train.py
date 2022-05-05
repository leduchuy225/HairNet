import os
import time
import logging
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import HairNetDataset
from model import Net, MyLoss, CollisionLoss, CurMSE, PosMSE


log = logging.getLogger("HairNet")
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)  # 32
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_step", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./weight/")
    parser.add_argument("--data", type=str, default="./")
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument("--test_step", type=int, default=0)
    return parser.parse_args()


def train(model, dataloader, optimizer, device):
    model.train()
    for i, data in enumerate(dataloader, 0):
        img, convdata, visweight = data

        img = img.to(device)
        convdata = convdata.to(device)
        visweight = visweight.to(device)
        # img (bs, 3, 128, 128); convdata (bs, 100, 4, 32, 32); visweight (bs, 100, 32, 32)

        optimizer.zero_grad()

        output = net(img)
        my_loss = loss(output, convdata, visweight)

        my_loss.backward()

        optimizer.step()

    return my_loss


def test(model, dataloader, device):
    pos_error = PosMSE().to(device)  # Position Loss
    cur_error = CurMSE().to(device)  # Curvature Loss
    col_error = CollisionLoss().to(device)  # Collision Loss

    tot_error = MyLoss().to(device)

    model.eval()
    for i, data in enumerate(dataloader, 0):
        img, convdata, visweight = data

        img = img.to(device)
        convdata = convdata.to(device)
        visweight = visweight.to(device)

        output = model(img)

        # cal loss
        pos = pos_error(output, convdata, visweight)
        cur = cur_error(output, convdata, visweight)
        col = col_error(output, convdata)

        tot = tot_error(output, convdata, visweight)

    return pos.item(), cur.item(), col.item(), tot.item()


if __name__ == "__main__":
    # load parameters
    opt = get_args()
    epochs, bs, lr, lr_step, save_dir, data, weight, test_step = (
        opt.epoch,
        opt.batch_size,
        opt.lr,
        opt.lr_step,
        opt.save_dir,
        opt.data,
        opt.weight,
        opt.test_step,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training args: {opt}")
    log.info(f"Training device: {device}")

    log.info("Initializing model and loss function ...")
    net = Net().to(device)
    loss = MyLoss().to(device)

    if weight != "":
        log.info("Loading model's weight ...")
        net.load_state_dict(torch.load(weight, map_location=torch.device(device)))

    # load data
    log.info("Loading data ...")
    train_data = HairNetDataset(project_dir=data, train_flag=1, noise_flag=1)
    train_loader = DataLoader(dataset=train_data, batch_size=bs)
    log.info(f"Train dataset: {len(train_data)} data points")

    if test_step != 0:
        test_data = HairNetDataset(project_dir=data, train_flag=0, noise_flag=0)
        test_loader = DataLoader(dataset=test_data, batch_size=bs)
        log.info(f"Test dataset: {len(test_data)} data points")

    # setup optimizer & lr schedualer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    t = time.localtime()
    save_path = save_dir + time.strftime("%H:%M:%S", t)
    os.mkdir(save_path)

    # train
    log.info("Training ...")
    pre_loss = 100000
    for epoch in range(epochs):
        # measure executive time
        # torch.cuda.synchronize()
        since = int(round(time.time() * 1000))

        train_loss = train(net, train_loader, optimizer, device)
        scheduler.step()

        # torch.cuda.synchronize()
        time_elapsed = int(round(time.time() * 1000)) - since

        # Logging
        log.info(f"Epoch {epoch+1} | Loss: {train_loss:.8f} | time: {time_elapsed}ms")
        if test_step != 0 and (epoch + 1) % test_step == 0:
            pos_loss, cur_loss, col_loss, tot_loss = test(net, test_loader, device)

            log.info(
                f"Epoch {epoch+1} | Loss[ Pos | Cur | Col | Total ]: "
                f"[ {pos_loss:.8f} | {cur_loss:.8f} | {col_loss:.8f} | {tot_loss:.8f} ]"
            )

        # Save model by performance
        if train_loss < pre_loss:
            pre_loss = train_loss
            torch.save(net.state_dict(), save_path + "/weight.pt")
