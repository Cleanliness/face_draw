import torch.nn

import face_dset
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import *
import cnn
import u_net

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# hyperparameters
batch_size = 25
iterations = 100
gen_lr = 0.001
discr_lr = 0.001
lam = 100  # weight for l1 cost term
bottleneck_size = 20
image_size = (200, 200)

# load dataset
dataset = face_dset.FaceLabels()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)


def train():
    gen = u_net.UnetGenerator(image_size).to(device)
    discr = cnn.ConvDiscriminator(image_size).to(device)

    optimizer_G = torch.optim.Adam(gen.parameters(), lr=gen_lr)
    optimizer_D = torch.optim.Adam(discr.parameters(), lr=discr_lr)
    l1 = torch.nn.L1Loss()
    loss_f = nn.BCELoss()

    curr_it = 0
    chosen_lbl = None  # face labels for tracking
    res = [0, 0]

    while curr_it < iterations:
        for i, batch in enumerate(dataloader, 0):

            l = batch["label"].to(device)
            f = batch["face"].to(device)
            batch_size = l.shape[0]

            # generate first half of batch and assign faces to labels
            gen_out_old = gen(l[:batch_size//2].to(device))
            gen_out = torch.cat((l[:batch_size//2].to(device), gen_out_old), dim=1)

            # track progress on generated faces
            if chosen_lbl is None:
                chosen_lbl = l[:8].clone().detach()
                test_out = gen_out_old[:8].clone().detach()
            else:
                test_out = gen(chosen_lbl).detach()

            res[0] = test_out.permute(0, 2, 3, 1).cpu()
            res[1] = chosen_lbl.clone().detach().cpu()

            # assign faces to second half of batch
            real_out = torch.cat((l[batch_size//2:].to(device), f[batch_size//2:].to(device)), dim=1)

            # forward pass on discriminator
            discr_out_gen = discr(gen_out.detach())
            discr_out_real = discr(real_out)

            # compute discriminator CE, backprop
            optimizer_D.zero_grad()
            lbl_left = torch.zeros(batch_size//2).to(device)
            lbl_right = torch.ones(math.ceil(batch_size/2)).to(device)
            lbl_discr = torch.cat((lbl_left, lbl_right))
            all_out = torch.cat((discr_out_gen, discr_out_real))

            lbl_discr = torch.unsqueeze(lbl_discr, -1)
            discr_cost = loss_f(all_out, lbl_discr) / 2
            discr_cost.backward()
            print(float(discr_cost))
            optimizer_D.step()

            # compute generator cross entropy + backprop
            optimizer_G.zero_grad()
            fake_gen_in = torch.cat((gen_out_old, f[:batch_size//2]), dim=1)
            discr_out_fake_gen = discr(fake_gen_in)

            fake_lbl = torch.ones(batch_size//2).to(device)
            gen_cost = loss_f(discr_out_fake_gen, torch.unsqueeze(fake_lbl, -1))
            gen_cost += lam*l1(gen_out_old, f[:batch_size//2])

            gen_cost.backward()
            optimizer_G.step()
            print(float(gen_cost))

        show_images(res[0], res[1].permute(0, 2, 3, 1), curr_it)
        print("epoch " + str(curr_it))
        curr_it += 1

    return gen, discr


def show_images(images, lbl, epoch) -> None:
    n: int = len(images)
    f = plt.figure()
    plt.axis("off")
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i])

    for i in range(len(lbl)):
        f.add_subplot(2, n, i + 1)
        plt.imshow(lbl[i])

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig("epoch_" + str(epoch) + ".png")


if __name__ == '__main__':
    train()

