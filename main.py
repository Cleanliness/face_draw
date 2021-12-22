import torch.nn

import face_dset
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import *
import cnn

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# hyperparameters
batch_size = 200
iterations = 10

gen_lr = 0.001
discr_lr = 0.001
lam = 100  # weight for l1 cost term

bottleneck_size = 20
image_size = (200, 200)

# load dataset
dataset = face_dset.FaceLabels()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)
loss_f = nn.BCELoss()

def train():
    gen = cnn.ConvGenerator(image_size, bottleneck_size)
    discr = cnn.ConvDiscriminator(image_size)

    optimizer_G = torch.optim.Adam(gen.parameters(), lr=gen_lr)
    optimizer_D = torch.optim.Adam(discr.parameters(), lr=discr_lr)
    l1 = torch.nn.L1Loss()

    curr_it = 0
    while curr_it < iterations:
        for i, batch in enumerate(dataloader, 0):

            l = batch["label"]
            f = batch["face"]

            batch_size = l.shape[0]

            # generate first half of batch
            gen_out_old = gen(l[:batch_size//2])
            test_out = torch.tensor(gen_out_old[:8])
            show_images(test_out.permute(0, 2, 3, 1))
            gen_out = torch.cat((gen_out_old, f[:batch_size//2]), dim=1)

            # assign faces to second half of batch
            real_out = l[batch_size//2:]
            real_out = torch.cat((real_out, f[batch_size//2:]), dim=1)

            discr_out_gen = discr(gen_out)
            discr_out_real = discr(real_out)

            # compute generator cross entropy cost + backprop
            optimizer_G.zero_grad()
            fake_lbl = torch.ones(batch_size//2)
            gen_cost = loss_f(discr_out_gen, torch.unsqueeze(fake_lbl, -1))
            gen_cost += lam*l1(gen_out_old, f[:batch_size//2])

            gen_cost.backward()
            optimizer_G.step()
            print(float(gen_cost))

            # compute discriminator CE, backprop
            optimizer_D.zero_grad()
            lbl_left = torch.zeros(batch_size//2)
            lbl_right = torch.ones(math.ceil(batch_size/2))
            lbl_discr = torch.cat((lbl_left, lbl_right))
            all_out = torch.cat((discr_out_gen.clone().detach(), discr_out_real))

            lbl_discr = torch.unsqueeze(lbl_discr, -1)
            discr_cost = loss_f(all_out, lbl_discr)
            discr_cost.backward()
            print(float(discr_cost))
            optimizer_D.step()

        print("epoch " + str(curr_it))
        curr_it += 1
    return gen, discr


def show_images(images) -> None:
    n: int = len(images)
    f = plt.figure()
    plt.axis("off")
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i])

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show(block=True)


if __name__ == '__main__':
    train()

    # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()

