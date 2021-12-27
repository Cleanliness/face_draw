Conditional GAN which converts labels to faces. Trained on <a href="https://store.mut1ny.com/product/face-head-segmentation-dataset-community-edition?v=3e8d115eb4b3" target="_blank">Mut1ny's face dataset</a>
using only the 2000 real faces/labels. Online demo <a href="https://cleanliness.github.io/js%20projects/facegen/facegen.html" target="_blank">here.</a>

# Generator architecture
Project consists of Pytorch implementations of a U-Net and a CNN-based autoencoder (baseline model). see cnn.py and u_net.py
todo add imgs

# Discriminator architecture
Includes Pytorch implementations of a single output discriminator and the patch discriminator described in pix2pix (patch size = 70).

# references
<a href="https://arxiv.org/pdf/1611.07004.pdf" target="_blank">pix2pix paper</a>
<a href="https://arxiv.org/abs/1701.00160" target="_blank">NIPS 2016 GAN tutorial</a>
<a href="https://arxiv.org/pdf/1505.04597v1.pdf" target="_blank">U-Net paper</a>
