import os

import torch

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(discrim_real, discrim_fake, discrim_interp,
                               interp, lamb):
    # TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    # loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    gradients = torch.autograd.grad(
        outputs=discrim_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(discrim_interp).cuda(),
        create_graph=True,
        retain_graph=True)[0]
    grad_penalty = (gradients.norm(p=2, dim=(1,2,3)) - 1.0)**2
    w_critic_loss = (discrim_fake-discrim_real)
    loss = w_critic_loss.mean() + lamb*grad_penalty.mean()
    return loss


def compute_generator_loss(discrim_fake):
    # TODO 1.5.1: Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    loss = -discrim_fake.mean()
    return loss


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.5.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
    )
