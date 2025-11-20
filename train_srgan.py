import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.srgan_discriminator import Discriminator
from models.srgan_generator import SRGANGenerator
from dataset.div2k import DIV2KDataset
from loss import FixedContentLoss, TensorAdversarialLoss
from utils import AverageMeter, clip_gradient

# -----------------------------
# Training parameters
# -----------------------------
data_folder = './data'
crop_size = 96
scaling_factor = 4
batch_size = 16
workers = 2
print_freq = 50

# Generator/Discriminator parameters
n_blocks_g = 16
n_channels_g = 64
n_blocks_d = 8
n_channels_d = 64
fc_size_d = 1024

# Training schedule
iterations_phase1 = 5500
iterations_phase2 = 5500
lr_phase1 = 1e-4
lr_phase2 = 1e-5
grad_clip = None
beta = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

#srresnet_checkpoint = './srresnet_final.pth.tar'
srresnet_checkpoint = './checkpoint_srresnet_iter_55000.pth.tar'
srgan_checkpoint = './srgan_final.pth.tar'

# -----------------------------
# Learning rate adjustment
# -----------------------------
def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale

# -----------------------------
# Checkpoint save function
# -----------------------------
def save_checkpoint(iteration, generator, discriminator, optimizer_g, optimizer_d, filename):
    torch.save({
        'iteration': iteration,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
    }, filename)
    print(f"Checkpoint saved at iteration {iteration}: {filename}")

# -----------------------------
# Main training function
# -----------------------------
def main():
    print("Initializing models...")
    generator = SRGANGenerator(
        in_channels=3,
        out_channels=3,
        channels=n_channels_g,
        num_residual_blocks=n_blocks_g,
        upscale_factor=scaling_factor
    )

    # Load SRResNet checkpoint safely
    if srresnet_checkpoint:
        print(f"Loading SRResNet weights from {srresnet_checkpoint}")
        checkpoint = torch.load(srresnet_checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        own_state = generator.state_dict()
        matched = 0
        for k, v in state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
                matched += 1
        generator.load_state_dict(own_state)
        print(f"Initialized {matched}/{len(own_state)} layers from SRResNet checkpoint")

    discriminator = Discriminator()

    # Move to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Optimizers
    optimizer_g = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr_phase1)
    optimizer_d = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr_phase1)

    # Loss functions
    content_loss_fn = FixedContentLoss().to(device)
    adversarial_loss_fn = TensorAdversarialLoss().to(device)

    # Dataset
    print("Loading dataset...")
    train_dataset = DIV2KDataset(
        lr_dir=f'{data_folder}/DIV2K_train_LR_bicubic/X{scaling_factor}',
        hr_dir=f'{data_folder}/DIV2K_train_HR',
        scale=scaling_factor,
        patch_size=crop_size
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    total_samples = len(train_dataset)
    batches_per_epoch = len(train_loader)
    total_iterations = iterations_phase1 + iterations_phase2
    estimated_epochs = total_iterations / batches_per_epoch

    print(f"Training on {total_samples} samples, estimated epochs: {estimated_epochs:.2f}")

    # Training loop
    generator.train()
    discriminator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_g_content = AverageMeter()
    losses_g_adv = AverageMeter()
    losses_g_total = AverageMeter()
    losses_d = AverageMeter()

    start_time = time.time()
    iteration = 0

    while iteration < total_iterations:
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
            iteration += 1
            if iteration > total_iterations:
                break

            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            data_time.update(time.time() - start_time)

            # Train generator
            optimizer_g.zero_grad()
            sr_imgs = generator(lr_imgs)
            content_loss = content_loss_fn(sr_imgs, hr_imgs)
            adv_loss = adversarial_loss_fn(discriminator(sr_imgs), torch.ones_like(discriminator(sr_imgs)))
            gen_loss = content_loss + beta * adv_loss
            gen_loss.backward()
            if grad_clip:
                clip_gradient(optimizer_g, grad_clip)
            optimizer_g.step()

            # Train discriminator
            optimizer_d.zero_grad()
            real_pred = discriminator(hr_imgs)
            fake_pred = discriminator(sr_imgs.detach())
            d_loss = (adversarial_loss_fn(real_pred, torch.ones_like(real_pred)) +
                      adversarial_loss_fn(fake_pred, torch.zeros_like(fake_pred))) / 2
            d_loss.backward()
            if grad_clip:
                clip_gradient(optimizer_d, grad_clip)
            optimizer_d.step()

            # -----------------------------
            # Save checkpoint every 500 iterations
            # -----------------------------
            if iteration % 500 == 0:
                save_checkpoint(
                    iteration,
                    generator,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    f"srgan_iter_{iteration}.pth.tar"
                )

            # Update metrics
            losses_g_content.update(content_loss.item(), lr_imgs.size(0))
            losses_g_adv.update(adv_loss.item(), lr_imgs.size(0))
            losses_g_total.update(gen_loss.item(), lr_imgs.size(0))
            losses_d.update(d_loss.item(), lr_imgs.size(0))
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            if iteration % print_freq == 0:
                print(f"Iter [{iteration}/{total_iterations}] "
                      f"Gen Total: {losses_g_total.avg:.4f} "
                      f"Gen Content: {losses_g_content.avg:.4f} "
                      f"Gen Adv: {losses_g_adv.avg:.4f} "
                      f"Disc Loss: {losses_d.avg:.4f}")
                losses_g_content.reset()
                losses_g_adv.reset()
                losses_g_total.reset()
                losses_d.reset()
                batch_time.reset()
                data_time.reset()

            if iteration == iterations_phase1:
                print("Phase 2: reducing learning rate")
                adjust_learning_rate(optimizer_g, lr_phase2 / lr_phase1)
                adjust_learning_rate(optimizer_d, lr_phase2 / lr_phase1)

    # Final checkpoint
    save_checkpoint(
        iteration,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        "srgan_final.pth.tar"
    )
    print("Training completed and final checkpoint saved!")

if __name__ == '__main__':
    main()

