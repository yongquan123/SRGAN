import time
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# Import your modules
from models.srresnet import SRResNet
from dataset.div2k import DIV2KDataset
from utils import AverageMeter, clip_gradient  # You need to implement these

# -----------------------------
# Training parameters
# -----------------------------
data_folder = './data'      # path to DIV2K dataset
crop_size = 96              # HR patch size
scaling_factor = 4          # upscaling factor
batch_size = 16
lr = 1e-4
grad_clip = None
num_workers = 4
print_freq = 50
total_iterations = 2500  # total updates #1E6 is too large to run for my computer


# Model parameters
in_channels = 3
out_channels = 3
n_channels = 64
n_blocks = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# -----------------------------
# Main function
# -----------------------------
def main():
    # -----------------------------
    # Initialize model
    # -----------------------------
    model = SRResNet(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=n_channels,
        num_residual_blocks=n_blocks,
        upscale_factor=scaling_factor
    ).to(device)

    # Loss function
    criterion = nn.MSELoss().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------
    # Dataset and DataLoader
    # -----------------------------
    train_dataset = DIV2KDataset(
        lr_dir=f'{data_folder}/DIV2K_train_LR_bicubic/X{scaling_factor}',
        hr_dir=f'{data_folder}/DIV2K_train_HR',
        scale=scaling_factor,
        patch_size=crop_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # -----------------------------
    # Training loop based on iterations
    # -----------------------------
    model.train()
    current_iteration = 0
    train_loader_iter = iter(train_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()
    while current_iteration < total_iterations:
        try:
            lr_imgs, hr_imgs = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            lr_imgs, hr_imgs = next(train_loader_iter)

        data_time.update(time.time() - start)

        # Move to device
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        # Forward pass
        sr_imgs = model(lr_imgs)

        # Compute MSE loss
        loss = criterion(sr_imgs, hr_imgs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optional gradient clipping
        if grad_clip:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        # Update metrics
        losses.update(loss.item(), lr_imgs.size(0))
        batch_time.update(time.time() - start)
        start = time.time()
        current_iteration += 1

        # Print progress
        if current_iteration % print_freq == 0:
            print(f"Iteration [{current_iteration}/{total_iterations}] "
                  f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                  f"Batch Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) "
                  f"Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s)")

    # Save final checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': current_iteration
    }, 'srresnet_final.pth.tar')


if __name__ == '__main__':
    main()
