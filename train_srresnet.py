import time
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# Import your modules
from models.srresnet import SRResNet
from dataset.div2k import DIV2KDataset
from utils import AverageMeter, clip_gradient

# -----------------------------
# Training parameters
# -----------------------------
data_folder = './data'
crop_size = 96
scaling_factor = 4
batch_size = 16
lr = 1e-4
grad_clip = None
num_workers = 4
print_freq = 50

total_iterations = int(1e6)   # updated
save_every = 500              # new: save ckpt every 500 iters

# Model parameters
in_channels = 3
out_channels = 3
n_channels = 64
n_blocks = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

def main():
    model = SRResNet(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=n_channels,
        num_residual_blocks=n_blocks,
        upscale_factor=scaling_factor
    ).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        losses.update(loss.item(), lr_imgs.size(0))
        batch_time.update(time.time() - start)
        start = time.time()
        current_iteration += 1

        if current_iteration % print_freq == 0:
            print(f"Iteration [{current_iteration}/{total_iterations}] "
                  f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                  f"Batch Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) "
                  f"Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s)")

        # -----------------------------
        # SAVE CHECKPOINT EVERY 500 ITERATIONS
        # -----------------------------
        if current_iteration % save_every == 0:
            checkpoint_path = f"checkpoint_srresnet_iter_{current_iteration}.pth.tar"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': current_iteration
            }, checkpoint_path)
            print(f"💾 Saved checkpoint at iteration {current_iteration} -> {checkpoint_path}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': current_iteration
    }, 'srresnet_final.pth.tar')
    print("🎉 Training completed! Saved final checkpoint.")

if __name__ == '__main__':
    main()



