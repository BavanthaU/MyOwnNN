import torch
import torchvision
from ADE20Kdataset import Ade20KDataset
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex


def save_checkpoint(state, filename="unet_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.loa_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = Ade20KDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Ade20KDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            preds = torch.argmax(preds, dim=1)  # dim =1 => over rows *
            num_correct += torch.eq(preds, y).int()  # the o/p here is 0 , 1 *
            num_pixels += torch.numel(preds)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsquuze(1), f"{folder}/pred_{idx}.png")
    model.train()
