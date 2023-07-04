from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2 as cv2
from torch_snippets import *
from Unet import Unet
from tqdm import tqdm
import wandb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-3
PATH = "./model_unet_20epoch.pth"


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


class SegmentationData(Dataset):
    def __init__(self, split):
        self.items = stems(f'ADEChallengeData2016/images/{split}')
        self.split = split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix):
        image = read(f'ADEChallengeData2016/images/{self.split}/{self.items[ix]}.jpg', 1)
        image = cv2.resize(image, (224, 224))

        mask = read(f'ADEChallengeData2016/annotations/{self.split}/{self.items[ix]}.png')
        mask = cv2.resize(mask, (224, 224))

        return image, mask

    def choose(self): return self[randint(len(self))]

    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))

        ims = torch.cat([get_transforms()(im.copy() / 255.)[None] for im in ims]).float().to(DEVICE)

        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(DEVICE)

        return ims, ce_masks


# Loading Data
def get_dataloaders():
    trn_ds = SegmentationData('training')
    val_ds = SegmentationData('validation')

    trn_dl = DataLoader(trn_ds, batch_size=16, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=val_ds.collate_fn)

    return trn_dl, val_dl


trn_dl, val_dl = get_dataloaders()

# Loss

ce = nn.CrossEntropyLoss()


def UnetLoss(preds, targets):
    ce_loss = ce(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc


# Training and Validation
class engine():
    def train_batch(model, data, optimizer, criterion):
        model.train()

        ims, ce_masks = data
        _masks = model(ims)
        optimizer.zero_grad()

        loss, acc = criterion(_masks, ce_masks)
        loss.backward()
        optimizer.step()

        return loss.item(), acc.item()

    @torch.no_grad()
    def validate_batch(model, data, criterion):
        model.eval()

        ims, masks = data
        _masks = model(ims)

        loss, acc = criterion(_masks, masks)

        return loss.item(), acc.item()


# Init Unet

def make_model():
    model = Unet(in_channels=3, out_classes=151).to(DEVICE)
    criterion = UnetLoss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, criterion, optimizer


model, criterion, optimizer = make_model()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))


def evaluate_model():
    for bx, data in tqdm(enumerate(val_dl), total=len(val_dl)):
        im, mask = data
        print("Image data", im, "End of image data")
        _mask = model(im)
        _, _mask = torch.max(_mask, dim=1)

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(im[0].permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.savefig("original_image.jpg")
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(mask.permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.savefig("original_mask.jpg")
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(_mask.permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.savefig("predicted_mask.jpg")
        plt.close()

        break


evaluate_model()
