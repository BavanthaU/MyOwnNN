from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2 as cv2
from torch_snippets import *
from Unet import Unet
from tqdm import tqdm
import wandb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-3
N_EPOCHS = 2


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
        # print(f'ADEChallengeData2016/images/{self.split}/{self.items[ix]}.jpg')
        image = read(f'ADEChallengeData2016/images/{self.split}/{self.items[ix]}.jpg', 1)
        image = cv2.resize(image, (224, 224))

        # print(f'ADEChallengeData2016/annotations/{self.split}/{self.items[ix]}.png')
        mask = read(f'ADEChallengeData2016/annotations/{self.split}/{self.items[ix]}.png', 0)
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

    trn_dl = DataLoader(trn_ds, batch_size=4, shuffle=True, collate_fn=trn_ds.collate_fn)
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

wandb.init(
    # set the wandb project where this run will be logged
    project="Unet-segmentation_indoor_mac",

    # track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "Unet",
        "dataset": "ADE20k-Indoor",
        "epochs": N_EPOCHS,
    }
)


def run():
    for epoch in range(N_EPOCHS):
        print("####################")
        print(f"       Epoch: {epoch}   ")
        print("####################")

        for bx, data in tqdm(enumerate(trn_dl), total=len(trn_dl)):
            train_loss, train_acc = engine.train_batch(model, data, optimizer, criterion)

        for bx, data in tqdm(enumerate(val_dl), total=len(val_dl)):
            val_loss, val_acc = engine.validate_batch(model, data, criterion)

        wandb.log(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
        )

        print()


run()
