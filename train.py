import os
import wandb
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from extreme_weather.data.utils import ExtremeEventDataset
from extreme_weather.data.losses import JDTLoss
from extreme_weather.data.metrics import get_cm, get_iou_perClass

from extreme_weather.models.cgnet import CGNetModule
from extreme_weather.models.segformer import Segformer

from lion_pytorch import Lion

config = {
    "experiment_name": "segformer_run",
    "architecture": "segformer",
    "lr": 1e-3, 
    "batch_size": 4,
    "gradient_accumulation": 16,
    "optimizer": "adam",
    "num_epochs": 20,
    "loss_temperature": 0.25,
    "dropout": 0.0,
    "weight_decay": 0.01,
    "img_size": (512, 1024),
    "train_fraction": 0.9,
    "checkpoints_path": "./checkpoints/", # where to save model weights
    "data_path": "/net/pf-pc69/scratch/andregr/dataset/" # where is the dataset downloaded to?
}

config["effective_batch_size"] = config["batch_size"] * config["gradient_accumulation"]

checkpoints_path = os.path.join(config["checkpoints_path"], config["experiment_name"])
os.makedirs(checkpoints_path, exist_ok=True)

wandb.init(project="ClimateNet-V2", config=config)

# get dataset
train = ExtremeEventDataset(
    config["data_path"], 
    config["train_fraction"], 
    "train",
    merge_annotations=True
)
test = ExtremeEventDataset(
    config["data_path"], 
    config["train_fraction"], 
    "test",
    merge_annotations=True
)

train_loader = torch.utils.data.DataLoader(train, batch_size=config["batch_size"], shuffle=True, num_workers=8, prefetch_factor=1, drop_last=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=config["batch_size"], shuffle=False, num_workers=8, prefetch_factor=1, drop_last=True)

if config["architecture"] == "cgnet":
    model = CGNetModule(classes=3, dropout=config["dropout"])

elif config["architecture"] == "segformer":
    model = Segformer(
        dims = (32, 64, 160, 256),      # dimensions of each stage
        heads = (1, 2, 5, 8),           # heads of each stage
        ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
        reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        num_layers = 2,                 # num layers of each stage
        decoder_dim = 256,              # decoder dimension
        num_classes = 3,                 # number of segmentation classes
        channels = 4
    )
    model = torch.nn.Sequential(
        model,
        torch.nn.Upsample(size=config["img_size"], mode='bilinear', align_corners=True),
    )

model = torch.nn.DataParallel(model).to("cuda")

if config["optimizer"] == "adam":
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
elif config["optimizer"] == "lion":
    optimizer = Lion(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

num_epochs = config["num_epochs"]
num_steps = num_epochs * len(train_loader) / config["gradient_accumulation"]
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

jaccard_loss = JDTLoss(T=config["loss_temperature"])

def validate(model, step):
    model.eval()

    val_loss = 0
    ious = []

    for it, (features, labels) in enumerate(tqdm(test_loader)): 
        features = features.to("cuda")
        labels = labels.to("cuda")

        with torch.inference_mode():

            # Interpolate size
            labels = F.interpolate(labels, size=config["img_size"])
            features = F.interpolate(features, size=config["img_size"])

            preds = model(features)

            # validation soft jaccard loss
            loss = jaccard_loss(preds, labels)
            val_loss += loss.item()

            # validation IoU (based on union of annotations)
            _, preds = torch.max(preds, dim=1, keepdims=True)
            _, labels = torch.max(labels, dim=1, keepdims=True) #FIXME: make sure this works
            
            cm = get_cm(
                preds.flatten(start_dim=1), 
                labels.flatten(start_dim=1), 
                n_classes=3
            )            
            ious.append(get_iou_perClass(cm))

    ious = np.mean(ious, axis=0)
    wandb.log({"Validation mIoU": np.mean(ious)}, step=step)
    wandb.log({"Validation BG IoU": ious[0]}, step=step)
    wandb.log({"Validation AR IoU": ious[1]}, step=step)
    wandb.log({"Validation TC IoU": ious[2]}, step=step)
    wandb.log({"val_loss": val_loss / len(test_loader)}, step=(step))

    size = (120, 240)
    features = F.interpolate(features, size=size)[:8, 2].cpu().numpy()
    features *= (1.0/features.max())
    preds = F.interpolate(preds.float(), size=size)[:8,0].cpu().numpy()
    labels = F.interpolate(labels.float(), size=size)[:8,0].cpu().numpy()

    # Save 
    class_labels = {0: "Background", 1: "Atmospheric River", 2: "Tropical Cyclone"}
    wandb.log(
        {"segmentations" : [wandb.Image(f, masks={
            "predictions" : {
                "mask_data" : p,
                "class_labels" : class_labels
            },
            "ground_truth" : {
                "mask_data" : l,
                "class_labels" : class_labels
            }
        }) for f, p, l in zip(features, preds, labels)]}, step=step)

global_step = 0
for epoch in range(1, num_epochs+1):   
    validate(model, global_step)
    model.train()

    for it, (features, labels) in enumerate(tqdm(train_loader), 1):

        features = features.to("cuda")
        labels = labels.to("cuda")

        labels = F.interpolate(labels, size=config["img_size"])
        features = F.interpolate(features, size=config["img_size"])

        preds = model(features) 
        
        loss = jaccard_loss(preds, labels)
        loss.backward()

        wandb.log({"loss": loss.item()}, step=global_step)

        if (it % config["gradient_accumulation"] == 0) or (it == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        global_step += config["batch_size"]

    save_path = os.path.join(checkpoints_path, f"model_{epoch}.pth")
    torch.save(model.state_dict(), save_path)