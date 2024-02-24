import os
import wandb
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from extreme_weather.data.utils import ExtremeEventDataset
from extreme_weather.data.losses import JDTLoss
from extreme_weather.data.metrics import get_cm, get_iou_perClass

from extreme_weather.models.diffusion import Unet, MedSegDiff

from lion_pytorch import Lion

config = {
    "experiment_name": "unet_run",
    "architecture": "unet",
    "lr": 1e-4, 
    "train_batch_size": 4,
    "eval_batch_size": 16,
    "gradient_accumulation": 8,
    "optimizer": "adam",
    "num_epochs": 500,
    "loss_temperature": 0.25,
    "dropout": 0.0,
    "weight_decay": 0.01,
    "img_size": (512 // 2, 1024 // 2),
    "train_fraction": 0.9,
    "diffusion_timesteps": 100,
    "epochs_per_eval": 10, # only evaluate every n epochs
    "checkpoints_path": "./checkpoints/", # where to save model weights
    "data_path": "/net/pf-pc69/scratch/andregr/dataset/" # where is the dataset downloaded to?
}

config["effective_batch_size"] = config["train_batch_size"] * config["gradient_accumulation"]

checkpoints_path = os.path.join(config["checkpoints_path"], config["experiment_name"])
os.makedirs(checkpoints_path, exist_ok=True)

wandb.init(project="ClimateNet-V2", config=config)

# get dataset
train = ExtremeEventDataset(
    config["data_path"], 
    config["train_fraction"], 
    "train",
    merge_annotations=False
)
test = ExtremeEventDataset(
    config["data_path"], 
    config["train_fraction"], 
    "test",
    merge_annotations=False
)

train_loader = torch.utils.data.DataLoader(train, batch_size=config["train_batch_size"], shuffle=True, num_workers=8, prefetch_factor=1, drop_last=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=config["eval_batch_size"], shuffle=False, num_workers=8, prefetch_factor=1, drop_last=True)

if config["architecture"] == "unet":
    model = Unet(
    dim = 32,
    image_size = config["img_size"],
    mask_channels = 2,
    input_img_channels = 4,
    dim_mults = (1, 2, 4, 8)
) 
else:
    raise Exception

model = torch.nn.DataParallel(model).to("cuda")

if config["optimizer"] == "adam":
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
elif config["optimizer"] == "lion":
    optimizer = Lion(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

num_epochs = config["num_epochs"]
num_steps = num_epochs * len(train_loader) / config["gradient_accumulation"]
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

jaccard_loss = JDTLoss(T=config["loss_temperature"])

diff = MedSegDiff(
    model,
    timesteps = config["diffusion_timesteps"]
).cuda()

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

            # validation soft jaccard loss
            loss = diff(labels, features)
            val_loss += loss.item()

            #features = einops.repeat(features, "b c h w -> (b r) c h w", r=config["validation_samples"])
            #labels = einops.repeat(labels, "b c h w -> (b r) c h w", r=config["validation_samples"])

            preds = diff.sample(features)

        #preds = einops.rearrange(labels, "(b r) c h w -> b r c h w", r=config["validation_samples"])
        #preds = preds.mean(dim=1)
        
        # validation IoU (based on union of annotations)
        
        #_, preds = torch.max(preds, dim=1, keepdims=True)
        preds = preds > 0.5
        mask = (preds.sum(1) == 0)[:,None]
        preds = torch.cat([mask, preds], dim=1)

        mask = (labels.sum(1) == 0)[:,None]
        labels = torch.cat([mask, labels], dim=1)

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

    if epoch % config["epochs_per_eval"] == 0:
        validate(model, global_step)
    model.train()

    for it, (features, labels) in enumerate(tqdm(train_loader), 1):

        features = features.to("cuda")
        labels = labels.to("cuda")

        labels = F.interpolate(labels, size=config["img_size"])
        features = F.interpolate(features, size=config["img_size"])

        loss = diff(labels, features)
        loss.backward()

        wandb.log({"loss": loss.item()}, step=global_step)

        if (it % config["gradient_accumulation"] == 0) or (it == len(train_loader)):

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        global_step += config["train_batch_size"]

    save_path = os.path.join(checkpoints_path, f"model_{epoch}.pth")
    torch.save(model.state_dict(), save_path)