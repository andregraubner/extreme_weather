from os import path
import xarray as xr
import torch
import glob

#TODO add script to download data

class ExtremeEventDataset(torch.utils.data.Dataset):
  
    def __init__(self, data_path: str, split_fraction=0.9, split="train", merge_annotations=True):

        self.data_path: str = data_path
        self.merge_annotations: bool = merge_annotations

        self.files = glob.glob(path.join(self.data_path, "full",  "*.nc"))

        self.mean = xr.load_dataset(path.join(data_path, f"mean.nc"))
        self.std = xr.load_dataset(path.join(data_path, f"std.nc"))

        split_number = int(len(self.files)*split_fraction)
        if split == "train":
            self.files = self.files[:split_number]
            print(f"Using {len(self.files)} for training.")
        elif split == "test":
            self.files = self.files[split_number:]
            print(f"Using {len(self.files)} for testing.")
        else:
            raise Exception

        # we have one file per time step,
        # and two annotations per file
        self.length = len(self.files)
        if not self.merge_annotations:
            self.length *= 2

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):                            
            
            if self.merge_annotations:
                ds = xr.load_dataset(self.files[idx])

                features = ds[['msl', 'tcwv', 'p72.162', 'p71.162']]
                labels = ds['label'].mean(dim="annotator")

            else:
                file_idx = idx // 2
                annotation_idx = idx % 2

                ds = xr.load_dataset(self.files[file_idx])

                # TODO: Potentially correct features (correct IVT)
                features = ds[['msl', 'tcwv', 'p72.162', 'p71.162']]
                labels = ds['label'].sel(annotator=annotation_idx)


            # Normalize data
            features -= self.mean
            features /= self.std

            features = features.fillna(self.mean)

            features = features.to_array().values
            features = torch.tensor(features, dtype=torch.float32)

            labels = labels.values
            labels = torch.tensor(labels, dtype=torch.float32)

            if self.merge_annotations:
                # if we're not training the diffusion model, we want an output channel for the 'background' class
                mask = (labels.sum(0) == 0)[None]
                labels = torch.cat([mask, labels], dim=0)
            
            labels = labels[:, :720]
            features = features[:, :720]

            return features, labels

    @staticmethod 
    def collate(batch):
        data, labels = map(list, zip(*batch))
        return torch.stack(data), torch.stack(labels)