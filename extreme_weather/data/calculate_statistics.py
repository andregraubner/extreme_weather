from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet, CGNetModule
from climatenet.utils.utils import Config
from climatenet.track_events import track_events
from climatenet.analyze_events import analyze_events
from climatenet.visualize_events import visualize_events
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass

from os import path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from os import listdir, path
import xarray as xr
from climatenet.utils.utils import Config
import pandas as pd
import torch
import numpy as np
import torchvision

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import glob
from dask import delayed

from dask.utils import SerializableLock
from dask.diagnostics import ProgressBar


root_path = "/net/pf-pc69/"
data_path = f"{root_path}/scratch/andregr/dataset" 
  
chunk_1 = xr.open_dataset(f"{root_path}/scratch/lukaska/data/chunks/chunk_random_1980_1_1_00:00-2023_1_1_00:00_samples_5000_02.nc", chunks={'time': 10})
chunk_2 = xr.open_dataset(f"{root_path}/scratch/lukaska/data/chunks/chunk_random_1980_1_1_00:00-2023_1_1_00:00_samples_5000_01.nc", chunks={'time': 10})

data = xr.concat([chunk_1, chunk_2], dim='time').sortby("time")
data = data.isel(time=slice(0,5000))

print("mean")
mean = data.mean(dim=['time', 'latitude', 'longitude'])
print("std")
std = data.std(dim=['time', 'latitude', 'longitude'])

mean.to_netcdf(path.join(data_path, f"mean.nc"))
std.to_netcdf(path.join(data_path, f"std.nc"))