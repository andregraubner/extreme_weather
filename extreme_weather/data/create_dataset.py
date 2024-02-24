from download_data import download_ar_tc
import xarray as xr
import pandas as pd

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

# download data:
chunk = xr.open_mfdataset("data_temp/*.nc", concat_dim="time", combine='nested')
dates = pd.Series(chunk.time.values)

download_ar_tc(dates[:3], "features.nc")

features = xr.open_dataset("features.nc")

for ts in chunk.time.values:

    labels = ds.sel(time=ts)
    features = data.sel(time=ts)

    full = xr.merge([features, labels])

    comp = dict(zlib=True, complevel=5)
    encoding = {"label": comp}

    full.to_netcdf(path.join("data", f"{ts}.nc"), encoding=encoding)