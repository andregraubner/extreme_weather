
# Advancing the Study of Extreme Weather Events with Data, Deep Learning Methods and Climate Analysis

## How to get the data?
In order to run model training or to simply play around with the dataset yourself, download our annotations from [here](https://polybox.ethz.ch/index.php/s/nBr0t1cuZM6SrgM).
This only includes the label masks for each time step. 
To download and pre-process the ERA5 features that are required for model training, run the following scripts:

    extreme_weather/data/create_dataset.py 
    extreme_weather/data/calculate_statistics.py

These scripts download the corresponding features from the ERA5 web-api, combine everything into a unified file format and pre-calculates per-feature means and standard deviations for data normalisation.

## How to train your own models?
Now, in order to train a model, we can use 

    train.py
   or
   

    train_diffusion.py