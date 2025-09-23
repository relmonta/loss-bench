import os
import yaml
import torch
import pandas as pd
import xarray as xr
import numpy as np
from scipy.stats import gamma
from torch.utils.data import Dataset
from torch.nn import functional as F
from tqdm import tqdm
import pickle
import fcntl
import matplotlib.pyplot as plt
from typing import List
import warnings
import gc

def get_netcdf(path, var_name, year):
    """Loads a single NetCDF file from the local directory or the archive."""
    local_path = os.path.join(path, f"{var_name}_1d/{var_name}_1d_{year}_ERA5.nc")
    if not os.path.exists(local_path):
        print(f"File {local_path} not found. Try downloading from zenodo...")
        # zenodo url
        zenodo_url = f"https://zenodo.org/records/17098120/files/{var_name}_1d_{year}_ERA5.nc"
        # try to download the file using wget
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            os.system(f"wget -O {local_path} {zenodo_url}")
        except Exception as e:
            print(f"Error downloading {zenodo_url} using wget: {e}")
            print("Trying using urllib ... If this fails or take too long, please download the file manually from zenodo.") 
            try:
                import urllib.request as requests
                requests.urlretrieve(zenodo_url, local_path)
            except Exception as e:
                raise FileNotFoundError(f"File {local_path} not found and download failed: {e}")
    ds = xr.open_dataset(local_path)

    # Adjust longitude coordinates to [-180, 180] if needed
    lon = ds.longitude.values
    if np.all(lon >= 0) and np.all(lon <= 360):
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby('longitude')
    return ds


class SingleVariableDataset(Dataset):
    """
    PyTorch Dataset for downscaling single ERA5 variable.

    Args:
        data_path (str): Path to the ERA5 data directory.
        variable (list): The variabl name to load (e.g., "uas", "vas").
        start_date (str): Start date of the dataset, format 'YYYY-MM-DD'.
        end_date (str): End date of the dataset, format 'YYYY-MM-DD'.
        frequency (str): Temporal frequency ('D' = daily, 'M' = monthly, 'H' = hourly).
        extent (list): Geographic extent [lon_min, lon_max, lat_min, lat_max].
        normalize (bool): Whether to normalize inputs.
        path_exp (str): Path to experiment directory.
    """

    def __init__(self, data_path, variable: str, start_date: str,
                 end_date: str, frequency: str = 'D', extent: List[float] = [-19, 45, 17, 81],
                 normalize: bool = False, train_start_date: str = None, train_end_date: str = None,
                 stats_file: str = None, downscaling_factor: int = 4, standardize: bool = False,
                 inference: bool = False, output_shape=(256, 256),
                 apply_log: bool = False, require_gamma_params: bool = False):
        """
        Initialize the dataset.
        """
        super(SingleVariableDataset, self).__init__()
        self.data_path = data_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.apply_log_flag = apply_log
        self.variable = variable
        self.require_gamma_params = require_gamma_params
        self.era5_info_path = 'data/era5_variables.yaml'

        # Training period details (usefull for validation and test datasets)
        self.train = train_start_date == start_date if train_start_date else True
        # if inference, do not load train datasets
        self.inference = inference
        self.train_start_date = pd.to_datetime(
            train_start_date) if not self.train else self.start_date
        self.train_end_date = pd.to_datetime(
            train_end_date) if not self.train else self.end_date
        self.train_num_years = self.train_end_date.year - self.train_start_date.year + 1

        self.frequency = frequency
        self.extent = extent
        self.normalize = normalize
        self.downscaling_factor = downscaling_factor  # Downscaling factor
        self.standardize = standardize
        self.interpolation_mode = 'nearest'

        self.date_index = pd.date_range(
            self.start_date, self.end_date, freq=self.frequency)
        self.formats = {"H": "%Y-%m-%d %H:%M",
                        "D": "%Y-%m-%d", "M": "%Y-%m", "Y": "%Y"}
        self.date_index = self.date_index.strftime(
            self.formats.get(self.frequency, "%Y-%m-%d"))

        self.stats_file = stats_file
        self.normalization_stats = self._load_normalization_stats_()
        self.stats_updated = False
        # Normalisation parameters
        self._load_era5_data_()

        self.output_shape = output_shape
        self.done = False

        del self.normalization_stats
        gc.collect()
        # select extent in lat and lon
        if hasattr(self, 'latitudes') and hasattr(self, 'longitudes'):
            self.latitudes = self.latitudes[(self.latitudes >= extent[2]) & (
                self.latitudes <= extent[3])][:-1]
            self.longitudes = self.longitudes[(self.longitudes >= extent[0]) & (
                self.longitudes <= extent[1])][:-1]

    def __len__(self):
        return len(self.date_index)

    def _load_era5_data_(self) -> dict:
        """Load ERA5 data for the specified variables."""

        with open(self.era5_info_path) as f:
            era5_vars = yaml.safe_load(f)
        var_name = self.variable
        self.data = {}
        self.mean_x, self.std_x = 0, 0
        self.var_details = {}

        if not self.inference: # self.inference concerns only the training dataset
            print(
                f"Loading {self.variable.upper()} data from {self.start_date.year} to {self.end_date.year}...")
            for year in tqdm(range(self.start_date.year, self.end_date.year + 1), desc="Loading data", colour="blue"):
                ds = get_netcdf(self.data_path, var_name, year)
                self.latitudes = ds.latitude.values
                self.longitudes = ds.longitude.values
                # apply log(x + 1)
                if self.apply_log_flag:
                    ds[var_name] = self._apply_log_(ds[var_name])

                if self.train:  # Train mode
                    self.__update_stats__(ds, var_name, year)

                ds = ds.sel(longitude=slice(
                    self.extent[0], self.extent[1]), latitude=slice(self.extent[3], self.extent[2]))
                self.data[year] = ds

            # Save the updated stats to the pickle file for future use
            if self.stats_updated:
                print("Saving updated normalization stats...")
                with open(self.stats_file, "wb") as f:
                    pickle.dump(self.normalization_stats, f)

        # In train, validation or test mode, always use stats from training period
        var_name_ = var_name + "_log" if self.apply_log_flag else var_name
        for year in range(self.train_start_date.year, self.train_end_date.year + 1):
            self.mean_x += self.normalization_stats[var_name_][year]["mean"]
            self.std_x += self.normalization_stats[var_name_][year]["std"]
        self.mean_x /= self.train_num_years
        self.std_x /= self.train_num_years

        # Variable details
        self.var_details['name'] = var_name
        self.var_details['log'] = self.apply_log_flag
        self.var_details['title'] = era5_vars[var_name+'_1d']['short']
        self.var_details['explanation'] = era5_vars[var_name+'_1d']['long']
        self.var_details['unit'] = era5_vars[var_name+'_1d']['unit']

        # Min and max values
        self.vmin = None
        self.vmax = None

        # Use the same mean and std for target variables
        self.mean_y, self.std_y = self.mean_x, self.std_x

    def _load_normalization_stats_(self) -> dict:
        """Load or compute normalization stats for ERA5 data."""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, "rb") as f:
                # Acquire an exclusive lock before reading
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    data = pickle.load(f)
                finally:
                    # Release the lock
                    fcntl.flock(f, fcntl.LOCK_UN)
                return data
        return {}

    def __update_stats__(self, ds, var_name, year):
        # Determine variable name (log or raw)
        var_name_ = var_name + "_log" if self.apply_log_flag else var_name

        if var_name_ not in self.normalization_stats:
            self.normalization_stats[var_name_] = {}
            self.stats_updated = True

        if year not in self.normalization_stats[var_name_]:
            # Compute mean and std
            mean_val = ds[var_name].mean().values.item()
            std_val = ds[var_name].std().values.item()

            stats_dict = {"mean": mean_val, "std": std_val}
            self.stats_updated = True
        else:
            stats_dict = self.normalization_stats[var_name_][year]

        # If the variable is precipitation, compute alpha and beta for rainy days (>1 mm)
        if var_name == "pr" and not self.inference and self.require_gamma_params:
            if "alpha" not in stats_dict:
                print("Gamma parameters not found in stats; This may take a while...")
                values = ds["pr"].values
                # Compute Gamma parameters per grid point (rainy days > 1 mm)
                time, lat, lon = values.shape
                alpha_grid = np.full((lat, lon), np.nan)
                beta_grid = np.full((lat, lon), np.nan)

                for i in range(lat):
                    for j in range(lon):
                        rainy_days = values[:, i, j][values[:, i, j] > 1.0]
                        # print(" num rainy_days:", len(rainy_days))
                        if len(rainy_days) == 2:
                            # Use method of moments
                            mean = np.mean(rainy_days)
                            var = np.var(rainy_days)
                            alpha = mean**2 / var if var > 0 else 1.0
                            beta = var / mean if mean > 0 else 1.0
                            alpha_grid[i, j] = alpha
                            beta_grid[i, j] = beta
                        elif len(rainy_days) == 1:
                            # Only one sample → degenerate case
                            alpha_grid[i, j] = 1.0  # Exponential distribution
                            # Scale parameter equals the observed value
                            beta_grid[i, j] = rainy_days[0]
                        elif len(rainy_days) == 0:
                            alpha_grid[i, j] = 1.0   # Exponential distribution
                            # Arbitrary scale (won’t matter much since no rain)
                            beta_grid[i, j] = 1.0
                        else:
                            try:
                                # Fit Gamma distribution with loc=0
                                alpha, loc, beta = gamma.fit(
                                    rainy_days, floc=0)
                                alpha_grid[i, j] = alpha
                                beta_grid[i, j] = beta
                            except Exception as e:
                                print(
                                    f"Gamma fit failed at pixel ({i},{j}): {e}")
                                alpha_grid[i, j] = 1
                                beta_grid[i, j] = 1
                stats_dict.update({"alpha": alpha_grid, "beta": beta_grid})
                self.stats_updated = True
        # Store computed values
        self.normalization_stats[var_name_][year] = stats_dict

    def denormalize_x(self, x):
        assert len(x.shape) == 4, f"Expected 4D tensor, given {len(x.shape)}D"
        if self.normalize:
            x[:, 0] = x[:, 0] * self.std_x + self.mean_x
        # Apply the inverse of log(x + 1) to get rhe the physical space
        x[:, 0] = self._apply_exp_(x[:, 0])
        return x

    def denormalize_y(self, x):
        assert len(x.shape) == 4, f"Expected 4D tensor, given {len(x.shape)}D"
        if self.normalize:
            x[:, 0] = x[:, 0] * self.std_y + self.mean_y
        x[:, 0] = self._apply_exp_(x[:, 0])
        return x

    def _apply_exp_(self, x):
        if self.apply_log_flag:
            return torch.expm1(x)
        return x

    def _apply_log_(self, x):
        if self.apply_log_flag:
            if isinstance(x, torch.Tensor):
                return torch.log1p(x)
            elif isinstance(x, xr.DataArray):
                return np.log1p(x)
            else:
                raise ValueError(
                    "Unsupported type for x. Expected torch.Tensor or xr.DataArray.")
        return x

    def _precompute_static_variables_(self, h, w):
        # Create longitude and latitude grids efficiently
        self.lon_grid = torch.linspace(
            self.extent[0], self.extent[1], w).repeat(h, 1).unsqueeze(0)
        self.lat_grid = torch.linspace(self.extent[2], self.extent[3], h).unsqueeze(
            1).repeat(1, w).unsqueeze(0)
        self.static_vars = torch.cat([self.lat_grid, self.lon_grid], dim=0)
        self.done = True

    def upscaling(self, y):
        x = F.avg_pool2d(y, self.downscaling_factor)
        x = F.interpolate(
            x, size=y.shape[-2:], mode=self.interpolation_mode)
        return x

    def __getitem__(self, idx, date=None):
        if date is not None:
            date = pd.to_datetime(date).strftime(
                self.formats.get(self.frequency, "%Y-%m-%d"))
            # if idx != 0 get date + idx
            if idx != 0:
                date_idx = self.date_index.get_loc(date) + idx
                date = self.date_index[date_idx]
        else:
            date = self.date_index[idx]
        year = pd.to_datetime(date).year

        y = self.data[year][self.variable]
        y = y.sel(time=date)
        y = y.sum(
            dim="time").values[:-1, :-1] if 'time' in y.dims else y.values[:-1, :-1]

        # Get low resolution data
        y = torch.tensor(y).unsqueeze(0).unsqueeze(0).float()
        orig_y = y.squeeze(0).clone()
        x = self.upscaling(y)

        # Normalisation
        if self.normalize:
            x = (x - self.mean_x) / self.std_x
            y = (y - self.mean_y) / self.std_y

        if not self.done:  # Compute spatial coordinate grids
            self._precompute_static_variables_(*y.shape[-2:])

        time_features = torch.tensor([pd.to_datetime(date).year,
                                      pd.to_datetime(date).month,
                                      pd.to_datetime(date).day,
                                      pd.to_datetime(date).hour]).unsqueeze(0)
        x = x.squeeze(0)
        y = y.squeeze(0)
        if self.standardize:
            # Standardize input and output data
            mean_x = x.mean(dim=(-2, -1), keepdim=True)
            std_x = x.std(dim=(-2, -1), keepdim=True)
            x = (x - mean_x) / std_x
            y = (y - mean_x) / std_x
            # Return means and stds for denormalization, for validation and test sets
            return x, self.static_vars, time_features, y, mean_x, std_x, orig_y

        return x, self.static_vars, time_features, y
