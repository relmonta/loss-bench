import os
import yaml
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import cycle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from models.vision_transformer import VisionTransformer
from models.losses import get_criterion
from data.data_module import DatasetSetup
from evaluation.utils import inference_mode, COLORS_CB_FRIENDLY, loss_display_name
import pickle
from training.utils import inference_step


class Evaluator:
    def __init__(self, experiments, config_path, var_name, comparison_title,
                 metrics=None, start_date="2020-01-01", end_date="2020-12-31", start_date_plot=None, end_date_plot=None, overwrite=False, figformat="png"):
        """
        Initialize the Evaluator class.

        Args:
            experiments (list): List of experiment names to evaluate.
            config_path (str): Path to the YAML configuration file containing training config.
            metrics (list, optional): List of metrics to evaluate. Defaults to None, which uses metrics from config.
            folder (str, optional): Folder name to save plots. Defaults to None.    
            start_date (str, optional): Start date for evaluation period. Defaults to "2020-01-01".
            end_date (str, optional): End date for evaluation period. Defaults to "2020-12-31".
            overwrite (bool, optional): Whether to overwrite existing results. Defaults to False.
            figformat (str, optional): Format for saving figures. Defaults to "png".
        """
        self.experiments = experiments
        self.config_path = config_path
        self.var_name = var_name
        self.config = self._load_yaml_config(config_path)['training']
        self.loss_config = self._load_yaml_config(
            self.config['loss_config_path'])
        self.metrics = metrics or self.config['metrics']
        self.plot_path = os.path.join(
            self.config['plot_path'], figformat, f"{var_name}/{comparison_title}")
        os.makedirs(self.plot_path, exist_ok=True)
        self.start_date = start_date
        self.end_date = end_date
        self.start_date_plot = start_date_plot or self.start_date
        self.end_date_plot = end_date_plot or self.end_date
        self.inference_res_path = os.path.join(
            self.config['inference_res_path'], var_name, f"start_{self.start_date}_end_{self.end_date}")
        self.all_scores_path = os.path.join(
            self.config['plot_path'], figformat, f"{var_name}/all_scores.csv")

        self.line_styles = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": (0, (2, 1, 1, 1)),
            "dense_dashdotdot": (0, (3, 1, 1, 1, 1, 1)),
            "dashdotdotted": (0, (3, 1, 1, 1, 1, 3)),
            "dashdotdot": (0, (3, 2, 1, 2)),
            "dense_dashes": (0, (3, 1)),
            "dash_space_dot": (0, (5, 2, 1, 2)),
            "loosely_dashed": (0, (5, 10))
        }
        self.colors_dict = {"truth": "gray"}
        for i, exp in enumerate(self.experiments):
            self.colors_dict[exp] = COLORS_CB_FRIENDLY[i]

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device".upper())
        print("-----------------------------------------------------")
        print(f"Comparison title: {comparison_title.upper()}")
        print(f"Variable: {var_name.upper()}")
        print("-----------------------------------------------------")
        # Check if inference results already exist
        self.results = {}
        self.y_preds = {}
        self.y_true = None
        self.common_data = None
        self.exp_names = {"truth": "Ground Truth"}
        for exp in self.experiments:
            if overwrite or not os.path.exists(os.path.join(
                    self.inference_res_path, exp, f'results.pkl')):
                if overwrite:
                    print(
                        f"Overwriting existing results for experiment: {exp}")
                print(f"Evaluating experiment: {exp}")
                self.evaluate_model(exp)
            else:
                print(
                    f"Results for experiment {exp} already exist. Skipping evaluation.")

            # Load predictions and ground truth
            self.y_preds[exp] = torch.load(os.path.join(
                self.inference_res_path, exp, f'y_pred.pt'), weights_only=False)
            if self.y_true is None:
                self.y_true = torch.load(os.path.join(
                    self.inference_res_path, f'y_true.pt'), weights_only=False)
            # Load results dictionary and common data
            with open(os.path.join(self.inference_res_path, exp, f'results.pkl'), 'rb') as f:
                self.results[exp] = pickle.load(f)
            if self.common_data is None:
                with open(os.path.join(self.inference_res_path, f'common_data.pkl'), 'rb') as f:
                    self.common_data = pickle.load(f)

            self.exp_names[exp] = loss_display_name(self.loss_config,  exp)
            print("---")
            self.figformat = figformat

    def evaluate_model(self, exp):
        exp_config = self._load_yaml_config(self.config_path)
        # Dataset configuration tweaks per experiment
        if "nllbg" in exp:
            exp_config['data']['kwargs_train_val']['normalize'] = False
            exp_config['data']['kwargs_train_val']['standardize'] = False

        exp_config['data']['kwargs_train_val']['apply_log'] = exp.startswith(
            "log_")

        # Limit the time range to the specified period
        exp_config['data']['val']['start_date'] = self.start_date
        exp_config['data']['val']['end_date'] = self.end_date

        dss = DatasetSetup(exp_config, inference=True)
        dss.setup()

        val_dataset = dss.get_val_ds()
        train_dataset = dss.get_train_ds()

        val_dataloader = DataLoader(val_dataset, batch_size=16)

        ckpt_file = os.path.join(exp_config["training"]["weights_path"],
                                 exp_config["name"], f"weights-{exp}.ckpt")
        if not os.path.exists(ckpt_file):
            # Download from zenodo
            download_weights_from_zenodo(self.var_name, file_name=f"weights-{exp}.ckpt", save_path=ckpt_file)

        state_dict = torch.load(ckpt_file, weights_only=True, map_location=self.device)[
            'state_dict']
        # Remove "model." prefix and filter out criterion keys
        state_dict = {k.replace(
            "model.", "", 1): v for k, v in state_dict.items() if "criterion." not in k}
        model_args = exp_config["model"]["params"]
        if self.var_name == "pr":
            model_args["bernoulli_gamma"] = "nllbg" in exp

        model = VisionTransformer(**model_args).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        metrics_res, y_pred, y_true, dates = self.inference_on_val_data(
            model, val_dataloader, train_dataset)

        common_data = {
            "longitudes": train_dataset.longitudes,
            "latitudes": train_dataset.latitudes,
            "extent": train_dataset.extent,
            "dates": dates
        }
        if self.var_name == "pr":
            common_data["rx1day_true"] = np.max(y_true.numpy())
        # Save inference results
        os.makedirs(os.path.join(self.inference_res_path, exp), exist_ok=True)
        torch.save(y_pred, os.path.join(
            self.inference_res_path, exp, f'y_pred.pt'))
        # Save the results dictionary using pickle
        with open(os.path.join(
                self.inference_res_path, exp, f'results.pkl'), 'wb') as f:
            pickle.dump(metrics_res, f)
        # Save ground truth and dates for reference
        if not os.path.exists(os.path.join(
                self.inference_res_path, f'y_true.pt')):
            torch.save(y_true, os.path.join(
                self.inference_res_path, f'y_true.pt'))
        # Save common data if not already saved
        if not os.path.exists(os.path.join(
                self.inference_res_path, f'common_data.pkl')):
            with open(os.path.join(
                    self.inference_res_path, f'common_data.pkl'), 'wb') as f:
                pickle.dump(common_data, f)

    def inference_on_val_data(self, model, val_dataloader, train_dataset):
        validation_items = {'x': [], 'y_pred': [], 'y_true': [], 'dates': []}
        with inference_mode("Inference"):
            for batch in tqdm(val_dataloader, desc="Validation", colour="green"):
                low_res, high_res, outputs, dates = inference_step(
                    batch, model, train_dataset, self.device, dates=True)
                validation_items['y_pred'].append(outputs.detach().cpu())
                validation_items['y_true'].append(high_res.detach().cpu())
                validation_items['x'].append(low_res.detach().cpu())
                validation_items['dates'].append(dates.detach().cpu())

        y_pred = torch.cat(validation_items['y_pred'], dim=0)
        y_true = torch.cat(validation_items['y_true'], dim=0)
        dates = torch.cat(validation_items['dates'], dim=0)
        while len(dates.shape) > 2:
            dates = dates.squeeze(1)
        dates = np.array([np.datetime64(datetime(year, month, day))
                          for year, month, day, *_ in dates])

        # Convert to pandas datetime
        dates = pd.to_datetime(dates, format="%Y-%m-%d")
        metrics_list = ['mae', 'mse', 'ssim', 'gdl',
                        'spectral', 'wavelet', 'wavelet_complex', 'wasserstein']
        metrics_res = {name: get_criterion(
            name, **self.loss_config['losses'].get(name, {}) or {})(y_pred, y_true).item() for name in metrics_list}

        if self.var_name == "pr":
            metrics_res["rx1day"] = np.max(y_pred.numpy())
        return metrics_res, y_pred.squeeze(1), y_true.squeeze(1), dates

    def _load_yaml_config(self, yaml_config):
        """Load YAML configuration."""
        with open(yaml_config, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def get_scores_matrix(self):
        score_matrix = np.zeros(
            (len(self.experiments), len(self.metrics)))

        for i, exp in tqdm(enumerate(self.experiments), desc="Getting scores matrix"):
            for j, metric in enumerate(self.metrics):
                score_matrix[i, j] = self.results[exp][metric]
        return score_matrix

    def plot_radar_chart(self, title, higher_is_better=False, min_radius=0.25):
        """
        Plots the radar chart comparing model performance for each experiment.
        Ensures the weakest score is not zero by shifting the normalized values.
        """
        # Convert scores to array: shape (num_experiments, num_metrics)
        score_matrix = self.get_scores_matrix()

        # Per-metric (column-wise) normalisation
        min_vals = np.nanmin(score_matrix, axis=0)
        max_vals = np.nanmax(score_matrix, axis=0)

        normalized_matrix = (score_matrix - min_vals) / \
            (max_vals - min_vals + 1e-8)

        if higher_is_better:
            normalized_matrix = 1 - normalized_matrix

        # Shift so the weakest score corresponds to min_radius
        normalized_matrix = normalized_matrix * (1.0 - min_radius) + min_radius

        # Radar chart angles
        num_metrics = len(self.metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics,
                             endpoint=False).tolist()
        angles += angles[:1]  # Ensure the chart closes
        line_styles = cycle(self.line_styles.values())
        # Plot setup
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.spines['polar'].set_color("teal")     # outer circle border
        # ax.spines['polar'].set_linewidth(2)

        ax.grid(color="teal")
        ax.tick_params(colors="teal")            # tick labels

        # Plot each experiment
        for i, exp in tqdm(enumerate(self.experiments), desc="Plotting radar chart"):
            values = normalized_matrix[i].tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, label=self.exp_names[exp],
                    linewidth=3,
                    color=self.colors_dict.get(exp, None),
                    # marker='.',
                    linestyle=next(line_styles)
                    )
            ax.fill(angles, values, alpha=0.1,
                    color=self.colors_dict.get(exp, None))

        ax.set_title(title, fontsize=16)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12)

        # Axis setup
        # Axis setup
        metrics_labels = [loss_display_name(self.loss_config,  metric)
                          for metric in self.metrics]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels, fontsize=14, fontweight='bold')
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels([])  # '0.25', '0.5', '0.75'], fontsize=12)
        ax.set_ylim(0, 1)

        # Improve label rotation for readability
        for label, angle in zip(ax.get_xticklabels(), angles):
            angle_deg = np.degrees(angle)
            if 90 <= angle_deg <= 270:
                label.set_rotation(angle_deg + 180)
                label.set_verticalalignment('center')
                label.set_horizontalalignment('right')
            else:
                label.set_rotation(angle_deg)
                label.set_verticalalignment('center')
                label.set_horizontalalignment('left')

        # Title and legend
        ax.set_title(title, size=18, pad=30,
                     fontweight='bold', color='darkblue')

        # Add legends to the bottom, outside the plot
        ncols = 2
        to_anchor_y = -0.3 - (0.1 * len(self.experiments) // ncols)
        legend = ax.legend(loc='lower center', fontsize=14, frameon=True,
                           framealpha=0.7, ncols=ncols, bbox_to_anchor=(0.5, to_anchor_y))
        legend.get_frame().set_edgecolor("teal")
        # Final layout adjustments
        plt.tight_layout()

        plt.savefig(os.path.join(
            self.plot_path, f'val_radar.pdf'), bbox_inches='tight')
        plt.close()
