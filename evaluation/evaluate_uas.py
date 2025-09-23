import os
import yaml
import pandas as pd
from evaluation.evaluate_uas_utils import plot_distributions_max, plot_div_plot, plot_temporal_evolution
from evaluation.utils import loss_display_name, plot_qq, plot_distributions, plot_event_full, plot_event_zoom, plot_power_spectra
from evaluation.evaluator import Evaluator


class EvaluatorWind(Evaluator):
    def __init__(self, experiments, config_path, var_name, comparison_title,
                 metrics=None, start_date="2020-01-01", end_date="2020-12-31", start_date_plot=None, end_date_plot=None, overwrite=False, figformat="png"):
        super().__init__(experiments, config_path, var_name, comparison_title,
                         metrics, start_date, end_date, start_date_plot, end_date_plot, overwrite, figformat)

    def draw_plots(self):
        # Draw temporal evolution, spectrum, etc.
        # Dataframe with metrics for each experiment
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df.index.name = 'Model'
        df.to_csv(os.path.join(self.plot_path,
                  'metrics_summary.csv'), float_format='%.3f')
        print(
            f"Saved {os.path.join(self.plot_path, 'metrics_summary.csv')}")

        plot_temporal_evolution(self.plot_path, self.y_preds, self.y_true,
                                self.experiments, self.exp_names, self.common_data['dates'], self.start_date_plot, self.end_date_plot, self.colors_dict, figformat=self.figformat)

        relative_hf_delta = plot_power_spectra(self.plot_path, self.y_preds, self.y_true,
                                               self.experiments, self.exp_names, self.line_styles, self.colors_dict, figformat=self.figformat, hp=30)

        plot_event_zoom(self.plot_path, self.y_preds, self.y_true,
                        self.experiments, self.exp_names, self.common_data, variable="uas", title="Wind", label=r"Zonal wind [m.s$^{-1}$]", figformat=self.figformat, zoom_extent=[15, 45, 35, 52])  # 50 -> 41.5

        plot_event_full(self.plot_path, self.y_preds, self.y_true,
                        self.experiments, self.exp_names, self.common_data, variable="uas", title="Wind", label=r"Zonal wind [m.s$^{-1}$]", figformat=self.figformat)

        plot_qq(self.plot_path, self.y_preds, self.y_true,
                self.experiments, self.exp_names, self.colors_dict, scales=["linear"], var_title="zonal wind", unit=r"[m.s$^{-1}$]", figformat=self.figformat)

        max_emd_scores = plot_distributions_max(self.plot_path, self.y_preds, self.y_true,
                                                self.experiments, self.exp_names, self.colors_dict, figformat=self.figformat)

        plot_distributions(self.plot_path, self.y_preds, self.y_true,
                           self.experiments, self.exp_names, self.results, self.colors_dict, var_title=r"Zonal wind", unit=r"m.s$^{-1}$", figformat=self.figformat)

        e_div_scores = plot_div_plot(self.plot_path, self.y_preds, self.y_true,
                                     self.experiments, self.exp_names, estimation_thresholds=[1, 2, 3], figformat=self.figformat)
        #
        # Load existing all scores if available
        if os.path.exists(self.all_scores_path):
            all_scores_df = pd.read_csv(self.all_scores_path, index_col=0)
        else:
            all_scores_df = pd.DataFrame()
        # Add new scores
        for exp in self.experiments:
            # check if all metrics columns are in all_scores_df, if not add them
            exp_name = loss_display_name(self.loss_config,  exp, weights=False)
            for col in self.results[exp].keys():
                col_ = col if col != 'wasserstein' else 'emd'
                col_ = col_.replace('_', '-')  # replace _ by -
                if col_ not in all_scores_df.columns:
                    all_scores_df[col_] = None
                all_scores_df.loc[exp_name, col_] = self.results[exp][col]
            # Add new metrics
            all_scores_df.loc[exp_name, 'max-emd'] = max_emd_scores[exp]
            for threshold in e_div_scores.keys():
                all_scores_df.loc[exp_name,
                                  f'max-under-estim-{threshold}'] = e_div_scores[threshold][exp]["under"]
                all_scores_df.loc[exp_name,
                                  f'max-over-estim-{threshold}'] = e_div_scores[threshold][exp]["over"]
            all_scores_df.loc[exp_name,
                              'spectrum-delta'] = relative_hf_delta[exp]

        # Save updated all scores
        formatted_df = all_scores_df.applymap(
            lambda x: f"{x:.3g}" if isinstance(x, float) else x)
        formatted_df.index.name = 'Training'
        formatted_df.to_csv(self.all_scores_path)
        print(f"Saved {self.all_scores_path}")


if __name__ == "__main__":
    import os
    # load comparison config
    with open('evaluation/compare_uas.yaml', 'r') as f:
        config = yaml.safe_load(f)

    var_name = config['variable']
    for comparison_title, experiments in config['comparisons'].items():

        evaluator = EvaluatorWind(
            experiments, os.path.join('configs', f'exp_config_{var_name}.yaml'), var_name, comparison_title, **config["evaluator_kwargs"])

        evaluator.plot_radar_chart(
            title=None, higher_is_better=True)
        evaluator.draw_plots()
