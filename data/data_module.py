from data.dataset import SingleVariableDataset
import os


class DatasetSetup:
    def __init__(self, exp_config,
                 inference=False,
                 require_gamma_params=False):
        self.config = exp_config['data']
        self.inference = inference
        self.require_gamma_params = require_gamma_params

    def setup(self):
        pass

    def get_train_ds(self):
        return SingleVariableDataset(
            self.config['data_path'],
            self.config['variable'],
            self.config['train']['start_date'],
            self.config['train']['end_date'],
            require_gamma_params=self.require_gamma_params,
            inference=self.inference,
            **self.config['common_kwargs']
        )

    def get_val_ds(self):
        return SingleVariableDataset(
            self.config['data_path'],
            self.config['variable'],
            self.config['val']['start_date'],
            self.config['val']['end_date'],
            train_start_date=self.config['train']['start_date'],
            train_end_date=self.config['train']['end_date'],
            require_gamma_params=self.require_gamma_params,
            **self.config['common_kwargs']
        )

    def get_test_ds(self):
        return SingleVariableDataset(
            self.config['data_path'],
            self.config['variable'],
            self.config['test']['start_date'],
            self.config['test']['end_date'],
            train_start_date=self.config['train']['start_date'],
            train_end_date=self.config['train']['end_date'],
            require_gamma_params=self.require_gamma_params,
            **self.config['common_kwargs']
        )
