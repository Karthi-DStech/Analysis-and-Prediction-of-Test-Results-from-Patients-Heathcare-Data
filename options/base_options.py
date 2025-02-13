import argparse
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseOptions:
    """
    This class defines options used during all types of experiments.
    """
    def __init__(self) -> None:
        """
        Initializes the BaseOptions class

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self) -> None:
        self.parser.add_argument(
            "--experiment_name",
            type=str,
            default="Analysis and Prediction of Test Results from Patients Heathcare Data",
            help="Name of the experiment",
        ),

        self.parser.add_argument(
            "--data_path",
            type=str,
            default="/Users/karthik/Desktop/Kaggle-Healthcare-Data/healthcare_dataset.csv",
            help="Path to the data file",
        )

        self.parser.add_argument(
            "--wrapper_width",
            type=int,
            default=50,
            help="Width of the wrapper",
        )
        
        self.parser.add_argument(
            "--saved_model_path",
            type=str,
            default="/Users/karthik/My-Github-Repos/ml-indus/artifacts/models/",
            help="Path to save the trained model",
        )
        
        self.parser.add_argument(
            "--log_path",
            type=str,
            default="/Users/karthik/My-Github-Repos/ml-indus/artifacts/logs/",
            help="Path to save the logs",
        )

        self.initialized = True

    def parse(self):
        """
        Parses the arguments passed to the script

        Parameters
        ----------
        None

        Returns
        -------
        opt: argparse.Namespace
            The parsed arguments
        """
        if not self.initialized:
            self.initialize()
        self._opt = self.parser.parse_args()

        return self._opt
