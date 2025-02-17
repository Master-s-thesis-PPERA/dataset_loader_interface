import os
import numpy as np
import pandas as pd

from BaseDatasetLoader import BaseDatasetLoader
from typing import Tuple, Dict, List, Union




class AmazonSalesDataset(BaseDatasetLoader):
    """
    Dataset loader for the Amazon Sales dataset.
    """

    def __init__(self, data_path: str = "datasets/AmazonSales"):
        super().__init__(data_path)
        self.dataset_file = f"{self.data_path}/amazon.csv"
        self.AmazonSalesPathTest = "datasets/AmazonSales/testDataset"
        self.AmazonSalesPathTrain = "datasets/AmazonSales/trainDataset"

    # def load_dataset(self) -> pd.DataFrame:
    #     dataset_df = pd.read_csv(self.dataset_file)
    #     return dataset_df
    
    def load_dataset_useful_columns(self) -> pd.DataFrame:
        dataset_df = pd.read_csv(self.dataset_file)
        return dataset_df[["product_id", "product_name", "category", "rating", "user_id", "review_content"]]

# To zostaje - split na test data i train data więc uniwersalne do każdego dataeu
    def get_train_test_split(self, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataset_df = self.load_dataset()
        train_df = dataset_df.sample(frac=1 - test_size, random_state=seed)
        test_df = dataset_df.drop(train_df.index)
        
        save_path_train = self.AmazonSalesPathTrain
        save_path_test = self.AmazonSalesPathTest

        train_file = os.path.join(save_path_train, "train.csv")
        test_file = os.path.join(save_path_test, "test.csv")

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        return train_df, test_df
    
    def hide_information(self, data: pd.DataFrame,  # Add 'self' here
                     hide_type: str = "columns",
                     columns_to_hide: Union[str, List[str]] = None,
                     fraction_to_hide: float = 0.0,
                     records_to_hide: List[int] = None,
                     seed: int = 42) -> pd.DataFrame:

        df = data.copy()
        np.random.seed(seed)

        if hide_type == "columns":
            if columns_to_hide is None:
                raise ValueError("Must specify 'columns_to_hide' for hide_type='columns'.")
            if isinstance(columns_to_hide, str):
                columns_to_hide = [columns_to_hide]
            if not all(col in df.columns for col in columns_to_hide):
                raise ValueError("One or more 'columns_to_hide' not found in DataFrame.")
            df = df.drop(columns=columns_to_hide)

        elif hide_type == "records_random":
            if not 0.0 <= fraction_to_hide <= 1.0:
                raise ValueError("'fraction_to_hide' must be between 0.0 and 1.0.")
            num_to_hide = int(len(df) * fraction_to_hide)
            indices_to_hide = np.random.choice(df.index, size=num_to_hide, replace=False)
            df = df.drop(index=indices_to_hide)

        elif hide_type == "records_selective":
            if records_to_hide is None:
                raise ValueError("Must specify 'records_to_hide' for hide_type='records_selective'.")
            if not all(idx in df.index for idx in records_to_hide):
                raise ValueError("One or more 'records_to_hide' indices not found in DataFrame.")
            df = df.drop(index=records_to_hide)

        elif hide_type == "values_in_column":
            if columns_to_hide is None:
                raise ValueError("Must specify 'columns_to_hide' for hide_type='values_in_column'.")
            if not 0.0 <= fraction_to_hide <= 1.0:
                raise ValueError("'fraction_to_hide' must be between 0.0 and 1.0.")
            if isinstance(columns_to_hide, str):
                columns_to_hide = [columns_to_hide]
            if not all(col in df.columns for col in columns_to_hide):
                raise ValueError("One or more 'columns_to_hide' not found in DataFrame.")

            for col in columns_to_hide:
                num_to_hide = int(len(df) * fraction_to_hide)
                indices_to_hide = np.random.choice(df.index, size=num_to_hide, replace=False)
                df.loc[indices_to_hide, col] = np.nan
        else:
            raise ValueError(f"Invalid 'hide_type': {hide_type}")

        return df