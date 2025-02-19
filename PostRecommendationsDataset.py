import os
from typing import List, Optional
import pandas as pd

from BaseDatasetLoader import BaseDatasetLoader





class PostRecommendationsDataset(BaseDatasetLoader):
    """
    Dataset loader for the PostRecommendation dataset.
    """

    def __init__(self, data_path: str = "datasets/PostRecommendations"):
        super().__init__(data_path)
        self.userData_file = f"{self.data_path}/user_data.csv"
        self.viewData_file = f"{self.data_path}/view_data.csv"
        self.postData_file = f"{self.data_path}/post_data.csv"

        self.merge_file = f"{self.data_path}/merge_file.csv"

        self.test_path = "datasets/PostRecommendations/testDataset"
        self.train_path = "datasets/PostRecommendations/trainDataset"

    def merge_datasets(self) -> pd.DataFrame:
        user_df = pd.read_csv(self.userData_file)
        view_df = pd.read_csv(self.viewData_file)
        post_df = pd.read_csv(self.postData_file)

        merge_file = pd.merge(user_df, view_df, on="user_id", how="left")
        final_merge_file = pd.merge(merge_file, post_df, on="post_id", how="left")
        final_merge_file.to_csv(self.merge_file, index=False)
        return final_merge_file