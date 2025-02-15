import os
from BaseDatasetLoader import BaseDatasetLoader
from typing import Tuple, Dict, List

import pandas as pd


class PostRecommendationsDataset(BaseDatasetLoader):
    """
    Dataset loader for the PostRecommendation dataset.
    """

    def __init__(self, data_path: str = "datasets/PostRecommendations"):
        super().__init__(data_path)
        self.userData_file = f"{self.data_path}/user_data.csv"
        self.movies_file = f"{self.data_path}/post_data.csv"

        self.PostRecommendationPathTest = "datasets/PostRecommendations/testDataset"
        self.PostRecommendationPathTrain = "datasets/PostRecommendations/trainDataset"

    def load_userData(self) -> pd.DataFrame:
        userData_df = pd.read_csv(self.userData_file)
        # userData_df.rename(
        #     columns={"userId": "user_id", "movieId": "item_id"}, inplace=True
        # )
        return userData_df[["user_id", "first_name", "city"]]  # Select only necessary columns

# To zostaje - split na test data i train data więc uniwersalne do każdego dataeu
    def get_train_test_split(self, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        userData_df = self.load_userData()
        train_df = userData_df.sample(frac=1 - test_size, random_state=seed)
        test_df = userData_df.drop(train_df.index)
        
        save_path_train = self.PostRecommendationPathTrain
        save_path_test = self.PostRecommendationPathTest

        train_file = os.path.join(save_path_train, "train.csv")
        test_file = os.path.join(save_path_test, "test.csv")

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        return train_df, test_df