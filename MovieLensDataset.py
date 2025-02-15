import os
from BaseDatasetLoader import BaseDatasetLoader
from typing import Tuple, Dict, List

import pandas as pd


class MovieLensDataset(BaseDatasetLoader):
    """
    Dataset loader for the MovieLens 20M dataset.
    """

    def __init__(self, data_path: str = "datasets/MovieLens"):
        super().__init__(data_path)
        self.ratings_file = f"{self.data_path}/rating.csv"
        self.movies_file = f"{self.data_path}/movie.csv"

        self.MovieLensPathTest = "datasets/MovieLens/testDataset"
        self.MovieLensPathTrain = "datasets/MovieLens/trainDataset"

    def load_ratings(self) -> pd.DataFrame:
        ratings_df = pd.read_csv(self.ratings_file)
        ratings_df.rename(
            columns={"userId": "user_id", "movieId": "item_id"}, inplace=True
        )
        return ratings_df[["user_id", "item_id", "rating"]]  # Select only necessary columns

    def load_item_features(self) -> pd.DataFrame:
        movies_df = pd.read_csv(self.movies_file)
        movies_df.rename(columns={"movieId": "item_id"}, inplace=True)
        # Convert genres to a list of genres
        movies_df["genres"] = movies_df["genres"].str.split("|")
        return movies_df[["item_id", "genres", "title"]] # Include title

# Jak dany użytkownik ocenił poszczególne filmy
    def get_user_item_interactions(self) -> Dict[int, List[Tuple[int, float]]]:
        ratings_df = self.load_ratings()
        interactions = {}
        for user_id, group in ratings_df.groupby("user_id"):
            interactions[user_id] = list(zip(group["item_id"], group["rating"]))
        return interactions


# To zostaje - split na test data i train data więc uniwersalne do każdego dataeu
    def get_train_test_split(self, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ratings_df = self.load_ratings()
        train_df = ratings_df.sample(frac=1 - test_size, random_state=seed)
        test_df = ratings_df.drop(train_df.index)
        
        save_path_train = self.MovieLensPathTrain
        save_path_test = self.MovieLensPathTest

        train_file = os.path.join(save_path_train, "train.csv")
        test_file = os.path.join(save_path_test, "test.csv")

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        return train_df, test_df