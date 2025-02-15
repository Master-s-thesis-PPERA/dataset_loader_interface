import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Optional

class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    """

    def __init__(self, data_path: str):
        """
        Initializes the DatasetLoader.

        Args:
            data_path: The path to the directory containing the dataset.
        """
        self.data_path = data_path

    @abstractmethod
    def load_ratings(self) -> pd.DataFrame:
        """
        Loads the user-item interaction data (e.g., ratings).

        Returns:
            A Pandas DataFrame with at least 'user_id', 'item_id',
            and 'rating' columns.
        """
        pass

    @abstractmethod
    def load_item_features(self) -> pd.DataFrame:
        """
        Loads item features (e.g., movie genres, product descriptions).

        Returns:
            A Pandas DataFrame with at least 'item_id' and feature columns.
        """
        pass

    @abstractmethod
    def get_user_item_interactions(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Gets a dictionary mapping user IDs to a list of (item_id, rating) tuples.
        This is useful for CF algorithms.

        Returns:
            A dictionary where keys are user IDs and values are lists of
            (item_id, rating) tuples.
        """
        pass

    @abstractmethod
    def get_train_test_split(self, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
       """
       Splits the interaction dataset into training and testing sets.

       Args:
           test_size (float): Proportion of data to use for testing.
           seed (int): Random seed for reproducibility.

       Returns:
           Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.
        """
       pass


    def get_item_features_for_item(self, item_id: int) -> Optional[Dict]:
        """
        (Optional) Get features for a single item.  Useful for L2R or RL
        when you need to represent the current state.

        Args:
            item_id: The ID of the item.

        Returns:
            A dictionary of item features, or None if the item is not found.
        """
        item_features_df = self.load_item_features()
        if item_id in item_features_df['item_id'].values:
            return item_features_df[item_features_df['item_id'] == item_id].iloc[0].to_dict()
        else:
            return None

    def get_user_history(self, user_id: int) -> List[Tuple[int, float]]:
        """
        (Optional) Get the interaction history for a single user. Useful for RL.

        Args:
            user_id: The ID of the user.

        Returns:
            A list of (item_id, rating) tuples.
        """
        interactions = self.get_user_item_interactions()
        return interactions.get(user_id, [])