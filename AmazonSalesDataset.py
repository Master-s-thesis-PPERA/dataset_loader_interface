import os
from BaseDatasetLoader import BaseDatasetLoader

class AmazonSalesDataset(BaseDatasetLoader):
    """
    Dataset loader for the Amazon Sales dataset.
    """

    def __init__(self, data_path: str = "datasets/AmazonSales"):
        super().__init__(data_path)
        self.dataset_file = f"{self.data_path}/amazon.csv"

        self.merge_file = self.dataset_file

        self.train_path = os.path.join(self.data_path, "trainDataset")
        self.test_path = os.path.join(self.data_path, "testDataset")
    