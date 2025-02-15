# Initialize the loader
import AmazonSalesDataset

def amazonsales():
  loader = AmazonSalesDataset.AmazonSalesDataset()

  # Load ratings
  ratings = loader.load_dataset()
  print(ratings.head())

  # Get train/test split
  train_data, test_data = loader.get_train_test_split()
  print(f"Train data shape: {train_data.shape}")
  print(f"Test data shape: {test_data.shape}")