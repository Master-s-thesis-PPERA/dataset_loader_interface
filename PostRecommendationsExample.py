# Initialize the loader
import PostRecommendationsDataset

def postRecommendation():
  loader = PostRecommendationsDataset.PostRecommendationsDataset()

  # Load ratings
  userdata = loader.load_userData()
  print(userdata.head())


  # Get train/test split
  train_data, test_data = loader.get_train_test_split()
  print(f"Train data shape: {train_data.shape}")
  print(f"Test data shape: {test_data.shape}")