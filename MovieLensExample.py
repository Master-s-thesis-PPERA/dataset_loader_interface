# Initialize the loader
import MovieLensDataset

def movielens():
  loader = MovieLensDataset.MovieLensDataset()

  # Load ratings
  ratings = loader.load_ratings()
  print(ratings.head())

  # Load item features (movie genres)
  item_features = loader.load_item_features()
  print(item_features.head())

  # Get user-item interactions
  interactions = loader.get_user_item_interactions()
  print(f"User 1 interactions: {interactions.get(1, [])[:5]}")  # Show first 5

  # Get train/test split
  train_data, test_data = loader.get_train_test_split()
  print(f"Train data shape: {train_data.shape}")
  print(f"Test data shape: {test_data.shape}")

  # Get features for a specific item
  item_features_single = loader.get_item_features_for_item(1)  # Movie ID 1
  print(f"Features for item 1: {item_features_single}")

  # Get user history
  user_history = loader.get_user_history(1) # User ID 1
  print(f"User 1 history (first 5): {user_history[:5]}")