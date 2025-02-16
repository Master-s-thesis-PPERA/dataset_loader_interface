# Initialize the loader
import MovieLensDataset
import pandas as pd

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


  print("MovieLens example completed.")
  # 1. Hide the 'rating' column:
  df_hidden_column = loader.hide_information(test_data, hide_type="columns", columns_to_hide="rating")
  print(df_hidden_column)

  # 2. Hide 50% of the records randomly:
  df_hidden_records_random = loader.hide_information(test_data, 
  hide_type="records_random", fraction_to_hide=0.5)
  print(df_hidden_records_random)

  # 3. Hide records with indices 1 and 3:
  df_hidden_records_selective = loader.hide_information(test_data, hide_type="records_selective", records_to_hide=[1, 3])
  print(df_hidden_records_selective)

  # 4. Hide 30% of the values in the 'rating' column:
  df_hidden_values = loader.hide_information(test_data, hide_type="values_in_column", columns_to_hide='rating', fraction_to_hide=0.3)
  print(df_hidden_values)

  # 5. Hide 30% of the values in the 'rating' and 'genre' columns:
  df_hidden_values_multiple = loader.hide_information(test_data, hide_type="values_in_column", columns_to_hide=['rating', 'genre'], fraction_to_hide=0.3)
  print(df_hidden_values_multiple)