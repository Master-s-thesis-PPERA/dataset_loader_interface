import PostRecommendationsDataset

def postRecommendation():
  loader = PostRecommendationsDataset.PostRecommendationsDataset()

  # Load ratings
  userdata = loader.load_dataset()
  print(userdata.head())


  # Get train/test split
  train_data, test_data = loader.get_train_test_split()
  print(f"Train data shape: {train_data.shape}")
  print(f"Test data shape: {test_data.shape}")


  print("MovieLens example completed.")
  # 1. Hide the 'rating' column:
  df_hidden_column = loader.hide_information(test_data, hide_type="columns", columns_to_hide="first_name")
  print(df_hidden_column)

  # 2. Hide 50% of the records randomly:
  df_hidden_records_random = loader.hide_information(test_data, 
  hide_type="records_random", fraction_to_hide=0.5)
  print(df_hidden_records_random)

  # 3. Hide records with indices 1 and 3:
  # df_hidden_records_selective = loader.hide_information(test_data, hide_type="records_selective", records_to_hide=[1, 3])
  # print(df_hidden_records_selective)

  # 4. Hide 30% of the values in the 'rating' column:
  df_hidden_values = loader.hide_information(test_data, hide_type="values_in_column", columns_to_hide='first_name', fraction_to_hide=0.3)
  print(df_hidden_values)

  # 5. Hide 30% of the values in the 'rating' and 'genre' columns:
  df_hidden_values_multiple = loader.hide_information(test_data, hide_type="values_in_column", columns_to_hide=['user_id', 'gender'], fraction_to_hide=0.3)
  print(df_hidden_values_multiple)