# Dataset Loader Interface

This repository contains a dataset loader interface for various datasets, including Amazon Sales, MovieLens, and Post Recommendations. The interface provides a unified way to load, preprocess, and split datasets for machine learning tasks.

## Project Structure

```
DATASET_LOADER_INTERFACE/
│── datasets/                    # Directory containing various datasets
│   ├── AmazonSales/             # Amazon Sales dataset
│   │   ├── testDataset/         # Test dataset for Amazon Sales
│   │   ├── trainDataset/        # Training dataset for Amazon Sales
│   ├── MovieLens/               # MovieLens dataset
│   │   ├── testDataset/         # Test dataset
│   │   ├── trainDataset/        # Training dataset
│   │   ├── genome_scores.csv
│   │   ├── genome_tags.csv
│   │   ├── link.csv
│   │   ├── movie.csv
│   │   ├── rating.csv
│   │   ├── tag.csv
│   ├── PostRecommendations/      # Post Recommendations dataset
│   │   ├── testDataset/
│   │   ├── trainDataset/
│   │   ├── post_data.csv
│   │   ├── user_data.csv
│   │   ├── view_data.csv
│── .gitignore                    # Ignore unnecessary files (e.g., pycache/)
│── AmazonSalesDataset.py         # Loader for Amazon Sales dataset
│── AmazonSalesExample.py         # Example usage of Amazon Sales dataset
│── BaseDatasetLoader.py          # Base class for dataset loading
│── main.py                       # Main script
│── MovieLensDataset.py           # Loader for MovieLens dataset
│── MovieLensExample.py           # Example usage of MovieLens dataset
│── PostRecommendationsDataset.py # Loader for Post Recommendations dataset
│── PostRecommendationsExample.py # Example usage of Post Recommendations dataset
│── README.md                     # Project documentation

```

## Files and Directories

- `AmazonSalesDataset.py`: Contains the `AmazonSalesDataset` class for loading and processing the Amazon Sales dataset.
- `AmazonSalesExample.py`: Example script to demonstrate the usage of `AmazonSalesDataset`.
- `BaseDatasetLoader.py`: Abstract base class for dataset loaders.
- `main.py`: Main script to select and run examples for different datasets.
- `MovieLensDataset.py`: Contains the `MovieLensDataset` class for loading and processing the MovieLens dataset.
- `MovieLensExample.py`: Example script to demonstrate the usage of `MovieLensDataset`.
- `PostRecommendationsDataset.py`: Contains the `PostRecommendationsDataset` class for loading and processing the Post Recommendations dataset.
- `PostRecommendationsExample.py`: Example script to demonstrate the usage of `PostRecommendationsDataset`.
- `datasets/`: Directory containing subdirectories for each dataset with train and test splits.
- `.gitignore`: Git ignore file to exclude certain files and directories from version control.
- `README.md`: This file.

## Usage

1. Clone the repository:

   ```sh
   git clone <repository-url>
   cd dataset_loader_interface
   ```

2. Install the required dependencies:

   ```sh
   pip install pandas
   ```

3. Run the main script:

   ```sh
   python main.py
   ```

4. Follow the prompts to select a dataset and see the example usage.

## Kaggle Dataset Links

### Amazon Sales Dataset

🔗 [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)  
To run the Amazon Sales dataset example, select option `1` when prompted.

### MovieLens Dataset

🔗 [MovieLens Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)  
To run the MovieLens dataset example, select option `2` when prompted.

### Post Recommendations Dataset

🔗 [Post Recommendations Dataset](https://www.kaggle.com/datasets/vatsalparsaniya/post-pecommendation)  
To run the Post Recommendations dataset example, select option `3` when prompted.
