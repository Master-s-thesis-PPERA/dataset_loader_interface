from MovieLensExample import movielens
from AmazonSalesExample import amazonsales
from PostRecommendationsExample import postRecommendation

def main():
    while True:
        try:
            dataset = int(input("Select Dataset (1) Amazon Sales, (2) MovieLens 20M, (3) Post Recommendation, (0) Exit: "))
            match dataset:
                case 1:
                    print("Amazon Sales Selected")
                    amazonsales()
                    break
                case 2:
                    print("MovieLens Selected")
                    movielens()
                    break
                case 3:
                    print("Post Recommendation Selected")
                    postRecommendation()
                    break
                case 0:
                    print("Exit")
                    break
                case _:
                    print("Invalid selection. Please choose 1, 2, 3, or 0 to exit.")
        except ValueError:
            print("Invalid input. Please enter a number (1, 2, 3, or 0).")

if __name__ == "__main__":
    main()