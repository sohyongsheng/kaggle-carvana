# kaggle-carvana
Image segmentation for the Kaggle competition [Carvana Image Masking Challenge: Automatically identify the boundaries of the car in an image](https://www.kaggle.com/c/carvana-image-masking-challenge). This solution uses a batch size of 1 with a large momentum of 0.99.

## Pre-requisites and downloads
- Install [PyTorch](http://pytorch.org/) with Python 3.
- We shall reference the root directory where `my_solution.py` as `./`
- Make a directory `./data/` and download the data from the [competition's website](https://www.kaggle.com/c/carvana-image-masking-challenge/data) into this directory. The data directory should have something like the following tree structure:
```
./data/
├── test
├── test_hq
├── test_samples
├── train
├── train_hq
├── train_masks
├── train_masks_samples
├── train_samples
├── metadata.csv
├── sample_submission.csv
└── train_masks.csv

8 directories, 3 files
```

## Running the solution
To run the solution, use the following command in terminal:
```
python3 my_solution.py
```
Most of the important parameters are after the line `if __name__ == "__main__":`. The results are stored in the relevant `./results/my_solution/`.

To view the learning curves while training, use the following command in terminal:
```
python3 plot_learning_curves.py
```
