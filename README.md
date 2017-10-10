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

## Results

### Dice coefficients
Evaluation of predicted masks are done using the Dice coefficient, as described [here](https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation). Cross validation is done on first 16 cars.

| Early stopping at | Epoch 50 | Epoch 110 |
| --- | ---| --- |
| Cross validation | 0.99646 | 0.99665 |
| Public LB | 0.996314 | 0.996509 |
| Private LB | 0.996070 | 0.996288 |

### Test images
![0004d4463b50_01](https://user-images.githubusercontent.com/7287899/31338391-991d28cc-ad31-11e7-87a5-4d090b8a5444.jpg)
![0004d4463b50_02](https://user-images.githubusercontent.com/7287899/31338386-99163530-ad31-11e7-8d7d-5396799e0a16.jpg)
![0004d4463b50_03](https://user-images.githubusercontent.com/7287899/31338388-99174628-ad31-11e7-9e97-9c657d78030d.jpg)
![0004d4463b50_04](https://user-images.githubusercontent.com/7287899/31338387-9916ff38-ad31-11e7-8f79-d9528d749a05.jpg)
![0004d4463b50_05](https://user-images.githubusercontent.com/7287899/31338389-99197998-ad31-11e7-83a5-d2d2cdccbe6a.jpg)
![0004d4463b50_06](https://user-images.githubusercontent.com/7287899/31338390-991d5040-ad31-11e7-953f-ae17a4d26bcd.jpg)
![0004d4463b50_07](https://user-images.githubusercontent.com/7287899/31338396-9973bad4-ad31-11e7-91e9-1cfae7f78933.jpg)
![0004d4463b50_09](https://user-images.githubusercontent.com/7287899/31338395-9952d2f6-ad31-11e7-91af-2294dcde4c7c.jpg)
![0004d4463b50_10](https://user-images.githubusercontent.com/7287899/31338392-99520c9a-ad31-11e7-8111-98bd694a7f27.jpg)
![0004d4463b50_11](https://user-images.githubusercontent.com/7287899/31338393-9952612c-ad31-11e7-9fa8-38716031b820.jpg)
![0004d4463b50_12](https://user-images.githubusercontent.com/7287899/31338402-99bea274-ad31-11e7-9ce0-5a71fb4eb428.jpg)
![0004d4463b50_13](https://user-images.githubusercontent.com/7287899/31338397-99867192-ad31-11e7-8d06-496e91e64f04.jpg)
![0004d4463b50_14](https://user-images.githubusercontent.com/7287899/31338398-99869fdc-ad31-11e7-9885-394ee9890f5c.jpg)
![0004d4463b50_15](https://user-images.githubusercontent.com/7287899/31338399-99879130-ad31-11e7-8254-430ae93848bc.jpg)
![0004d4463b50_16](https://user-images.githubusercontent.com/7287899/31338400-998996b0-ad31-11e7-8463-9c7637c99952.jpg)

### Predictions using epoch 50
![0004d4463b50_01](https://user-images.githubusercontent.com/7287899/31338134-bb0e4728-ad30-11e7-9de7-48f6dcba2e4e.jpg)
![0004d4463b50_02](https://user-images.githubusercontent.com/7287899/31338140-bb220470-ad30-11e7-85c3-ded7a251f4df.jpg)
![0004d4463b50_03](https://user-images.githubusercontent.com/7287899/31338135-bb10958c-ad30-11e7-8ab5-2576f61d2daf.jpg)
![0004d4463b50_04](https://user-images.githubusercontent.com/7287899/31338136-bb1107f6-ad30-11e7-932a-25b91510b123.jpg)
![0004d4463b50_05](https://user-images.githubusercontent.com/7287899/31338138-bb1c9968-ad30-11e7-8d4d-68079a6fbfd5.jpg)
![0004d4463b50_06](https://user-images.githubusercontent.com/7287899/31338137-bb19b734-ad30-11e7-81b6-eaa8e54c0c00.jpg)
![0004d4463b50_07](https://user-images.githubusercontent.com/7287899/31338144-bb451cf8-ad30-11e7-84b8-3675a4a338b0.jpg)
![0004d4463b50_08](https://user-images.githubusercontent.com/7287899/31338145-bb458a30-ad30-11e7-95ce-b2fa9cabd36c.jpg)
![0004d4463b50_09](https://user-images.githubusercontent.com/7287899/31338143-bb44c24e-ad30-11e7-84ec-f64b76118b65.jpg)
![0004d4463b50_10](https://user-images.githubusercontent.com/7287899/31338150-bb62cb68-ad30-11e7-9e6c-312e89b88963.jpg)
![0004d4463b50_11](https://user-images.githubusercontent.com/7287899/31338146-bb4dd14a-ad30-11e7-86ee-2a914aeda070.jpg)
![0004d4463b50_12](https://user-images.githubusercontent.com/7287899/31338147-bb51efb4-ad30-11e7-8d3e-1595b66cd92d.jpg)
![0004d4463b50_13](https://user-images.githubusercontent.com/7287899/31338152-bb7530e6-ad30-11e7-84bd-81573d7f1e29.jpg)
![0004d4463b50_14](https://user-images.githubusercontent.com/7287899/31338158-bbd5e044-ad30-11e7-8779-20c077897b92.jpg)
![0004d4463b50_15](https://user-images.githubusercontent.com/7287899/31338153-bb756b42-ad30-11e7-8636-68f4455b250f.jpg)
![0004d4463b50_16](https://user-images.githubusercontent.com/7287899/31338154-bb7dd304-ad30-11e7-9e09-09fa38152184.jpg)

### Predictions using epoch 110
![0004d4463b50_01](https://user-images.githubusercontent.com/7287899/31338244-0adc78a6-ad31-11e7-97b1-c147aa1b774d.jpg)
![0004d4463b50_02](https://user-images.githubusercontent.com/7287899/31338258-0b91a51e-ad31-11e7-9dd2-c2541bcdc17d.jpg)
![0004d4463b50_03](https://user-images.githubusercontent.com/7287899/31338245-0aeb56a0-ad31-11e7-802d-960a599dc6df.jpg)
![0004d4463b50_04](https://user-images.githubusercontent.com/7287899/31338247-0aee8186-ad31-11e7-9f02-2e58f16cc8e4.jpg)
![0004d4463b50_05](https://user-images.githubusercontent.com/7287899/31338246-0aee7470-ad31-11e7-9f86-b3402a0ef70f.jpg)
![0004d4463b50_06](https://user-images.githubusercontent.com/7287899/31338248-0af6389a-ad31-11e7-8360-aee2b3180731.jpg)
![0004d4463b50_07](https://user-images.githubusercontent.com/7287899/31338249-0b0ba6bc-ad31-11e7-87a3-e1c4e78b2bf1.jpg)
![0004d4463b50_08](https://user-images.githubusercontent.com/7287899/31338253-0b32e240-ad31-11e7-9259-1d7505a0d998.jpg)
![0004d4463b50_09](https://user-images.githubusercontent.com/7287899/31338252-0b2e471c-ad31-11e7-83d4-eaa207b1f1f5.jpg)
![0004d4463b50_10](https://user-images.githubusercontent.com/7287899/31338250-0b27b73a-ad31-11e7-984f-5d4fd2f5e9ef.jpg)
![0004d4463b50_11](https://user-images.githubusercontent.com/7287899/31338251-0b2e5284-ad31-11e7-9721-be40b109ba49.jpg)
![0004d4463b50_12](https://user-images.githubusercontent.com/7287899/31338257-0b83c3b8-ad31-11e7-97e6-e51a57a34636.jpg)
![0004d4463b50_13](https://user-images.githubusercontent.com/7287899/31338254-0b6050ea-ad31-11e7-907c-d1dfb7cd2499.jpg)
![0004d4463b50_14](https://user-images.githubusercontent.com/7287899/31338255-0b6dedb8-ad31-11e7-96ba-764689d4457b.jpg)
![0004d4463b50_15](https://user-images.githubusercontent.com/7287899/31338259-0bf8e2ce-ad31-11e7-93c9-cd30ebf83bc6.jpg)
![0004d4463b50_16](https://user-images.githubusercontent.com/7287899/31338256-0b70ac10-ad31-11e7-8cd9-eb3c14828a34.jpg)
