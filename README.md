# [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data)

Use satellite data to track the human footprint in the Amazon rainforest.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

```
keras(backend tensorflow)
numpy
pandas
tqdm
PIL
```

Python 2.7 environment was used for the development.

### Training

Run the main.py file in src to start training. VGG-Net fine tuned with conv layers frozen. The top of the model is replaced with 2 FC layers for classification.

```
python ./src/main.py
```

Please place the train-jpg, test-jpg, test-jpg-additional, test_v2.csv in `./`
## Running the tests

You can monitor the loss and metrics on tensorboard. The logs are found in the `./logs/`. The demo to test images can be found in `./demo/demo.ipynb`

Monitor the loss and metrics using tensorboard

```
tensorboard --logdir=20180628-122047
```

## Authors

* **[Maneesh Bilalpur](https://bmaneesh.github.io/bmaneesh/)**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. The data_helper function might be mostly adopted from the Kaggle submission(s), adjusted for the current solution requirements.
