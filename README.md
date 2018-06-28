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
OpenCV
```

### Training

Run the main.py file in src to start training. VGG-Net fine tuned with conv layers frozen. The top of the model is replaced with 2 FC layers for classification.

```
python ./src/main.py
```

## Running the tests

You can monitor the loss and metrics on tensorboard. The logs are found in the ./logs/ dir. The demo to test images can be found in ./demo/demo.ipynb.

## Authors

* **[Maneesh Bilalpur](https://bmaneesh.github.io/bmaneesh/)**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

