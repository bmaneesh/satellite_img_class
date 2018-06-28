# Planet: Understanding the Amazon from Space

Use satellite data to track the human footprint in the Amazon rainforest.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them

```
keras(backend tensorflow)
numpy
pandas
tqdm
PIL
OpenCV
```

### Training

Run the main.py file in src to start training. VGG-Net fine tuned with conv layers frozen. The top of the model is replaced with 2 FC layers with 17 class classification.

Say what the step will be

```
python ./src/main.py
```

## Running the tests

You can monitor the loss and metrics on tensorboard. The demo to test images can be found in ./demo/demo.ipynb.

## Authors

* **Maneesh Bilalpur** - *Initial work* - [Website](https://bmaneesh.github.io/bmaneesh/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

