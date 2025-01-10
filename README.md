# Description
This repository contains the code for the Deep Learning challenge, as well as instructions on how to train, test, and run the model.
The prediction pipeline consists of a simple Multi Layer Perceptron on top of the open-source foundation model [Phikon-v2](https://arxiv.org/abs/2409.09173).

# Dependencies
This project uses the `poetry` package manager. If you also use poetry, you can directly run `poetry install` at the root of this repository. Otherwise, the list of requirements can be found in `pyproject.toml`.

# How to run the code
### A - Training and testing a new model on the challenge data
The first part consists in generating and saving the embeddings for easier and faster training.
1) Locate the dataset folder which contains the TIF images in the `class1`,`class2`,`class3`,`class4` subfolders.
2) Run the script `python create_embeddings.py /path/to/challenge/images/` once. This will generate an `embeddings/` folder at the project root with the same structure as the challenge data, and save the `.pt` file corresponding to each image.

For the sake of simplicty, training and testing the model on the available data can be done directly in a notebook.
3) Run the whole `train_test.ipynb` notebook. This will train a fresh MLP, show training/validation loss curves, and evaluate the performance on a test set. The trained MLP model will be exported at the project root as `mlp.pth`

### B - Testing a trained model on any image
You can also run the model on the challenge data, or any other image.
1) Ensure that the trained weights `mlp.pth` are located at the root of this project.
2) Run `python classify_image.py /path/to/any/other/image.tif`
3) The program will output the predicted class in the console.

# Author

Antoine Schieb (antoine.schieb@cpe.fr)
