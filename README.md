# Description
This repository contains the code for the Deep Learning challenge, as well as instructions on how to train, test, and run the model.
The prediction pipeline consists of a simple Multi Layer Perceptron on top of the open-source foundation model [Phikon-v2](https://arxiv.org/abs/2409.09173). 

# Dependencies
This project uses the `poetry` package manager. The requirements can be found in `pyproject.toml`

# How to run the code
## Training and testing a new model on the challenge data
For the sake of simplicty, training and testing the model on the available data can be done directly in a notebook.
1) Locate the dataset folder which contains the `class1`,`class2`,`class3`,`class4` subfolders at its root.
2) Edit the first cell of the `train_test.ipynb` notebook and change the environment variable `%env DATA_ROOT=/path/to/data/`
3) Run the whole notebook. The trained MLP model will be exported at the project root as `mlp.pth`

## Testing a trained model on any image
You can also run the model on the challenge data, or any other image.
1) Ensure that the trained weights `mlp.pth` are located at the root of this project.
2) Run `python classify_image.py /path/to/any/other/image.tif`
3) The program will output the predicted class

# Author

Antoine Schieb (antoine.schieb@cpe.fr)
