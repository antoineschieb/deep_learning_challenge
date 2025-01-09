# Description
This repository contains the code for the Deep Learning challenge, as well as instructions on how to train, test, and run the model.

# Dependencies
This project uses the `poetry` package manager. The requirements can be found in `pyproject.toml`

# How to run the code
## Training and testing on the available data
For the sake of simplicty, training the model on the available data can be done directly in a notebook. 

1) Locate the dataset folder which should contain the `class1`,`class2`,`class3`,`class4` subfolders and set the following environment variable:
    `export DATA_ROOT = /path/to/folder/`
2) Run the `train_test.ipynb` notebook.

## Testing the model on any other image
You can also run the model on any other image.
1) Ensure that the trained weights `mlp.pth` are located at the root of this project.
2) Run python classify_image.py /path/to/any/other/image.tif

# Author

Antoine Schieb (antoine.schieb@cpe.fr)