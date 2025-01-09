import os

DATA_ROOT = os.environ.get('DATA_ROOT')
if DATA_ROOT is None:
    raise KeyError("Ensure the DATA_ROOT environment variable points to the challenge data")

LABELS_MAP = {
    0: "class1",
    1: "class2",
    2: "class3",
    3: "class4"
}
