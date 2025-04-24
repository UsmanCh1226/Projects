import kaggle
import numpy as np
import pandas as pd
import random

kaggle.api.authenticate()

kaggle.api.dataset_download_files("masoudnickparvar/brain-tumor-mri-dataset",path='.',unzip=True)

kaggle.api.dataset_metadata("masoudnickparvar/brain-tumor-mri-dataset",path='.')

print(kaggle.api.dataset_list_files("masoudnickparvar/brain-tumor-mri-dataset").files)
