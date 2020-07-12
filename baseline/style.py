import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from utils import yaml_load, files_to_vector_array
from matplotlib import pyplot

if __name__ == "__main__":
    # load config from config.yaml
    config = yaml_load("./config.yaml")

    base_cfg = config.get("base", {})
    dev_path = base_cfg["dev_data"]
    test_path = base_cfg["test_data"]

    # load development data
    names, datas, labels = files_to_vector_array(dev_path, base_cfg)

    # Using kmeans for unsupervised clustering
    clf = KMeans(n_clusters=2)
    clf.fit(datas)

    # load test data and predict
    names, datas, labels = files_to_vector_array(test_path, base_cfg, False, True)
    predict = clf.predict(datas)

    # add style result to result.csv
    dataframe = pd.read_csv("result.csv")
    dataframe["style"] = predict
    dataframe.to_csv("result.csv", index=False)
