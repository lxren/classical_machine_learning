from mlcroissant import Dataset
import itertools
import pandas as pd

def build_dataset():
    ds = Dataset(jsonld="https://www.kaggle.com/datasets/jocelyndumlao/cardiovascular-disease-dataset/croissant/download")
    records = ds.records(ds.metadata.record_sets[0].id)
    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    build_dataset()

#References: https://huggingface.co/docs/datasets-server/en/mlcroissant; TFDS's CroissantBuilder does not work