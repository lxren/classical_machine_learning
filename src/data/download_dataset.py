import os
import mlcroissant as mlc

def download_dataset():
    ds = mlc.Dataset("data/external/cardiovascular-disease-dataset-metadata.json")
    metadata = ds.metadata.to_json()
    print(f"{metadata['name']}: {metadata['description']}")

    for x in ds.records(record_set="Cardiovascular_Disease_Dataset.csv"):
        print(x)

if __name__ == '__main__':
    download_dataset()