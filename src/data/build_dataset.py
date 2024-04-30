import tensorflow_datasets as tfds

def build_dataset():
    builder = tfds.core.dataset_builders.CroissantBuilder(
        file="data\\external\\cardiovascular-disease-dataset-metadata.json",
        record_set_names=["Cardiovascular_Disease_Dataset.csv"],
        file_format="array_record",
        data_dir="data\\raw"
        )
    builder.download_and_prepare()
    ds = builder.as_data_source()
    print(ds['default'][0])
    # print(f"Dataset's description:\n{builder.info.description}\n")
    # print(f"Dataset's citation:\n{builder.info.citation}\n")
    # print(f"Dataset's features:\n{builder.info.features}")


if __name__ == '__main__':
    build_dataset()