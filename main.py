import os
import zipfile
from pprint import pprint

import requests
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    chunk_size = 8196

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size)):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def load_data(dest, dest_file, file_id):
    if not os.path.exists(dest):
        os.makedirs(dest)

    if not os.path.exists(os.path.join(dest, dest_file)):
        download_file_from_google_drive(file_id, os.path.join(dest, dest_file))

    for filename in os.listdir(dest):
        if filename.endswith(".zip"):
            filename = os.path.join(dest, filename)
            name = os.path.splitext(os.path.basename(filename))[0]
            if not os.path.isdir(name):
                zip = zipfile.ZipFile(filename)
                zip.extractall(path=dest)

    return os.path.join(dest, str([e.name for e in os.scandir(dest) if os.path.isdir(e)][0]))


if __name__ == "__main__":
    file_id = '0Bwo0SFiZwl3JVGRlWGZUaW5va00'
    dest, dest_file = 'data', 'file.zip'
    data_folder = load_data(dest, dest_file, file_id)

    # Create a dataset.
    dataset = keras.preprocessing.image_dataset_from_directory(
        data_folder, batch_size=64, image_size=(200, 200))

    pprint({i+1: d for i, d in enumerate(dataset.class_names)})

    # # For demonstration, iterate over the batches yielded by the dataset.
    # for i, (data, labels) in enumerate(dataset):
    #     print(f"{i}. data: shape: {data.shape} type: {data.dtype}")
    #     print(f"label: shape: {labels.shape} type: {labels.dtype}")
