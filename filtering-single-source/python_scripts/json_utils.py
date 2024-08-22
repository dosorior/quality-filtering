import json
import os

import numpy as np


def get_json_filenames(dir):
    files = os.listdir(dir)
    for f in files:
        assert f.endswith(".json")
    return files


def remove_extensions(files):
    return [".".join(f.split(".")[:-1]) for f in files]


def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def save_json(dictionary, filepath):
    dirpath = "/".join(filepath.split("/")[:-1])
    os.makedirs(dirpath, exist_ok=True)
    with open(filepath, 'w+') as file:
        file.write(json.dumps(dictionary))


def load_and_process_fingerprints(input_directory):
    json_files = get_json_filenames(input_directory)
    processed_fingerprints = dict()
    for filename in json_files:
        minutiae = load_json(input_directory + filename)
        processed_fingerprints[filename] = dict()
        for minu_key, scores in minutiae.items():
            processed_fingerprints[filename][minu_key] = scores
    return processed_fingerprints


def get_scores(fingerprint_data):
    return np.array([score for fp in fingerprint_data.values() for scores in fp.values() for score in scores])


def get_means(fingerprint_data):
    return np.array([np.mean(minu_scores) for fp in fingerprint_data.values() for minu_scores in fp.values()])


def get_stds(fingerprint_data):
    return np.array([np.std(minu_scores) for fp in fingerprint_data.values() for minu_scores in fp.values()])


def _convert_json_FineNet_scores(minu_quality_json):
    new_json = dict()
    for key, value in minu_quality_json.items():
        assert len(value) == 100
        new_json[key] = [float(s) for s in value]
    return new_json


def convert_all_FineNet_scores(input_directory, output_directory):
    json_files = get_json_filenames(input_directory)
    for filename in json_files:
        minutiae = load_json(input_directory + filename)
        minutiae = _convert_json_FineNet_scores(minutiae)
        save_json(minutiae, output_directory + filename)
