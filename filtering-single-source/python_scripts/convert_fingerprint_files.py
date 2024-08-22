import os

import numpy as np

import json_utils

"""
Calculates minutia quality scores from MiDeCon output files and adds them to fingerprints given in .mnt format (MinutiaNet output format).
The result is then saved in a single json file, containing an array of fingerprint objects. 
The minutia quality scores in the output are normalized to [0, 1]. Beware that different datasets may contain a different max / min value
of minutia quality! It is important to process all fingerprints with one call of "convert()",
otherwise the quality may not be directly comparable, although the difference for large numbers of fingerprints will probably be neglible.
"""

def get_mnt_filenames(dir):
    files = os.listdir(dir)
    for f in files:
        assert f.endswith(".mnt")
    return sorted(files)


def load_mnt(filepath):
    fingerprint = dict()
    with open(filepath, 'r') as file:
        lines = file.readlines()
        n_minutiae = int(lines[1].split()[0])
        assert n_minutiae == len(lines) - 2
        minutiae = []
        for i, line in enumerate(lines[2:-1]):
            minu = dict()
            elems = line.split()
            assert len(elems) == 4
            minu["id"] = i
            minu["x"] = float(elems[0])
            minu["y"] = float(elems[1])
            minu["dir"] = float(elems[2])
            minutiae.append(minu)
        fingerprint["minutiae"] = minutiae
    return fingerprint

def load_and_process_mnt_files(input_directory):
    mnt_files = get_mnt_filenames(input_directory)
    fingerprints = []
    for i, filename in enumerate(mnt_files):
        fingerprint = load_mnt(input_directory + filename)
        fingerprint["name"] = filename[:-4]
        fingerprints.append(fingerprint)
    return fingerprints

def add_minutia_quality(fingerprints, midecon_results, quality_function):
    fingerprints = [fp for fp in fingerprints if (fp["name"] + ".json") in midecon_results]
    minqual = np.infty
    maxqual = -np.infty
    for fp in fingerprints:
        name = fp["name"] + ".json"
        for minu in fp["minutiae"]:
            qual = quality_function(midecon_results[name][str(minu["id"])])
            minqual = min(minqual, qual)
            maxqual = max(maxqual, qual)
            minu["qual"] = qual
    normalize_qualities(fingerprints, minqual, maxqual)
    return fingerprints

def add_id(fingerprints):
    id = 0
    processed = []
    for fp in fingerprints:
        fp["id"] = id
        id += 1
        processed.append(fp)
    return processed

def normalize_qualities(fingerprints, minqual, maxqual):
    for fp in fingerprints:
        for minu in fp["minutiae"]:
            minu["qual"] = (minu["qual"] - minqual) / (maxqual - minqual)

def mean_minus_std(scores):
    return np.mean(scores) - np.std(scores)

def convert(MinutiaNet_results_dir, MiDeCon_results_dir, output_file, quality_function):
    fingerprints = load_and_process_mnt_files(MinutiaNet_results_dir)
    midecon_results = json_utils.load_and_process_fingerprints(MiDeCon_results_dir)
    fingerprints = add_minutia_quality(fingerprints, midecon_results, quality_function)
    json_utils.save_json({"fingerprints": add_id(fingerprints)}, output_file)
