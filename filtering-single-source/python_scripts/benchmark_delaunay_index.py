import numpy as np
import sys
import json
import os
import random
sys.path.insert(
    0, "C:/Users/tim/Google Drive/h_da/BMS/improved-fingerprint-indexing/build_win/Debug/")
os.add_dll_directory("C:/Program Files (x86)/opencv/build/x64/vc15/bin/")

from DelaunayIndex import DelaunayIndex, setFingerprintImageDir, setDebugOutputDir, loadFingerprintsFromJson
import convert_fingerprint_files
import json_utils

BASE_DIR = "C:/Users/tim/Google Drive/h_da/BMS/improved-fingerprint-indexing/"
FVC2006_IMG_DIR = BASE_DIR + "data/FVC2006/"

DEBUG_IMG_OUTDIR = BASE_DIR + "data/debug_img/"
BINDIST_OUTDIR = BASE_DIR + "data/bin_distributions/"
RESULT_OUTDIR = BASE_DIR + "benchmark_results/"

FVC2006_SUBSETS = ["DB1_A", "DB2_A", "DB3_A", "DB4_A"]
FVC2006_SUBSET = FVC2006_SUBSETS[3]
FVC2006_SUBSET_DIR = FVC2006_SUBSET + "/"

CONVERT_AGAIN = False
MINUTIANET_JSON_DIR = BASE_DIR + "data/FVC2006_MinutiaNet_Output/" + FVC2006_SUBSET_DIR # MinutiaNet output (.mnt)
MIDECON_JSON_DIR = BASE_DIR + "data/FVC2006_MiDeCon_Output/" + FVC2006_SUBSET_DIR # MiDeCon quality scores (100 per minutia, .json)
FINGERPRINT_JSON_OUTDIR = BASE_DIR + "data/FPJson_MEANminusSTD/" # The converted fingerprint files go here
FINGERPRINT_JSON_FILE = FINGERPRINT_JSON_OUTDIR + FVC2006_SUBSET + "_fingerprints.json" # The converted fingerprint file

SELECTION_MODES = [
    DelaunayIndex.MinutiaSelection.QUALITY_BEST05,
    DelaunayIndex.MinutiaSelection.QUALITY_BEST10,
    DelaunayIndex.MinutiaSelection.QUALITY_BEST15,
    DelaunayIndex.MinutiaSelection.QUALITY_BEST20,
    DelaunayIndex.MinutiaSelection.QUALITY_BEST30,
    DelaunayIndex.MinutiaSelection.QUALITY_BEST40,
    DelaunayIndex.MinutiaSelection.QUALITY_BEST50,
    DelaunayIndex.MinutiaSelection.QUALITY_BEST60,
    DelaunayIndex.MinutiaSelection.KEEP_ALL]

HASHING_MODES = [
    DelaunayIndex.Hashtable.HASH_GEOM,
    DelaunayIndex.Hashtable.HASH_RFDD,
    DelaunayIndex.Hashtable.HASH_GEOM_RFDD,
    DelaunayIndex.Hashtable.HASH_ALL]

def make_all_outdirs():
    all_outdirs = [RESULT_OUTDIR]
    all_outdirs += [DEBUG_IMG_OUTDIR + db + "/" for db in FVC2006_SUBSETS]
    all_outdirs += [BINDIST_OUTDIR + db + "/" for db in FVC2006_SUBSETS]
    all_outdirs += [FINGERPRINT_JSON_OUTDIR]

    for dir in all_outdirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def separateIndexAndQuery(fingerprints):
    random.seed(42)
    random.shuffle(fingerprints)
    subjectInIndex = {fp.subjectId: False for fp in fingerprints}
    index = []
    query = []
    for fp in fingerprints:
        if not subjectInIndex[fp.subjectId]:
            index.append(fp)
            subjectInIndex[fp.subjectId] = True
        else:
            query.append(fp)
    return index, query


def results_to_json(results, hashing_mode, selection_mode, db_name, db_size):
    results_json = {"hashing_mode": hashing_mode,
                    "selection_mode": selection_mode, "database": db_name}
    results_json["found"] = [r[0] for r in results]
    results_json["num_comparisons"] = [r[1] for r in results]
    results_json["database_size"] = db_size
    return results_json


def benchmark(index_data, query_data, hashing_mode, selection_mode, db_name):
    hashing_mode_str = str(hashing_mode).split(".")[-1]
    selection_mode_str = str(selection_mode).split(".")[-1]
    print("Run benchmark for Delaunay Index with parameters: " +
          hashing_mode_str + " and " + selection_mode_str + " on " + db_name)
    
    # Create Index and report bin distribution
    delaunay_index = DelaunayIndex(hashing_mode, selection_mode, index_data)
    delaunay_index.reportBinDistribution(
        BINDIST_OUTDIR + db_name + "/" + hashing_mode_str + "_" + selection_mode_str + ".json")
    
    # Run benchmark
    results = [delaunay_index.searchExhaustive(fp) for fp in query_data]
    print("Average penetration rate: " +
          str(sum([int(r[1]) for r in results]) / len(query_data) / len(index_data)))
    print("Rate of fingerprints found " +
          str(sum([int(r[0]) for r in results]) / len(query_data)))
    
    # Save results as json
    json_filename = db_name + "-" + hashing_mode_str + \
        "-" + selection_mode_str + ".json"
    with open(RESULT_OUTDIR + json_filename, 'w') as file:
        results_json = results_to_json(
            results, hashing_mode_str, selection_mode_str, db_name, len(index_data))
        file.write(json.dumps(results_json))

def main():
    make_all_outdirs()

    if (CONVERT_AGAIN):
        convert_fingerprint_files.convert(
            MINUTIANET_JSON_DIR, MIDECON_JSON_DIR, FINGERPRINT_JSON_FILE, convert_fingerprint_files.mean_minus_std)
        print("Fingerprints converted...")

    # This is for debugging purposes only, i.e. for plotting the delaunay triangulation
    setFingerprintImageDir(FVC2006_IMG_DIR + FVC2006_SUBSET_DIR)
    setDebugOutputDir(DEBUG_IMG_OUTDIR + FVC2006_SUBSET_DIR)

    # Separate the data for index and query
    fps = loadFingerprintsFromJson(FINGERPRINT_JSON_FILE)
    
    print(FVC2006_SUBSET)
    nminus = [len(f.minutiae) for f in fps]
    print(sum(nminus) / len(fps))

    index_data, query_data = separateIndexAndQuery(fps)

    
    return 0

    for selection_mode in SELECTION_MODES:
        for hashing_mode in HASHING_MODES:
            benchmark(index_data, query_data, hashing_mode,
                      selection_mode, FVC2006_SUBSET)
    return 0


if __name__ == "__main__":
    main()
