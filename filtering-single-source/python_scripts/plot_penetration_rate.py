import pandas
import seaborn
import json
import os
import sys

from matplotlib import pyplot as plt
from benchmark_delaunay_index import FVC2006_SUBSETS, SELECTION_MODES, HASHING_MODES


BASE_DIR = "/Users/daile.osorio/Projects/Terms_paper_topics/2022/Materials_IDT/code_git/improved-fingerprint-indexing"
RESULT_DIR = BASE_DIR + "benchmark_results/"
PLOT_OUTDIR = BASE_DIR + "plots/"
SAVE_PLOT = True

# This is needed for importing the C++ module "DelaunayIndex" into python.
# Set "build/Debug" to the "build" directory of your C++ project (where the file DelaunayIndex.lib is located) 
sys.path.insert(
    0, BASE_DIR + "build_win/Debug/")
from DelaunayIndex import DelaunayIndex

LABEL_SELECTION_MODE = {
    DelaunayIndex.MinutiaSelection.QUALITY_BEST05: "BEST_5",
    DelaunayIndex.MinutiaSelection.QUALITY_BEST10: "BEST_10",
    DelaunayIndex.MinutiaSelection.QUALITY_BEST15: "BEST_15",
    DelaunayIndex.MinutiaSelection.QUALITY_BEST20: "BEST_20",
    DelaunayIndex.MinutiaSelection.QUALITY_BEST30: "BEST_30",
    DelaunayIndex.MinutiaSelection.QUALITY_BEST40: "BEST_40",
    DelaunayIndex.MinutiaSelection.QUALITY_BEST50: "BEST_50",
    DelaunayIndex.MinutiaSelection.QUALITY_BEST60: "BEST_60",
    DelaunayIndex.MinutiaSelection.KEEP_ALL: "ALL"
}
LABEL_HASHING_MODE = {
    DelaunayIndex.Hashtable.HASH_GEOM: "GEMO",
    DelaunayIndex.Hashtable.HASH_RFDD: "RFDD",
    DelaunayIndex.Hashtable.HASH_GEOM_RFDD: "GEMO_RFDD",
    DelaunayIndex.Hashtable.HASH_ALL: "FULL"
}

DB_SIZE = 140 # Number of fingerprints enrolled in the database that was benchmarked

def get_partial_sums(values):
    current_sum = 0
    partial_sums = []
    for v in values:
        current_sum += v
        partial_sums.append(current_sum)
    return partial_sums


def get_num_comparisons(results):
    zipped = zip(results["found"], results["num_comparisons"])
    num_comparisons = [z[1] for z in zipped if z[0]]
    num_comparisons.sort()
    return num_comparisons


def get_average_pen_rate(results):
    filtered = [results["num_comparisons"][i]
                for i, found in enumerate(results["found"]) if found]
    return sum(filtered) / len(filtered) / DB_SIZE


def plot_pen_over_selection_mode(label_data_map, filename="CompareSelection"):
    seaborn.set_style("whitegrid")
    ax = plt.subplot()
    for label, data in label_data_map.items():
        seaborn.lineplot(x=list(range(1, len(data) + 1)),
                         y=data, label=label, marker="v", ax=ax)
    plt.legend()
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([LABEL_SELECTION_MODE[sel]
                       for sel in SELECTION_MODES], rotation=45, fontweight="bold")
    plt.xlabel("Minutia selection mode", fontweight="bold")
    plt.ylabel("Penetration rate", fontweight="bold")
    plt.subplots_adjust(bottom=0.2)
    if SAVE_PLOT:
        plt.savefig(PLOT_OUTDIR + filename + ".png")
    else:
        plt.show()
    plt.close()


def get_all_pen_rates(fvc_subset, hashing_mode):
    hashing_mode_str = str(hashing_mode).split(".")[-1]
    pen_rates = []
    for selection_mode in SELECTION_MODES:
        selection_mode_str = str(selection_mode).split(".")[-1]
        filename = fvc_subset + "-" + hashing_mode_str + \
            "-" + selection_mode_str + ".json"
        with open(RESULT_DIR + filename, "r") as file:
            data = json.load(file)
            pen_rates.append(get_average_pen_rate(data))
    return pen_rates


def compare_selection_modes(fvc_subset):
    # Needs rewriting: It is better to plot penetration rate over number of minutiae
    label_data_map = dict()
    for hashing_mode in HASHING_MODES:
        label_data_map[LABEL_HASHING_MODE[hashing_mode]
                       ] = get_all_pen_rates(fvc_subset, hashing_mode)
    plot_pen_over_selection_mode(
        label_data_map, "CompareSelectionModes_" + fvc_subset)


def plot_hit_over_pen(label_data_map, filename="CompareHashing"):
    seaborn.set_style("whitegrid")
    ax = plt.subplot()
    for label, data in label_data_map.items():
        num_comparisons = get_num_comparisons(data)
        pen_rate = [s / len(num_comparisons) /
                    DB_SIZE for s in get_partial_sums(num_comparisons)]
        hit_rate = [i / len(num_comparisons)
                    for i in range(len(num_comparisons))]
        seaborn.lineplot(x=pen_rate, y=hit_rate, label=label, ax=ax)
    plt.legend()
    plt.yscale("linear")
    plt.xlabel("Penetration rate", fontweight="bold")
    plt.ylabel("Hit rate", fontweight="bold")
    if SAVE_PLOT:
        plt.savefig(PLOT_OUTDIR + filename + ".png")
    else:
        plt.show()
    plt.close()


def find_best_selection_mode(fvc_subset, hashing_mode):
    hashing_mode_str = str(hashing_mode).split(".")[-1]
    best_mode = None
    lowest_pen_rate = 1.0
    for selection_mode in SELECTION_MODES:
        selection_mode_str = str(selection_mode).split(".")[-1]
        filename = fvc_subset + "-" + hashing_mode_str + \
            "-" + selection_mode_str + ".json"
        with open(RESULT_DIR + filename, "r") as file:
            data = json.load(file)
            pen_rate = get_average_pen_rate(data)
            if pen_rate < lowest_pen_rate:
                lowest_pen_rate = pen_rate
                best_mode = selection_mode
    return best_mode


def compare_hashing_modes(fvc_subset):
    label_data_map = dict()
    for hashing_mode in HASHING_MODES:
        hashing_mode_str = str(hashing_mode).split(".")[-1]
        selection_mode = find_best_selection_mode(fvc_subset, hashing_mode)
        selection_mode_str = str(selection_mode).split(".")[-1]
        filename = fvc_subset + "-" + hashing_mode_str + \
            "-" + selection_mode_str + ".json"
        with open(RESULT_DIR + filename, "r") as file:
            label_data_map[LABEL_SELECTION_MODE[selection_mode] + "+" +
                           hashing_mode_str] = json.load(file)
    plot_hit_over_pen(label_data_map, "CompareHashingModes_" + fvc_subset)


def main():
    for fvc_subset in FVC2006_SUBSETS:
        compare_selection_modes(fvc_subset)
        # compare_hashing_modes(fvc_subset)


if __name__ == "__main__":
    main()
