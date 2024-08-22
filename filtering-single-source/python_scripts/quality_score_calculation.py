from datetime import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from json_utils import *


def get_datetime_string():
    return datetime.today().strftime("%y%m%d_") + datetime.now().strftime("%H_%M_%S")


def plot_kde(observed_values, title, xlab):
    sns.set_style('whitegrid')
    p = sns.kdeplot(observed_values, bw=0.15) 
    p.set_xlabel(xlab)
    p.set_ylabel("Estimated density")
    plt.title(title)
    plt.show()


def normalize_data(observed_values):
    mu = np.mean(observed_values)
    sd = np.std(observed_values)
    return (observed_values - mu) / sd 


def main():
    MiDeCon_results_dir = "../data/FVC2006_MiDeCon_Output/DB2_B/"
    database_name = MiDeCon_results_dir.split("/")[-2] # Last item is empty
    output_dir = "../data/output_QualityScore_" + get_datetime_string() + "/" + database_name + "/"

    processed = load_and_process_fingerprints(MiDeCon_results_dir)
    scores = get_scores(processed)
    means = get_means(processed)
    stds = get_stds(processed)

    title_data_descr = "\nExtracted by CoarseNet from FVC2006 dataset (DB2_B). N = " + str(len(scores))
    title_scores = "FineNet scores across all minutiae." + title_data_descr
    plot_kde(scores, title=title_scores, xlab="FineNet scores")

    title_data_descr = "\nExtracted by CoarseNet from FVC2006 dataset (DB2_B). N = " + str(len(means))

    title_mean = "FineNet score mean for each minutia." + title_data_descr
    plot_kde(means, title=title_mean, xlab="Mean")
    title_std = "FineNet score standard deviation for each minutia." + title_data_descr
    plot_kde(stds, title=title_std, xlab="Standard deviation")

    qualities = means + stds
    title_qualities = "MiDeCon quality score for each minutia." + title_data_descr
    plot_kde(qualities, title=title_qualities, xlab="Quality score")

    qualities_alt = means - stds
    title_qualities = "MiDeCon quality score (alternative) for each minutia." + title_data_descr
    plot_kde(qualities_alt, title=title_qualities, xlab="Quality score")
    
    
    qualities_alt_norm = normalize_data(means) - normalize_data(stds)
    title_qualities = "MiDeCon quality score (alternative, norm) for each minutia." + title_data_descr
    plot_kde(qualities_alt_norm, title=title_qualities, xlab="Quality score")
    

if __name__ == "__main__":
    main()