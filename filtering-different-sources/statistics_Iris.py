import argparse
from pathlib import Path
from enrolment.IrisQualitySystem import QualitySystemIris
import os
import json
from collections import defaultdict
import numpy as np
import csv
import pandas as pd
import pathlib
import scipy
from scipy.stats import describe


def mean_confidence_interval(input_list, confidence=0.95):

        a = 1.0 * np.array(input_list)

        n = len(a)

        m, se = np.mean(a), scipy.stats.sem(a)

        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

        return m

def gettingdata(dataset):

    subjects_selected = []

    feat_selected = []

    list_path_emb = []
    
    for e in dataset:

        info = dataset[e]

        list_samples = []

        [list_samples.append(i[0]) for i in info]

        [subjects_selected.append(list_samples[i])for i in range(0, len(list_samples))]
    
    subjects_selected.sort(key=lambda i:(-i[0], i[1]))

    final_list = {}

    final_list = defaultdict(list)

    for e in range(0, len(subjects_selected)):

        key = subjects_selected[e][1]
                
        if key in final_list:

            final_list[key].append(subjects_selected[e])
                
        else:

            final_list[key] = [subjects_selected[e]]
    
    return final_list

"""Quality filtering applied to the Iris-based source """

parser = argparse.ArgumentParser(description='Quality Filtering',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-e', '--embeddings', type=str, help='path to the face embeddings extracted')

parser.add_argument('-o', '--output', type=str, help= 'path to save the results')

parser.add_argument('-m', '--model', type=str, help= 'path where are the files with the quality information')

parser.add_argument('-n', '--name', type=str, help= 'name of the quality estimator used: e.g. MagFace or e.g. SER-FIQ ')

parser.add_argument('-s', '--samples', type=int, help= 'number of samples to select per subject')

args = parser.parse_args()

save_path_csv = os.path.join(args.output, "Iris_Statistics{}.csv".format(args.name))

path_scores_total = os.path.join(args.output, "iris_scores_q_total_{}.csv".format(args.name))

path_scores_used = os.path.join(args.output, "iris_scores_q_used_{}.csv".format(args.name))

with open(save_path_csv, 'w', newline='') as f:

    fieldnames = ['Biometric','Quality', 'Sensor', 'Total_bins','Max_bin','Min_bin' ,'Bin_not_similar','Percent_bin_not_similar' ,'Bin_most_similar','Percent_bin_most_similar' ,'#_Comparisons_Bin_most_similar','Comp_Ave_Total','comp_bin_non_visited','Total']

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()

    variance_total = []

    comp_total = []

    variance_final_comp = 0

    comp_final = 0

    penetration_rate_total = 0

    list_p_r_total = []

    variance_p_r = 0

    variance_total = []

    count_e = 0

    labels_s = []

    list_comp_bin_non_visited = []

    list_comp_bin_total = []

    list_counter_not_similar_sf = []

    list_counter_s_sf = []

    list_ave_comp_most_similar_sf = []

    list_min_bin_sf = []

    list_max_bin_sf = []

    list_total_bins_sf = []

    score_q_s = []

    for round in range(0,10):

        enrol_model = QualitySystemIris()

        penetration_rate = 0

        #loading info_json
        path_file_json = os.path.join(args.model,'area_percent_iris_completed.csv')

        data_oirg = pd.read_csv(path_file_json,sep=',')  ### this works
    
        info_data = pd.DataFrame(data_oirg)

        dataset,scores_q_normalized = enrol_model.normalisation(info_data)

        statistic_info = describe(scores_q_normalized)

        max = statistic_info.minmax[1]

        min = statistic_info.minmax[0] 

        score_q_s, labels_s, count_e, list_scores_q_used = enrol_model.building_enrol_random(dataset)

        enrol_model.organizing_enrol()

        counter_not_similar_sf, counter_s_sf,ave_comp_most_similar_sf, min_bin_sf, max_bin_sf,total_bins_sf,value,count_comp_non_visited = enrol_model.Statistics_general(score_q_s, labels_s)

        list_comp_bin_non_visited.append(count_comp_non_visited)

        list_comp_bin_total.append(value)

        list_counter_not_similar_sf.append(counter_not_similar_sf)

        list_counter_s_sf.append(counter_s_sf)

        list_ave_comp_most_similar_sf.append(ave_comp_most_similar_sf)

        list_min_bin_sf.append(min_bin_sf)

        list_max_bin_sf.append(max_bin_sf)

        list_total_bins_sf.append(total_bins_sf)   


    
    mean_comp_bin_non_visited = mean_confidence_interval(list_comp_bin_non_visited)

    mean_counter_not_similar_sf =  mean_confidence_interval(list_counter_not_similar_sf)

    percent_counter_not_similar_sf = (mean_counter_not_similar_sf * 100) / len(score_q_s)

    mean_counter_s_sf =  mean_confidence_interval(list_counter_s_sf)

    percent_counter_s_sf = (mean_counter_s_sf * 100) / len(score_q_s)

    mean_ave_comp_most_similar_sf =  mean_confidence_interval(list_ave_comp_most_similar_sf)

    mean_min_bin_sf =  mean_confidence_interval(list_min_bin_sf)

    mean_max_bin_sf = mean_confidence_interval(list_max_bin_sf)

    mean_total_bins_sf = mean_confidence_interval(list_total_bins_sf)
    
    mean_comp_bin_total = mean_confidence_interval(list_comp_bin_total)
    
    writer.writerow({'Biometric': "Iris",'Quality':"Visible Iris Area", 'Sensor': "infrared", 'Total_bins':mean_total_bins_sf,'Max_bin':mean_max_bin_sf,'Min_bin':mean_min_bin_sf ,'Bin_not_similar':mean_counter_not_similar_sf,'Percent_bin_not_similar':percent_counter_not_similar_sf ,'Bin_most_similar':mean_counter_s_sf, 'Percent_bin_most_similar':percent_counter_s_sf,'#_Comparisons_Bin_most_similar':mean_ave_comp_most_similar_sf,'Comp_Ave_Total':mean_comp_bin_total ,'comp_bin_non_visited':mean_comp_bin_non_visited,'Total':len(score_q_s)})
