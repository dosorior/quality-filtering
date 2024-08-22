import argparse
from pathlib import Path
from enrolment.QualitySystemStatisticalFace import QualitySystemStatsFace
import os
import json
from collections import defaultdict
import numpy as np
import csv
import pandas as pd
import pathlib
import scipy

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

    scores_used = []
    
    for e in dataset:

        info = dataset[e]

        if len(info) > 1:   ##original >=2

            list_samples = []

            [list_samples.append(i[0]) for i in info]

            [scores_used.append(i[0][0]) for i in info]

            [subjects_selected.append(list_samples[i])for i in range(0, len(list_samples))]
    
    final_list = {}

    final_list = defaultdict(list)

    for e in range(0, len(subjects_selected)):

        key = ""

        name = subjects_selected[e][1].split('_')

        for i in range(0, len(name)-1):

            key += name[i] + "_" 
                
        if key in final_list:

            final_list[key].append(subjects_selected[e])
                
        else:

            final_list[key] = [subjects_selected[e]]
    
    return final_list,scores_used

"""Quality filtering applied to the Face-based source """
parser = argparse.ArgumentParser(description='QualityIndexingOneToFirst',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-e', '--embeddings', type=str, help='path to the face embeddings extracted')

parser.add_argument('-o', '--output', type=str, help= 'path to save the results')

parser.add_argument('-m', '--model', type=str, help= 'path where are the files with the quality information')

parser.add_argument('-n', '--name', type=str, help= 'name of the quality estimator used: e.g. MagFace or e.g. SER-FIQ ')

parser.add_argument('-s', '--samples', type=int, help= 'number of samples to select per subject')

args = parser.parse_args()

save_path_csv = os.path.join(args.output, "Face_Statistics_{}.csv".format(args.name))

path_scores_total = os.path.join(args.output, "face_scores_q_total_selected_{}".format(args.name))

path_scores_used = os.path.join(args.output, "face_scores_q_used_{}.csv".format(args.name))

with open(save_path_csv, 'w', newline='') as f:

    fieldnames = ['Biometric','Quality', 'Sensor', 'Total_bins','Max_bin','Min_bin' ,'Bin_not_similar','Percent_bin_not_similar' ,'Bin_most_similar','Percent_bin_most_similar' ,'#_Comparisons_Bin_most_similar','Comp_Ave_Total','Total']

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()

    list_counter_not_similar_sf,list_counter_s_sf,list_ave_comp_most_similar_sf,list_min_bin_sf,list_max_bin_sf,list_total_bins_sf,score_q_s_sf,list_comp_bin_total = [],[],[],[],[],[],[],[]

    for round in range(0,10):

        embeddings_path = list(Path(args.embeddings).rglob('*.npy'))

        enrol_model = QualitySystemStatsFace()

        penetration_rate = 0

        #loading info_json
        path_file_json = os.path.join(args.model,'211006-1547-Extracted-FIQAA-LFW-data.json') ###loading quality information for X(e.g. LFW) Database

        file_json = open(path_file_json)

        info_json = json.load(file_json)

        ### Defining quality estimators for Face
        dataset_fq, scores_q_normalized_fq = enrol_model.normalisation(info_json,'FaceQnet_v1')

        dataset_mf, scores_q_normalized_mf = enrol_model.normalisation(info_json,'MagFace')

        dataset_sf, scores_q_normalized_sf = enrol_model.normalisation(info_json,'SER-FIQ')

        selected_subjects_fq,scores_used_fq = gettingdata(dataset_fq) ###selecting data

        selected_subjects_mf,scores_used_mf = gettingdata(dataset_mf) ###selecting data

        selected_subjects_sf,scores_used_sf = gettingdata(dataset_sf) ###selecting data

        selected_fq = list(selected_subjects_fq)

        selected_fq.sort()

        selected_mf = list(selected_subjects_mf)

        selected_mf.sort()

        selected_sf = list(selected_subjects_sf)

        selected_sf.sort()
        
        score_q_s_sf, labels_s_sf, count_e = enrol_model.building_enrol_fusion(selected_fq,selected_subjects_fq, selected_subjects_mf, selected_subjects_sf, args.name) ###building enrolment

        total_search = len(labels_s_sf)
        
        enrol_model.organizing_enrol_fusion(args.name)

        counter_not_similar_sf, counter_s_sf,ave_comp_most_similar_sf, min_bin_sf, max_bin_sf,total_bins_sf,value = enrol_model.Statistics_general(score_q_s_sf, labels_s_sf, args.name)

        list_comp_bin_total.append(value)

        list_counter_not_similar_sf.append(counter_not_similar_sf)

        list_counter_s_sf.append(counter_s_sf)

        list_ave_comp_most_similar_sf.append(ave_comp_most_similar_sf)

        list_min_bin_sf.append(min_bin_sf)

        list_max_bin_sf.append(max_bin_sf)

        list_total_bins_sf.append(total_bins_sf)   

    ###Computing stat
    mean_comp_bin_total = mean_confidence_interval(list_comp_bin_total)

    mean_counter_not_similar_sf =  mean_confidence_interval(list_counter_not_similar_sf)

    percent_counter_not_similar_sf = (mean_counter_not_similar_sf * 100) / len(score_q_s_sf)

    mean_counter_s_sf =  mean_confidence_interval(list_counter_s_sf)

    percent_counter_s_sf = (mean_counter_s_sf * 100) / len(score_q_s_sf)

    mean_ave_comp_most_similar_sf =  mean_confidence_interval(list_ave_comp_most_similar_sf)

    mean_min_bin_sf =  mean_confidence_interval(list_min_bin_sf)

    mean_max_bin_sf = mean_confidence_interval(list_max_bin_sf)

    mean_total_bins_sf = mean_confidence_interval(list_total_bins_sf)

    writer.writerow({'Biometric': "Face",'Quality':args.name, 'Sensor': "Visible", 'Total_bins':mean_total_bins_sf,'Max_bin':mean_max_bin_sf,'Min_bin':mean_min_bin_sf ,'Bin_not_similar':mean_counter_not_similar_sf,'Percent_bin_not_similar':percent_counter_not_similar_sf ,'Bin_most_similar':mean_counter_s_sf, 'Percent_bin_most_similar':percent_counter_s_sf,'#_Comparisons_Bin_most_similar':mean_ave_comp_most_similar_sf,'Comp_Ave_Total':mean_comp_bin_total ,'Total':len(score_q_s_sf)})



