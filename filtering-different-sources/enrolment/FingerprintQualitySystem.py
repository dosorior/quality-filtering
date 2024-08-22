from collections import defaultdict
from math import inf
from os import lseek
from posixpath import join
import numpy as np
from numpy.core.numeric import full
import random
from math import modf
import csv
from random import randrange
import scipy.stats
import collections
from scipy.stats import describe
import math

"""Class to work on quality estimators applied to fingerprint: NFIQ2.0"""

class QualitySystemFingerPrint:

    def __init__(self):

        self.list_info_subject = []

        self.features = {}

        self.enrolled_subjects = []

    def mean_confidence_interval(self,input_list,confidence=0.95):

        a = 1.0 * np.array(input_list)

        n = len(a)

        m, se = np.mean(a), scipy.stats.sem(a)

        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

        return m, h
    
    def truncate(self,f, n):

        return math.floor(f * 10 ** n) / 10 ** n

        
    def normalisation(self, info):
        #getting json and organize info in self.dataset

        dataset = {}

        list_id = []

        list_score_q = []

        first_col = info['File name']

        second_col = info['NFIQ2 score']

        second_col = list(second_col)

        statistic_info = describe(second_col)

        max = statistic_info.minmax[1]

        min = statistic_info.minmax[0] 

        names = []

        dataset = defaultdict(list)

        scores_q = []


        for f,v in zip(list(first_col),list(second_col)):

            key = f.split('/')[-1].split('_')[0] +'_'+ f.split('/')[-1].split('_')[1] + '_' + f.split('/')[-1].split('_')[2]

            full_name = f.split('/')[-1]

            v_normalized = ((v - min)/ (max-min))

            norm_integer = [self.truncate(v_normalized, n) for n in range(7)]

            norm_integer = int(norm_integer[2] * 100)

            tuple_value = [(norm_integer,key)]

            scores_q.append(norm_integer)

            if key in dataset:

                dataset[key].append(tuple_value)
                           
            else:

                dataset[key] = [tuple_value]

        return dataset,scores_q
     
    
    def prepare_system(self,protocol,dataset,pos):

        search = []

        for p in protocol:

            if p =='Protocol1':
               
               for e in dataset:

                   info = dataset[e]

                   if len(info) >=0:

                        value = 0

                        name = ""

                        list_samples = []

                        [list_samples.append(i[0]) for i in info] #si

                        list_samples.sort(key=lambda i:(-i[0], i[1]))

                        f = list_samples[0:pos]

                        for s in range(0,pos):

                            value +=list_samples[s][0]

                            name = list_samples[s][1]
                        
                        value /= pos

                        tuple = (value,name)

                        self.enrolled_subjects.append(tuple)

                        ####Selecting a sample with higher quality score for query####

                        list_diff = list_samples[pos:]

                        search.append(list_diff[0])

        return list(search)
    

    def building_enrol (self, number,selected_s,dataset):

        enrol_s = []

        list_total_scores_q = []

        list_scores_e = []

        list_names_e = []

        search_s = []

        list_scores_s = []

        list_names_s = []

        for key in dataset:

            list_sub = dataset[key]

            enrol_s.append(list_sub[number][0][0])

            list_total_scores_q.append(list_sub[number][0][0])

            list_scores_e.append(list_sub[number][0][0])

            list_names_e.append(list_sub[number][0][1])

            for q in range(0,len(list_sub)):

                if q != number:

                    info_q_id_s = [list_sub[q][0][0], list_sub[q][0][1]]
                    
                    search_s.append(list_sub[q][0][0]) 

                    list_total_scores_q.append(list_sub[q][0][0])

                    list_scores_s.append(list_sub[q][0][0])

                    list_names_s.append(list_sub[q][0][1])
        
        self.enrolment_total = list(zip(enrol_s,list_names_e))

        self.enrolment_total.sort(key=lambda i:(-i[0], i[0]))

        count_enrolment = len(self.enrolment_total)

        return search_s,list_names_s,count_enrolment, list_total_scores_q

    def building_enrol_random(self, selected_s,dataset):

        enrol_s = []

        list_total_scores_q = []

        list_scores_e = []

        list_names_e = []

        search_s = []

        list_scores_s = []

        list_names_s = []

        for key in dataset:

            list_sub = dataset[key]

            random_index = randrange(len(list_sub))

            enrol_s.append(list_sub[random_index][0][0])

            list_total_scores_q.append(list_sub[random_index][0][0])

            list_scores_e.append(list_sub[random_index][0][0])

            list_names_e.append(list_sub[random_index][0][1])

            for q in range(0,len(list_sub)):

                if q != random_index:

                    info_q_id_s = [list_sub[q][0][0], list_sub[q][0][1]]
                    
                    search_s.append(list_sub[q][0][0]) 

                    list_total_scores_q.append(list_sub[q][0][0])

                    list_scores_s.append(list_sub[q][0][0])

                    list_names_s.append(list_sub[q][0][1])
        
        self.enrolment_total = list(zip(enrol_s,list_names_e))

        self.enrolment_total.sort(key=lambda i:(-i[0], i[0]))

        count_enrolment = len(self.enrolment_total)

        return search_s,list_names_s,count_enrolment, list_total_scores_q


    def organizing_enrol(self):

        self.mark = {}

        self.hash_table_q = {}

        for e in self.enrolment_total: 

            score = e[0]

            id = e[1]

            if score in self.hash_table_q:

                self.hash_table_q[score].append(id)
            
            else:

                self.hash_table_q[score] = [(id)]

                self.mark[score] = False

        collections.OrderedDict(sorted(self.hash_table_q.items()))


    def __search_tree_improved(self, score_probe, labels_e):

        cost = 0

        finished = False

        for item in self.mark:

            self.mark[item] = False

        keys = list(self.hash_table_q.keys())

        queue = []

        result = []

        higher = 0

        lower = 0

        dist_prob_lower = 0

        dist_prob_higher = 0

        list_labels_final = []

        if not score_probe in self.hash_table_q:

            keys_dict = np.array(keys)

            try:
                higher = keys_dict[keys_dict > score_probe].min()  

            except ValueError:  #raised if `y` is empty.
                pass

            try:
                lower = keys_dict[keys_dict < score_probe].max()

            except ValueError:  #raised if `y` is empty.
                pass

            if lower > 0:
                
                dist_prob_lower = score_probe - lower
            
            if higher > 0:

                dist_prob_higher = score_probe - higher


            if ((dist_prob_lower == dist_prob_higher) and (dist_prob_lower != 0 and dist_prob_lower != 0)):

                queue.append(lower)

                queue.append(higher)
            
            else:

                if ((dist_prob_higher < dist_prob_lower) and (dist_prob_higher != 0)):
                
                    queue.append(higher)
                
                if ((dist_prob_higher > dist_prob_lower) and (dist_prob_lower != 0)):

                    queue.append(lower)

        else:

            queue.append(score_probe)

        while(len(queue) > 0) :

            p = queue.pop(0)

            if(p in self.hash_table_q and not self.mark[p]):

                self.mark[p] = True

                ref_values = self.hash_table_q[p]

                k = keys.index(p)

                cost += len(ref_values)

                if labels_e in ref_values:

                    finished = True

                    list_labels_final.append(labels_e)

                    break
                
                else:

                    if(k + 1 < len(keys)):

                        queue.append(keys[(k + 1)])
                    
                    if(k - 1 >= 0):

                        queue.append(keys[(k - 1)])          
                    
                        
        return cost, list_labels_final


    def nearest_improved(self,score_q_s, labels_s):

        list_comp = []

        list_final = []

        for score_probe, label in zip(score_q_s,labels_s):

            count_comp,label_f = self.__search_tree_improved(score_probe, label)

            list_comp.append(count_comp)

            list_final.append(label_f)

        ave_comp, variance = self.mean_confidence_interval(list_comp)

        print("Size recorrido {}".format(len(list_final)))

        return ave_comp, variance
    
    def __statistics(self, score_probe, labels_e):

        cost = 0

        counter_search = 0

        bin_visited = -1
        
        for item in self.mark:

            self.mark[item] = False

        keys = list(self.hash_table_q.keys())

        queue = []

        not_similar_bin = 0

        if not score_probe in self.hash_table_q:

            not_similar_bin = 1

            return not_similar_bin,counter_search,cost, bin_visited
          
        else:

            queue.append(score_probe)
        
            p = queue.pop(0)

            if(p in self.hash_table_q and not self.mark[p]):

                self.mark[p] = True

                ref_values = self.hash_table_q [p]
                        
                if labels_e in ref_values:

                    counter_search = 1

                    cost = len(ref_values)

                    bin_visited = p
                            
            return not_similar_bin, counter_search,cost,bin_visited

    def Statistics_general(self,score_q_s, labels_s):

        list_comp = []

        list_final = []

        counter_s = 0

        counter_not_similar = 0

        list_cost_most_similar = []

        length_dict = {key: len(value) for key, value in self.hash_table_q.items()}

        values_key = length_dict.values()

        list_bin_most_similar_visited = []

        value = 0

        Comp_tota = 0

        for e in values_key:

            value += e
        
        Comp_tota = value

        value /= len(values_key)

        min_bin = min(values_key)

        max_bin = max(values_key)

        total_bins = len(length_dict)

        count_comp_non_visited = 0

        for score_probe, label in zip(score_q_s,labels_s):

            not_similar_bin,counter_search,cost,bin_visited = self.__statistics(score_probe, label)

            if counter_search != 0:

                counter_s +=1

                list_cost_most_similar.append(cost)

            if not_similar_bin !=0:
                
                counter_not_similar +=1

            if bin_visited != -1:

                list_bin_most_similar_visited.append(bin_visited)

        unique_bin_visited = np.unique(list_bin_most_similar_visited)
       
        ave_comp_most_similar, variance_comp_most_similar = self.mean_confidence_interval(list_cost_most_similar)

        return counter_not_similar, counter_s,ave_comp_most_similar, min_bin, max_bin,total_bins,value,count_comp_non_visited


    def __counter_visited(self, score_probe, labels_e):

        cost = 0

        bin_visited = 0

        list_bin_visited = []
        
        list_cost = []

        for item in self.mark:

            self.mark[item] = False

        keys = list(self.hash_table_q.keys())

        queue = []

        result = []

        higher = 0

        lower = 0

        dist_prob_lower = 0

        dist_prob_higher = 0

        list_labels_final = []

        if not score_probe in self.hash_table_q:

            # return 0,[]

            keys_dict = np.array(keys)

            try:
                higher = keys_dict[keys_dict > score_probe].min()  

            except ValueError:  #raised if `y` is empty.
                pass

            try:
                lower = keys_dict[keys_dict < score_probe].max()

            except ValueError:  #raised if `y` is empty.
                pass

            if lower > 0:
                
                dist_prob_lower = score_probe - lower
            
            if higher > 0:

                dist_prob_higher = score_probe - higher    

        else:

            queue.append(score_probe)

        while(len(queue) > 0) :

            p = queue.pop(0)

            if(p in self.hash_table_q and not self.mark[p]):

                bin_visited +=1

                list_bin_visited.append(bin_visited)

                self.mark[p] = True

                ref_values = self.hash_table_q[p]

                k = keys.index(p)

                cost += len(ref_values)

                list_cost.append(cost)

                if labels_e in ref_values:

                    list_labels_final.append(labels_e)

                    break
                
                else:

                    if(k + 1 < len(keys)):

                        queue.append(keys[(k + 1)])
                    
                    if(k - 1 >= 0):

                        queue.append(keys[(k - 1)])          
                    
                        
        return list_bin_visited,list_cost

    def bins_visited(self,score_q_s, labels_s, model):

        list_bin_visited = []

        list_cost  = []

        count_bin_high = 0

        list_higher_bin = []

        list_higher_cost = []

        for score_probe, label in zip(score_q_s,labels_s):

            list_bin_visited,list_cost = self.__counter_visited(score_probe, label)

            if len(list_bin_visited) > count_bin_high:

                count_bin_high = len(list_bin_visited)

                list_higher_bin = [*list_higher_bin, *list_bin_visited]

                list_higher_cost = [*list_higher_cost, *list_cost]

    
        return list_bin_visited, list_cost      
    


       


    



