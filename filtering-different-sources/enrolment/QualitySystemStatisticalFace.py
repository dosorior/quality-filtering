from collections import defaultdict
from genericpath import exists
from math import cos, inf
from os import lseek, remove
from posixpath import join
import numpy as np
from numpy.core.fromnumeric import var
from numpy.core.numeric import full
import random
from math import modf
import math
import pathlib
import os
from scipy.spatial import distance
import collections
from scipy.stats import describe
import operator
from random import randrange
import scipy.stats
from numpy.lib.function_base import append

"""Class to work on quality estimators applied to face: FaceQnet_v1, FaceQnet_v0, SER-FIQ, and MagFace"""

class QualitySystemStatsFace:

    def __init__(self):

        self.list_info_subject = []

        self.features = {}

        self.enrolled_subjects = []

        self.mark_fq = {}

        self.mark_sf = {}

        self.mark_mf = {}

        self.mark = {}

        self.enrolment_total = []

        self.enrolment_total_fq = []

        self.enrolment_total_mf = []

        self.enrolment_total_sf = []

        self.hash_table_q = {}

        self.hash_table_q_rand = {}

        self.hash_table_q_fq = {}

        self.hash_table_q_mf = {}

        self.hash_table_q_sf = {}

    
    def truncate(self,f, n):

        return math.floor(f * 10 ** n) / 10 ** n

    def cleaner(self, name):

        new_name = ""

        full_name = name.split('_')

        for i in range(0, len(full_name)-1):

            new_name += full_name[i]+'_' 
        
        return new_name.strip()

    def mean_confidence_interval(self,input_list, confidence=0.95):

        a = 1.0 * np.array(input_list)

        n = len(a)

        m, se = np.mean(a), scipy.stats.sem(a)

        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

        return m, h
    

    def IntersecOfSets(self,arr1, arr2):
    # Converting the arrays into sets
        s1 = set(arr1)
        s2 = set(arr2)
        final_list = []
        # on set1 and s3
        set1 = s1.intersection(s2) 
        # Converts resulting set to list
        final_list = list(set1)
    
        return final_list
    

    def difference(self,arr1, arr2):

        s1 = set(arr1)
        s2 = set(arr2)

        difference_three = []

        difference = s1.symmetric_difference(s2)

        difference_three = list(difference)

        return difference_three

    ###Organise and normalize quality scores according to the quality estimator
    def normalisation(self, file, model):
        #getting json and organize info in self.dataset

        dataset = {}

        scores_processed_norm,list_id,list_score_q,values_scores = [],[],[],[]

        structure = dict(file)

        dataset = defaultdict(list)

        q_score = 0

        for qa in structure:

            if model in qa:

                data_subject = structure[qa]

                values_scores = list(data_subject.values())

                statistic_info = describe(values_scores)

                max = statistic_info.minmax[1]

                min = statistic_info.minmax[0] 

                for data in data_subject:

                    if model == 'FaceQnet_v1':

                        score_normalized = ((data_subject[data] - min)/ (max-min))

                    if model == 'SER-FIQ' :

                        ###normalize bwt 0 and 1

                        score_normalized = ((data_subject[data] - min)/ (max-min))

                    elif model == 'MagFace':

                        score_normalized = ((data_subject[data] - min)/ (max-min))
                    
                    if model == 'FaceQnet_v1' or model == 'FaceQnet_v0' or model == 'SER-FIQ' or model == 'MagFace':

                        q_score = [self.truncate(score_normalized, n) for n in range(7)]

                        q_score = int(q_score[2] * 100)

                        scores_processed_norm.append(q_score)


                    key = data.split('/')[0]

                    name = " "

                    full_name = data.split('/')[1]

                    full_name = full_name.split('_')

                    for i in range(0, len(full_name)):

                        name += full_name[i]+'_' 

                    tuple_value = [(q_score,(name.split('.jpg')[0]).strip())]

                    list_id.append(key)

                    list_score_q.append(q_score)

                    if key in dataset:

                        dataset[key].append(tuple_value)
                    
                    else:

                        dataset[key] = [tuple_value]
        
        return dataset,scores_processed_norm

    ###Prepare the filtering on the system
    def prepare_system(self,dataset,enrol_selected, path_emb):

        enrol_general = {}

        feat_selected,enrol_g_select = [],[]

        for key in enrol_selected:

            enrol_general[key] = dataset[key]

        for e in enrol_general:

            list_e = enrol_general[e]

            for l in list_e:

                id = l[1] + '.npy'

                tuple = [l[0],l[1]]

                file = pathlib.Path(os.path.join(path_emb, id))

                if file.exists ():

                    feat = np.load(str(file))

                    feat_selected.append(feat)

                    enrol_g_select.append(tuple)


                else:

                    print(id)

        return enrol_g_select,feat_selected
    
    ###Prepare enrolment
    def building_enrol(self, number_s, selected_s,feat):

        enrol_s,search_s,feat_e,list_names_e,list_names_s,feat_s = [],[],[],[],[],[]

        temp = {}

        temp = defaultdict(list)

        for d,f in zip(selected_s,feat):

            key = " "

            id = d[1].split('_')

            for i in range(0, len(id)-1):

                key += id[i]+'_' 

            tuple = [d, f]

            if key in temp:

                temp[key].append(tuple)
                    
            else:

                temp[key] = [tuple]

        list_total_scores_q = []

        list_scores_e = []

        list_scores_s = []

        for t in temp:

            list_sub = temp[t]

            enrol_s.append(list_sub[number_s][0][0])

            list_total_scores_q.append(list_sub[number_s][0][0])

            list_scores_e.append(list_sub[number_s][0][0])

            list_names_e.append(list_sub[number_s][0][1])

            feat_e.append(list_sub[number_s][1])

            for q in range(0,len(list_sub)):

                if q != number_s:
                    
                    search_s.append(list_sub[q][0][0]) 

                    list_total_scores_q.append(list_sub[q][0][0])

                    list_scores_s.append(list_sub[q][0][0])

                    name_fixed = self.cleaner(list_sub[q][0][1])

                    list_names_s.append(name_fixed)

                    feat_s.append(list_sub[q][1])
        
        self.enrolment_total = list(zip(enrol_s,list_names_e,feat_e))

        self.enrolment_total.sort(key=lambda i:(-i[0], i[0]))

        count_enrolment = len(self.enrolment_total)

        return search_s,list_names_s, feat_s,count_enrolment, list_total_scores_q

    def building_enrol_three(self, number_s, selected_s,selected_subjects):

        enrol_s = []

        search_s = []

        temp = {}

        list_names_e = []

        list_names_s = []

        temp = defaultdict(list)

        list_total_scores_q = []

        list_scores_e = []

        list_scores_s = []

        for t in selected_s:

            list_sub = selected_subjects[t]

            enrol_s.append(list_sub[number_s][0])

            list_total_scores_q.append(list_sub[number_s][0])

            list_scores_e.append(list_sub[number_s][0])

            list_names_e.append(list_sub[number_s][1])

            for q in range(0,len(list_sub)):

                if q != number_s:
                    
                    search_s.append(list_sub[q][0]) 

                    list_total_scores_q.append(list_sub[q][0])

                    list_scores_s.append(list_sub[q][0])

                    name_fixed = self.cleaner(list_sub[q][1])

                    list_names_s.append(name_fixed)
        
        self.enrolment_total = list(zip(enrol_s,list_names_e))

        self.enrolment_total.sort(key=lambda i:(-i[0], i[0]))

        count_enrolment = len(self.enrolment_total)

        return search_s,list_names_s,count_enrolment, list_total_scores_q

    def building_enrol_fusion(self, selected_s,selected_subjects_fq, selected_subjects_mf, selected_subjects_sf, model):

        enrol_s_fq = []

        enrol_s_mf = []

        enrol_s_sf = []

        search_s = []

        list_names_e = []

        list_names_s = []

        for t in selected_s:

            list_sub_fq = selected_subjects_fq[t]

            list_sub_mf = selected_subjects_mf[t]

            list_sub_sf = selected_subjects_sf[t]

            random_index = randrange(len(list_sub_fq))

            enrol_s_fq.append(list_sub_fq[random_index][0])

            enrol_s_mf.append(list_sub_mf[random_index][0])

            enrol_s_sf.append(list_sub_sf[random_index][0])

            list_names_e.append(list_sub_fq[random_index][1])

            for q in range(0,len(list_sub_sf)):

                if q != random_index:

                    if model == 'FaceQnet_v1':
                    
                        search_s.append(list_sub_fq[q][0]) 

                    elif model == 'SER-FIQ':

                        search_s.append(list_sub_sf[q][0]) 
                    
                    else:

                        search_s.append(list_sub_mf[q][0]) 

                    name_fixed = self.cleaner(list_sub_fq[q][1])

                    list_names_s.append(name_fixed)
        
        self.enrolment_total_fq = list(zip(enrol_s_fq,list_names_e))

        self.enrolment_total_sf = list(zip(enrol_s_sf,list_names_e))

        self.enrolment_total_mf = list(zip(enrol_s_mf,list_names_e))

        count_enrolment = len(self.enrolment_total_mf)

        return search_s,list_names_s,count_enrolment


    def building_enrol_random(self, selected_s, dataset):

        enrol_s = []

        search_s = []

        feat_e = []

        feat_s = []

        temp = {}

        list_names_e = []

        list_names_s = []

        list_scores_q_used = []

        list_total_scores_q = []

        list_scores_e = []

        list_scores_s = []

        for t in selected_s:

            list_sub = dataset[t]

            # enrol_s.append(list_sub[number_s][0][0])

            random_index = randrange(len(list_sub))

            enrol_s.append(list_sub[random_index][0])

            list_total_scores_q.append(list_sub[random_index][1])

            list_scores_e.append(list_sub[random_index][0])

            list_names_e.append(list_sub[random_index][1])

            for q in range(0,len(list_sub)):

                if q != random_index:

                    info_q_id_s = [list_sub[q][0], list_sub[q][1]]
                    
                    search_s.append(list_sub[q][0]) 

                    list_total_scores_q.append(list_sub[q][0])

                    list_scores_s.append(list_sub[q][0])

                    list_scores_e.append(list_sub[q][0])

                    name_fixed = self.cleaner(list_sub[q][1])

                    list_names_s.append(name_fixed)
        
        self.enrolment_total = list(zip(enrol_s,list_names_e))

        self.enrolment_total.sort(key=lambda i:(-i[0], i[0]))

        count_enrolment = len(self.enrolment_total)

        return search_s,list_names_s,count_enrolment,list_scores_e

    def organizing_enrol(self):

        self.mark = {}

        self.hash_table_q = {}

        for e in self.enrolment_total: 

            score = e[0]

            id = e[1]

            id = self.cleaner(id)

            if score in self.hash_table_q:

                self.hash_table_q[score].append(id)

            else:

                self.hash_table_q[score] = [(id)]


                self.mark[score] = False

        collections.OrderedDict(sorted(self.hash_table_q.items()))
    
    def organizing_enrol_fusion(self, model):

        self.mark_fq = {}

        self.mark_sf = {}

        self.mark_mf = {}

        self.hash_table_q_sf = {}

        self.hash_table_q_fq = {}

        self.hash_table_q_mf = {}

        for fq,mf,sf in zip(self.enrolment_total_fq,self.enrolment_total_mf,self.enrolment_total_sf): 

            score_fq = fq[0]

            score_sf = sf[0]

            score_mf = mf[0]

            id = fq[1]

            id = self.cleaner(id)

            if score_fq in self.hash_table_q_fq:

                self.hash_table_q_fq[score_fq].append(id)
            
            else:

                self.hash_table_q_fq[score_fq] = [(id)]

                self.mark_fq[score_fq] = False
            
            if score_mf in self.hash_table_q_mf:

                self.hash_table_q_mf[score_mf].append(id)

            else:

                self.hash_table_q_mf[score_mf] = [(id)]

                self.mark_mf[score_mf] = False

            
            if score_sf in self.hash_table_q_sf:

                self.hash_table_q_sf[score_sf].append(id)

            else:

                self.hash_table_q_sf[score_sf] = [(id)]

                self.mark_sf[score_sf] = False
            
        if model == 'FaceQnet_v1':

            self.hash_table_q = self.hash_table_q_fq

            self.mark = self.mark_fq

        elif model == 'SER-FIQ':

            self.hash_table_q = self.hash_table_q_sf

            self.mark = self.mark_sf
        
        else:

            self.hash_table_q = self.hash_table_q_mf

            self.mark = self.mark_mf

        
        self.hash_table_q_fq  = sorted(self.hash_table_q_fq.items(),key=operator.itemgetter(0),reverse=True)

        self.hash_table_q_fq = dict(self.hash_table_q_fq)

        self.hash_table_q_mf  = sorted(self.hash_table_q_mf.items(),key=operator.itemgetter(0),reverse=True)

        self.hash_table_q_mf = dict(self.hash_table_q_mf)

        self.hash_table_q_sf  = sorted(self.hash_table_q_sf.items(),key=operator.itemgetter(0),reverse=True)

        self.hash_table_q_sf = dict(self.hash_table_q_sf)

        self.hash_table_q = sorted(self.hash_table_q.items(),key=operator.itemgetter(0),reverse=True)

        self.hash_table_q = dict(self.hash_table_q)




    def organizing_random(self):

        self.mark = {}

        self.hash_table_q = {}

        for e in self.enrolment_total: 

            score = e[0]

            id = e[1]

            feat = e[2]

            if score in self.hash_table_q:

                self.hash_table_q[score].append((id,feat))
            
            else:

                self.hash_table_q[score] = [(id,feat)]

                self.mark[score] = False

        list_keys = list(self.hash_table_q.keys())

        random.shuffle(list_keys)

        for key_r in list_keys:
            
            self.hash_table_q_rand[key_r] = self.hash_table_q[key_r]  

    
    def sorting_ascending_keys(self):

        self.hash_table_q = dict(sorted(self.hash_table_q.items(), key=lambda x: x[0]))

    def __search_tree_min_bin(self, score_probe, labels_e, model):

        cost = 0

        contador_search = 0

        for item in self.mark_fq:

            self.mark_fq[item] = False

        for item in self.mark_mf:

            self.mark_mf[item] = False

        for item in self.mark_sf:

            self.mark_sf[item] = False
        
        for item in self.mark:

            self.mark[item] = False

        keys = list(self.hash_table_q.keys())

        queue = []

        higher = 0

        lower = 0

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

            if lower != 0:
                
                queue.append(lower)

            elif higher !=0:

                queue.append(higher)
          
        else:

            queue.append(score_probe)
        
        while(len(queue) > 0) :

            p = queue.pop(0)

            if(p in self.hash_table_q and not self.mark[p]):

                self.mark[p] = True

                ref_values = []

                count = 0

                count_sf = 0

                count_mf = 0

                count_fq = 0

                count = len(self.hash_table_q[p])

                menor = count

                               
                if model == 'FaceQnet_v1':

                    if(p in self.hash_table_q_sf):

                        if len(self.hash_table_q_sf[p]) < menor:

                            menor = len(self.hash_table_q_sf[p])

                            ref_values = self.hash_table_q_sf[p]
                    
                    if(p in self.hash_table_q_mf):
                        
                        if len(self.hash_table_q_mf[p]) < menor:

                            menor = len(self.hash_table_q_mf[p])

                            ref_values = self.hash_table_q_mf[p]
                
                elif model == 'SER-FIQ':

                    if(p in self.hash_table_q_fq):

                        if len(self.hash_table_q_fq[p]) < menor:

                            menor = len(self.hash_table_q_fq[p])

                            ref_values = self.hash_table_q_fq[p]
                    
                    if(p in self.hash_table_q_mf):
                        
                        if len(self.hash_table_q_mf[p]) < menor:

                            menor = len(self.hash_table_q_mf[p])

                            ref_values = self.hash_table_q_mf[p]
                
                elif model == 'MagFace':

                    if(p in self.hash_table_q_sf):

                        if len(self.hash_table_q_sf[p]) < menor:

                            menor = len(self.hash_table_q_sf[p])

                            ref_values = self.hash_table_q_sf[p]
                    
                    if(p in self.hash_table_q_fq):
                        
                        if len(self.hash_table_q_fq[p]) < menor:

                            menor = len(self.hash_table_q_fq[p])

                            ref_values = self.hash_table_q_fq[p]


                if len(ref_values) == 0:

                    ref_values = self.hash_table_q[p]

                k = keys.index(p)

                cost += len(ref_values)

                if labels_e in ref_values:

                    contador_search = 1

                    break
                
                else:

                    if(k + 1 < len(keys)):

                        queue.append(keys[(k + 1)])
                    
                    if(k - 1 >= 0):

                        queue.append(keys[(k - 1)])    

                        
        return cost,contador_search
    
    def fusion_min_bin(self,score_q_s, labels_s, model):

        list_comp = []

        list_final = []

        number_encontrado = 0

        for score_probe, label in zip(score_q_s,labels_s):

            count_comp, contador_s = self.__search_tree_min_bin(score_probe, label, model)

            list_comp.append(count_comp)

            if contador_s != 0:

                number_encontrado += 1
        
        print("La cantidad de elementos encontrados por fusion is {}".format(number_encontrado))

        ave_comp, variance = self.mean_confidence_interval(list_comp)

        print("Size recorrido {}".format(len(list_final)))

        return ave_comp, variance

    def __search_tree_union(self, score_probe, labels_e, model):

        cost = 0

        counter_search = 0

        for item in self.mark_fq:

            self.mark_fq[item] = False

        for item in self.mark_mf:

            self.mark_mf[item] = False

        for item in self.mark_sf:

            self.mark_sf[item] = False
        
        for item in self.mark:

            self.mark[item] = False

        keys = list(self.hash_table_q.keys())

        queue = []

        higher = 0

        lower = 0

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

            if lower != 0:
                
                queue.append(lower)

            elif higher !=0:

                queue.append(higher)
          
        else:

            queue.append(score_probe)
        
        while(len(queue) > 0) :

            p = queue.pop(0)

            if(p in self.hash_table_q and not self.mark[p]):

                self.mark[p] = True

                ref_values = []

                temp = self.hash_table_q[p]

                if model == 'FaceQnet_v1':

                    if(p in self.hash_table_q_sf):

                        intersecction_list = []

                        difference_list = []

                        temp_sf = self.hash_table_q_sf[p]

                        intersecction_list = self.IntersecOfSets(temp,temp_sf)

                        difference_list = self.difference(temp,temp_sf)

                        ref_values = intersecction_list + difference_list

            
                elif model == 'SER-FIQ':

                    if (p in self.hash_table_q_mf):

                        intersecction_list = []

                        difference_list = []

                        temp_mf = self.hash_table_q_mf[p]

                        if len(ref_values) > 0:

                            intersecction_list = self.IntersecOfSets(ref_values,temp_mf)

                            difference_list = self.difference(ref_values,temp_mf)

                            ref_values = intersecction_list + difference_list
                        
                        else:

                            intersecction_list = self.IntersecOfSets(temp,temp_mf)

                            difference_list = self.difference(temp,temp_mf)

                            ref_values = intersecction_list + difference_list

                      
                else:
                    
                    if(p in self.hash_table_q_fq):

                        intersecction_list = []

                        difference_list = []

                        temp_fq = self.hash_table_q_fq[p]

                        intersecction_list = self.IntersecOfSets(temp,temp_fq)

                        difference_list = self.difference(temp,temp_fq)

                        ref_values = intersecction_list + difference_list
                    
                if len(ref_values) == 0:

                    ref_values = temp

                      
                k = keys.index(p)

                cost += len(ref_values)

                if labels_e in ref_values:

                    counter_search = 1


                    break
                
                else:

                    if(k + 1 < len(keys)):

                        queue.append(keys[(k + 1)])
                    
                    if(k - 1 >= 0):

                        queue.append(keys[(k - 1)])    

                        
        return cost, counter_search

    def fusion_union(self,score_q_s, labels_s, model):

        list_comp = []

        list_final = []

        counter_s = 0

        for score_probe, label in zip(score_q_s,labels_s):

            count_comp,counter = self.__search_tree_union(score_probe, label, model)

            list_comp.append(count_comp)

            if counter != 0:

                counter_s +=1

        print("La cantidad de elementos encontrados por union is {}".format(counter_s))

        ave_comp, variance = self.mean_confidence_interval(list_comp)

        print("Size recorrido {}".format(len(list_final)))

        return ave_comp, variance
    
    def __statistics(self, score_probe, labels_e, model):

        cost = 0

        counter_search = 0
        
        for item in self.mark:

            self.mark[item] = False

        keys = list(self.hash_table_q.keys())

        queue = []

        not_similar_bin = 0

        if not score_probe in self.hash_table_q:

            not_similar_bin = 1

            return not_similar_bin,counter_search,cost
          
        else:

            queue.append(score_probe)
        
            p = queue.pop(0)

            if(p in self.hash_table_q and not self.mark[p]):

                self.mark[p] = True

                ref_values = self.hash_table_q [p]
                        
                if labels_e in ref_values:

                    counter_search = 1

                    cost = len(ref_values)
                            
            return not_similar_bin, counter_search,cost

    def Statistics_general(self,score_q_s, labels_s, model):

        list_comp = []

        list_final = []

        counter_s = 0

        counter_not_similar = 0

        list_cost_most_similar = []

        length_dict = {key: len(value) for key, value in self.hash_table_q.items()}

        print(type(length_dict))

        values_key = length_dict.values()

        value = 0

        Comp_tota = 0

        for e in values_key:

            value += e
        
        value /= len(values_key)

        # length_dict = list(length_dict)

        min_bin = min(values_key)

        max_bin = max(values_key)

        total_bins = len(length_dict)



        # max_bin =

        for score_probe, label in zip(score_q_s,labels_s):

            not_similar_bin,counter_search,cost = self.__statistics(score_probe, label, model)

            if counter_search != 0:

                counter_s +=1

                list_cost_most_similar.append(cost)


            if not_similar_bin !=0:
                
                counter_not_similar +=1


        ave_comp_most_similar, variance_comp_most_similar = self.mean_confidence_interval(list_cost_most_similar)

        return counter_not_similar, counter_s,ave_comp_most_similar, min_bin, max_bin,total_bins,value
    
    def __search_tree_improved(self, score_probe, labels_e):

        cost = 0

        finished = False

        contador_search = 0

        self.hash_table_q = self.hash_table_q_fq

        self.mark = self.mark_fq

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

            if lower != 0:
                
                queue.append(lower)

            elif higher !=0:

                queue.append(higher)
          
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

                    contador_search += 1

                    break
                
                else:

                    if(k + 1 < len(keys)):

                        queue.append(keys[(k + 1)])
                    
                    if(k - 1 >= 0):

                        queue.append(keys[(k - 1)])    

                        
        return cost, contador_search


    def nearest_improved(self,score_q_s, labels_s, model):

        list_comp = []

        list_final = []

        counter= 0

        for score_probe, label in zip(score_q_s,labels_s):

            count_comp,counter_search = self.__search_tree_improved(score_probe, label)

            list_comp.append(count_comp)

            if counter_search != 0:

                counter +=1

        ave_comp, variance = self.mean_confidence_interval(list_comp)

        return ave_comp, variance
    
    def __counter_visited(self, score_probe, labels_e):

        cost = 0

        bin_visited = 0

        list_cost = []

        cost_sep = 0

        list_bin_visited = []

        for item in self.mark:

            self.mark[item] = False

        keys = list(self.hash_table_q.keys())

        queue = []

        higher = 0

        lower = 0

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

            if lower != 0:
                
                queue.append(lower)

            elif higher !=0:

                queue.append(higher)
          
        else:

            queue.append(score_probe)

        while(len(queue) > 0) :

            # cost = 0

            p = queue.pop(0)

            if(p in self.hash_table_q and not self.mark[p]):

                bin_visited +=1

                list_bin_visited.append(bin_visited)

                self.mark[p] = True

                ref_values = self.hash_table_q[p]

                k = keys.index(p)

                cost += len(ref_values)

                cost_sep = len(ref_values)

                list_cost.append(cost_sep)

                if labels_e in ref_values:

                    break
                
                else:

                    if(k + 1 < len(keys)):

                        queue.append(keys[(k + 1)])
                    
                    if(k - 1 >= 0):

                        queue.append(keys[(k - 1)])    

                        
        return list_bin_visited, list_cost

    def bins_visited(self,score_q_s, labels_s, model):

        list_bin_visited = []

        list_cost  = []

        count_bin_high = 0

        list_higher_bin = []

        list_higher_cost = []

        for score_probe, label in zip(score_q_s,labels_s):

            list_bin_visited,list_cost = self.__counter_visited(score_probe, label)

            if len(list_bin_visited) > 10:

                list_bin_visited = list_bin_visited

                list_cost = list_cost


                break

        return list_bin_visited, list_cost


    def search_one_to_first(self, score_q_s, feat_s,labels):

        total_comparisons = 0

        penetration_rate = 0

        list_comp = []

        for score_probe, feat_probe, label in zip(score_q_s,feat_s,labels):

            count_comp = self.__one_to_first(score_probe, feat_probe, label)

            list_comp.append(count_comp)

            total_comparisons += count_comp

        ave_comp, variance = self.mean_confidence_interval(list_comp)

        total_comparisons /= len(labels)

        return ave_comp, variance


    def exhaustive_search(self,labels_s, list_names_e):

        list_count_comp = []

        for l in labels_s:

            count = 0

            for label_e in list_names_e:

                label_e = self.cleaner(label_e)

                if label_e != l:

                    count += 1

                else:

                    break
            
            list_count_comp.append(count)

        ave_comp, variance = self.mean_confidence_interval(list_count_comp)

        return ave_comp,variance









        















