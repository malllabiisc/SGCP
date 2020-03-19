import argparse
import os
from nltk import agreement
import pickle
import glob
import ipdb as pdb
import numpy as np

PICKLE_PATH = '/scratche/home/ashutosh/HumanEvaluation/Results'

def get_tag(x):
    '''
    if x == 0 or x == 1:
        return '0'

    return '1'
    '''
    '''
    if x == 0:
        return str(x)
    if x == 1 or x == 2:
        return '1'
    else:
        return '2'
    '''
    return str(x)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', choices = ['quora', 'paranmt'], required=True)
    args = parser.parse_args()
    '''
    all_files = glob.glob(os.path.join(PICKLE_PATH, args.dataset, 'abhinav*-allscores.pkl')) + \
                glob.glob(os.path.join(PICKLE_PATH, args.dataset, 'chandrahas*-allscores.pkl'))
    '''

    #all_files = ['abhinav-236-allscores.pkl', 'Akash-288-allscores.pkl', 'shikhar-409-allscores.pkl']
    #all_files = [os.path.join(PICKLE_PATH, args.dataset, file) for file in all_files]
    
    #all_files = glob.glob(os.path.join(PICKLE_PATH, args.dataset, '*-allscores.pkl'))

    all_files = glob.glob(os.path.join(PICKLE_PATH, args.dataset, '*_allranks.pkl'))
    print(all_files)

    if len(all_files) < 2:
        print("Not enough Annotations. Bye Bye!")
        exit()
    annotators  =   []
    annotations =   []
    for pfile in all_files:
        annotators.append(pfile.split('/')[-1].split('-')[0])
        with open(pfile, 'rb') as pk:
            annotations.append(pickle.load(open(pfile,'rb')))
    score2id = {'semantic':0, 'syntactic':1, 'readability':2}
    gen2id = {'sgcp':0, 'sgcp_rouge':1, 'scpn':2, 'cgen': 3}

    '''

    for score in score2id.keys():
        for gen in gen2id.keys():
            taskdata = []
            points = 0
            cnt = 0
            for i,annotation in enumerate(annotations):
                for j in range(len(annotation)):
                    taskdata.append([annotators[i], str(j), get_tag(annotation[j,gen2id[gen],score2id[score]])])
                    points += annotation[j,gen2id[gen],score2id[score]]
                    cnt += 1
            net_average_scores[gen2id[gen], score2id[score]] = points / cnt
            #pdb.set_trace()
            ratingtask                  =   agreement.AnnotationTask(data=taskdata)
            cohen_kappa                 =   ratingtask.kappa()
            fleiss_kappa                =   ratingtask.multi_kappa() 
            cohen_kappas[(score, gen)]  =   cohen_kappa
            fleiss_kappas[(score, gen)] =   fleiss_kappa
    '''
    cohen_kappas = {}
    fleiss_kappas = {}
    net_average_scores = np.zeros((4))
    for gen in gen2id.keys():
        taskdata = []
        points = 0
        cnt = 0
        for i, annotation in enumerate(annotations):
            for j in range(len(annotation)):
                taskdata.append([annotators[i], str(j), get_tag(annotation[j, gen2id[gen]])])
                points += annotation[j, gen2id[gen]]
                cnt += 1
        net_average_scores[gen2id[gen]] = points / cnt
        ratingtask                  =   agreement.AnnotationTask(data=taskdata)
        cohen_kappa                 =   ratingtask.kappa()
        fleiss_kappa                =   ratingtask.multi_kappa() 
        cohen_kappas[ gen]  =   cohen_kappa
        fleiss_kappas[gen] =   fleiss_kappa


    print(net_average_scores)
    print("Cohen Kappas: ")
    print(cohen_kappas)
    print("--------------------------")
    print("Fleiss Kappas: ")
    print(fleiss_kappas)
    print("--------------------------")
    with open(os.path.join(PICKLE_PATH, args.dataset, 'cohen.pkl'),'wb') as pk:
        pickle.dump(cohen_kappas, pk)

    with open(os.path.join(PICKLE_PATH, args.dataset, 'fleiss.pkl'),'wb') as pk:
        pickle.dump(fleiss_kappas, pk)
