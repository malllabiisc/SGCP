import csv
import numpy as np
import pdb
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Convert trees file to sentence file')
    parser.add_argument('-data_dir',required = True,  help = '')
    parser.add_argument('-gen_dir', required = True, help = ' ')
    parser.add_argument('-gen_file',required = True, help = 'name of the file')
    parser.add_argument('-ref_file', default = 'ref.txt', help = 'name of reference file')
    args = parser.parse_args()

    print("Opening files")
    with open(os.path.join(args.data_dir, args.ref_file)) as f:
        refs = f.read().split('\n')
    with open(os.path.join(args.gen_dir, args.gen_file)) as f:
        gens = f.read().split('\n')
    print("Done!")

    print("Cleaning generation file")
    prev_src = None
    prev_temp = None
    ref_it = 0
    results_file = open(os.path.join(args.gen_dir, 'clean_'+args.gen_file.split('.')[0]+'.csv'), 'w')
    writer = csv.writer(results_file)
    writer.writerow(['height', 'source', 'template', 'syn_paraphrase', 'reference'])
    i = 0
    while i < len(gens):
        if len(gens[i]) == 0:
            i += 1
            continue
        if gens[i][0] == "*":
            ref_it += 1
            writer.writerow(["NEXT"]*5)
            i += 1
            continue
        try:
            ht = gens[i][8:]
            src = gens[i+1][17:]
        except:
            pdb.set_trace()
        temp_n_syn_tokens = gens[i+2].split()
        temp_start_id = 2
        temp_end_id = temp_n_syn_tokens.index('Syntactic')
        temp = ' '.join(temp_n_syn_tokens[temp_start_id : temp_end_id])
        syn = ' '.join(temp_n_syn_tokens[temp_end_id + 3:])
        ref = refs[ref_it] 
        prev_src = src
        prev_temp = temp
        writer.writerow([ht, src, temp, syn, ref])
        i += 4
    print("Done")





    



        

