import numpy as np
import os, sys
from six.moves import cPickle as pickle


def read_word_embedding(embed_path, embed_file, pickle_fname="glove50.pkl"):
    '''
    This function will load the word embedding vector and simultaneously
    give us the vocablary dictionary for preprocessing of the dataset.
    '''
    print("Loading the word_embedding:",os.path.join(embed_path, embed_file))
    word2index={}
    wvec_list=[]
    with open(os.path.join(embed_path, embed_file), 'r', encoding='utf-8', errors='ignore') as fhandle:
        for widx,line in enumerate(fhandle):
            elements = line.split(' ')
            word = elements[0]
            vector = [float(elements[i]) for i in range(1,len(elements))]

            word2index[word] = np.array(vector, dtype=np.float32)

    print("Pickling the word vectors")
    with open(os.path.join(embed_path, pickle_fname), "wb") as outfile:
        pickle.dump(word2index, outfile, protocol=pickle.HIGHEST_PROTOCOL)

def load_word_embedding(embed_path, pickle_fname="random.pkl"):
    '''
    This function will load the word embedding pickeled as tuple.
    '''
    with open(os.path.join(embed_path, pickle_fname),"rb") as infile:
        embedding = pickle.load(infile)

    return embedding

if __name__ == "__main__":
    read_word_embedding(os.path.join('data','glove'), sys.argv[1], '{}-pkl'.format(sys.argv[1]))
    test = load_word_embedding(os.path.join('data','glove'), '{}-pkl'.format(sys.argv[1]))
