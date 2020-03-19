
import argparse

def build_parser():
    parser = argparse.ArgumentParser(description = 'For parsing arguements for Paraphrase detection')

    #General argumets
    parser.add_argument('-mode', type = str, default = 'train', help = 'train or test?')
    parser.add_argument('-device', type = str, default = 'cuda:0', help = 'device to use i.e cpu or cuda:<device_id>')
    parser.add_argument('-run_name', type = str, default = 'ppd', help = 'name of experiment')
    parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')
    parser.add_argument('-dataset', type = str, default = 'QQP', help = 'Dataset to use')
    parser.add_argument('-datatype', type = str, default = 'train', help = 'Dataset to use')
    parser.add_argument('-test_file', type = str, default = 'Generations/cgeh512d3v60k21-09-2019/src_gen.csv')

    #Training hyperparamters
    parser.add_argument('-lr', type = float, default = 2e-5, help = 'learning rate for training the model')
    parser.add_argument('-n_epochs', type = int, default = 10, help = 'Number of epochs to train')
    parser.add_argument('-display_freq', type = int, default = 50, help = 'After how many iterations print loss')
    parser.add_argument('-batch_size', type = int, default = 32, help = 'batch size')

    return parser