import argparse


def build_parser():
    parser = argparse.ArgumentParser(description='Arguments for Syntactic Paraphraser')

    # Training data/mode
    parser.add_argument('-mode', type=str, required=True, choices=['train', 'decode'], help='Modes: train, decode')
    parser.add_argument('-datatype', type=str, default = 'test', choices=['test','trainrep', 'trainrep500'], help='Modes: train, decode')
    parser.add_argument('-len_sort', action='store_true', help='Sort training data by length')
    parser.add_argument('-dataset', type=str, default='QQPPos', choices=['QQPPos', 'ParaNMT50m'], help='Dataset to use')
    parser.add_argument('-display_freq', type=int, default=200, help='number of batches after which to display loss')
    parser.add_argument('-config_file', type=str, default='config.p', help='Configuration pickle file to save params')
    parser.add_argument('-project_name', type=str, default='Syntactic-paraphrase', help='Project name for this code')
    parser.add_argument('-debug', action='store_true', help='Debugging mode')
    parser.add_argument('-pretrained_encoder', type=str, default=None, choices=['bert_cls', 'bert_avg', 'bert_max', 'bert_all', 'infersent', 'use'], help='Which pretrained encoder to use?')
    parser.add_argument('-cov_after_ep', default=1000, type=int, help='Epoch to start coverage mechanism')
    ###########################################################################################################
    ###########################################################################################################
    # NEVER USE "." or "_" in run_name
    parser.add_argument('-run_name', type=str, default='SYNPar-debug', help='Enter the run name')
    ###########################################################################################################
    ###########################################################################################################


    # Device Configuration
    parser.add_argument('-gpu', type=str, required=True, help='Specify the gpu to use')
    parser.add_argument('-seed', type=int, default=1023, help='Default seed to set')
    parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')
    parser.add_argument('-date_fmt', type=str, default='%Y-%m-%d-%H:%M:%S', help='Format of the date')
    ###########################################################################################################
    ###########################################################################################################
    # Don't change ckpt name. Synpar can be replaced with alphanumeric characters only. The rest should remain as is
    parser.add_argument('-ckpt_file', type=str, default='Synpar_0.pth.tar', help='Checkpoint file name')
    ###########################################################################################################
    ###########################################################################################################


    # Input files
    parser.add_argument('-vocab_size', type=int, default=50000, help='Vocabulary size to consider')
    parser.add_argument('-save_res', action='store_true', help='Save results?')
    parser.add_argument('-res_file', type=str, default='generations.txt', help='File name to save results in')
    parser.add_argument('-res_folder', type=str, default='Generations', help='Folder name to save results in')
    parser.add_argument('-single_side', action = 'store_true', help = 'Whether to chain source to target and target to source')


    # Seq2Seq parameters
    parser.add_argument('-height_dec', type=int, default=5, help='Decoding height')
    parser.add_argument('-use_ptr', action='store_true', help='Use pointer networks')
    parser.add_argument('-cov_wt', type=float, default=1.0, help='Coverage loss weight')
    parser.add_argument('-cell_type', type=str, default='gru', help='RNN cell for encoder and decoder, default: gru')
    parser.add_argument('-use_attn', action='store_true', help='To use attention mechanism?')
    parser.add_argument('-attn_type', type=str, default='general', help='Attention mechanism: (general, concat), default: general')
    parser.add_argument('-hidden_size', type=int, default=512, help='Number of hidden units in each layer')
    parser.add_argument('-depth', type=int, default=3, help='Number of layers in each encoder and decoder')
    parser.add_argument('-emb_size', type=int, default=300, choices=[50,100,200,300], help='Embedding dimensions of encoder and decoder inputs')
    parser.add_argument('-max_length', type=int, default=60, help='Specify max decode steps: Max length string to output')
    parser.add_argument('-s2sdprate', type=float, default=0.2, help='Dropout probability for input/output/state units (0.0: no dropout)')
    parser.add_argument('-init_range', type=float, default=0.10, help='Initialization range for seq2seq model')
    parser.add_argument('-tree_height', type=str, default='3,4,5,6,7', help='Height of trees in tree decoder')
    parser.add_argument('-lambda_dec_eop', type=float, default=0.1, help='Decode lambda eop')

    # Training parameters
    # Add code for shuffle after epoch
    # LEARNING RATES
    parser.add_argument('-lr_enc', type=float, default=0.00007, help='Learning rate')
    parser.add_argument('-lr_tree', type=float, default=0.00007, help='Learning rate')
    parser.add_argument('-lr_rnndec', type=float, default=0.00007, help='Learning rate')
    parser.add_argument('-max_grad_norm', type=float, default=2.00, help='Clip gradients to this norm')
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-max_epochs', type=int, default=50, help='Maximum # of training epochs')
    parser.add_argument('-load_from_ep', type=int, default=None, help='Load epoch for decoding')
    parser.add_argument('-opt', type=str, default='adam', choices=['adam', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')
    parser.add_argument('-tfr', type=float, default=0.9, help='Teacher forcing ratio')
    parser.add_argument('-bidirectional', action='store_true', help='Initialization range for seq2seq model')
    parser.add_argument('-dynamic_tree', action='store_true', help='Use dynamic tree height for training')
    parser.add_argument('-bpe', action = 'store_true', help = 'Use byte pair encoding')

    parser.add_argument('-use_word2vec', action='store_true', help='Initialization Embedding matrix with word2vec vectors')
    parser.add_argument('-use_glove', action='store_true', help='Initialization Embedding matrix with glove vectors')
    parser.add_argument('-beam_width', type=int, default=1, help='Specify the beam width for decoder')
    parser.add_argument('-tree_height2', default=40, type=int, help='Tree height for evaluation')

    # GCN parameters
    parser.add_argument('-gcn_dim', type=int, default=768, help='GCN embedding size')
    parser.add_argument('-gcn_layers', type=int, default=3, help='No. of GCN layers')
    parser.add_argument('-max_labels', type=int, default=45, help='No. of dependency labels')
    parser.add_argument('-max_nodes', type=int, default=20, help='Specify max no. of nodes. Same as max sentence length.')
    parser.add_argument('-gcn_gating', dest='gcn_gating', action='store_true')
    parser.add_argument('-no_gcn_gating', dest='gcn_gating', action='store_false')
    parser.set_defaults(gcn_gating=True)
    return parser
