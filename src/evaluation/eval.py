import argparse
import rouge

from src.evaluation.eval_utils import Meteor, stanford_parsetree_extractor, \
    compute_tree_edit_distance
from tqdm import tqdm
import ipdb as pdb
import subprocess

MULTI_BLEU_PERL = 'src/evaluation/apps/multi-bleu.perl'

def run_multi_bleu(input_file, reference_file):
    bleu_output = subprocess.check_output(
        "./{} -lc {} < {}".format(MULTI_BLEU_PERL, reference_file, input_file),
        stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    bleu = float(
        bleu_output.strip().split("\n")[-1]
        .split(",")[0].split("=")[1][1:])
    return bleu


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str)
parser.add_argument('--ref_file', '-r', type=str)
parser.add_argument('--temp_file', '-t', type = str)
args = parser.parse_args()

n_ref_line = len(list(open(args.ref_file)))
n_inp_line = len(list(open(args.input_file)))
print("#lines - ref: {}, inp: {}".format(n_ref_line, n_inp_line))
assert n_inp_line == n_ref_line, \
    "#ref {} != #inp {}".format(n_ref_line, n_inp_line)

bleu_score = run_multi_bleu(args.input_file, args.ref_file)
print("bleu", bleu_score)
spe = stanford_parsetree_extractor()
input_parses = spe.run(args.input_file)
ref_parses = spe.run(args.ref_file)
temp_parses = spe.run(args.temp_file)
spe.cleanup()
assert len(input_parses) == n_inp_line
assert len(ref_parses) == n_inp_line

all_meteor = []
all_ted = []
all_ted_t = []
all_rouge1 = []
all_rouge2 = []
all_rougel = []
preds = []

rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                         max_n=2,
                         limit_length=True,
                         length_limit=100,
                         length_limit_type='words',
                         apply_avg=False,
                         apply_best=False,
                         alpha=0.5,  # Default F1_score
                         weight_factor=1.2,
                         stemming=True)
meteor = Meteor()
pbar = tqdm(zip(open(args.input_file),
            open(args.ref_file),
            input_parses,
            ref_parses,
            temp_parses))
i = 0
for input_line, ref_line, input_parse, ref_parse,temp_parse in pbar:
    ted = compute_tree_edit_distance(input_parse, ref_parse)
    ted_t = compute_tree_edit_distance(input_parse, temp_parse)
    ms = meteor._score(input_line.strip(), [ref_line.strip()])
    rs = rouge_eval.get_scores([input_line.strip()], [ref_line.strip()])

    all_rouge1.append(rs['rouge-1'][0]['f'][0])
    all_rouge2.append(rs['rouge-2'][0]['f'][0])
    all_rougel.append(rs['rouge-l'][0]['f'][0])
    all_meteor.append(ms)
    all_ted.append(ted)
    all_ted_t.append(ted_t)
    pbar.set_description(
        "bleu: {:.3f}, rouge-1: {:.3f}, rouge-2: {:.3f}, "
        "rouge-l: {:.3f}, meteor: {:.3f}, syntax-TED: {:.3f}, Template-TED: {:.3f}".format(
            bleu_score,
            sum(all_rouge1) / len(all_rouge1) * 100,
            sum(all_rouge2) / len(all_rouge1) * 100,
            sum(all_rougel) / len(all_rouge1) * 100,
            sum(all_meteor) / len(all_meteor) * 100,
            sum(all_ted) / len(all_ted),
            sum(all_ted_t) / len(all_ted_t)))

print(
    "bleu: {:.3f}, rouge-1: {:.3f}, rouge-2: {:.3f}, "
    "rouge-l: {:.3f}, meteor: {:.3f}, syntax-TED: {:.3f}, Template-TED: {:.3f}".format(
        bleu_score,
        sum(all_rouge1) / len(all_rouge1) * 100,
        sum(all_rouge2) / len(all_rouge1) * 100,
        sum(all_rougel) / len(all_rouge1) * 100,
        sum(all_meteor) / len(all_meteor) * 100,
        sum(all_ted) / len(all_ted),
        sum(all_ted_t) / len(all_ted_t)))
