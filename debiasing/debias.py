import argparse
import numpy as np
import json

from util import convert_legacy_to_keyvec, load_legacy_w2v, pruneWordVecs, debias_train_json, write_evaluation_results
from biasOps import identify_bias_subspace, equalize_and_soften
from loader import load_def_sets, load_eval_terms
from evalBias import multiclass_evaluation
from scipy.stats import ttest_rel
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--defining_path', type=str, required=True, help="Path to defining set JSON file")
parser.add_argument('--training_path', type=str, required=True, help="Path to training JSON file")
parser.add_argument('--topn', type=int, choices=[3,5,10], default=10,
                    help="Which top-N selection label to record in evaluation CSV")
parser.add_argument('--dataset', type=str,
                    choices=['IMDB','RealToxicityPrompts','CNN','XSum'], required=True,
                    help="Dataset name for the evaluation CSV")
parser.add_argument('--category', type=str,
                    choices=['toxicity', 'insult', 'threat', 'identity_attack'], required=True,
                    help="Category name for the evaluation CSV")
parser.add_argument('-embeddingPath', default='reddit.US.txt.tok.clean.cleanedforw2v_0.w2v')
parser.add_argument('-soft', action='store_true')
parser.add_argument('-v', action='store_true')
parser.add_argument('-k', type=int, default=2)
parser.add_argument('-g', action='store_true') #visualize
parser.add_argument('-printLimit', type=int, default=500)

args = parser.parse_args()

print("Loading embeddings from {}".format(args.embeddingPath))
word_vectors, embedding_dim = load_legacy_w2v(args.embeddingPath)
# Save original keyed vectors for neighbor comparison
keyedVecs_orig = convert_legacy_to_keyvec(word_vectors)

# Debiasing Subspace
path = args.defining_path
with open(path, "r", encoding="utf-8") as f:
    def_data = json.load(f)
mode = 'role' if 'role' in def_data.get("analogy_templates", {}) else 'attribute'

# EvalSet Setting
print("Evaluation Set based on category labels")
evalTargets, evalAttrs = load_eval_terms(path, mode)

print("Loading vocabulary from {}".format(path))

defSets = load_def_sets(path)

# Collect neutral words from the defining sets
neutral_words = []
for pair in defSets.values():
    neutral_words.extend(pair)

print("Pruning Word Vectors... Starting with", len(word_vectors))
word_vectors = pruneWordVecs(word_vectors)
print("\tEnded with", len(word_vectors))

print("Identifying bias subspace")
subspace = identify_bias_subspace(word_vectors, defSets, args.k, embedding_dim)

final_subspace = subspace

if args.soft:
    print("Equalizing and Softening")
    new_soft_word_vectors = equalize_and_soften(
        word_vectors,
        neutral_words,
        final_subspace,
        embedding_dim,
        verbose=args.v
    )

    print("Making Output File")
    train_path=args.training_path
    category = args.category
    debiased_kv = convert_legacy_to_keyvec(new_soft_word_vectors)
    debias_train_json(
        train_path=train_path,
        debiased_keyedvecs=debiased_kv,
        output_path=f'{train_path}_{category}_debiased.json'
    )
    
    # ------- Evaluation -------
    # Compute Bias Aware Classification (MAC) before and after
    biasedMAC, biasedDistribution = multiclass_evaluation(
        word_vectors, evalTargets, evalAttrs
    )
    debiasedMAC, debiasedDistribution = multiclass_evaluation(
        new_soft_word_vectors, evalTargets, evalAttrs
    )
    # Paired t-test on cosine distances
    _, pvalue = ttest_rel(biasedDistribution, debiasedDistribution)
    # Write results to CSV
    write_evaluation_results(
        {
            'biased': biasedMAC,
            'soft': debiasedMAC,
            'pvalue_soft': pvalue,
        },
        args.topn,
        args.dataset
    )