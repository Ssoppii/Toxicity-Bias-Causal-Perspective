import string
import numpy as np
from gensim.models.keyedvectors import Word2VecKeyedVectors

import csv
import os
import json

def load_legacy_w2v(w2v_file, dim=50):
    vectors = {}
    with open(w2v_file, 'r') as f:
        for line in f:
            vect = line.strip().rsplit()
            word = vect[0]
            vect = np.array([float(x) for x in vect[1:]])
            if(dim == len(vect)):
                vectors[word] = vect
        
    return vectors, dim

def convert_legacy_to_keyvec(legacy_w2v):
    dim = len(next(iter(legacy_w2v.values())))
    vectors = Word2VecKeyedVectors(vector_size=dim)
    
    ws = []
    vs = []

    for word, vect in legacy_w2v.items():
        ws.append(word)
        vs.append(vect)
        assert(len(vect) == dim)
    vectors.add_vectors(ws, np.array(vs))
    return vectors

def pruneWordVecs(wordVecs):
    newWordVecs = {}
    for word, vec in wordVecs.items():
        valid=True
        if(not isValidWord(word)):
            valid = False
        if(valid):
            newWordVecs[word] = vec
    return newWordVecs

def isValidWord(word):
    return all([c.isalpha() for c in word])

def write_evaluation_results(results, topn, dataset, output_path='output/evaluation_results.csv'):
    """
    Write MAC scores and (optional) p-values for biased/hard/soft runs.
    `results` is a dict with keys:
       'biased', 'soft', 'pvalue_hard', 'pvalue_soft'
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_header = not os.path.exists(output_path)
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['topn', 'dataset', 'setting', 'MAC', 'pvalue'])
        writer.writerow([topn, dataset, 'biased', results.get('biased',''), ''])
        if 'soft' in results and results['soft'] is not None:
            writer.writerow([topn, dataset, 'soft', results['soft'], results.get('pvalue_soft','')])

def debias_train_json(train_path, debiased_keyedvecs, output_path='output/train_debiased.json'):
    """
    Load the original train JSON, replace each category token list with its debiased word form
    using `debiased_keyedvecs` to find the nearest neighbor, and save to `output_path`.
    """
    import os, json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    categories = ['toxicity', 'threat', 'insult', 'identity_attack']
    for record in data:
        for cat in categories:
            key = f"{cat}_tokens"
            if key not in record:
                continue
            new_list = []
            for w in record.get(key, []):
                if w in debiased_keyedvecs.key_to_index:
                    try:
                        candidate = debiased_keyedvecs.most_similar([w], topn=1)[0][0]
                    except Exception:
                        candidate = w
                else:
                    candidate = w
                new_list.append(candidate)
            record[key] = new_list
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)