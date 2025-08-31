from scipy import spatial
import numpy as np
from polysemy import vocabCheck
import itertools


def generateAnalogies_parallelogram(analogyTemplates, keyedVecs):
    expandedAnalogyTemplates = []
    for A, stereotypes in analogyTemplates.items():
        for B in analogyTemplates:
            if A != B:
                for stereotype in stereotypes:
                    expandedAnalogyTemplates.append([[B, stereotype], [A]])

    analogies = []
    for positive, negative in expandedAnalogyTemplates:
        try:
            words = keyedVecs.most_similar(positive=positive, negative=negative, topn=100)
        except KeyError:
            words = []
        for word, score in words:
            analogy = f"{negative[0]} is to {positive[1]} as {positive[0]} is to {word}"
            analogyRaw = [negative[0], positive[1], positive[0], word]
            analogies.append([score, analogy, analogyRaw])
    analogies.sort(key=lambda x: -x[0])
    return analogies

def scoredAnalogyAnswers(a, b, x, keyedVecs, thresh=200):
    # Ensure all probe words are in vocabulary
    words = []
    for w in keyedVecs.key_to_index:
        try:
            if np.linalg.norm(np.array(keyedVecs[w]) - np.array(keyedVecs[x])) < thresh:
                words.append(w)
        except KeyError:
            continue
    def cos(a, b, x, y):
        aVec, bVec, xVec, yVec = (np.array(keyedVecs[w]) for w in (a, b, x, y))
        num = (aVec - bVec).dot(xVec - yVec)
        denom = np.linalg.norm(aVec - bVec) * np.linalg.norm(xVec - yVec) or 1e-6
        return num / denom
    return sorted(
        [(cos(a, b, x, y), a, b, x, y) for y in words],
        key=lambda x: -x[0]
    )

def generateAnalogies(analogyTemplates, keyedVecs):
    print("KeyedVecs", keyedVecs)
    expanded = []
    for A, stereotypes in analogyTemplates.items():
        for B in analogyTemplates:
            if A != B:
                for stereotype in stereotypes:
                    expanded.append([A, stereotype, B])
    analogies, groups = [], []
    for a, b, x in expanded:
        outputs = scoredAnalogyAnswers(a, b, x, keyedVecs)
        formatted = []
        for score, a_w, b_w, x_w, y_w in outputs:
            analogy = f"{a_w} is to {b_w} as {x_w} is to {y_w}"
            analogies.append([score, analogy, [a_w, b_w, x_w, y_w]])
            formatted.append([score, analogy, [a_w, b_w, x_w, y_w]])
        groups.append(formatted)
    analogies.sort(key=lambda x: -x[0])
    print("Analogies generated:", len(analogies))
    return analogies, groups

def multiclass_evaluation(embeddings, targets, attributes):
    results = []
    for targetSet in targets:
        for target in targetSet:
            for attrSet in attributes:
                score = _unary_s(embeddings, target, attrSet)
                if score is not None:
                    results.append(score)
    if not results:
        return 0.0, []
    return np.mean(results), results

def _unary_s(embeddings, target, attributes):
    if not vocabCheck(embeddings, target):
        return None
    attrs = [w.lower() for w in attributes if vocabCheck(embeddings, w)]
    return np.mean([spatial.distance.cosine(embeddings[target], embeddings[ai]) for ai in attrs])
