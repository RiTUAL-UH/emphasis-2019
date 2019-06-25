import numpy as np
from scipy.stats import entropy
from sklearn_crfsuite import metrics
import itertools
import pickle
import scipy
from sklearn.metrics import mean_squared_error

def js(p, q):
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2
def Average(lst):
    return sum(lst) / len(lst)
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def fix_padding(scores_numpy, label_probs,  mask_numpy):
    if len(scores_numpy) != len(mask_numpy):
        print("Error: len(scores_numpy) != len(mask_numpy)")
    assert len(scores_numpy) == len(mask_numpy)
    if len(label_probs) != len(mask_numpy):
        print("len(label_probs) != len(mask_numpy)")
    assert len(label_probs) == len(mask_numpy)

    all_scores_no_padd = []
    all_labels_no_pad = []
    for i in range(len(mask_numpy)):
        all_scores_no_padd.append(scores_numpy[i][:mask_numpy[i].sum()])
        all_labels_no_pad.append(label_probs[i][:mask_numpy[i].sum()])

    assert len(all_scores_no_padd) == len(all_labels_no_pad)
    return all_scores_no_padd, all_labels_no_pad

def topK(all_scores_no_padd, all_labels_no_pad):
    topk = [1,2,3,4]

    for k in topk:
        score_lst =[]
        for s in all_scores_no_padd:
            # if it contains several top values with the same amount
            h = k
            if len(s) > h:
                while (s[np.argsort(s)[-h]] == s[np.argsort(s)[-(h + 1)]] and h<(len(s)-1)):
                    h += 1
            s = np.array(s)
            ind_score = np.argsort(s)[-h:]
            score_val = np.zeros(len(s))
            score_val[ind_score] = 1
            score_lst.append(score_val.tolist())

        ############################################### computing labels:

        label_lst_topk = []
        for l in all_labels_no_pad:
            h = k
            if len(l) > h:
                while (l[np.argsort(l)[-h]] == l[np.argsort(l)[-(h + 1)]] and h<(len(l)-1) ):
                    h += 1
            l = np.array(l)
            ind_label = np.argsort(l)[-h:]
            label_val = np.zeros(len(l))
            label_val[ind_label] = 1
            label_lst_topk.append(label_val.tolist())

        ############################################### computing topk_label - topk_score:

        print("K----> ",k)
        print("binary f-score: ", metrics.flat_f1_score(label_lst_topk, score_lst, average="binary"))

def match_M(all_scores_no_padd, all_labels_no_pad):

    top_m = [1, 2, 3, 4]

    for m in top_m:
        intersects_lst = []
        # exact_lst = []
        score_lst = []
        ############################################### computing scores:
        for s in all_scores_no_padd:
            if len(s) <=m:
                continue
            h = m
            # if len(s) > h:
            #     while (s[np.argsort(s)[-h]] == s[np.argsort(s)[-(h + 1)]] and h < (len(s) - 1)):
            #         h += 1

            s = np.array(s)

            ind_score = np.argsort(s)[-h:]
            score_lst.append(ind_score)

        ############################################### computing labels:
        label_lst = []
        for l in all_labels_no_pad:
            if len(l) <=m:
                continue
            # if it contains several top values with the same amount
            h = m
            if len(l) > h:
                while (l[np.argsort(l)[-h]] == l[np.argsort(l)[-(h + 1)]] and h < (len(l) - 1)):
                    h += 1
            l = np.array(l)
            ind_label = np.argsort(l)[-h:]
            label_lst.append(ind_label)

        ############################################### :

        for i in range(len(score_lst)):
            intersect = intersection(score_lst[i], label_lst[i])
            intersects_lst.append((len(intersect))/(min(m, len(score_lst[i]))))
            # sorted_score_lst = sorted(score_lst[i])
            # sorted_label_lst =  sorted(label_lst[i])
            # if sorted_score_lst==sorted_label_lst:
            #     exact_lst.append(1)
            # else:
            #     exact_lst.append(0)

        print("m----> ", m)
        print("approx_m: ", Average(intersects_lst))

def MSE( label_probs ,scores_probs):
    scores_flat = list(itertools.chain(*scores_probs))
    labels_flat = list(itertools.chain(*label_probs))
    print("[LOG] - - - MSE: ", mean_squared_error(labels_flat, scores_flat))

def Jensen_Shannon(scores_probs, label_probs):
    scores_flat = list(itertools.chain(*scores_probs))
    labels_flat = list(itertools.chain(*label_probs))
    Jensen = js(np.array(scores_flat), np.array(labels_flat))
    print("[LOG] - - - Jensen_Shannon: ", Jensen)


if __name__ == '__main__':
    file = "../Evals/test1/"

    scores_numpy = pickle.load(open(file + "score_pobs.pkl", "rb"))
    label_probs = pickle.load(open(file + "label_pobs.pkl", "rb"))
    mask_numpy = pickle.load(open(file + "mask_pobs.pkl", "rb"))


    all_scores_no_padd, all_labels_no_pad = fix_padding(scores_numpy, label_probs, mask_numpy)
    print()
    print("[LOG] - - - TOPK: ")
    topK(all_scores_no_padd, all_labels_no_pad)



    #print(">>>MSE:")
    print()
    MSE(all_labels_no_pad, all_scores_no_padd)


    #print(">>>Jensen_Shannon Divergence:")
    print()
    Jensen_Shannon(all_scores_no_padd, all_labels_no_pad)


    print()
    print("[LOG] - - - Match_m: ")
    match_M(all_scores_no_padd, all_labels_no_pad)
    print()
    print("[LOG] Reading from: ", file)
