import numpy as np
import matplotlib.pyplot as plt

predicted_labels = []

f = open('pseudo_labels/train_labels.txt')
for i in f:
    predicted_labels.append(int(i[:-1]))
f.close()

with open('pseudo_labels/train_scores_and_feat_mat.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)

exp_b = np.exp(b)
c = np.expand_dims(np.sum(exp_b, axis=1), axis=1)
prob = exp_b/c
log_prob = np.log(prob)
entropy = -1*np.sum(prob*log_prob,axis=1) / np.log(2)

percentages = [ 10*i for i in range(1,10)]
percentages.append(99.999)

with open('pseudo_labels/train.tsv','r') as fil_text:
    arr_text = [lin.strip() for lin in fil_text]


print(len(arr_text))
print(entropy.size)

entropy_pos = []
entropy_neg = []
arr_text_pos = []
arr_text_neg = []

for i in range(len(predicted_labels)):
    if predicted_labels[i] == 1:
        entropy_pos.append(entropy[i])
        arr_text_pos.append(arr_text[i])
    else:
        entropy_neg.append(entropy[i])
        arr_text_neg.append(arr_text[i])

entropy_pos = np.asarray(entropy_pos, dtype = np.float64)
entropy_neg = np.asarray(entropy_neg, dtype = np.float64)

for percentage in percentages:
    num_pos_samples = int(percentage*0.01*len(arr_text_pos))
    idx_pos = entropy_pos.argpartition(num_pos_samples)[:num_pos_samples]

    num_neg_samples = int(percentage*0.01*len(arr_text_neg))
    idx_neg = entropy_neg.argpartition(num_neg_samples)[:num_neg_samples]

    for i in idx_pos:
        with open('train_top_rel'+str(percentage)+'.tsv','a+',encoding="utf-8-sig") as fil_train_pos:
            fil_train_pos.write(F'{1}\t{arr_text_pos[i]}\n')
    
    for i in idx_neg:
        with open('train_top_rel'+str(percentage)+'.tsv','a+',encoding="utf-8-sig") as fil_train_neg:
            fil_train_neg.write(F'{-1}\t{arr_text_neg[i]}\n')


            