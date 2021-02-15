import numpy as np
import matplotlib.pyplot as plt

original_labels = []
predicted_labels = []

f = open('pseudo_labels/dev_labels.txt')
for i in f:
    original_labels.append(int(i[:-1]))
f.close()

f = open('pseudo_labels/dev_pred.tsv')
for i in f:
    predicted_labels.append(int(i[:-1]))
f.close()

with open('pseudo_labels/dev_scores_and_feat_mat.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)

exp_b = np.exp(b)
c = np.expand_dims(np.sum(exp_b, axis=1), axis=1)
prob = exp_b/c

log_prob = np.log(prob)

entropy = -1*np.sum(prob*log_prob,axis=1) / np.log(2)

percentages = [i for i in range(1,100)]
percentages.append(99.999)


entropy_pos = []
entropy_neg = []
orig_lab_acc_pos = []
orig_lab_acc_neg = []

for i in range(len(predicted_labels)):
    if predicted_labels[i]==1:
        entropy_pos.append(entropy[i])
        orig_lab_acc_pos.append(original_labels[i])
    else:
        entropy_neg.append(entropy[i])
        orig_lab_acc_neg.append(original_labels[i])

entropy_pos = np.asarray(entropy_pos, dtype = np.float64)
entropy_neg = np.asarray(entropy_neg, dtype = np.float64)

per = []
pre = []
rec = []
f1s = []

for percentage in percentages:
    num_pos_samples = int(percentage*0.01*len(orig_lab_acc_pos))
    idx_pos = entropy_pos.argpartition(num_pos_samples)[:num_pos_samples]

    num_neg_samples = int(percentage*0.01*len(orig_lab_acc_neg))
    idx_neg = entropy_neg.argpartition(num_neg_samples)[:num_neg_samples]


    count_f = 0
    count_s = 0
    count_t = 0
    count_fo = 0

    for i in idx_pos:
        if orig_lab_acc_pos[i] == 1:
            count_f += 1
        else:
            count_t += 1

    for i in idx_neg:
        if orig_lab_acc_neg[i] == -1:
            count_fo += 1
        else:
            count_s += 1

    precision = (count_f+0.0)/(count_f+count_t+0.0)
    recall = (count_f+0.0)/(count_f+count_s+0.0)
    f1 = (2*precision*recall)/(precision+recall+0.0)

    per.append(percentage)
    pre.append(precision)
    rec.append(recall)
    f1s.append(f1)
    print("Percentage:", percentage)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)
    print("")

plt.xlabel("Percentage of points")
plt.ylabel("Scores")
plt.title("Scores on dev set when choosing points according to lowest entropies")
plt.plot(per,pre,label = "Precision")
plt.plot(per,rec, label = "Recall")
plt.plot(per,f1s, label = "F1")
plt.legend()
plt.savefig("score_variation_on_devset.png")