import numpy as np
import matplotlib.pyplot as plt

with open('pseudo_labels/dev_scores_and_feat_mat.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)


f = open('pseudo_labels/dev_labels.txt')
classes = []
for i in f:
    classes.append(int(i[:-1]))

u,s,vh = np.linalg.svd(a)

dim_reduced_a = u[:,:2]

x_neg = []
y_neg = []
x_pos = []
y_pos = []

for i in range(dim_reduced_a.shape[0]):
    if classes[i] == -1:
        x_neg.append(dim_reduced_a[i,[0]])
        y_neg.append(dim_reduced_a[i,[1]])
    else:
        x_pos.append(dim_reduced_a[i,[0]])
        y_pos.append(dim_reduced_a[i,[1]])

plt.title("Dimensionality Reduction using SVD on devset, original labels")
plt.scatter(x_neg,y_neg, color = 'red',s=1, label = '-1')
plt.scatter(x_pos,y_pos, color = 'blue',s=1, label = '1')
plt.legend()
plt.savefig('svd_dev_orig_lab.png')
print(dim_reduced_a.shape)