import time
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

np.random.seed(32)
with open('pseudo_labels/dev_scores_and_feat_mat.npy', 'rb') as f:
    X = np.load(f)
    b = np.load(f)

#f = open('SFDA-APM/lin/margin20_top100/dev_pred.tsv')
f = open('pseudo_labels/dev_labels.txt')


y = np.zeros(X.shape[0])
index = 0
classes = []
for i in f:
    y[index] = int(i[:-1])
    classes.append(int(i[:-1]))
    index += 1

print(X.shape, y.shape)

u,s,vh = np.linalg.svd(X)

dim_reduced_X = u[:,:50]

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity = 40)
tsne_results = tsne.fit_transform(dim_reduced_X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


dim_reduced_a = tsne_results

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

plt.title("tsne on devset, original labels")
plt.scatter(x_neg,y_neg, color = 'red',s=1, label = '-1')
plt.scatter(x_pos,y_pos, color = 'blue',s=1, label = '1')
plt.legend()
#plt.show()
plt.savefig('tsne_dev_orig_lab.png')
print(dim_reduced_a.shape)