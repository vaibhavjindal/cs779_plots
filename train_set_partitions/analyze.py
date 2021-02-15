import numpy as np
import matplotlib.pyplot as plt
# a = np.load('pseudo_labels/train_scores_and_feat_mat.npy')
# b = np.load('pseudo_labels/train_scores_and_feat_mat.npy')

with open('pseudo_labels/train_scores_and_feat_mat.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)

exp_b = np.exp(b)
c = np.expand_dims(np.sum(exp_b, axis=1), axis=1)
prob = exp_b/c

log_prob = np.log(prob)

entropy = -1*np.sum(prob*log_prob,axis=1) / np.log(2)

print(np.max(entropy),np.min(entropy))


idx = entropy.argpartition(2000)#(int(0.1*entropy.shape[0]))
print(entropy[idx[:5]])

for i in range(10):
    with open('pseudo_labels/train.tsv','r') as fil_text:
        arr_text = [lin.strip() for lin in fil_text]
        with open('pseudo_labels/train_labels.txt','r') as fil_labs:
            
            arr_lab = [lin.strip() for lin in fil_labs]
            
            idx = entropy.argpartition(int(((i+1)/10)*entropy.shape[0]))

            all_idx = idx[:int(((i+1)/10)*entropy.shape[0])]
            
            print((i+1)*10, np.max(entropy[all_idx]), np.min(entropy[all_idx]))
            with open('train_top'+str((i+1)*10)+'.tsv','w+',encoding="utf-8-sig") as fil_train:
                for j in all_idx:
                    fil_train.write(F'{arr_lab[j]}\t{arr_text[j]}\n')
            
        






















# count = 0
# x = []
# y = []
# for i in range(1000):
#     count = 0
#     for j in range(entropy.shape[0]):
#         if(entropy[j]<0.001*i):
#             count += 1
#     x.append(0.001*i)
#     y.append((count*100)/entropy.shape[0])
# print(x)
# print(y)

# plt.xlabel('Entropy')
# plt.ylabel('Percentage of samples')
# plt.plot(x,y)
# plt.savefig('freq.png')