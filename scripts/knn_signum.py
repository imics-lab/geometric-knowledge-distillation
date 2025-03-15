import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

#X_train = np.load('data_numpy/X_train.npy')
#X_test = np.load('data_numpy/X_test.npy')
y_train = np.load('../data_numpy/y_train.npy')
y_test = np.load('../data_numpy/y_test.npy')
distance_arr = np.load('../distances_test_left_right.npy')

k = 15
acc_list = []
prec_list = []
rec_list = []
f1_list = []
k_list = []
start, end = 0, len(y_test)
for k in range(1, 30):
    knn_result = []
    for i in tqdm(range(start, end), desc="Processing test sequences"):
        indices = np.argpartition(distance_arr[i], k)[:k]
        values = y_train[indices]
        unique_elements, counts = np.unique(values, return_counts=True)
        knn_result.append(unique_elements[np.argmax(counts)])


    knn_result = np.array(knn_result)
    acc_score = accuracy_score(y_test[start:end], knn_result)
    prec_score = precision_score(y_test[start:end], knn_result, average='macro')
    rec_score = recall_score(y_test[start:end], knn_result, average='macro')
    f1 = f1_score(y_test[start:end], knn_result, average='macro')
    print('Accuracy: ' ,acc_score)
    print('Precision: ' ,prec_score)
    print('Recall: ', rec_score)
    print('F1: ' ,f1)
    acc_list.append(acc_score)
    prec_list.append(prec_score)
    rec_list.append(rec_score)
    f1_list.append(f1)
    k_list.append(k)

print(acc_list)    
plt.plot(k_list, acc_list, linestyle='-', label='Accuracy', color='b')
plt.plot(k_list, prec_list,  linestyle='-', label='Precision', color='g')
plt.plot(k_list, rec_list,  linestyle='-', label='Recall', color='r')
plt.plot(k_list, f1_list,  linestyle='-', label='F1-score', color='purple')
plt.title('k-NN Metrics vs k Value')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.savefig('knn_accuracy_plot_all_left.png', format='png')


    