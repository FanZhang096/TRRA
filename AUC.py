from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = r'results/efficientnet_b1_learning_rate_5e-05_weight_decay_0.001_labels_3/fold-0/cnn_classification/best_accuracy/test_image_level_prediction.tsv'
true_label_1 = pd.read_csv(path, sep='\t', usecols=['true_label'])
sub_proba1 = pd.read_csv(path, sep='\t', usecols=['sub_proba1'])
y_label_1 = []
y_pre_1 = []
for i in range(0, 133):
    # print(true_label_1['true_label'][i])
    y_label_1.append(true_label_1['true_label'][i])
    y_pre_1.append(sub_proba1['sub_proba1'][i])

fpr, tpr, thersholds = roc_curve(y_label_1, y_pre_1)

# for i, value in enumerate(thersholds):
#     print("%f %f %f" % (fpr[i], tpr[i], value))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
mean_tpr += np.interp(mean_fpr, fpr, tpr)
# print(mean_tpr)

roc_auc = auc(fpr, tpr)
print('AUC is', roc_auc)
plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
plt.plot(mean_fpr, mean_tpr, label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
