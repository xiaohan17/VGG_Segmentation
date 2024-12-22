import numpy as np
import logging
from tabulate import tabulate

logging.basicConfig(
    filename='deep_learning.log',
    format='[%(asctime)s][%(filename)s][%(levelname)s][%(message)s]',
    level=logging.INFO,
    filemode="w"
)


def condusion_matrix_Index(confusion_matrix,classes):
    confusion_matrix=np.array(confusion_matrix)
    #类别数
    num_class=confusion_matrix.shape[0]
    #样本数
    total_samples=np.sum(confusion_matrix)

    per_class_values=[]
    for i in range(num_class):
        #每个类的TP数
        TP_i=confusion_matrix[i,i]

        #每个类的准确率
        col_i_sum=np.sum(confusion_matrix[:,i])#计算第i列的总和
        precision_i=TP_i/col_i_sum if col_i_sum!=0 else 0

        #每个类别召回率
        row_i_sum=np.sum(confusion_matrix[i,:])#计算第i行的总和
        Recall_i=TP_i/row_i_sum if row_i_sum!=0 else 0

        #计算每个类别的F1得分
        F1_score_i=(2*(precision_i*Recall_i)/(precision_i+Recall_i)) if (precision_i+Recall_i)!=0 else 0
        per_class_values.append((classes[i],precision_i,Recall_i,F1_score_i))

    #计算总的准确率（Accurary）
    accurary=np.trace(confusion_matrix)/total_samples if total_samples !=0 else 0
    _,precision_list,Recall_list,F1_score_list=zip(*per_class_values)

    header=["Class","Val_precision","Val_Recall","Val_F1_score"]
    table_str = tabulate(per_class_values, headers=header, tablefmt="grid")
    logging.info("\n" + table_str)
    return precision_list,Recall_list,F1_score_list,accurary



