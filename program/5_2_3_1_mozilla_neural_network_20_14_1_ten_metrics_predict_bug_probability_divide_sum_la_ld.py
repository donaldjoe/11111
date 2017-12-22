#!/user/bin/env python
#-*- coding: utf-8 -*-

"""
Created on %(date)s  Sat Apr 08 20:23:14 2017

@author: %(username)s  qiaolei
"""

"""
*******导入模块******* 
"""

from keras.models import Sequential

""""随机划分训练集和测试集
train_test_split是交叉验证中常用的函数，功能是从样本中
随机的按比例选取train data和testdata"""
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from StringIO import StringIO
import scipy.io as sio
import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt # 可视化模块
import random
import xlwt
import os


from decimal import Decimal as D
from decimal import getcontext 

getcontext().prec = 9
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


"""
******第一步：构造数据/加载数据***需要根据模型训练时需要的数据格式来构造数据的shape****** 
# load dataset
"""
#x = np.random.random((3000,15))
#y = np.random.random((3000,1))
print("Loading data...")

#filepath = 'J:/genetic_algorithm/data_processing/data_spilite_into_ten_fold_output/bugzilla/bugzilla_data_first_taking_natural_logarithms_second_max_min_normalization/'
filepath = 'J:/genetic_algorithm/data_processing/data_spilite_into_ten_fold_output/mozilla/mozilla_data_first_taking_natural_logarithms_second_max_min_normalization/'



train_filename_list = []
test_filename_list = []



train_filename_list = ['mozilla_train_dataset_first_fold_1','mozilla_train_dataset_second_fold_2','mozilla_train_dataset_third_fold_3','mozilla_train_dataset_fourth_fold_4', 'mozilla_train_dataset_fifth_fold_5','mozilla_train_dataset_sixth_fold_6','mozilla_train_dataset_seventh_fold_7','mozilla_train_dataset_eighth_fold_8', 'mozilla_train_dataset_ninth_fold_9','mozilla_train_dataset_tenth_fold_10']
test_filename_list = ['mozilla_test_dataset_first_fold_1','mozilla_test_dataset_second_fold_2','mozilla_test_dataset_third_fold_3','mozilla_test_dataset_fourth_fold_4','mozilla_test_dataset_fifth_fold_5','mozilla_test_dataset_sixth_fold_6','mozilla_test_dataset_seventh_fold_7','mozilla_test_dataset_eighth_fold_8', 'mozilla_test_dataset_ninth_fold_9','mozilla_test_dataset_tenth_fold_10']

                      
#train_filename_list = ['mozillaultimate_mozilla_everyvaluemultiplied1000_train_dataset_first_fold_1','mozillaultimate_mozilla_everyvaluemultiplied1000_train_dataset_second_fold_2','mozillaultimate_mozilla_everyvaluemultiplied1000_train_dataset_third_fold_3','mozillaultimate_mozilla_everyvaluemultiplied1000_train_dataset_fourth_fold_4', 'mozillaultimate_mozilla_everyvaluemultiplied1000_train_dataset_fifth_fold_5','mozillaultimate_mozilla_everyvaluemultiplied1000_train_dataset_sixth_fold_6','mozillaultimate_mozilla_everyvaluemultiplied1000_train_dataset_seventh_fold_7','mozillaultimate_mozilla_everyvaluemultiplied1000_train_dataset_eighth_fold_8', 'mozillaultimate_mozilla_everyvaluemultiplied1000_train_dataset_ninth_fold_9','mozillaultimate_mozilla_everyvaluemultiplied1000_train_dataset_tenth_fold_10']
#test_filename_list = ['mozillaultimate_mozilla_everyvaluemultiplied1000_test_dataset_first_fold_1','mozillaultimate_mozilla_everyvaluemultiplied1000_test_dataset_second_fold_2','mozillaultimate_mozilla_everyvaluemultiplied1000_test_dataset_third_fold_3','mozillaultimate_mozilla_everyvaluemultiplied1000_test_dataset_fourth_fold_4','mozillaultimate_mozilla_everyvaluemultiplied1000_test_dataset_fifth_fold_5','mozillaultimate_mozilla_everyvaluemultiplied1000_test_dataset_sixth_fold_6','mozillaultimate_mozilla_everyvaluemultiplied1000_test_dataset_seventh_fold_7','mozillaultimate_mozilla_everyvaluemultiplied1000_test_dataset_eighth_fold_8', 'mozillaultimate_mozilla_everyvaluemultiplied1000_test_dataset_ninth_fold_9','mozillaultimate_mozilla_everyvaluemultiplied1000_test_dataset_tenth_fold_10']

train_complete_filename = [] 
test_complete_filename = [] 
train_complete_file_path_name = []
test_complete_file_path_name = []


train = []
#全部的训练集
train_except_transactionid_commitdate = []
#完整的训练集中除去transactionid和commitdate这两列的训练集
train_except_rexp = []
#完整的训练集中除去rexp这一列数值的训练集
train_except_transactionid_commitdate_rexp = []
#完整的训练集中除去transactionid和commitdate，rexp这三列的训练集

test = []
#全部的测试集
test_except_transactionid_commitdate = []
#完整的测试集中除去transactionid和commitdate这两列的训练集
test_except_rexp = []
#完整的测试集中除去rexp这一列数值的训练集
test_except_transactionid_commitdate_rexp = []
#完整的测试集中除去transactionid和commitdate，rexp这三列的训练集



for i in range(len(train_filename_list)):
    
    """对训练集的处理，将十折的训练集读入内存中，并分别保存成4个数据集，完整的数据集，完整数据集中除去日期和时间的数据集"""
    """对训练集的处理，将十折的训练集读入内存中，并分别保存成4个数据集，去除rexp这一列的数据集，完整数据集中除去rexp，日期和时间这三列的数据集"""
    train_complete_filename_intermediate_variable = train_filename_list[i] + '.csv'
    train_complete_filename.append(train_complete_filename_intermediate_variable) 
    
    train_current_file_name = train_complete_filename[i] 
    #test_current_file_name = test_complete_filepath_name {i}
    train_complete_file_path_name_intermediate_variable = os.path.join(filepath,train_current_file_name)
    train_complete_file_path_name.append(train_complete_file_path_name_intermediate_variable)
    
    train_current_file = train_complete_file_path_name[i]
    #original_train = np.genfromtxt(train_current_file, dtype="float", skip_header = True, delimiter="," ) 
    original_train = np.genfromtxt(train_current_file, dtype="float", skip_header = 0, delimiter="," )
    """这样数据就不会少一行了,第一行表头也可以读进来"""
    
    
    train.append(original_train)
    #全部的原始的训练集
    train_except_transactionid_commitdate_temp = original_train[:,7:]
    train_except_transactionid_commitdate.append(train_except_transactionid_commitdate_temp)
    #去掉完整的训练集中除去transactionid和commitdate这两列的数值。
    train_current_rexp_previous = original_train[:,0:19]
    #取出来完整的数据集中的rexp这一列之前的数据。
    train_current_rexp_following = original_train[:,20:]
    #取出来完整的数据集中的rexp这一列之后的数据。
    train_current_except_rexp = np.column_stack((train_current_rexp_previous, train_current_rexp_following))
    #将取出来完整的数据集中的rexp之前的数据和rexp之后的数据在列的方向上，左右两部分进行合并 C = [A B]。
    train_except_rexp.append(train_current_except_rexp)
    #train_except_rexp这里边用来存放所有的去除rexp这一列的数据集。                                                  
    train_except_transactionid_commitdate_rexp_temp = train_current_except_rexp[:,7:]
    #将取出来完整的数据集中除去rexp这一列的数据再去除id和commitdate之后的数据集。
    train_except_transactionid_commitdate_rexp.append(train_except_transactionid_commitdate_rexp_temp)
    
    
    
    
    
    
    """对测试集的处理，将十折的训练集读入内存中，并分别保存成4个数据集，完整的数据集，完整数据集中除去日期和时间的数据集"""
    """对训练集的处理，将十折的训练集读入内存中，并分别保存成4个数据集，去除rexp这一列的数据集，完整数据集中除去rexp，日期和时间这三列的数据集"""
    test_complete_filename_intermediate_variable = test_filename_list[i] + '.csv'
    test_complete_filename.append(test_complete_filename_intermediate_variable) 
    
    test_current_file_name = test_complete_filename[i] 
    #test_current_file_name = test_complete_filepath_name {i}
    test_complete_file_path_name_intermediate_variable = os.path.join(filepath,test_current_file_name)
    test_complete_file_path_name.append(test_complete_file_path_name_intermediate_variable)
    
    test_current_file = test_complete_file_path_name[i]
    #original_test = np.genfromtxt(test_current_file, dtype="float", skip_header = True, delimiter="," ) 
    original_test = np.genfromtxt(test_current_file, dtype="float", skip_header = 0, delimiter="," )
    """这样数据就不会少一行了,第一行表头也可以读进来"""
    test.append(original_test)
    #全部的原始的训练集
    test_except_transactionid_commitdate_temp = original_test[:,7:]
    test_except_transactionid_commitdate.append(test_except_transactionid_commitdate_temp)
     #去掉完整的测试集中除去transactionid和commitdate这两列的数值。
    test_current_rexp_previous = original_test[:,0:19]
    #取出来完整的测试集中的rexp这一列之前的数据。
    test_current_rexp_following = original_test[:,20:]
    #取出来完整的测试集中的rexp这一列之后的数据。
    test_current_except_rexp = np.column_stack((test_current_rexp_previous, test_current_rexp_following))
    #将取出来完整的测试集中的rexp之前的数据和rexp之后的数据在列的方向上，左右两部分进行合并 C = [A B]。
    test_except_rexp.append(test_current_except_rexp)
    #train_except_rexp这里边用来存放所有的去除rexp这一列的数据集。                                                  
    test_except_transactionid_commitdate_rexp_temp = test_current_except_rexp[:,7:]
    #将取出来完整的测试集中除去rexp这一列的数据再去除id和commitdate之后的数据集。
    test_except_transactionid_commitdate_rexp.append(test_except_transactionid_commitdate_rexp_temp)
  
print("计算出所有的第10%行的行号，", train)
print("计算出所有的第10%行的行号，", test)






#
#original_dataset = np.genfromtxt("J:/my_neural_network/input_normalized/ultimate_mozilla_everyvaluemultiplied1000.csv", 
#                        dtype="float", skip_header = True, delimiter="," ) 



#dataset = dataset1.decode('gb2312')
"""
dataframe = pd.read_csv("J:/my_neural_network/input_normalized/mozilla_everyvaluemultiplied1000.csv", 
                        delimiter="," ,skiprows =0,header=None)
dataset = dataframe.values
"""
"""
dataset = np.loadtxt("J:/my_neural_network/input_normalized/mozilla_everyvaluemultiplied1000.csv", 
                       delimiter="," ) 
"""  

"""
dataset = np.genfromtxt("J:/my_neural_network/input_normalized/mozilla_everyvaluemultiplied1000.csv", 
                        dtype="U75", skip_header=True, delimiter=",") 
"""


#dtype="U75",dtype="float" 
#J:/my_neural_network/input_normalized/mozilla.csv
#J:/my_neural_network/input/select_mozilla.csv
#print(original_dataset);
##x_train = dataset[:,1:16]
#dataset = original_dataset[:,2:20]
#
#print(dataset)
#dataset_len = len(dataset)
#print(dataset_len)






"""随机化数据集"""
#random.seed (4)
##设置随机种子，保证实验可以重现
#random_dataset_rows_list =  random.sample(range(len(dataset)), len(dataset))
##生成的随机数的个数是：数据集长度，len(dataset)。 
##生成的随机数的范围是0到 数据集长度个：range(len(dataset))，也就是生成的随机数的行号。
#print(random_dataset_rows_list)
#linenumbers = len(random_dataset_rows_list)
#print(linenumbers)
#
#random_dataset = dataset[random_dataset_rows_list]
##根据数据集的行号，对数据集进行随机化。
#print(random_dataset)
#
#random_dataset_len = random_dataset.shape[0]
#random_dataset_width = random_dataset.shape[1]
#
#print(random_dataset_len)
#print(random_dataset_width)
#
#
#random_dataset_longitude = random_dataset.shape[0]
#random_dataset_len = len(random_dataset)
#print(random_dataset_longitude)
#print(random_dataset_len)
#
#ten_per_length = random_dataset_longitude * 0.1
#temp_ten_per_length = round(ten_per_length,0)
##用round进行四舍五入
#integer_ten_per_length = int(temp_ten_per_length)
##用int取整数
#print(integer_ten_per_length)
#nine_per_length = random_dataset_longitude - integer_ten_per_length
#print(nine_per_length)


"""
#第一次取前90%作为训练集，10%作为测试集
train = random_dataset[:nine_per_length]
print(random_dataset)
len_train = len(train)
print(len_train)
test = random_dataset[nine_per_length:random_dataset_longitude]
print(test)
len_test = len(test)
print(len_test)
"""

"""10折交叉验证算法的实现"""
#获取样本空间10等份的每一等份的行号
#nth_rowsnum_list = []
#nth_rows_number = 0
#for i in range(len(random_dataset)):
#    nth_rows_number = nth_rows_number + integer_ten_per_length
#    if(nth_rows_number <= random_dataset_longitude):
#       nth_rowsnum_list.append(nth_rows_number)
#print("计算出所有的第10%行的行号，", nth_rowsnum_list)
#
#
#dataset_splite_by_ten_percentage = np.vsplit(random_dataset,nth_rowsnum_list)
##将学习样本空间 C 分为大小相等的 K 份  
#print(dataset_splite_by_ten_percentage)
#
#
#wide_height = len(dataset_splite_by_ten_percentage)
#print(wide_height)  



#k_fold = 10
#test = []
##测试集
#train = []
#训练集
#for i in range(k_fold):   
#    tem_dataset = dataset_splite_by_ten_percentage[i]
#    # 取第i份作为测试集
#    test.append(tem_dataset)
#    temp_train = np.array([], dtype=float).reshape(-1,random_dataset_width)
#    #数据集中有18列所以此处是18,用一个变量random_dataset_width来表示，免得数据集一旦多一列就会出错。
#    #random_dataset_width = random_dataset.shape[1]用此种方法获得，即可动态变化
#    for j in range(k_fold):
#        if(j != i):   
#           temp_train = np.vstack((temp_train,dataset_splite_by_ten_percentage[j]))
#           #

#           print(temp_train)
#    train.append(temp_train)
#print(train)

        
        
#test_first_fold = test[0]
#train_first_fold = train[0]     
#
#test_second_fold = test[1]
#train_second_fold = train[1]    
#
#test_third_fold = test[2]
#train_third_fold = train[2]     
#
#test_fourth_fold = test[3]
#train_fourth_fold = train[3]  
#
#test_fifth_fold = test[4]
#train_fifth_fold = train[4]     
#
#test_sixth_fold = test[5]
#train_sixth_fold = train[5]   
#
#test_seventh_fold = test[6]
#train_seventh_fold = train[6]  
#
#test_eighth_fold = test[7]
#train_eighth_fold = train[7]     
#
#test_ninth_fold = test[8]
#train_ninth_fold = train[8]  
#
#test_tenth_fold = test[9]
#train_tenth_fold = train[9]     



#print(test_first_fold)
#print(train_first_fold)
#print(test_second_fold)
#print(train_second_fold)
#print(test_third_fold)
#print(train_third_fold)
#print(test_tenth_fold)
#print(train_tenth_fold)


my_classification_neural_network_test_score_global = [ ]
#保存十折的神经网络的score的数值
my_classification_neural_network_test_accuracy_global = [ ]
#保存十折的神经网络的accuracy的数值
my_classification_neural_network_precision_result_global = [ ]
#保存我的分类神经网络实验结果的precision的值
my_classification_neural_network_recall_result_global = [ ]
#保存我的分类神经网络实验结果的recall值
my_classification_neural_network_F1_measure_result_global = [ ]
##保存我的分类神经网络实验结果的F1_measure的值
bug_predect_label_confusion_matrix_test = [ ]
#保存十折的四列结果数值，测试集中的bug，predect，predect_label,confusion_matrix


#保存我的神经网络的预测模型的测试集和机器学习的输出结果bug/la+ld+1(也就是defect sensity在左右方向进行合并的)果的原始数据集，没有进行任何排序
my_nn_model_original_test_dataset_predict_density_global = [ ]
#保存我的神经网络的预测模型中的测试集和机器学习输出的defect density（bug/la+ld+1）合并之后的数据集按照bug/la+ld+1降序排序之后的数据集
my_nn_model_test_dataset_density_decent_orderby_predict_density_global  = [ ]


TP_global = [ ]
FN_global = [ ]
FP_global = [ ]
TN_global = [ ]

my_neural_network_exp_result_global = [ ]
#保存我的实验结果的全局变量
nju_exp_result_global = [ ]
#保存南京大学实验结果的全局变量



"""***********************以下是自己的方法*****************"""
"""***********************以下是自己的方法*****************"""
"""***********************以下是自己的方法*****************"""
#函数statistics_function(a,b)，需要传入两个参数，返回一个统计结果的list，list中保存的是
#统计结果的6个list。
#需要传入两个参数，一个参数是sum_la_ld的这一列，获取这一个列向量可以计算达到总的la+ld的
#20%时的行数。
#一个是按照某一列进行排序之后的整个数据集，通过这个数据集可以获取测试集的长度和
#测试集中的bug这一列的数值。
def statistics_function(sum_lald_vector, decent_orderby_outputml222):
    
    nth_lines_number_list = []
    #用来存放前20到第几行
    testdata_total_lines_number_list = []
    #用来存放测试集中所有行的数量
    nth_lines_percent_list = []
    #用来存放la+ld的和达到总行数20%时行数与测试集中总行数的百分比
    twenty_percent_nonzero_lines_list = []
    #用来存放la+ld的和达到总行数20%时行数中bug为非0的行数
    total_nonzero_lines_list = []
    #用来存放测试集中全部的bug非0行数
    nonzero_lines_percent_list = []
    #用来存放测试集中la+ld的和达到测试集中全部的la+ld的和的20%时，
    #前20%行中bug为非0的数量与测试集中bug为非0的数量的比值

    sum_lald_colum = sum_lald_vector
    print(sum_lald_colum)
    #sum_lald_colum存放的是按照行号从小到大排序之后的la+ld的结果
    decent_orderby_outputml = decent_orderby_outputml222
    
    sumlald = 0
    sumlaldsecond = 0
    for i in range(len(sum_lald_colum)):
        sumlald = sumlald + sum_lald_colum[i]
    sumlaldsecond = sumlald    
    print(sumlald)
    print(sumlaldsecond)
    
    total_lald = np.sum(sum_lald_colum,axis=0) 
    
    
    
    tatal_lald_first = np.sum(sum_lald_colum)
    #total_lald_second = np.sum(sum_lald_colum,axis=1) 
    #计算出所有的行和la+ld
    #total_lald = np.sum(sum_lald_colum,axis=0) 
    print(total_lald)
    print(tatal_lald_first)
    #print(total_lald_second)
    
    
    print("total_lald的类型是",type(total_lald))              
    
    
    twentypercent_lald = (total_lald * 2)/10
    print(twentypercent_lald)
                         
    
    
    
    nthrows_lald = 0
    rowsnumber = 0
    
    
    for i in range(len(sum_lald_colum)): 
        nthrows_lald = sum_lald_colum[i] + nthrows_lald 
        if(nthrows_lald >=  twentypercent_lald):
           print("计算出从第一行加到第n行的数值小于等于la+ld的总数的20%是第多少行",i)
           rowsnumber = i + 1
           break  
    print("计算出从第一行加到第n行的数值小于等于la+ld的总数的20%是第多少行",rowsnumber) 
    
    
    
    nth_lines_number_list.append(rowsnumber)
    print(nth_lines_number_list)
   
    
    
    total_lines_testdata = len(decent_orderby_outputml)
    print("测试集的总行数是：",total_lines_testdata) 
    
    
    
    testdata_total_lines_number_list.append(total_lines_testdata)
    print(testdata_total_lines_number_list)
    
    
    
    rate_percentage_tweperc = D(rowsnumber)/D(total_lines_testdata)
    percentage_tweperc = float(rate_percentage_tweperc)
    print("计算出从第一行加到第n行的数值小于等于la+ld的总数的20%是所占的比例",percentage_tweperc)
    print(percentage_tweperc)
    percentage_tweperc_final = percentage_tweperc*100
    print("%.6f%%" % percentage_tweperc_final)
    
    
    
    nth_lines_percent_list.append(percentage_tweperc)
    print(nth_lines_percent_list)
    
    
    #获取按照某一列进行排序后的测试集中的bug的这一列
    test_data_bug_column = decent_orderby_outputml[:,15]
    print(test_data_bug_column)
    
    
    tweperc_non_zero_num = 0
    counter_twe = 0
    counter_total =  0
    for m in range(rowsnumber): 
        if(test_data_bug_column[m] !=  0):
           counter_twe = counter_twe + 1
    print("计算出la+ld达到总的20%时的非0行数",counter_twe)
    tweperc_non_zero_num = counter_twe
    print("计算出la+ld达到总的20%时的非0行数",tweperc_non_zero_num) 
    
    
    
    twenty_percent_nonzero_lines_list.append(counter_twe)
    print(twenty_percent_nonzero_lines_list)
    
    
    
    total_non_zero_num = 0
    for n in range(len(test_data_bug_column)): 
        if(test_data_bug_column[n] !=  0):
           counter_total = counter_total + 1
    print("计算出所有的非0行数",counter_total)
    total_non_zero_num = counter_total
    print("计算出所有的非0行的行数",total_non_zero_num) 
    
    
    
    total_nonzero_lines_list.append(counter_total)
    print(total_nonzero_lines_list)
   
    
    rate_nonzero = D(tweperc_non_zero_num)/D(total_non_zero_num)
    percentage_nonzero = float(rate_nonzero)
    
    print("计算出前20%的非0行（没有缺陷）占所有非0行（没有缺陷）的比例",percentage_nonzero)
    print(percentage_nonzero)
    percentage_nonzero_final = percentage_nonzero*100
    print("%.6f%%" % percentage_nonzero_final)
    
    
    
    nonzero_lines_percent_list.append(percentage_nonzero)
    print(nonzero_lines_percent_list)
    
    return(rowsnumber,total_lines_testdata,percentage_tweperc,
           counter_twe,counter_total,percentage_nonzero)

    #return(nth_lines_number_list,testdata_total_lines_number_list,nth_lines_percent_list,
           #twenty_percent_nonzero_lines_list,total_nonzero_lines_list,nonzero_lines_percent_list)






"""开始十折交叉验证，程序开始循环10次"""
for i in range(len(train)):
    train_current = train_except_transactionid_commitdate[i]
    test_current = test_except_transactionid_commitdate[i]
    print(train_current)
    print(test_current)
    #对训练集进行处理
    nth_colnum_sort_by = 14
    #按照bug/(la+ld)的数值进行排序，按照0和非0排序
    train_sortby_bug_decent_order = train_current[np.argsort(-train_current[:,nth_colnum_sort_by])]
    #按照bug的数值进行降序排序，按照1和0进行排序，bug在第15列
    #进行降序排序需在前边加上-减号
    
    
    #计算出bug不是0的行数 
    nth_bug_colnum = 14     
    bug_column = train_sortby_bug_decent_order[:,nth_bug_colnum]
    nonzero_num = 0
    for j in range(len(bug_column)): 
        print(bug_column[j])
        if(bug_column[j] ==  0):
           print("计算出非0行到第几行",j)
           nonzero_num = j
           break  
    print(nonzero_num)
    
    longitude = train_sortby_bug_decent_order.shape[0]
    #经度（竖直）计算出整个训练集中的总行数
    print(longitude)
    
    
    #latitude = traing_datasets_orderby_decent.shape[1]
    #纬度（横线 ）
    #将数据集分成两部分，bug为1的是一组，bug是0的是一组
    train_bug_nonzero = train_sortby_bug_decent_order[0:nonzero_num]
    #bug是1的数据集
    train_bug_zero = train_sortby_bug_decent_order[nonzero_num:longitude]
    #bug是0的数据集
    print(train_bug_nonzero)
    print(train_bug_zero)
    
    
    
    #在bug为1的数据集中（也就是非0数据集中），按照 la+ld的大小从小到大进行排序。
#    nth_colnum_sort_by_sumlald = 16
#    #la+ld的和在数据集的第16列
#    train_nonzero_sortby_sumlald_ascending_order = train_bug_nonzero[np.argsort(train_bug_nonzero[:,nth_colnum_sort_by_sumlald])]
#    #按照la+ld的数值进行升序排序，从小到大排序。
#    #开一个新的向量进行编号
#    
#    #求出训练集中bug为1的矩阵的高度
#    train_nonzero_longitude = train_nonzero_sortby_sumlald_ascending_order.shape[0]
#    print(train_nonzero_longitude)
#    train_nonzero_sn = 0
#    
#    #对训练集中bug为1的向量进行编号从1开始到train_nonzero_longitude
#    train_output_nonzero_label = []
#    for i in range(train_nonzero_longitude):
#        train_nonzero_sn = train_nonzero_sn + 1
#        train_output_nonzero_label.append(train_nonzero_sn)
#    print(train_output_nonzero_label)
#    
#    #将list转换为矩阵
#    train_output_nonzero_label_vec = np.array(train_output_nonzero_label)
#    #traing_output_nonzero_label_vec = np.array(traing_output_nonzero_label, dtype='float64')
#    print(train_output_nonzero_label_vec)
#    
#    
#    train_datasets_nonzero_final = np.column_stack((train_nonzero_sortby_sumlald_ascending_order, train_output_nonzero_label_vec))
#    #将编的行号添加到最后一列，也就是第18列上
#    print(train_datasets_nonzero_final)
#    
    
    
    
#    nth_colnum_bug_zero_sort_by_sumlald = 16
#    #在bug是0的数据集中按照，按照la+ld的和的数值这一列进行排序
#    train_zero_sortby_sumlald_ascending_order = train_bug_zero[np.argsort(train_bug_zero[:,nth_colnum_bug_zero_sort_by_sumlald])]
#    #在bug为0的数据集中（也就是0数据集中），按照 la+ld的大小从小到大进行排序。
#    
#    """
#    开一个新的向量,对bug为0的数据集的行数进行编号，从2n+1开始编号
#    """
#    train_zero_longitude = train_zero_sortby_sumlald_ascending_order.shape[0]
#    #获取bug为0的数据集的长度
#    print(train_zero_longitude)
#    train_zero_sn = 2*train_nonzero_longitude
#    #对bug为0的数据集的行数进行编号，从2n+1开始编号
#    train_output_zero_label = []
#    for i in range(train_zero_longitude):
#        train_zero_sn = train_zero_sn + 1
#        train_output_zero_label.append(train_zero_sn)
#    print(train_output_zero_label)
#    
#    train_output_zero_label_vec = np.array(train_output_zero_label)
#    #traing_output_nonzero_label_vec = np.array(traing_output_nonzero_label, dtype='float64')
#    print(train_output_zero_label_vec)
#    #对bug为0的数据集从2n+1开始进行编号
#    
#    train_datasets_zero_final = np.column_stack((train_zero_sortby_sumlald_ascending_order, train_output_zero_label_vec))
#    #对bug为0的数据集增加行号
#    print(train_datasets_zero_final)
    
    
    
    """处理数据不平衡，对bug为0的数据集进行采样，对bug为0的进行采样之后的数据和bug为1的数据集一样"""
    
    random.seed (1)
    #设置随机种子，保证实验可以重现
    sampling_zero_rows_list =  random.sample(range(len(train_bug_zero)), nonzero_num)
    #生成的随机数的个数是：nonzero_num，训练集中bug为非0的行数。 
    #生成的随机数的范围是0到 bug为0的训练集长度个：range(len(train_datasets_zero_final))
    #也就是生成nonzero_num个，范围是0到bug为0的数据集的长度个行号。
    print(sampling_zero_rows_list)
    sampling_zero_line_numbers = len(sampling_zero_rows_list)
    print(sampling_zero_line_numbers)
    
    sampling_train_datasets_zero = train_bug_zero[sampling_zero_rows_list]
    #对bug为0的数据集进行采样，采样的个数和bug不为0的个数一样。将采样的行号传入数据集生成采样的数据集矩阵
    print(sampling_train_datasets_zero)
    
    sampling_train_datasets_zero_length = sampling_train_datasets_zero.shape[0]
    sampling_train_datasets_zero_width = sampling_train_datasets_zero.shape[1]
    
    print(sampling_train_datasets_zero_length)
    print(sampling_train_datasets_zero_width)
    
    
    #ultimate_train_datasets = np.vstack((train_datasets_nonzero_final,sampling_train_datasets_zero))
    #将bug为1的训练集矩阵和采样后的bug为0的训练集矩阵合并，在竖直方向上进行合并，生成最终的训练集
    
    ultimate_train_datasets = np.vstack((train_bug_nonzero,sampling_train_datasets_zero))
    #将bug为1的训练集和bug为0的训练集合并，在竖直方向上合并。
    print(ultimate_train_datasets)
    
    len_ultimate_traing_datasets = len(ultimate_train_datasets)
    print(len_ultimate_traing_datasets)
    
    
    
    """
    数据集的格式是：
    第几列：
    1    2   3     4       5   6    7    8     9    10   11     12   13    14        15            16       17        18      19        20
    在内存中的索引： 
    0    1   2     3       4   5    6    7     8    9    10     11   12    13        14            15       16        17      18        19         20           21
    ns  nm  nf  entropy   la	 ld	 lt 	fix 	ndev 	 pd   npt	  exp  rexp  sexp 	 bug/sum(la+ld+1)  bug   sum(la+ld)  lt_raw   la_raw   ld_raw  predict_label   1/lt_raw   
    
    """
    
    
    
    without_sampling_X_train = train_current[:,:14]
    #将训练集中前14个度量项作为训练集的输入
    without_sampling_original_remove_nd_la_ld_rexp_X_train = np.delete(without_sampling_X_train, [1,4,5,12], axis=1)
    #用没有采样的数据集来调整阈值
    without_sampling_original_y_bug_label_train = train_current[:,15]
    #训练集中bug这一列在第15列，
    
    
    
    without_sampling_decent_sortby_X_train = train_sortby_bug_decent_order[:,:14]
    #没有采样的训练集按照bug进行降序排序之后的数据集
    without_sampling_sortby_bug_remove_nd_la_ld_rexp_X_train = np.delete( without_sampling_decent_sortby_X_train, [1,4,5,12], axis=1)
    #用没有采样的数据集来调整阈值
    without_sampling_sortby_bug_y_bug_label_train = train_sortby_bug_decent_order[:,15]    
    #训练集中bug这一列在第15列，
    
    
    
    
    
    ultimate_X_train = ultimate_train_datasets[:,:14]
    #将训练集中前14个度量项作为训练集的输入
    
    remove_nd_la_ld_rexp_X_train = np.delete(ultimate_X_train, [1,4,5,12], axis=1)
    #要移除掉的4列是  """移除掉ND,REXP,LA,LD四个度量项"""
    #其中ND这一个度量项指的就是NM这一个度量项。 目前认为NM就是ND
    #没有ND这一个度量项，只有LA,LD,REXP这三个度量项,移除掉这4列
    #其中LA在第4列，LD在第5列值，REXP在第12列
    #这4列在我得训练集和测试集中分别是在第 4,5,12列。
    #sum_la_ld_train = ultimate_train_datasets[:,16]
    #将取出来训练集中的第14列也就是la+ld这个度量项作为新的度量项，它在第16列
    #remove_nd_la_ld_rexp_add_sum_la_ld_X_train = np.column_stack((remove_nd_la_ld_rexp_X_train, sum_la_ld_train))
    
    
    ultimate_y_train = ultimate_train_datasets[:,14]
    #将训练集中的bug/(la+ld+1)，在训练集中的第14列。
    
    ultimate_y_bug_label_train = ultimate_train_datasets[:,15]
    #训练集中bug这一列在第15列，
    
    ultimate_X_test = test_current[:,:14]
    #将测试集中的前14列度量项作为训练集的输入
    
    remove_nd_la_ld_rexp_X_test = np.delete(ultimate_X_test, [1,4,5,12], axis=1)
    #要移除掉的4列是  """移除掉ND,REXP,LA,LD四个度量项"""
    #其中ND这一个度量项指的就是NM这一个度量项。 目前认为NM就是ND
    #没有ND这一个度量项，只有LA,LD,REXP这三个度量项
    #其中LA在第4列，LD在第5列值，REXP在第12列
    #这4列在我得训练集和测试集中分别是在第 4,5,12列。
    sum_la_ld_test = test_current[:,16]
    #将取出来训练集中的第14列也就是la+ld这个度量项作为新的度量项，它在第16列
    remove_nd_la_ld_rexp_add_sum_la_ld_X_test = np.column_stack((remove_nd_la_ld_rexp_X_test, sum_la_ld_test))
    ultimate_y_test = test_current[:,14]
    #将测试集中的第14列也就是bug/la+ld+1这个度量项作为测试集的输出，它在第14列
    #print(ultimate_X_train)
    
    ultimate_y_bug_label_test = test_current[:,15]
    #将测试集中的bug这一列的数值作为测试集的bug这一列，在测试集中第15列
    
    print(ultimate_y_train)
    
    
    
    

    """     
    直接使用 Numpy 库的 loadtxt() 方法加载数据,一共 8 个输出变量和 1 个输出变量（最后一列）。
    加载之后我们就可以把数据分离为 X（输出变量）和 Y（输出分类）    
     split into input (X) and output (Y) variables
    delimiter：数据之间的分隔符。如使用逗号","。
    
    这是因为numpy在读取元素时，默认是按照float格式来读取的，对于不能转换为float类型的数据会读取为nan（not a number），
    对于留空的数据则显示为na（not available），为了正确的读取数据，可以通过增加参数：
    dtype参数用来指定读取数据的格式，这里的U75表示将每一个数据都读取为75个byte的unicode数据格式
    skip_header参数用来跳过文件的第一行
    delimiter参数用来指定每行数据的分隔符
    """
    
    
    """
    *******第二步：构建神经网络模型***（定义）构造一个神经网络模型******* 
    """
    

    model = Sequential()
    #初始化构造一个神经网络模型,单支线性网络模型
    #Layer1
    model.add(Dense( output_dim = 20, input_dim = 10, kernel_initializer = 'uniform'))
    #为第一层指定输入数据的shape的参数  add 添加一层神经网 
    """
    #输入层有784个神经元
    #第一个隐层有1个神经元，激活函数为ReLu，Dropout比例为0.2
    Dense是全连接网络,只有第一层需要设定输入层的结点个数,其他都不需要
    Dense’表示全连接结构；’Activation’表示每层的激活函数，
    即每层的输入和输出的对应关系，
    常用的有’tanh’、’relu’、’sigmoid’函数；
    ’input_dim=6’表示输入参数为6维，
    ’15000’表示第一层的输出为15000维
    添加的是Dense全连接神经层。
    参数有两个，（注意此处Keras 2.0.2版本中有变更）
    一个是输入数据的维度，输入层，28*28=784 
    另一个units代表神经元数，即输出单元数。
    参数表示该层神经元的数量。
    # init是关键字，’uniform’表示用均匀分布去初始化 
    上面代码我们定义了一个简单的多层感知模型：具有2个入参的输入层，
    具有5个神经元的隐含层，具有1个神经元的输出层。
    """

    model.add(Activation('tanh')) 
    #model.add(Dropout(0.5))
    #model.add(Activation('tanh')) 
    #激活函数层 
    """
    #每一个神经网络层都需要一个激活函数
    """  
    #Layer2
    model.add(Dense(output_dim = 10, kernel_initializer='normal'))
    model.add(Activation('relu')) 
    #model.add(Dropout(0.25))
    #model.add(Dense(output_dim = 8, init = 'uniform'))
    #第一个隐藏层有14个神经元，Dropout比例为0.5
    #激活函数层   


    #Layer3
#    model.add(Dense(output_dim = 6, kernel_initializer='normal'))
#    #第二个隐藏层有10个神经元，
#    model.add(Activation('relu'))
    #激活函数层 
    #model.add(Dropout(0.25))
    #Dropout比例为0.5

    #Layer4
    #model.add(Dense(output_dim = 4,init = 'uniform'))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
   #第三个隐层有6个神经元，激活函数为ReLu，Dropout比例为0.2

   #Layer5
   #model.add(Dense(output_dim = 6,init = 'uniform'))
   #model.add(Activation('relu'))
   #model.add(Dropout(0.25))
   #第四个隐层有6个神经元，激活函数为ReLu，Dropout比例为0.2


   #Layer6
   #model.add(Dense(output_dim = 3, init = 'uniform'))
   #model.add(Activation('softmax'))
   #model.add(Dropout(0.5))
  #第五个隐层有3个神经元，激活函数为ReLu，Dropout比例为0.2


    #Layer5
    model.add(Dense(1, kernel_initializer='normal'))
    """
    因为是回归问题，在输出层作者并没有使用激活函数。
    """
    #输出层有1个神经元，激活函数为sigmoid，得到分类结果
    #全连接层，神经元个数为200个
    #最后一层也是输出层

    # 输出模型的整体信息
    # 总共参数数量为784*512+512 + 512*512+512 + 512*10+10 = 669706
    #model.summary()
                
    """
    *******第三步：编译模型*******
    模型编译时必须指明损失函数和优化器，配置模型的学习过程 
    model.compile来激活模型，参数中，误差函数用的是mse均方误差；
    优化器用的是 sgd 随机梯度下降法。metrics（评估模型的指标）
    编译模型,使用后端的代码tensorflow或theano去实现 
    """

    model.compile(optimizer='adam',loss='mean_squared_error')
    #model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])


    history = model.fit( remove_nd_la_ld_rexp_X_train, ultimate_y_train, batch_size = 30, nb_epoch = 150, verbose=1) 
#    history = model.fit( ultimate_X_train, ultimate_y_train, batch_size = 30, nb_epoch = 150, verbose=1)
#    myoutputpredy = model.predict( ultimate_X_test )
#    loss_and_metrics = model.evaluate(ultimate_X_train, ultimate_y_train, batch_size=128)
    
    
    
#     def my_baseline_model():
#        model = Sequential()
#        #初始化构造一个神经网络模型,单支线性网络模型
#        #Layer1
#        model.add(Dense( output_dim = 14, input_dim = 14, kernel_initializer = 'uniform'))
#        #为第一层指定输入数据的shape的参数  add 添加一层神经网 
#        """
#        #输入层有784个神经元
#        #第一个隐层有1个神经元，激活函数为ReLu，Dropout比例为0.2
#        Dense是全连接网络,只有第一层需要设定输入层的结点个数,其他都不需要
#        Dense’表示全连接结构；’Activation’表示每层的激活函数，
#        即每层的输入和输出的对应关系，
#        常用的有’tanh’、’relu’、’sigmoid’函数；
#        ’input_dim=6’表示输入参数为6维，
#        ’15000’表示第一层的输出为15000维
#        添加的是Dense全连接神经层。
#        参数有两个，（注意此处Keras 2.0.2版本中有变更）
#        一个是输入数据的维度，输入层，28*28=784 
#        另一个units代表神经元数，即输出单元数。
#        参数表示该层神经元的数量。
#        # init是关键字，’uniform’表示用均匀分布去初始化 
#        上面代码我们定义了一个简单的多层感知模型：具有2个入参的输入层，
#        具有5个神经元的隐含层，具有1个神经元的输出层。
#        """
#    
#        model.add(Activation('tanh')) 
#        #model.add(Dropout(0.5))
#        #model.add(Activation('tanh')) 
#        #激活函数层 
#        """
#        #每一个神经网络层都需要一个激活函数
#        """  
#        #Layer2
#        model.add(Dense(output_dim = 8, kernel_initializer='normal'))
#        model.add(Activation('relu')) 
#        #model.add(Dropout(0.25))
#        #model.add(Dense(output_dim = 8, init = 'uniform'))
#        #第一个隐藏层有14个神经元，Dropout比例为0.5
#        #激活函数层   
#    
#    
#        #Layer3
#        model.add(Dense(output_dim = 6, kernel_initializer='normal'))
#        #第二个隐藏层有10个神经元，
#        model.add(Activation('relu'))
#        #激活函数层 
#        #model.add(Dropout(0.25))
#        #Dropout比例为0.5
#    
#        #Layer4
#        #model.add(Dense(output_dim = 4,init = 'uniform'))
#        #model.add(Activation('relu'))
#        #model.add(Dropout(0.5))
#       #第三个隐层有6个神经元，激活函数为ReLu，Dropout比例为0.2
#    
#       #Layer5
#       #model.add(Dense(output_dim = 6,init = 'uniform'))
#       #model.add(Activation('relu'))
#       #model.add(Dropout(0.25))
#       #第四个隐层有6个神经元，激活函数为ReLu，Dropout比例为0.2
#    
#    
#       #Layer6
#       #model.add(Dense(output_dim = 3, init = 'uniform'))
#       #model.add(Activation('softmax'))
#       #model.add(Dropout(0.5))
#      #第五个隐层有3个神经元，激活函数为ReLu，Dropout比例为0.2
#    
#    
#        #Layer5
#        model.add(Dense(1, kernel_initializer='normal'))
#        """
#        因为是回归问题，在输出层作者并没有使用激活函数。
#        """
#        #输出层有1个神经元，激活函数为sigmoid，得到分类结果
#        #全连接层，神经元个数为200个
#        #最后一层也是输出层
#    
#        # 输出模型的整体信息
#        # 总共参数数量为784*512+512 + 512*512+512 + 512*10+10 = 669706
#        #model.summary()
#                    
#        """
#        *******第三步：编译模型*******
#        模型编译时必须指明损失函数和优化器，配置模型的学习过程 
#        model.compile来激活模型，参数中，误差函数用的是mse均方误差；
#        优化器用的是 sgd 随机梯度下降法。metrics（评估模型的指标）
#        编译模型,使用后端的代码tensorflow或theano去实现 
#        """
#    
#        model.compile(optimizer='adam',loss='mean_squared_error')
#        #model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
#        return model

    """
    *******第四步：训练模型******
    传入要训练的数据和标签，并指定训练的一些参数，然后进行模型训练
    训练模型
    batch_size：指定梯度下降时每个batch包含的样本数
    nb_epoch：训练的轮数，nb指number 
    verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为epoch输出一行记录
    validation_data：指定验证集
    fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，
    如果有验证集的话，也包含了验证集的这些指标变化情况 
    """
    
#    print('\n Keras预测数值 ------------')
    
    
    
    # fix random seed for reproducibility
#    seed = 7
#    numpy.random.seed(seed)
#    # evaluate model with standardized dataset
#    estimator = KerasRegressor(build_fn = my_baseline_model, nb_epoch=100, batch_size=20,verbose=1)
#    print(estimator) 
    
    """
     We will use 10-fold cross validation to evaluate my_base_line_model.
    """
#    print("\n We will use 10-fold cross validation to evaluate my_base_line_model.")
#    kfold = KFold(n_splits = 10, random_state = seed)
#    results = cross_val_score(estimator, ultimate_X_train, ultimate_y_train, cv=kfold)
#    print("\n evaluate this baseline model")
#    print("\n The result reports the mean squared error including the average and standard deviation (average variance) across all 10 folds of the cross validation evaluation")
#    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    
#    print('\n Training ------------')
    
    #estimator.fit( ultimate_X_train, ultimate_y_train, epochs=360)
    
    #history = model.fit(train, train_target, epochs=20 ) #batch_size=10,verbose = 1
    #history = model.fit(X_train, y_train, epochs=100, batch_size=16,verbose = 1) 
    #batch_size=10,verbose = 1
    
    """
    我们将迭代20次、批处理大小为128。这些参数可以通过试错来选择。
    """
    
    
    """
    *********第五步：测试数据*********
    用测试数据测试已经训练好的模型，并可以获得测试结果，从而对模型进行评估。
     展示模型在验证数据上的效果
    返回：误差率或者是(误差率，准确率)元组（if show_accuracy=True）
    参数：和fit函数中的参数基本一致，其中verbose取1或0，表示有进度条或没有
    。一个典型的划分是训练集占总样本的50％，而其它各占25％，三部分都是从样本中随机抽取。
    
    
    """
    #按batch计算在某些输入数据上模型的误差
    #evaluate
    
    #print ("*********",X_test[0])
    
    #print ("type(X_test[0]****)",type(X_test[0]))
    #myoutputpredy = estimator.predict( ultimate_X_test )
#    print('########output predect#######:', myoutputpredy)
#    r2 = r2_score( ultimate_y_test, myoutputpredy )
#    rmse = mean_squared_error( ultimate_y_test, myoutputpredy )
#    print("##################################################################")
#    print("##################################################################")
#    print("##################################################################")
#    print( "KERAS: R2 : {0:f}, RMSE : {1:f}".format( r2, rmse ) )
#    
#    #print( "KERAS: R2 : {0:f}, RMSE : {1:f}".format( r2, rmse ) )
#    #print( "KERAS: R2 : {0:f}, RMSE : {1:f}".format( rmse ) )
#    
#    print('\nTesting ------------')
    #score = model.evaluate(X_test, y_test, batch_size = 32, verbose = 1, sample_weight = None)
    
    #score111 = mean_squared_error(y_test, estimator.predict(X_test))
    #score222 = mean_absolute_error(y_test, estimator.predict(X_test))
    
    #print("score111**************",score111)
    #print(len(score111))
    #print("score222**************",score222)
    
    
    
    # 输出训练好的模型在测试集上的表现
    
    #print("score**************",score)
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1])
    
    
    #predict
    #predict = model.predict(X_test, batch_size = 32, verbose = 0)
    #print('Test predect:', predict)
    #print('Test predect:','%.9f' %predict)
    #proba = model.predict_proba(X_test, batch_size=1)
    #print(proba)
    #classes = model.predict_classes(X_test, batch_size=1)
    #print(classes)
    
    
    
    #ax.plot(X_test, predict, 'k--', lw=4)  
    #ax.set_xlabel('Measured')  
    #ax.set_ylabel('Predicted')  
    #plt.show()  
    #figures(history)  
    
    """
    本函数按batch获得输入数据对应的输出，其参数有：
    函数的返回值是预测值的numpy array
    """







    #nf_current_column = X_test[:,2]
    #la_current_column = X_test[:,4]
    #ld_current_column = X_test[:,5]
    #lt_current_column = X_test[:,6]
    #
    #print(nf_current_column)
    #print(la_current_column)
    #print(ld_current_column)
    #print(lt_current_column)
    #
    #nf_normalization_column = nf_current_column/1000
    #la_normalization_column = la_current_column/1000
    #ld_normalization_column = ld_current_column/1000
    #lt_normalization_column = lt_current_column/1000
    #
    #print(nf_normalization_column)
    #print(la_normalization_column)
    #print(ld_normalization_column)
    #print(lt_normalization_column)
    #
    #
    #nf_min = 1
    #nf_max = 2817
    #nf_dvalue = nf_max - nf_min
    #nf_raw = ( nf_normalization_column * nf_dvalue) + nf_min
    #
    #print(nf_raw)
    #         
    #      
    #
    #lt_min = 0
    #lt_max = 38980
    #lt_dvalue = lt_max - lt_min
    #lt_norma = ( lt_normalization_column * lt_dvalue) + lt_min
    #
    #print(lt_norma)
    #
    #  
    #lt_raw = lt_norma * nf_raw
    #la_raw = la_normalization_column * lt_raw
    #ld_raw = ld_normalization_column * lt_raw
    #
    #
    #print(la_raw)
    #print(ld_raw)
    #
    #ltraw_ldraw_vector = np.column_stack((la_raw,ld_raw))
    #sum_laraw_ldraw = ltraw_ldraw_vector.sum(axis=1)#计算矩阵的列和
    #
    #
    #                                        
    #temp111 = []
    #for i in range(len(sum_laraw_ldraw)):
    #    temp111.append(sum_laraw_ldraw[i])
    #print(temp111)
    #
    #
    #                                        
    #
    #final_vector = np.column_stack((twentyper_rowsnum_datasets, sum_laraw_ldraw))
    #
    #print(final_vector)












    
    """将测试集的数据输入进行预测"""
    output_predict_defect_probolity_testdata = model.predict_proba( remove_nd_la_ld_rexp_X_test )
    
    
    
    
    predict_approach_datasets = []
    temp_predict_approach_datasets = []
    temp_predict_approach_datasets = test_current
    
    
    sum_lald_colum = temp_predict_approach_datasets[:,16]
    #la+ld的和在数据集的第16列。  
    print(sum_lald_colum)
    
    sum_lald_one_colum = sum_lald_colum + 1
    #la+ld的数值可能会是0，所以要加上1.

    output_probolity_tuple1 = output_predict_defect_probolity_testdata.shape
    sum_lald_one_colum_transfer_to_column = sum_lald_one_colum.reshape(output_probolity_tuple1)
 
    defect_density_probability = np.divide(output_predict_defect_probolity_testdata,sum_lald_one_colum_transfer_to_column)
    
    
    
    
    predict_approach_datasets = np.column_stack((temp_predict_approach_datasets, defect_density_probability))
    print(predict_approach_datasets)
    
    nth_col_num_sorting_according = 20
    #神经网络模型预测的分类结果，也就是defect的数值的降序进行排序，它的数值在数据集的最后一列，也就是第20列。
    my_nn_model_original_test_dataset_predict_density_global.append(predict_approach_datasets)
    #将测试集中的原始数值和神经网络预测的标签保存起来



    decent_orderby_outputml = predict_approach_datasets[np.argsort(-predict_approach_datasets[:,nth_col_num_sorting_according]) ]
    
    """保存我的神经网络的预测模型中的测试集和机器学习输出的defect density（bug/la+ld+1）合并之后的数据集按照bug/la+ld+1降序排序之后的数据集"""
    my_nn_model_test_dataset_density_decent_orderby_predict_density_global.append(decent_orderby_outputml)
    
    sum_lald_colum_after_sorting = decent_orderby_outputml[:,16]
    #la+ld的和在第16列。
    print(sum_lald_colum_after_sorting)
    
    my_neu_result1,my_neu_result2,my_neu_result3,my_neu_result4,my_neu_result5,my_neu_result6 = statistics_function(sum_lald_colum_after_sorting,decent_orderby_outputml)
    my_neural_network_exp_result_global.append([my_neu_result1,my_neu_result2,my_neu_result3,my_neu_result4,my_neu_result5,my_neu_result6])





















#数据集合并成想要的结果

my_neural_network_output_result_final = []
tem_my_neural_network_output = np.array([], dtype=float).reshape(-1,6)
#定义一个空的矩阵
#数据集中有6列所以此处是6,用一个变量random_dataset_width来表示，免得数据集一旦多一列就会出错。
#random_dataset_width = random_dataset.shape[1]用此种方法获得，即可动态变化
#训练集
for i in range(len(my_neural_network_exp_result_global)):   
    #tem_my_neural_row = my_neural_network_exp_result_global[i]
    #取得list中第一列的数值
    tem_myneural_result = np.array(my_neural_network_exp_result_global[i])
    tem_my_neural_network_output = np.vstack((tem_my_neural_network_output,tem_myneural_result))
    #将第list中第一个数值在垂直方向上和别的数值进行合并
    print(tem_my_neural_network_output)
my_neural_network_output_result_final.append(tem_my_neural_network_output)
print(my_neural_network_output_result_final)




#
##数据集合并成想要的结果
#
#nju_output_result_final = []
#tem_nju_output = np.array([], dtype=float).reshape(-1,6)
##定义一个空的矩阵
##数据集中有6列所以此处是6,用一个变量random_dataset_width来表示，免得数据集一旦多一列就会出错。
##random_dataset_width = random_dataset.shape[1]用此种方法获得，即可动态变化
##训练集
#for i in range(len(nju_exp_result_global)):   
#    #tem_my_neural_row = my_neural_network_exp_result_global[i]
#    #取得list中第一列的数值
#    tem_nju_result = np.array(nju_exp_result_global[i])
#    tem_nju_output = np.vstack((tem_nju_output,tem_nju_result))
#    #将第list中第一个数值在垂直方向上和别的数值进行合并
#    print(tem_nju_output)
#nju_output_result_final.append(tem_nju_output)
#print(nju_output_result_final)

   
    



my_neural_network_ultimate_output_result = my_neural_network_output_result_final 
#nju_ultimate_output_result = nju_output_result_final


print(my_neural_network_ultimate_output_result)
#print(nju_ultimate_output_result)



def store_exp_result(filepath,filename,exp_result,):                                 
    file_path_tem = filepath
    my_file_name = filename
    my_path_file_name = file_path_tem + my_file_name
    my_path_file_name_final = my_path_file_name + '.xls'
    ultimate_output_result = exp_result
    workbook=xlwt.Workbook(encoding='utf-8')  
    booksheet=workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)  
    #创建sheet
    
    title_row0 = [u'nth_rows_num	',u'total_lines',u'percent',
                  u'twe_per_nonzero_lines',u'total_nonzero_lines',u'ACC']
    #生成第一行
    for i in range(0,len(title_row0)):
        booksheet.write(0,i,title_row0[i])
        #booksheet.write(0,i,title_row0[i],set_style('Times New Roman',220,True))
    
    #将数据写进csv文件并保存
    i = 1
    j = 0
    for ultimate_output_result_temp in ultimate_output_result:
        for i,row in enumerate(ultimate_output_result_temp):  
            for j,col in enumerate(row):  
                booksheet.write(i,j,col)  
    workbook.save(my_path_file_name_final)

store_path = 'M:/paper/experiment_result_defect_probability_divide_sum_la_ld/recall_ten_metrics_nn_predect_bug_probolity_diveded_sum_la_ld_one/' 
project_name = 'mozilla_recall_10_20_10_1_150_test_three_layers_bug_probo_divide_sum_la_ld_nn'   
store_exp_result(store_path,project_name,my_neural_network_ultimate_output_result)










"""以下代码是2017-07-30添加的，主要用来保存我得预测模型的中间结果，来计算popt指标"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""#########################################"""
"""1.将十折的原始的数据集保存到十个csv文件中"""
"""2.将十折的数据集按照bug/la+ld+1进行降序排序之后的结果保存到十个xls文件中"""

####原始数据集没有经过任何操作的文件路径
test_data_density_file_path =  'J:/genetic_algorithm/data_processing/my_nn_model_original_test_data_density/'
project_name_string_list = ['bugzilla','columba','jdt','platform','mozilla','postgres']
project_test_middle_name = "_test_dataset_nn_predict_density_"
nth_fold_name_string_list = ['first_fold_1','second_fold_2','third_fold_3','fourth_fold_4','fifth_fold_5','sixth_fold_6','seventh_fold_7','eighth_fold_8', 'ninth_fold_9','tenth_fold_10']
file_suffix = '.xls'
file_suffix2 = '.csv'

####数据集按照预测出来的bug/la+ld+1的数值的降序排序之后的文件路径   
test_data_density_decent_orderby_density_file_path =  'K:/prediction_results/my_nn_test_dataset_density_decent_orderby_predict_density/'
project_test_data_decent_orderby_density_middle_name = "_test_dataset_nn_density_decent_orderby_density_"



"""文件路径的拼接,list的长度为6"""
""""拼接测试集原始的数据集"""
"""拼接成M:/genetic_algorithm/data_processing/my_nn_model_original_test_data_density/bugzilla/的形成"""
all_project_file_path_list = []
for i in range(len(project_name_string_list)):   
    current_project_name = project_name_string_list[i]
    #取得list中第一列的数值
    file_path_name_temp = test_data_density_file_path + current_project_name + '/'
    all_project_file_path_list.append(file_path_name_temp)
print(all_project_file_path_list)



"""文件路径的拼接,list的长度为6"""  
""""拼接测试集和最后一列的圣经网络输出的density，按照输出的density降序排序之后的测试集"""
"""拼接成K:/my_nn_test_data_nn_predict_decent_orderby_predect_density/bugzilla的形成"""
all_project_file_decent_orderby_density_path_list = []
for i in range(len(project_name_string_list)):   
    current_project_name = project_name_string_list[i]
    #取得list中第一列的数值
    file_path_name_temp = test_data_density_decent_orderby_density_file_path + current_project_name + '/'
    all_project_file_decent_orderby_density_path_list.append(file_path_name_temp)
print(all_project_file_decent_orderby_density_path_list)





"""文件名的拼接，将工程名字和中间的字符串进行拼接，拼接成bugzilla_test_dataset_nnpredict_density_的形式"""
"""文件名的拼接，将工程名字和中间的字符串进行拼接，拼接成bugzilla_test_dataset_density_decent_orderby_density_的形式"""
all_project_test_file_name_middle = []
all_project_decent_order_test_file_name_middle = []
for i in range(len(project_name_string_list)):   
    current_project_file_name = project_name_string_list[i]
    #取得list中第一列的数值
    
    test_file_name_temp = current_project_file_name + project_test_middle_name
    decent_order_test_file_name_temp = current_project_file_name + project_test_data_decent_orderby_density_middle_name
    all_project_test_file_name_middle.append(test_file_name_temp)
    all_project_decent_order_test_file_name_middle.append(decent_order_test_file_name_temp)
print(all_project_test_file_name_middle)
print(all_project_decent_order_test_file_name_middle)


"""文件名的拼接，拼接成bugzilla_test_dataset_nnpredict_density_first_fold_1的形式"""
all_projects_test_date_nn_predict_density_filenames = []
for i in range(len(all_project_test_file_name_middle)):
    for j in range(len(nth_fold_name_string_list)):   
        current_test_file_name_middle = all_project_test_file_name_middle[i]
        current_fold_name = nth_fold_name_string_list[j]
        ultimate_test_date_nn_predect_density_filenames = current_test_file_name_middle + current_fold_name
        all_projects_test_date_nn_predict_density_filenames.append(ultimate_test_date_nn_predect_density_filenames)
print(all_projects_test_date_nn_predict_density_filenames)




"""文件名的拼接，拼接成bugzilla_test_dataset_density_decent_orderby_density_first_fold_1的形式"""
all_projects_decent_orderby_density_test_density_filenames = []
for i in range(len(all_project_decent_order_test_file_name_middle)):
    for j in range(len(nth_fold_name_string_list)):   
        current_decent_test_file_name_middle = all_project_decent_order_test_file_name_middle[i]
        current_decent_fold_name = nth_fold_name_string_list[j]
        ultimate_decent_test_date_nn_predict_density_filenames = current_decent_test_file_name_middle + current_decent_fold_name
        all_projects_decent_orderby_density_test_density_filenames.append(ultimate_decent_test_date_nn_predict_density_filenames)
print(all_projects_decent_orderby_density_test_density_filenames)






"""文件名的拼接，使文件都带有后缀，拼接成bugzilla_test_dataset_nn_predict_density_first_fold_1.xls带有后缀的形式"""
all_projects_test_date_nn_predict_density_filenames_suffix = []
for i in range(len(all_projects_test_date_nn_predict_density_filenames)):   
    current_test_date_nn_predect_density_filenames_suffix = all_projects_test_date_nn_predict_density_filenames[i] + file_suffix
    #取得当前的文件名
    all_projects_test_date_nn_predict_density_filenames_suffix.append(current_test_date_nn_predect_density_filenames_suffix)
print(all_projects_test_date_nn_predict_density_filenames_suffix)



"""文件名的拼接，使文件都带有后缀，拼接成bugzilla_test_dataset_density_decent_orderby_density_first_fold_1.xls带有后缀的形式"""
all_projects_decent_orderby_density_test_density_filenames_suffix = []
for i in range(len(all_projects_decent_orderby_density_test_density_filenames)):   
    current_decent_test_date_nn_predect_density_filenames_suffix = all_projects_decent_orderby_density_test_density_filenames[i] + file_suffix
    #取得当前的文件名
    all_projects_decent_orderby_density_test_density_filenames_suffix.append(current_decent_test_date_nn_predect_density_filenames_suffix)
print(all_projects_decent_orderby_density_test_density_filenames_suffix)






"""获取当前想要保存的原始的测试集和预测输出的bug/la+ld(defect density)的路径"""
current_file_save_path_bugzilla = all_project_file_path_list[0]
current_file_save_path_columba = all_project_file_path_list[1]
current_file_save_path_jdt = all_project_file_path_list[2]
current_file_save_path_platform = all_project_file_path_list[3]
current_file_save_path_mozilla = all_project_file_path_list[4]
current_file_save_path_postgres = all_project_file_path_list[5]


"""获取当前想要保存的原始的测试集和density的文件名,每一个都是一个长度为10的list，例如bugzilla_test_dataset_nnpredict_density_first_fold_1"""
current_test_data_density_file_name_bugzilla = all_projects_test_date_nn_predict_density_filenames[0:10]
current_test_data_density_file_name_columba = all_projects_test_date_nn_predict_density_filenames[10:20]
current_test_data_density_file_name_jdt = all_projects_test_date_nn_predict_density_filenames[20:30]
current_test_data_density_file_name_platform = all_projects_test_date_nn_predict_density_filenames[30:40]
current_test_data_density_file_name_mozilla = all_projects_test_date_nn_predict_density_filenames[40:50]
current_test_data_density_file_name_postgres = all_projects_test_date_nn_predict_density_filenames[50:60]



"""获取当前想要保存的原始的测试集和density的带有后缀的文件名,每一个都是一个长度为10的list，例如bugzilla_test_dataset_nnpredict_density_first_fold_1.xls"""
current_test_data_density_file_suffix_name_bugzilla = all_projects_test_date_nn_predict_density_filenames_suffix[0:10]
current_test_data_density_file_suffix_name_columba = all_projects_test_date_nn_predict_density_filenames_suffix[10:20]
current_test_data_density_file_suffix_name_jdt = all_projects_test_date_nn_predict_density_filenames_suffix[20:30]
current_test_data_density_file_suffix_name_platform = all_projects_test_date_nn_predict_density_filenames_suffix[30:40]
current_test_data_density_file_suffix_name_mozilla = all_projects_test_date_nn_predict_density_filenames_suffix[40:50]
current_test_data_density_file_suffix_name_postgres = all_projects_test_date_nn_predict_density_filenames_suffix[50:60]




"""#####################################################################"""
"""以下是获取想要保存的按照降序排序排列的测试集和density保存起来的数据的文件路径和文件名"""
"""以下是获取想要保存的按照降序排序排列的测试集和density保存起来的数据的文件路径和文件名"""
"""以下是获取想要保存的按照降序排序排列的测试集和density保存起来的数据的文件路径和文件名"""

"""获取当前想要保存的原始的测试集和预测输出的bug/la+ld(defect density)的路径"""
current_decent_file_save_path_bugzilla = all_project_file_decent_orderby_density_path_list[0]
current_decent_file_save_path_columba = all_project_file_decent_orderby_density_path_list[1]
current_decent_file_save_path_jdt = all_project_file_decent_orderby_density_path_list[2]
current_decent_file_save_path_platform = all_project_file_decent_orderby_density_path_list[3]
current_decent_file_save_path_mozilla = all_project_file_decent_orderby_density_path_list[4]
current_decent_file_save_path_postgres = all_project_file_decent_orderby_density_path_list[5]


"""获取当前想要保存的原始的测试集和density的文件名,每一个都是一个长度为10的list，例如bugzilla_test_dataset_nnpredict_density_first_fold_1"""
current_decent_test_data_density_file_name_bugzilla = all_projects_decent_orderby_density_test_density_filenames[0:10]
current_decent_test_data_density_file_name_columba = all_projects_decent_orderby_density_test_density_filenames[10:20]
current_decent_test_data_density_file_name_jdt = all_projects_decent_orderby_density_test_density_filenames[20:30]
current_decent_test_data_density_file_name_platform = all_projects_decent_orderby_density_test_density_filenames[30:40]
current_decent_test_data_density_file_name_mozilla = all_projects_decent_orderby_density_test_density_filenames[40:50]
current_decent_test_data_density_file_name_postgres = all_projects_decent_orderby_density_test_density_filenames[50:60]



"""获取当前想要保存的原始的测试集和density的带有后缀的文件名,每一个都是一个长度为10的list，例如bugzilla_test_dataset_nnpredict_density_first_fold_1.xls"""
current_decent_test_data_density_file_suffix_name_bugzilla = all_projects_decent_orderby_density_test_density_filenames_suffix[0:10]
current_decent_test_data_density_file_suffix_name_columba = all_projects_decent_orderby_density_test_density_filenames_suffix[10:20]
current_decent_test_data_density_file_suffix_name_jdt = all_projects_decent_orderby_density_test_density_filenames_suffix[20:30]
current_decent_test_data_density_file_suffix_name_platform = all_projects_decent_orderby_density_test_density_filenames_suffix[30:40]
current_decent_test_data_density_file_suffix_name_mozilla = all_projects_decent_orderby_density_test_density_filenames_suffix[40:50]
current_decent_test_data_density_file_suffix_name_postgres = all_projects_decent_orderby_density_test_density_filenames_suffix[50:60]





"""自定义的保存数据的函数，需要传递三个参数，文件路径，文件名，数据集"""
def store_data(filepath,filename,exp_result): 
    file_path = filepath                               
    my_current_file_name = filename
    my_current_path_file_name = file_path + my_current_file_name
    #my_path_file_name_final = my_path_file_name + '.xls'
    ultimate_output_result = exp_result
    workbook=xlwt.Workbook(encoding='utf-8')  
    booksheet=workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)  
    #创建sheet
    
    title_row0 = [u'nth_rows_num	',u'total_lines',u'percent',
                  u'twe_per_nonzero_lines',u'total_nonzero_lines',u'ACC']
    #生成第一行
    for i in range(0,len(title_row0)):
        booksheet.write(0,i,title_row0[i])
        #booksheet.write(0,i,title_row0[i],set_style('Times New Roman',220,True))
    
    #将数据写进csv文件并保存
    i = 0
    j = 0
    for i,row in enumerate(ultimate_output_result):  
        for j,col in enumerate(row):  
            booksheet.write(i,j,col)  
    workbook.save(my_current_path_file_name)

"""将原始的数据集,testdata和机器学习输出的预测数值density在列的方向上，左右方向上合并之后，保存分别保存到10个xls文件中"""
"""保存我的神经网络的预测模型的测试集和机器学习的输出结果bug/la+ld+1(也就是defect sensity在左右方向进行合并的)果的原始数据集，没有进行任何排序"""

for i in range(len(my_nn_model_original_test_dataset_predict_density_global)):   
    #tem_my_neural_row = my_neural_network_exp_result_global[i]
    #取得list中第一列的数值
    current_file_path = current_file_save_path_mozilla
    current_file_name = current_test_data_density_file_suffix_name_mozilla[i]
                     
    current_my_original_testdata_nn_predict_density = my_nn_model_original_test_dataset_predict_density_global[i]
    #current_my_original_testdata_nn_predict_density = np.array(my_nn_model_original_test_dataset_predict_density_global[i])
    store_data(current_file_path,current_file_name,current_my_original_testdata_nn_predict_density)


"""将原始的数据集,testdata和机器学习输出的预测数值density在列的方向上，左右方向上合并之后，保存分别保存到10个xls文件中"""
"""保存我的神经网络的预测模型中的测试集和机器学习输出的defect density（bug/la+ld+1）合并之后的数据集按照bug/la+ld+1降序排序之后的数据集,
按照bug/la+ld+1，也就是最后一列进行了降序排序之后的数值
"""
 
for i in range(len(my_nn_model_test_dataset_density_decent_orderby_predict_density_global)):   
    #tem_my_neural_row = my_neural_network_exp_result_global[i]
    #取得list中第一列的数值
    current_decent_file_path = current_decent_file_save_path_mozilla
    current_decent_file_name = current_decent_test_data_density_file_suffix_name_mozilla[i]
    current_my_decent_testdata_nn_predict_density = my_nn_model_test_dataset_density_decent_orderby_predict_density_global[i]                   
    #current_my_decent_testdata_nn_predict_density = np.array(my_nn_model_test_dataset_density_decent_orderby_predict_density_global[i])
    store_data(current_decent_file_path,current_decent_file_name,current_my_decent_testdata_nn_predict_density)

















