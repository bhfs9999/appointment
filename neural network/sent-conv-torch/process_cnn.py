# coding:utf-8
import cPickle as pkl
import codecs
from sklearn.model_selection import train_test_split

departlist = list()
with codecs.open("../../../data/department_seg.txt", 'rb') as f:
    for line in f.readlines():
        depart = [i.strip() for i in line.split('：')[1].split('，')]
        departlist += depart

departdict = dict(zip(departlist, range(len(departlist))))
pkl.dump(departdict, open('../../../data/cnn_data/cnn_department.pkl', 'wb'))

data = list()
with codecs.open("../../../data/search_appointments_split_raw.txt", 'rb', 'utf-8') as f:
    for line in f.readlines():
        tline = line.split("\t")
        depart = tline[0].strip().encode('utf-8')

        if not departdict.has_key(depart):
            continue

        depart_index = departdict[depart]
        des = tline[1].strip()
        data.append((depart_index, des))

# 保存转换后的数据
with codecs.open("../../../data/cnn_data/data_for_cnn.txt", 'wb', 'utf-8') as f:
    for depart, des in data:
        aline = "{}\t{}\n".format(depart, des.encode('utf-8')).decode('utf-8')
        f.write(aline)

split_rate = 0.2

y = [i[0] for i in data]
x = [i[1] for i in data]

# 分割训练集测试集
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = split_rate)

# 保存训练集测试集
with codecs.open("../../../data/cnn_data/data_for_cnn_train.txt", 'wb', 'utf-8') as f:
    for i in range(len(xtrain)):
        aline = "{}\t{}\n".format(ytrain[i], xtrain[i].encode('utf-8')).decode('utf-8')
        f.write(aline)

with codecs.open("../../../data/cnn_data/data_for_cnn_test.txt", 'wb', 'utf-8') as f:
    for i in range(len(xtest)):
        aline = "{}\t{}\n".format(ytest[i], xtest[i].encode('utf-8')).decode('utf-8')
        f.write(aline)