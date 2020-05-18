#%%
import pandas as pd
import numpy as np
import os

path = r'D:\code\ml\ml2020spring-hw1'
data = pd.read_csv(os.path.join(path,'train.csv'),encoding='big5')

# pre
#%%
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# extract feature
# %%
# 分割到每个月
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# %%
# x为每前9小时的pm2.5 data，y是第10小时的pm2.5 data
# 每个月有480个小时，以每10个分割（前9个x，第10个y），可以分隔471份 data出来
def aggrate(month_data,hours = 9):
    dataSize = 480 - hours
    x = np.empty([12 * dataSize, 18 * hours], dtype = float)
    y = np.empty([12 * dataSize, 1], dtype = float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 23-hours:
                    continue
                x[month * dataSize + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + hours].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                y[month * dataSize + day * 24 + hour, 0] = month_data[month][hours, day * 24 + hour + hours] #value
    print(x)
    print(y)

    # 规范化
    mean_x = np.mean(x, axis = 0) #18 * 9 
    std_x = np.std(x, axis = 0) #18 * 9 
    for i in range(len(x)): #12 * 471
        for j in range(len(x[0])): #18 * 9 
            if std_x[j] != 0:
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    
    import math
    x_train_set = x[: math.floor(len(x) * 0.8), :]
    y_train_set = y[: math.floor(len(y) * 0.8), :]
    x_validation = x[math.floor(len(x) * 0.8): , :]
    y_validation = y[math.floor(len(y) * 0.8): , :]
    # print(x_train_set)
    # print(y_train_set)
    # print(x_validation)
    # print(y_validation)
    # print(len(x_train_set))
    # print(len(y_train_set))
    # print(len(x_validation))
    # print(len(y_validation))

    return x,y,x_validation,y_validation,std_x,mean_x




# %%
# 採用 Root Mean Square Error  均方根差
# eps 項是避免 adagrad 的分母為 0 而加的極小數值。
#藉由調整 learning rate、iter_time (iteration 次數)、取用 features 的多寡(取幾個小時，取哪些特徵欄位)，甚至是不同的 model 來超越 baseline。
#因為常數項的存在，所以 dimension (dim) 需要多加一欄
#%%
def train(x,y,f_count,ff_count,iter_time,learning_rate):
    """
    x:dim 18*9+1
    f_count:截取的特征量
    ff_count:需要几个特征量
    """
    dim = ff_count * f_count + 1 #163
    w = np.zeros([dim, 1])
    # 12个月 每个月有471个data
    size = x.shape[0]  
    # 截取特征量
    x = x[:,-f_count*18:]  
    if ff_count==1:
        x = x[:,10::18]  

    x = np.concatenate((np.ones([size, 1]), x), axis = 1).astype(float)
    
    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001
    losses = []
    for t in range(iter_time):        
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/size)#rmse
        losses.append(loss)
        if(t%100==0):
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps) 
    np.save('weight.npy', w)
    return losses

# iters = [1000,1500,2000,2500]
## 作业1题：画图
# learning_rate = [1,0.5,0.2,0.1]
# tables ={}
# for iter_num in learning_rate:
#     losses = train(x,25,iter_num)
#     tables[iter_num] = losses 
    
# df = pd.DataFrame(tables)
# df.plot()



def predict_loss(x,y,f_count,ff_count):
    w = np.load('weight.npy')   
    # 截取特征量
    x = x[:,-f_count*18:]  
    if ff_count==1:
        x = x[:,10::18]
    x = np.concatenate((np.ones([x.shape[0], 1]), x), axis = 1).astype(float)  
    vali_y = np.dot(x,w)
    loss = np.sqrt(np.sum(np.power(vali_y - y, 2))/x.shape[0])#rmse
    print('predict loss',loss)

#%%
# x_train_set,y_train_set,x_validation,y_validation = aggrate(month_data,9)
# r = train(x_train_set,y_train_set,9,18,10000,0.000005)
# print('train:',r[-1])
# predict_loss(x_validation,y_validation,9,18)
# 前5小时
# x_train_set,y_train_set,x_validation,y_validation = aggrate(month_data,5)
# r = train(x_train_set,y_train_set,5,18,1000,100)
# print('train:',r[-1])
# predict_loss(x_validation,y_validation,5,18)

#只取pm2.5
# r = train(x_train_set,y_train_set,9,1,1000,100)
# print('train:',r[-1])
# predict_loss(x_validation,y_validation,9,1)
#%%
# 尝试超越baseline
x_train_set,y_train_set,x_validation,y_validation,std_x ,mean_x= aggrate(month_data,9)
i=6
print('first hours',i)
r = train(x_train_set,y_train_set,i,18,5000,100)
print('train:',r[-1])
predict_loss(x_validation,y_validation,i,18)

# %%
# 加载test数据
# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')

testdata = pd.read_csv(os.path.join(path,'test.csv'), header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]

test_x
#%%
w = np.load('weight.npy')
test_x = test_x[:,-6*18:]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
ans_y = np.dot(test_x, w)
ans_y
# %%
import csv
with open('submit2.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
# %%
