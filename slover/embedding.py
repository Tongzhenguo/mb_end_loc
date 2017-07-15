from datetime import timedelta
import math
import keras
from keras import Input
from keras import losses
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Embedding, Flatten, merge, Dense
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle as pickle


BATCH_SIZE = 256

#产生每个batch的上下分界
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

#构造一个batch生成器
def batch_generator(train,y,batch_size=128,out_dim =2857, shuffle=True,isTrain=True):
    '''
    :param train:原始训练集，假定1,2,3,4列分别是bikeid,userid,start_loc,starttime
    :param y:聚类后得到的4000+的label
    :param batch_size:
    :param shuffle:
    :return:
    '''
    sample_size = train.shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for _, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch_bid = train[batch_ids,0]
            X_batch_uid = train[batch_ids,1]
            X_batch_start_loc = train[batch_ids, 2]
            X_starttime = train[batch_ids, 3]
            X_batch = [X_batch_bid,X_batch_uid,X_batch_start_loc,X_starttime]
            if isTrain:
                y_batch = y[batch_ids]
                y_binary = to_categorical(y_batch,out_dim)
                yield X_batch,y_binary
            else:
                yield X_batch

def embed_model( x1_dim,x2_dim,x3_dim,x4_dim,out_dim ):
    # define model
    bid = Input(shape=(1,), dtype='int64')
    uid = Input(shape=(1,), dtype='int64')
    start_loc = Input(shape=(1,), dtype='int64')
    starttime = Input(shape=(1,), dtype='int64')

    x1 = Embedding(output_dim=int(math.log(x1_dim,2))+1, input_dim=x1_dim, input_length=1)(bid)
    x2 = Embedding(output_dim=int(math.log(x2_dim,2))+1, input_dim=x2_dim, input_length=1)(uid)
    x3 = Embedding(output_dim=int(math.log(x3_dim,2))+1, input_dim=x3_dim, input_length=1)(start_loc)
    x4 = Embedding(output_dim=int(math.log(x4_dim,2))+1, input_dim=x4_dim, input_length=1)(starttime)
    x1 = Flatten()(x1)
    x2 = Flatten()(x2)
    x3 = Flatten()(x3)
    x4 = Flatten()(x4)
    x = merge([x1, x2,x3,x4], mode='concat')
    x = Dense(500, activation='relu', )(x)
    out = Dense(out_dim, activation='softmax',)(x)
    model = Model(input=[bid, uid,start_loc,starttime], output=out)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss=losses.categorical_crossentropy,
                  metrics=['accuracy'])

    model.summary()
    return model

def model_eval():
    print('-----------------data processing -------------------------------------')
    tr = pd.read_pickle('../data/tr_feat.pkl')

    user_le = LabelEncoder()
    tr['userid'] = user_le.transform(tr['userid'])

    print('userid uniq len', len(user_le.classes_))

    bike_le = LabelEncoder()
    tr['bikeid'] = bike_le.transform(tr['bikeid'])
    print('bikeid uniq len', len(bike_le.classes_))

    loc_le = LabelEncoder()
    tr['geohashed_start_loc'] = loc_le.transform(tr['geohashed_start_loc'].values)
    print('loc_start uniq len', len(loc_le.classes_))

    tr['starttime'] = tr[['weekday', 'hour', 'quarter']].apply(lambda dt: dt[0] + '-' + dt[1] + '-' + dt[2])
    time_le = LabelEncoder()
    tr['starttime'] = time_le.transform(tr['starttime'].values)
    print('starttime uniq len', len(time_le.classes_))

    y = pd.read_pickle('../data/label.pkl').values
    len_class = 2857 ##聚类中心个数
    tr_x = tr[['bikeid', 'userid', 'geohashed_start_loc', 'starttime']].values
    tr_gen = batch_generator(tr_x, y, BATCH_SIZE)

    te_x = te[['bikeid', 'userid', 'geohashed_start_loc', 'starttime']].values
    te_gen = batch_generator(te_x, y=None, batch_size=BATCH_SIZE, shuffle=False, isTrain=False)

    print('---------------------------------train a model -----------------------------------------')
    model = embed_model(len(bike_le.classes_) + 1, len(user_le.classes_) + 1, len(loc_le.classes_) + 1,
                        len(time_le.classes_) + 1, len_class)
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, min_delta=0.0005, mode='min')  # 最少下降0.0005才算一次提升
    model_checkpoint = ModelCheckpoint('../model/embed_v1.hdf5', save_best_only=False, save_weights_only=True,
                                       mode='min')

    model.fit_generator( tr_gen,
                         steps_per_epoch=int(tr_x.shape[0] / BATCH_SIZE) + 1,
                         epochs=1,
                         verbose=1,
                         validation_data=te_gen,
                         validation_steps=int(te_x.shape[0] / BATCH_SIZE) + 1,
                         max_q_size=10,
                         callbacks=[early_stopping, model_checkpoint]
                         )  #loss: 3.9545 - acc: 0.2087 - val_loss: 2.6289 - val_acc: 0.3047
                            #loss: 3.6766 - acc: 0.2337 - val_loss: 2.4322 - val_acc: 0.3255
                            #loss: 4.6960 - acc: 0.1508 - val_loss: 2.8093 - val_acc: 0.2880


#提交结果
def make_sub():
    print('-----------------data processing -------------------------------------')
    tr = pd.read_pickle('../data/tr_feat.pkl')
    te = pd.read_pickle('../data/te_feat.pkl')
    y = pd.read_pickle('../data/label.pkl').values
    model_path = '../model/embed_v1.hdf5'
    res_path = '../res/res.csv'
    cluter_3_hot = pd.read_csv('../data/cluter_3_hot.csv')
    len_class = 2857

    user_le = LabelEncoder()
    user_le.fit( list(set(tr['userid'].unique()).union(set(te['userid'].unique()))) )
    tr['userid'] = user_le.transform( tr['userid'] )
    te['userid'] = user_le.transform( te['userid'] )
    print( 'userid uniq len',len(user_le.classes_) )

    bike_le = LabelEncoder()
    bike_le.fit(list(set(tr['bikeid'].unique()).union(set(te['bikeid'].unique()))))
    tr['bikeid'] = bike_le.transform(tr['bikeid'])
    te['bikeid'] = bike_le.transform(te['bikeid'])
    print('bikeid uniq len', len(bike_le.classes_))

    loc_le = LabelEncoder()
    loc_le.fit(list(set(tr['geohashed_start_loc'].unique()).union(set(te['geohashed_start_loc'].unique()))))
    tr['geohashed_start_loc'] = loc_le.transform(tr['geohashed_start_loc'].values)
    te['geohashed_start_loc'] = loc_le.transform(te['geohashed_start_loc'].values)
    print('loc_start uniq len', len(loc_le.classes_))

    fn = lambda dt: str(dt[0]) + '-' + str(dt[1]) + '-' + str(dt[2])
    tr['starttime'] = tr[['weekday','hour','quarter']].apply( fn,raw=True,axis=1 )
    te['starttime'] = te[['weekday', 'hour', 'quarter']].apply( fn,raw=True,axis=1 )
    time_le = LabelEncoder()
    time_le = time_le.fit(list(set(tr['starttime'].unique()).union(set(te['starttime'].unique()))))
    tr['starttime'] = time_le.transform(tr['starttime'].values)
    te['starttime'] = time_le.transform(te['starttime'].values)
    print('starttime uniq len', len(time_le.classes_))


    tr_x = tr[['bikeid', 'userid', 'geohashed_start_loc', 'starttime']].values
    tr_gen = batch_generator(tr_x,y, BATCH_SIZE)
    te_x = te[['bikeid', 'userid', 'geohashed_start_loc', 'starttime']].values
    te_gen = batch_generator(te_x,y=None,batch_size=BATCH_SIZE,shuffle=False,isTrain=False)

    print('---------------------------------train a model -----------------------------------------')
    model = embed_model( len(bike_le.classes_)+1,len(user_le.classes_)+1,len(loc_le.classes_)+1,len(time_le.classes_)+1,len_class )
    # early_stopping = EarlyStopping(monitor='val_loss', patience=1, min_delta=0.0005, mode='min')  # 最少下降0.0005才算一次提升
    # model_checkpoint = ModelCheckpoint('../model/embed_v1.hdf5', save_best_only=False, save_weights_only=True,
    #                                    mode='min')
    # model.fit_generator( tr_gen,
    #                      steps_per_epoch=int(tr_x.shape[0] / BATCH_SIZE) + 1,
    #                      epochs=1,
    #                      verbose=1,
    #                      validation_data=None,
    #                      validation_steps=None,
    #                      max_q_size=10,
    #                      callbacks=[ model_checkpoint]
    #                      )
    model.load_weights(model_path)

    label_list = []
    for i in range(int(te_x.shape[0] / BATCH_SIZE) + 1):
        pred = model.predict_generator( te_gen,steps=1,max_q_size=10 )
        label_list.extend( list(pred.argpartition(-1)[:,-1]) )
    res = pd.DataFrame( )
    res['orderid'] = te['orderid']
    res['label'] = label_list


    pd.merge( res,cluter_3_hot,on='label' )[
        ['orderid','pred1','pred2','pred3']].to_csv(res_path,header=False,index=False)

    r = pd.read_csv('../res/20170714_B.csv', names=['orderid', 'pred1', 'pred2', 'pred3'])
    res = pd.read_csv('../res/res.csv', names=['orderid', 'pred1', 'pred2', 'pred3'])
    a = pd.DataFrame()
    a['orderid'] = list(set(r['orderid'].unique()) - set(res['orderid'].unique()))

    r = pd.merge(r, a, on='orderid')
    res = res.append(r[['orderid', 'pred1', 'pred2', 'pred3']]).drop_duplicates('orderid')
    res.to_csv('../res/res_embed.csv', header=None, index=None)
    print(len(res))

if __name__ == '__main__':
    make_sub()