import numpy as np
import mdat
import pandas as pd
import glob
from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from keras.utils import plot_model
import tensorflow as tf
from keras.initializers import he_normal
import datetime as dt

def huber_loss(y_true,y_pred,clip_delta=3):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta
    squared_loss = 0.5 * K.square(error)
    linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)
#0.105:町田
def custom_loss(y_true,y_pred,alpha=0.1):
    error = y_true - y_pred
    cost=huber_loss(y_true,y_pred)
    condition = K.less(error,0)
    overestimation_loss = alpha * cost 
    underestimation_loss = (1-alpha) * cost 
    return tf.where(condition, overestimation_loss, underestimation_loss)


def make_model():
    model = Sequential()
    model.add(Dense(40,activation="relu",kernel_initializer=he_normal(),input_shape=(79,)))
    model.add(Dropout(0.3))
    model.add(Dense(20,activation="relu",kernel_initializer=he_normal()))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    model.compile(optimizer='adam', 
                loss=custom_loss,
                metrics=['custom_loss'])
    plot_model(
        model,
        show_shapes=True,
        to_file=newpath+'model.png'
    )
    return model

if __name__=="__main__":
	lt=3
	newpath="/Users/e175764/Desktop/Okazaki-lab/week33/result2_10min/"
	path="/Users/e175764/Desktop/Okazaki-lab/data/"
	riv_datas=mdat.make_riv_data(path,["観測所","因原"])
	rain_datas=mdat.make_rain_data(path,["観測所","中野","下田所＿砂"])
	test_x,test_y,train_x,train_y,height = mdat.make_input_data(riv_datas,rain_datas,"因原",3,dt.datetime(2018,7,5),dt.datetime(2018,7,12))

	test_x=np.array(test_x).astype(float)
	test_y=np.array(test_y).astype(float)
	train_x=np.array(train_x).astype(float)
	train_y=np.array(train_y).astype(float)

	model=make_model()

	model.fit(train_x,train_y,epochs=10,verbose=1,batch_size=None)
	model.save(newpath+"test.h5")
	result=model.predict(test_x)

	pd.DataFrame(test_x).to_csv(newpath+"test_x.csv")
	df_now =pd.DataFrame(data=height,dtype=float)
	df_now.columns=["now"]
	df_now.to_csv(newpath+"water_level.csv")
	df_real=pd.DataFrame(data=test_y)
	df=pd.concat([df_real,pd.DataFrame(result)],axis=1)
	df.columns=["real","pred"]
	df.to_csv(newpath+"real_pred.csv")