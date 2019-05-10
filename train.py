import keras
import tensorflow as tf


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Input,Dense,Bidirectional,LSTM,Embedding,Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D,ZeroPadding2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard  
#from keras import backend as K
import numpy as np
import pandas as pd



df = pd.read_csv('train.csv')
x=df['x']
y = df['y']

mark_start = 'ssss '
mark_end = ' eeee'
def mark_captions(captions_list):
    captions_marked = [mark_start + caption + mark_end
                        for caption in captions_list]
                        
    
    return captions_marked

y = mark_captions(y)


data_text = x+y
num_words = 10000
tokenizer = Tokenizer(num_words = num_words)
tokenizer.fit_on_texts(data_text)


x_tokens = tokenizer.texts_to_sequences(x)
y_tokens = tokenizer.texts_to_sequences(y)


lens = [len(each) for each in x_tokens+y_tokens]
lens = np.array(lens)

max_tokens = np.mean(lens) + 2*np.std(lens)
max_tokens = int(max_tokens)

pad = 'pre'
pad = 'pre'
x_pad = pad_sequences(x_tokens,maxlen = max_tokens,padding = pad,truncating = pad)
pad = 'post'
y_pad = pad_sequences(y_tokens,maxlen = max_tokens,padding = pad,truncating = pad)

encoder_in_data = x_pad
decoder_in_data = y_pad
decoder_output_data = y_pad[:,1:]
decoder_out_data = []

for each in decoder_output_data:
	each = np.insert(each,len(each),0)
	decoder_out_data.append(each)
decoder_out_data = np.array(decoder_out_data)


x_data = \
{
    'encoder_input': encoder_in_data,
    'decoder_input': decoder_in_data
}

y_data = \
{
    'decoder_output': decoder_out_data
}



num_words = 10000
state_size = 512
embedding_size = 128
encoder_input = Input(shape=(None, ), name='encoder_input')
encoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='encoder_embedding')(encoder_input)
encoder_lstm1 = Bidirectional(LSTM(state_size, name='encoder_1',
                   return_sequences=True))(encoder_embedding)
encoder_lstm2 = Bidirectional(LSTM(state_size, name='encoder_2',
                   return_sequences=True))(encoder_lstm1)
encoder_output = LSTM(state_size, name='encoder_3',
                   return_sequences=False)(encoder_lstm2)


decoder_initial_state = Input(shape=(state_size,),
                              name='decoder_initial_state')
decoder_input = Input(shape=(None, ), name='decoder_input')
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')(decoder_input)
decoder_lstm1 = LSTM(state_size, name='decoder_1',
                   return_sequences=True)(decoder_embedding,initial_state = [encoder_output,encoder_output])
decoder_lstm2 = LSTM(state_size, name='decoder_2',
                   return_sequences=True)(decoder_lstm1,initial_state = [encoder_output,encoder_output])
decoder_lstm3 = LSTM(state_size, name='decoder_3',
                   return_sequences=True)(decoder_lstm2,initial_state = [encoder_output,encoder_output])
decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')(decoder_lstm3)
decoder_output = decoder_dense
chat_model = Model(inputs=[encoder_input, decoder_input],
                    outputs=[decoder_output])

print(chat_model.summary())

def sparse_cross_entropy(y_true, y_pred):
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

   
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

optimizer = RMSprop(lr=1e-3)
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
chat_model.compile(optimizer=optimizer,
                    loss=sparse_cross_entropy,
                    target_tensors=[decoder_target])

chat_model.fit(x=x_data,
                y=y_data,
                batch_size=256,
                epochs=10)

model.save_weights('chat_model.h5')
