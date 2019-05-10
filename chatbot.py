from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Input,Dense,Bidirectional,LSTM,Embedding,Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D,ZeroPadding2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard  
from keras import backend as K
import numpy as np
import pandas as pd


df = pd.read_csv('train.csv')
x=df['x']
y = df['y']
max_tokens = 33
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
#tokenizer.word_index


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



chat_model.load_weights("chat_model.h5")
test = ["You have my word.  As a gentleman"]
test = np.array(test)

test_tokens = tokenizer.texts_to_sequences(test)
pad = 'pre'
test_pad = pad_sequences(test_tokens,maxlen = max_tokens,padding = pad,truncating = pad)
test_d = [[0 for i in range(0,33)]]
test_d = np.array(test_d)

test_data = \
{
    'encoder_input': test_pad,
    'decoder_input': test_d
}
out = chat_model.predict(test_data)
output = []
count_tokens =0
while count_tokens<max_tokens:
    token_onehot = out[0, count_tokens, :]
    token_int = np.argmax(token_onehot)
    output.append(token_int)
    count_tokens+=1

for each in output:
    for word,index in tokenizer.word_index.items():
        if(index == each) :
            print(word)


