from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.callbacks import EarlyStopping
from keras.models import load_model
import numpy as np
import gensim

features_num = 500
vec_size = 600

def doc_features_generation():
    print("doc features start......")
    model = gensim.models.Doc2Vec.load('model_v600_w15.model')
    empty_word = np.zeros(model.vector_size, dtype='float16')
    content_list = np.load('testing2_content_list.npy', allow_pickle=True)
    word_set = model.wv.vocab.keys()

    doc_features = np.zeros((len(content_list), 500, 600), dtype='float16')

    for i in range(len(content_list)):
        doc_len = len(content_list[i])
        for j in range(features_num):
            if(j < doc_len):
                word = content_list[i][j]
                if(word in word_set):
                    word_vec = model.wv[word]
                    doc_features[i, j] = word_vec
            else:
                doc_features[i, j] = empty_word

    print("doc features finish......")
    #doc_features = np.array(doc_features)
    np.save('testing2_doc_features', doc_features)
            
    
def lstm_model():
    # model
    model = Sequential()
    # add conv1D layer
    model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                     activation='relu', input_shape=(features_num, vec_size)))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))

    for i in range(3):
    # add conv1D layer
        model.add(Conv1D(filters=32, kernel_size=3,
                        padding='same', activation='relu'))
        model.add(Dropout(0.2))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Dropout(0.2))
    
    # add LSTM layer
    model.add(LSTM(600, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam')

    return model

def training():
    doc_features_generation()
    model = lstm_model()
    train_data = np.load('training_doc_features.npy')
    train_label = np.load('s_tag_list.npy', allow_pickle=True)

    print("model training start......")
    model.fit(train_data, train_label, validation_split=0.2, epochs=10, batch_size=64)
    model.save("lstm_700_600.h5")


def testing():
    model = load_model('lstm_700_600.h5')
    print("testing start......")
    test_data = np.load('testing2_doc_features.npy')
    test_res = model.predict(test_data)
    test_res = np.array(test_res)
    
    np.save('lstm_testing2_result', test_res)

