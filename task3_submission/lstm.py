from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.callbacks import EarlyStopping
from keras.models import load_model
import numpy as np
import gensim
from result_handling import *

features_num = 600
vec_size = 600

def doc_features_generation(pos_file, neg_file='', model_type=''):
    print(model_type + " doc features start......")
    print(pos_file) 
    print(neg_file)

    model_loc = 'models/' + model_type + '_model.model'

    model = gensim.models.Doc2Vec.load(model_loc)
    empty_word = np.zeros(model.vector_size, dtype='float16')

    content_list = []
    pos_content_list = list(np.load(pos_file, allow_pickle=True))
    content_list = pos_content_list

    if(neg_file != ''):
        neg_content_list = list(np.load(neg_file, allow_pickle = True))
        content_list = content_list + neg_content_list

    word_set = model.wv.vocab.keys()
    doc_features = np.zeros((len(content_list), 600, 600), dtype='float16')

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

    print(model_type + " doc features finish......")
    #doc_features = np.array(doc_features)
    if(neg_file != ''):
        np.save('train_vec_files/' + model_type + '_train_vecs', doc_features)
    else:
        np.save('test_vec_files/' + 'test_vecs', doc_features)
            
    
def lstm_model():
    # model
    model = Sequential()
    # add conv1D layer
    model.add(Conv1D(filters=32, kernel_size=5, padding='same', dilation_rate=2,
                     activation='relu', input_shape=(features_num, vec_size)))
    #model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))

    for i in range(3):
    # add conv1D layer
        model.add(Conv1D(filters=32, kernel_size=5, dilation_rate=2,
                        padding='same', activation='relu'))
    
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2)) 

    for i in range(4):
        model.add(Conv1D(filters=64, kernel_size=5,
                        padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))

    for i in range(4):
        model.add(Conv1D(filters=128, kernel_size=3,
                        padding='same', activation='relu'))
    
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))
    
    for i in range(4):
        model.add(Conv1D(filters=128, kernel_size=3,
                        padding='same', activation='relu'))

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

def training(vec_files, tag_files, model_type): 
    #doc_features_generation()
    model = lstm_model()
    print("Model loaded......")
    assert(2 * len(vec_files) == len(tag_files))

    train_data = []
    train_label = []
    
    for i in range(len(vec_files)):
        train_data = train_data + list(np.load(vec_files[i], allow_pickle=True))

    for i in range(len(tag_files)):
        train_label = train_label + list(np.load(tag_files[i], allow_pickle=True))

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    print("Data loaded......")
    assert(len(train_data) == len(train_label))

    print(model_type + " model training start......")
    model.fit(train_data, train_label, validation_split=0.2, epochs=10, batch_size=64, \
            verbose=1, shuffle=True)
    print(model_type + " model training finished......")
    model.save('cnn_models/' + model_type + '_model.h5')


def testing(test_tag_loc, test_vec_file, model_type):
    model = load_model('cnn_models/' + model_type + '_model.h5')
    print(model_type + " testing start......")
    test_data = np.load(test_vec_file)
    test_res = model.predict(test_data)
    test_res = np.array(test_res)
    handling(test_tag_loc, test_res, model_type)

    #np.save('testing_res/' + model_type + '_result', test_res)


test_vec_file = 'test_vec_files/test_vecs.npy'
test_files = 'test_files/test_content_list.npy'
test_names = 'test_files/test_name_list.npy'
model_type = ['original', 'original_modified', 'original_extra']

#content file
ori_train_pos = 'train_files/original_pos_content_list.npy'
ori_train_neg = 'train_files/original_neg_content_list.npy'
mod_train_pos = 'train_files/modified_pos_content_list.npy'
mod_train_neg = 'train_files/modified_neg_content_list.npy'
ext_train_pos = 'train_files/extra_pos_content_list.npy'
ext_train_neg = 'train_files/extra_neg_content_list.npy'

#name file
ori_train_pos_name = 'train_files/original_pos_name_list.npy'
ori_train_neg_name = 'train_files/original_neg_name_list.npy'
mod_train_pos_name = 'train_files/modified_pos_name_list.npy'
mod_train_neg_name = 'train_files/modified_neg_name_list.npy'
ext_train_pos_name = 'train_files/extra_pos_name_list.npy'
ext_train_neg_name = 'train_files/extra_neg_name_list.npy'

#sentiment(tag) file
ori_train_pos_tag = 'train_files/original_pos_s_tag_list.npy'
ori_train_neg_tag = 'train_files/original_neg_s_tag_list.npy'
mod_train_pos_tag = 'train_files/modified_pos_s_tag_list.npy'
mod_train_neg_tag = 'train_files/modified_neg_s_tag_list.npy'
ext_train_pos_tag = 'train_files/extra_pos_s_tag_list.npy'
ext_train_neg_tag = 'train_files/extra_neg_s_tag_list.npy' 


'''
format:
    content loc list
    name loc list
    tag loc list
'''
ori_files = [ori_train_pos, ori_train_neg] 
ori_names = [ori_train_pos_name, ori_train_neg_name]
ori_tags = [ori_train_pos_tag, ori_train_neg_tag]
ori_vecs = ['train_vec_files/original_train_vecs.npy']


mod_files = [mod_train_pos, mod_train_neg]
mod_names = [mod_train_pos_name, mod_train_neg_name]
mod_tags = [mod_train_pos_tag, mod_train_neg_tag]
mod_vecs = ['train_vec_files/modified_train_vecs.npy']

ext_files = [ext_train_pos, ext_train_neg]
ext_names = [ext_train_pos_name, ext_train_neg_name]
ext_tags = [ext_train_pos_tag, ext_train_neg_tag]
ext_vecs = ['train_vec_files/extra_train_vecs.npy']


#original modified
ori_mod_files = ori_files + mod_files
ori_mod_names = ori_names + mod_names
ori_mod_tags = ori_tags + mod_tags
ori_mod_vecs = ori_vecs + mod_vecs

#original extra
ori_ext_files = ori_files + ext_files
ori_ext_names = ori_names + ext_names
ori_ext_tags = ori_tags + ext_tags
ori_ext_vecs = ori_vecs + ext_vecs


#test3
#doc_features_generation(ori_pos_files, ori_neg_files, model_type[0])
#doc_features_generation(test_files, '', model_type[0])
#training(ori_vecs, ori_tags, model_type[0])
testing(test_names, test_vec_file, model_type[0])

