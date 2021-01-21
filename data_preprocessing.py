import nltk
import glob
from nltk import tokenize
import gensim
import re
import numpy as np
from collections import OrderedDict
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import random

def read_data(files_loc, s_type='positive'):
    content_list = []
    tag_list = []
    s_tag_list = []
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    files_loc = files_loc + "*.txt"
    stemmer = SnowballStemmer("english")

    for f in glob.glob(files_loc):
        tag_list.append(f)
        s_tag_list.append(s_type)    

        with open(f, 'r', encoding='utf-8') as text:
            raw = text.read()
            content = re.sub(r'\d+', '', raw)
            content = tokenizer.tokenize(content)
            content = list(map(lambda x: x.lower(), content))
            content = list(map(lambda x: stemmer.stem(x), content))
            content_list.append(content)


    return content_list, tag_list, s_tag_list

def word_count(content_list):
    print("word count start.....\n")
    word_count_dict = OrderedDict()
    
    for content in content_list:
        for word in content:
            if word in word_count_dict:
                word_count_dict[word] += 1
            else:
                word_count_dict[word] = 1

    word_count_dict = OrderedDict(sorted(word_count_dict.items(), key=lambda x: -x[1]))
    
    # Generate a word key list, the order of keys is based on the frequency of words from most to least
    word_key_list = list(word_count_dict.keys()) 

    return word_count_dict, word_key_list


# This is NOT doc2vec
def docvec_generate(content_list, word_key_list):
    print("docvec generation start......\n")
    docvec_list = []

    for content in content_list:
        docvec = []
        for word in content:
            if word not in word_key_list:
                continue
            else:
                docvec.append(word_key_list.index(word))
        docvec_list.append(docvec)

    # return list of docvecs of all documents 
    return docvec_list

def c2i(x):
    if(x == 'positive'):
        return 1
    else:
        return 0

def preprocessing(pos='', neg='', data_type='original'):
    
    training_reading_done = False #False
    doc2vec = True #True
    testing_reading_done = True
    first_reading_done = False

    # global item initialization
    word_key_list = []
    content_list = []
    tag_list = []
    s_tag_list = []
    word_count_dict = {}

    if(training_reading_done == False and first_reading_done == False):
        print(data_type + ' first reading start......')
        #train_pos = "task1/train/positive/"
        #train_neg = "task1/train/negative/"   
        pos_content_list, pos_name_list, pos_s_tag_list = read_data(pos, "positive")
        neg_content_list, neg_name_list, neg_s_tag_list = read_data(neg, "negative")

        print("Reading finish, start saving......")

        data_type = "train_files/" + data_type

        np.save(data_type + "_" + "pos_content_list", pos_content_list)
        np.save(data_type + "_" + "neg_content_list", neg_content_list)
        np.save(data_type + "_" + "pos_name_list", pos_name_list)
        np.save(data_type + "_" + "neg_name_list", neg_name_list)

        pos_s_tag_list = np.array(list(map(c2i, pos_s_tag_list)))
        neg_s_tag_list = np.array(list(map(c2i, neg_s_tag_list)))
        np.save(data_type + "_" + "pos_s_tag_list", pos_s_tag_list)
        np.save(data_type + "_" + "neg_s_tag_list", neg_s_tag_list)

        print("Saving finish......\n")


        '''
        #content
        content_list = pos_content_list + neg_content_list
        #file path and name
        tag_list = pos_tag_list + neg_tag_list
        #sentiment tag
        s_tag_list = pos_s_tag_list + neg_s_tag_list
        s_tag_list = np.array(list(map(c2i, s_tag_list)))
        
        np.save("content_list", content_list)
        np.save("s_tag_list", s_tag_list)
        np.save("tag_list", tag_list)
        '''

    '''
    if(doc2vec == False and training_reading_done == False):
        # Naive docvec
        word_count_dict, word_key_list = word_count(content_list)
        docvec_list = docvec_generate(content_list, word_key_list)

        print("Naive preprocessing done, saving......\n")
        docvec_list = np.array(docvec_list)
        np.save("naive_docvec_list", docvec_list)
    '''
    
    '''
    elif(doc2vec == True and training_reading_done == False):
        # docvec from doc2vec model
        tag_list = np.load('tag_list.npy', allow_pickle=True)
        content_list = np.load('content_list.npy', allow_pickle=True)
        assert(len(content_list) == len(tag_list))
        TaggedDocument = gensim.models.doc2vec.TaggedDocument
        documents = [TaggedDocument(content_list[i], [tag_list[i]]) for i in range(len(content_list))]

        print("Doc2Vec model training start......")
        model = gensim.models.Doc2Vec(vector_size=600,\
                                      sample=1e-3, window=15, min_count=3, workers=6, epochs=10)

        model.build_vocab(documents)

        def perm(docs):
            random.shuffle(docs)
            return docs

        for i in range(10):
            print('Epoch: ' + str(i))
            model.train(perm(documents), total_examples=model.corpus_count, epochs=1)

        print("Doc2vec preprocessing done, saving......\n")
        model.save('model_v600_w15.model')
        print("model saved\n")
        
        docvec_list = []
        for filename in tag_list:
            docvec = model.docvecs[filename]
            docvec_list.append(docvec)
        docvec_list = np.array(docvec_list)
        np.save("doc2vec_docvec_list", docvec_list)

        '''
    

    '''
    #Generate Doc2Vec testing docvec
    if(testing_reading_done == False):
        if(doc2vec == True):
            model_loc = 'model_v600_w15.model'
            model = gensim.models.Doc2Vec.load(model_loc)

            #test1
            test1_loc = 'task1/test/'
            content_list, tag_list, s_tag_list = read_data(test1_loc)
            np.save('doc2vec_testing1_tag', tag_list)
            testing_docvec_list = []

            for content in content_list:
                docvec = model.infer_vector(content)
                testing_docvec_list.append(docvec)
            
            testing_docvec_list = np.array(testing_docvec_list)
            np.save("doc2vec_testing1", testing_docvec_list)

            #test2
            test2_loc = 'data2_task2/test/'
            content_list, tag_list, s_tag_list = read_data(test2_loc)
            np.save('doc2vec_testing2_tag', tag_list)
            testing_docvec_list = []

            for content in content_list:
                docvec = model.infer_vector(content)
                testing_docvec_list.append(docvec)
            
            testing_docvec_list = np.array(testing_docvec_list)
            np.save("doc2vec_testing2", testing_docvec_list)

        #Generate naive testing docvec
        if(doc2vec == False):
            test1_loc = 'task1/test/'
            test2_loc = 'data2_task2/test/'

            #test1 part 
            content_list, tag_list, s_tag_list = read_data(test1_loc)
            tag_list = np.array(tag_list)
            np.save('naive_testing1_tag', tag_list)
            docvec_list = docvec_generate(content_list, word_key_list)
            print("Testing 1 docvec finished, saving......\n")
            docvec_list = np.array(docvec_list)
            np.save('naive_testing1', docvec_list)
            print(len(docvec_list))
            
            #test2 part
            content_list, tag_list, s_tag_list = read_data(test2_loc)
            tag_list = np.array(tag_list)
            np.save('naive_testing2_tag', tag_list)
            docvec_list = docvec_generate(content_list, word_key_list)
            print("Testing 2 docvec finished, saving......\n")
            docvec_list = np.array(docvec_list)
            np.save('naive_testing2', docvec_list)
            print(len(docvec_list))
    '''

def model_train(file_list, file_name_list, model_type=''):
    content_list = []
    name_list = []

    assert(len(file_list) == len(file_name_list))

    for i in range(len(file_list)):
        content = list(np.load(file_list[i], allow_pickle=True))
        content_list = content_list + content
        
        name = list(np.load(file_name_list[i], allow_pickle=True))
        name_list = name_list + name
    
    assert(len(name_list) == len(content_list))
    print("Length of content_list: ", len(content_list))
    print("Length of name_list: ", len(name_list)) 

    TaggedDocument = gensim.models.doc2vec.TaggedDocument
    documents = [TaggedDocument(content_list[i], [name_list[i]]) for i in range(len(content_list))]

    print(model_type + " Doc2Vec model training start......")
    model = gensim.models.Doc2Vec(vector_size=600,\
                                    sample=1e-3, window=15, min_count=3, workers=6, epochs=10)

    model.build_vocab(documents)

    def perm(docs):
        random.shuffle(docs)
        return docs

    for i in range(10):
        print('Epoch: ' + str(i))
        model.train(perm(documents), total_examples=model.corpus_count, epochs=1)

    #model.save('model_v600_w15.model')
    model.save('models/' + model_type + '_model.model')
    print(model_type + " model saved\n")



def testing_doc_read(test1_loc):
    #test2_loc = 'data2_task2/test/'
    content_list, name_list, s_tag_list = read_data(test1_loc)
    np.save('test_files/test_content_list', content_list)
    np.save('test_files/test_name_list', name_list)
    '''
    content_list, tag_list, s_tag_list = read_data(test2_loc)
    np.save('testing2_content_list', content_list)
    '''


'''
ori_train_pos = 'task_3_4_data/train/original/pos/'
ori_train_neg = 'task_3_4_data/train/original/neg/'
mod_train_pos = 'task_3_4_data/train/modified/pos/'
mod_train_neg = 'task_3_4_data/train/modified/neg/'
ext_train_pos = 'task_3_4_data/train/original_extra/pos/'
ext_train_neg = 'task_3_4_data/train/original_extra/neg/'
testing = 'task_3_4_data/test/'

preprocessing(ori_train_pos, ori_train_neg, "original")
preprocessing(mod_train_pos, mod_train_neg, "modified")
preprocessing(ext_train_pos, ext_train_neg, "extra")
testing_doc_read(testing)
'''

ori_train_pos = 'train_files/original_pos_content_list.npy'
ori_train_neg = 'train_files/original_neg_content_list.npy'
mod_train_pos = 'train_files/modified_pos_content_list.npy'
mod_train_neg = 'train_files/modified_neg_content_list.npy'
ext_train_pos = 'train_files/extra_pos_content_list.npy'
ext_train_neg = 'train_files/extra_neg_content_list.npy'

ori_train_pos_name = 'train_files/original_pos_name_list.npy'
ori_train_neg_name = 'train_files/original_neg_name_list.npy'
mod_train_pos_name = 'train_files/modified_pos_name_list.npy'
mod_train_neg_name = 'train_files/modified_neg_name_list.npy'
ext_train_pos_name = 'train_files/extra_pos_name_list.npy'
ext_train_neg_name = 'train_files/extra_neg_name_list.npy'


ori_files = [ori_train_pos, ori_train_neg]
ori_names = [ori_train_pos_name, ori_train_neg_name]
mod_files = [mod_train_pos, mod_train_neg]
mod_names = [mod_train_pos_name, mod_train_neg_name]
ext_files = [ext_train_pos, ext_train_neg]
ext_names = [ext_train_pos_name, ext_train_neg_name]

ori_mod_files = ori_files + mod_files
ori_mod_names = ori_names + mod_names
ori_ext_files = ori_files + ext_files
ori_ext_names = ori_names + ext_names

#model_train(ori_files, ori_names, "original")
#model_train(ori_mod_files, ori_mod_names, "original_modified")
#model_train(ori_ext_files, ori_ext_names, "original_extra")
