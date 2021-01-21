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
            content = [word for word in content if word not in stopwords.words('english')]
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

def preprocessing():
    
    training_reading_done = True #False
    doc2vec = True #True
    testing_reading_done = False
    first_reading_done = True

    # global item initialization
    word_key_list = []
    content_list = []
    tag_list = []
    s_tag_list = []
    word_count_dict = {}

    if(training_reading_done == False and first_reading_done == False):
        print('first reading start......')
        train_pos = "task1/train/positive/"
        train_neg = "task1/train/negative/"   
        pos_content_list, pos_tag_list, pos_s_tag_list = read_data(train_pos, "positive")
        neg_content_list, neg_tag_list, neg_s_tag_list = read_data(train_neg, "negative")
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

    if(doc2vec == False and training_reading_done == False):
        # Naive docvec
        word_count_dict, word_key_list = word_count(content_list)
        docvec_list = docvec_generate(content_list, word_key_list)

        print("Naive preprocessing done, saving......\n")
        docvec_list = np.array(docvec_list)
        np.save("naive_docvec_list", docvec_list)
    

    elif(doc2vec == True and training_reading_done == False):
        # docvec from doc2vec model
        tag_list = np.load('tag_list.npy', allow_pickle=True)
        content_list = np.load('content_list.npy', allow_pickle=True)
        assert(len(content_list) == len(tag_list))
        TaggedDocument = gensim.models.doc2vec.TaggedDocument
        documents = [TaggedDocument(content_list[i], [tag_list[i]]) for i in range(len(content_list))]

        print("Doc2Vec model training start......")
        model = gensim.models.Doc2Vec(vector_size=600,\
                                      sample=1e-3, window=15, min_count=3, workers=4, epochs=10)

        model.build_vocab(documents)

        def perm(docs):
            random.shuffle(docs)
            return docs

        for i in range(10):
            print('Epoch: ' + str(i))
            model.train(perm(documents), total_examples=model.corpus_count, epochs=1)

        print("Doc2vec preprocessing done, saving......\n")
        model.save('model_v600_w15.model')

        docvec_list = []
        for filename in tag_list:
            docvec = model.docvecs[filename]
            docvec_list.append(docvec)
        docvec_list = np.array(docvec_list)
        np.save("doc2vec_docvec_list", docvec_list)


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


def testing_doc_read():
    test1_loc = 'task1/test/'
    test2_loc = 'data2_task2/test/'
    content_list, tag_list, s_tag_list = read_data(test1_loc)
    np.save('testing1_content_list', content_list)
    content_list, tag_list, s_tag_list = read_data(test2_loc)
    np.save('testing2_content_list', content_list)


#preprocessing()
#testing_doc_read()