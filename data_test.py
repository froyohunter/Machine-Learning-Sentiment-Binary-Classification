import gensim
import data_preprocessing
import numpy as np

def test(model_loc, test_loc):
    model = gensim.models.Doc2Vec.load(model_loc)
    tag_list = np.load('tag_list.npy', allow_pickle=True)    
    print(type(tag_list[0]))
    #t1 = model.docvecs[]

    #test_content_list, test_tag_list, test_s_tag_list = data_preprocessing.read_data(test_loc)
    
    #words = test_content_list[0]
    #words_docvec = model.infer_vector(words)
    #print(len(words_docvec))
    

model_loc = 'model_v200_w10.model'
test_loc = 'data1_task1/test/'
test(model_loc, test_loc)
