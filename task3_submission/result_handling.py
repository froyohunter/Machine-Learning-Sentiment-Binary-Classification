import numpy as np
import re

def extract_filename(x):
    l = []
    l = re.findall(r'\d+', x)
    return int(l[-1])

def match_result(tag_list_loc, result_list):
    tag_list = np.load(tag_list_loc, allow_pickle=True)
    tag_list = np.array(list(map(extract_filename, tag_list)))

    assert(len(result_list) == len(tag_list))
    tuple_list = list((zip(tag_list, result_list)))
    dtype = [('tag', int), ('sentiment', int)]

    tuple_list = np.array(tuple_list, dtype=dtype)
    tuple_list = np.sort(tuple_list, order='tag') 
    result_list = list(map(lambda x: str(x[1]), tuple_list))
    
    return tuple_list, result_list

def write_result(result_list, model_type):
    filename = model_type + '_testing_result.txt'
    with open(filename, 'w+') as f:
        f.write('\n'.join(result_list))
        f.close()


def handling(tag_list_loc, result_list, model_type):
    def res_check(x):
        res = x[0]
        if(res <= 0.5):
            return 0
        else:
            return 1    

    #test_res_list = np.load(result_list_loc)
    result_list = np.array(list(map(res_check, result_list)))
    tuple_list, result_list = match_result(tag_list_loc, result_list)

    write_result(result_list, model_type)


def clarify():

    def res_check(x):
        res = x[0]
        if(res <= 0.5):
            return 0
        else:
            return 1

    test_res_list = np.load('lstm_testing2_result.npy')
    test_res_list = np.array(list(map(res_check, test_res_list)))
    print(sum(test_res_list))
    np.save('lstm_res2', test_res_list)

'''
tag_loc = 'doc2vec_testing1_tag.npy'
res_loc = 'lstm_res1.npy'
handling(tag_loc, res_loc, 1)

tag_loc = 'doc2vec_testing2_tag.npy'
res_loc = 'lstm_res2.npy'
handling(tag_loc, res_loc, 2)
'''
