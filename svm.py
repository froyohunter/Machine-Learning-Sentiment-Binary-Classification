from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

def model_training_testing():
    boosting = False
    use_LR = True
    use_SVM = False

    docvec_list = np.load('doc2vec_docvec_list.npy', allow_pickle=True)
    s_tag_list = np.load('s_tag_list.npy', allow_pickle=True)

    if(boosting == False):
        grid_values = {'C':[30]}
        clf = GridSearchCV(LogisticRegression(penalty='l2', dual=False), grid_values, cv = 20)
        clf.fit(docvec_list, s_tag_list)
        #clf = load('svm_500.joblib')
        
        print(clf.best_score_)
        dump(clf.best_estimator_, 'LR.joblib')

        testing_list1 = np.load('doc2vec_testing1.npy', allow_pickle=True)
        testing_list2 = np.load('doc2vec_testing2.npy', allow_pickle=True)

        result_list1 = np.array(clf.predict(testing_list1))
        result_list2 = np.array(clf.predict(testing_list2))
        

        np.save('result_list1_600', result_list1)
        np.save('result_list2_600', result_list2)
        
        print("LR Done")

    elif(boosting == True):
            clf = None
            model_name = ''
            if(use_LR == True):
                clf = AdaBoostClassifier(LogisticRegression(penalty='l2', dual=False, solver='newton-cg'), n_estimators=1000,\
                                                        learning_rate=1.0)
                model_name = 'LR_boosting.joblib'
            if(use_SVM == True):
                clf = AdaBoostClassifier(svm.SVC(kernel='rbf', probability=True), n_estimators=100, learning_rate=1.0)
                model_name = 'SVM_boosting.joblib'

            assert(len(docvec_list) == len(s_tag_list))
            clf.fit(docvec_list, s_tag_list)
            dump(clf, model_name) 
            print('Fitting done')

            print("cross validation start......")
            print("20 fold score for " + model_name, np.mean(\
                    cross_val_score(clf, docvec_list, s_tag_list, cv=20, scoring='roc_auc')))


            testing_list1 = np.load('doc2vec_testing1.npy', allow_pickle=True)
            testing_list2 = np.load('doc2vec_testing2.npy', allow_pickle=True)

            result_list1 = np.array(clf.predict(testing_list1))
            result_list2 = np.array(clf.predict(testing_list2))

            np.save('boosting_result_list1', result_list1)
            np.save('boosting_result_list2', result_list2)
            
            print("Boosting Logistic Done")


model_training_testing()


    