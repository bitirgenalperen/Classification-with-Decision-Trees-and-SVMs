import sys
import copy
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from draw import draw_svm, MetricsStatDisplay

if __name__ == '__main__':
    current_part = str(sys.argv[1])
    c = [0.01, 0.1, 1.0, 10.0, 100.0]
    kernel_type = ["linear", "rbf", "poly", "sigmoid"]
    gamma_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    degree_range = [2,3,4,5,6]
    print("SVM for " + current_part)
    if(current_part == "linsep"):
        print("SVM process starting for " + current_part)
        training_data = np.load("hw3_data/linsep/train_data.npy")
        training_labels = np.load("hw3_data/linsep/train_labels.npy")

        for cur_c in c:
            classifier = SVC(kernel='linear', C= cur_c)
            classifier.fit(training_data, training_labels)    
            draw_svm(classifier, training_data, training_labels, -2.5, 2.5, -2.5, 2.5, "liner_kernel_c_value_tunning__"+str(cur_c)+".png")

    elif(current_part == "nonlinsep"):
        print("SVM process starting for " + current_part)
        training_data = np.load("hw3_data/nonlinsep/train_data.npy")
        training_labels = np.load("hw3_data/nonlinsep/train_labels.npy")

        for cur_kernel in kernel_type:
            classifier = SVC(kernel=cur_kernel)
            classifier.fit(training_data, training_labels)    
            draw_svm(classifier, training_data, training_labels, -2.5, 2.5, -2.5, 2.5, "kernel_implementation__"+str(cur_kernel)+".png")
    
    
    
    
    elif(current_part == "fashion"):
        print("SVM process starting for " + current_part)
        best_acc_holder = [0.0]
        training_data = np.load("hw3_data/fashion_mnist/train_data.npy")
        training_data = (training_data - np.min(training_data))/np.ptp(training_data)
        nsamples, nx, ny = training_data.shape
        training_data = training_data.reshape((nsamples,nx*ny))
        training_labels = np.load("hw3_data/fashion_mnist/train_labels.npy")

        testing_data = np.load("hw3_data/fashion_mnist/test_data.npy")
        testing_data = (testing_data - np.min(testing_data))/np.ptp(testing_data)
        testing_data = testing_data.reshape((-1, nx*ny))
        testing_labels = np.load("hw3_data/fashion_mnist/test_labels.npy")

                #kernel = 'poly'

        for cur_degree in degree_range:
            for cur_c in c:
                for cur_gamma in gamma_range:
                    classifier = SVC(kernel='poly', C=cur_c, gamma=cur_gamma, degree=cur_degree)
                    cv_results = cross_validate(classifier, training_data, training_labels, cv=5)
                    acc = sum(cv_results['test_score'])/len(cv_results['test_score'])
                    print("kernel=> poly", " - ",  "degree=> ", cur_degree, "c=> ", cur_c, " - ", "gamma=> ", cur_gamma, " - ",  "-- accuracy = ","{:.6f}".format(acc))
                    if(acc > best_acc_holder[0]):
                        best_acc_holder.clear()
                        best_acc_holder = [acc, 'poly', cur_c, cur_gamma, cur_degree]  

        #kernel = 'linear'
        for cur_c in c:
            classifier = SVC(kernel='linear', C=cur_c)
            cv_results = cross_validate(classifier, training_data, training_labels, cv=5)
            acc = sum(cv_results['test_score'])/len(cv_results['test_score'])
            print("kernel=> linear", " - ", "c=> ", cur_c, " - ", "-- accuracy = ","{:.6f}".format(acc))
            if(acc > best_acc_holder[0]):
                best_acc_holder.clear()
                best_acc_holder = [acc, 'linear', cur_c]    

        #kernel = 'rbf' or 'sigmoid'
        for cur_kernel in ['rbf', 'sigmoid']:
            for cur_c in c:
                for cur_gamma in gamma_range:
                    classifier = SVC(kernel=cur_kernel, C=cur_c, gamma=cur_gamma)
                    cv_results = cross_validate(classifier, training_data, training_labels, cv=5)
                    acc = sum(cv_results['test_score'])/len(cv_results['test_score'])
                    print("kernel=> ", cur_kernel, " - ", "c=> ", cur_c, " - ", "gamma=> ", cur_gamma, " - ", "-- accuracy = ","{:.6f}".format(acc)) 
                    if(acc > best_acc_holder[0]):
                        best_acc_holder.clear()
                        best_acc_holder = [acc, cur_kernel, cur_c, cur_gamma]  
    

        print("*************************************************************")
        print("Tunned kernel is " + str(best_acc_holder[1]))
        #testing for the best hyper-parameter tunes
        if(best_acc_holder[1] == 'linear'):
            print("Tunned C " + str(best_acc_holder[2]))
            classifier = SVC(kernel='linear',C = best_acc_holder[2])
        elif(best_acc_holder[1] == 'poly'):
            print("Tunned C " + str(best_acc_holder[2]))
            print("Tunned Gamma " + str(best_acc_holder[3]))
            print("Tunned Degree " + str(best_acc_holder[4]))
            classifier = SVC(kernel='poly', C=best_acc_holder[2], gamma=best_acc_holder[3], degree= best_acc_holder[4])
        else:
            print("Tunned C " + str(best_acc_holder[2]))
            print("Tunned Gamma " + str(best_acc_holder[3]))
            classifier = SVC(kernel=best_acc_holder[1], C=best_acc_holder[2], gamma= best_acc_holder[3])
            
        classifier.fit(training_data, training_labels)
        predicter = classifier.predict(testing_data)

        test_acc = np.sum(predicter == testing_labels)/len(predicter)
        print("Resulting test-accuracy with tunned hyper-parameters: ", test_acc)


    elif(current_part == "imbalanced"):
        print("SVM process starting for " + current_part)

        training_data = np.load("hw3_data/fashion_mnist_imba/train_data.npy")
        training_data = (training_data - np.min(training_data))/np.ptp(training_data)

        nsamples, nx, ny = training_data.shape
        training_data = training_data.reshape((nsamples,nx*ny))

        training_labels = np.load("hw3_data/fashion_mnist_imba/train_labels.npy")

        testing_data = np.load("hw3_data/fashion_mnist_imba/test_data.npy")
        testing_data = (testing_data - np.min(testing_data))/np.ptp(testing_data)
        testing_data = testing_data.reshape((-1,nx*ny))
        testing_labels = np.load("hw3_data/fashion_mnist_imba/test_labels.npy")

        training_data_oversample = copy.deepcopy(training_data)
        training_labels_oversample = (copy.deepcopy(training_labels)).reshape(-1,1)

        training_data_undersample = copy.deepcopy(training_data)
        training_labels_undersample = copy.deepcopy(training_labels).reshape(-1,1)

        # default
        pos, neg = np.sum(training_labels == 1), np.sum(training_labels == 0)
        classifier = SVC().fit(training_data, training_labels)
        predicter = classifier.predict(testing_data)
        print("Number of Positive Class Members: " + str(pos) + " Number of Negative Class Members: " + str(neg))
        MetricsStatDisplay(predicter, testing_labels, "Default Setup")

        # oversample
        

        inc = pos - neg
        start_idx = 0
        while (inc > 0 ):
            if(training_labels_oversample[start_idx] == 0):
                training_data_oversample = np.vstack((training_data_oversample, training_data_oversample[start_idx]))
                training_labels_oversample = np.vstack((training_labels_oversample, [0]))
                inc -= 1
            start_idx += 1
        
        classifier = SVC().fit(training_data_oversample, training_labels_oversample)
        predicter = classifier.predict(testing_data)
        print("Number of Positive Class Members: " + str(np.sum(training_labels_oversample == 1)) + " Number of Negative Class Members: " + str(np.sum(training_labels_oversample == 0)))
        MetricsStatDisplay(predicter, testing_labels, "With Oversampled Minority Class")

        # undersample
        inc = pos
        start_idx = 0
        undersample_idx = []
        while (inc > 0 ):
            if(training_labels_oversample[start_idx] == 1):
                undersample_idx.append(start_idx)
                inc -= 1
            start_idx += 1
        undersample_idx = random.sample(undersample_idx, pos-neg)
        undersample_idx.sort(reverse=True)

        for index in undersample_idx:
            training_data_undersample = np.delete(training_data_undersample, index, axis=0)
            training_labels_undersample = np.delete(training_labels_undersample, index, axis=0)

        classifier = SVC().fit(training_data_undersample, training_labels_undersample)
        predicter = classifier.predict(testing_data)
        print("Number of Positive Class Members: " + str(np.sum(training_labels_undersample == 1)) + " Number of Negative Class Members: " + str(np.sum(training_labels_undersample == 0)))
        MetricsStatDisplay(predicter, testing_labels, "With Oversampled Minority Class")

        # balanced weights
        classifier = SVC(class_weight='balanced').fit(training_data, training_labels)
        predicter = classifier.predict(testing_data)
        MetricsStatDisplay(predicter, testing_labels, "With Balanced Class Weight")
    
    else:
        print("Lack of requested functionality!")
        print("Error: No input layer called with name -:" + current_part+":-")