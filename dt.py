import numpy as np
from math import log2
from graphviz import Digraph

chiSquare = [[0, 2.706, 4.605, 6.251, 7.779, 9.236, 10.64, 12.02, 13.36],
            [0, 3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507],
            [0, 1.323, 2.773, 4.108, 5.385, 6.626, 7.841, 9.037, 10.219]]

def entropy(bucket):
    H, total_elem = 0.0, sum(bucket)
    for i in bucket:
        p = i / total_elem
        if(p > 0): #gives value domain error for <= 0
            H -= (p)*log2(p)
    return H 

def info_gain(parent_bucket, left_bucket, right_bucket):
    left, right = sum(left_bucket), sum(right_bucket)
    return entropy(parent_bucket) - (left*entropy(left_bucket) + right*entropy(right_bucket))/(left + right)

def gini(bucket):
    I, total_elem = 1.0, sum(bucket)
    for elem in bucket:
        I -= (elem/total_elem)**2
    return I

def avg_gini_index(left_bucket, right_bucket):
    left, right = sum(left_bucket), sum(right_bucket)
    return (left*gini(left_bucket) + right*gini(right_bucket))/(left + right)

def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    split_vals = []
    sorted_data, sorted_label = (list(t) for t in zip(*sorted(zip(data[:,attr_index], labels))))
    if(heuristic_name == 'info_gain'):
        for i in range(1,len(labels)):
            par_bucket = [sorted_label.count(x) for x in range(num_classes)]
            l_bucket, r_bucket = [sorted_label[:i].count(x) for x in range(num_classes)], [sorted_label[i:].count(x) for x in range(num_classes)]
            S = info_gain(par_bucket,l_bucket,r_bucket)
            ref_point = (sorted_data[i-1] + sorted_data[i]) / 2
            split_vals.append([ref_point, S]) 

    elif (heuristic_name == 'avg_gini_index'):
        for i in range(1,len(labels)):
            l_bucket, r_bucket = [sorted_label[:i].count(x) for x in range(num_classes)], [sorted_label[i:].count(x) for x in range(num_classes)]
            S = avg_gini_index(l_bucket, r_bucket)
            ref_point = (sorted_data[i-1] + sorted_data[i]) / 2
            split_vals.append([ref_point, S]) 
    else:
        print("Heuristic Name Error")
    return split_vals

def chi_squared_test(left_bucket, right_bucket):
    chi_val ,parent_bucket = 0.0, [(left_bucket[i] + right_bucket[i]) for i in range(len(left_bucket))]
    par_elem, l_elem, r_elem = sum(parent_bucket), sum(left_bucket), sum(right_bucket)
    #print(par_elem, l_elem, r_elem)
    p_expect = [elem/par_elem for elem in parent_bucket]
    l_expect, r_expect = [l_elem*p_expect[i] for i in range(len(left_bucket))],[r_elem*p_expect[i] for i in range(len(right_bucket))]
    for cur in range(len(parent_bucket)):
        if(l_expect[cur] != 0):
            l_val = (l_expect[cur] - left_bucket[cur])**2 / l_expect[cur]
            chi_val += l_val
        if(r_expect[cur] != 0):
            r_val = (r_expect[cur] - right_bucket[cur])**2 / r_expect[cur]
            chi_val += r_val
    return chi_val, (sum([1 if x > 0 else 0 for x in parent_bucket]) - 1)


class TreeNode:
    def __init__(self, feature = None, threshold = None, l_child = None, r_child = None, *, value = None):
        self.feature = feature 
        self.threshold = threshold
        self.l_child = l_child
        self.r_child = r_child
        self.value = value
    
    def isLeaf(self):
        return (self.value != None)

class ID3:
    def __init__(self,  heuristic,  pre_prune = False, confidence = None,n_feature = None):
        self.parent = None
        self.heuristic = heuristic
        self.n_feature = n_feature
        self.pre_prune = pre_prune
        self.confidence = confidence
        self.title = "Tree Graph with " + self.heuristic + " when Pre-prunning: " + str(self.pre_prune)
    def fit(self, data, label):
        self.q = len(np.unique(label))
        self.n_feature = data.shape[1] if not self.n_feature else min(self.n_feature, data.shape[1])
        self.parent = self.grow_tree(data, label)
        #self.render_tree()
    
    def predict(self,data):
        return np.array([self.leaf_selector(elem, self.parent) for elem in data])

    def grow_tree(self, data, label, depth = 0):
        sample_size, feature_size = data.shape
        uniq_label_size = len(np.unique(label))
        #print("Depth of tree: ",depth," Number of elements: ",len(label))
        #leaf node
        if(uniq_label_size == 1):
            int_val = self.most_common_class(label)
            print("Depth:",depth, "Leaf(1) labelled with: ",int_val," Number of elements: ",len(label))
            return TreeNode(value=int_val) 

        # default split
        best_attr, best_threshold = self.find_best_split(data, label, uniq_label_size, self.heuristic)

        l_idx, r_idx = self.split_holder_data(data[:,best_attr], best_threshold)
        if(not self.pre_prune):
            print("Left Size:",len(l_idx)," Right Size",len(r_idx))
            print("Depth: ",depth," Choosen Feature: ",best_attr," and choosen threshold: ",best_threshold," Number of elements: ",len(label))
            left_child = self.grow_tree(data[l_idx,:], label[l_idx], depth+1)
            right_child = self.grow_tree(data[r_idx,:], label[r_idx], depth+1)
            return TreeNode(best_attr,best_threshold, left_child, right_child)
        else:
            l_buc, r_buc = self.create_bucket(label, l_idx, r_idx)
            chi_val, df = chi_squared_test(l_buc, r_buc)
            if (chi_val < chiSquare[self.confidence][df]):
                int_val = self.most_common_class(label)
                print("CHI - Decisive Prune: Left: ", len(l_idx)," Right: ",len(r_idx))
                print("Depth:",depth,"Leaf labelled with(prune): ",int_val," Number of elements: ",len(label))
                return TreeNode(value=int_val)
            else:
                print("Left Size:",len(l_idx)," Right Size",len(r_idx))
                print("Depth: ",depth," Choosen Feature: ",best_attr," and choosen threshold: ",best_threshold," Number of elements: ",len(label))
                left_child = self.grow_tree(data[l_idx,:], label[l_idx], depth+1)
                right_child = self.grow_tree(data[r_idx,:], label[r_idx], depth+1)
                return TreeNode(best_attr,best_threshold, left_child, right_child)

    def split_holder_data(self, col_data, threshold):
        return list(np.argwhere(col_data <= threshold).flatten()), list(np.argwhere(col_data > threshold).flatten()) 

    def create_bucket(self, label,l_idx, r_idx):
        left_bucket = [np.sum(label[l_idx] == i) for i in range(self.q)]
        right_bucket = [np.sum(label[r_idx] == i) for i in range(self.q)]
        return left_bucket, right_bucket

    def find_best_split(self, data, label, num_labels, heuristic):
        best_at = -1
        if(heuristic == 'avg_gini_index'):
            best_gain = np.infty
            for i in range(data.shape[1]):
                split_values = calculate_split_values(data, label, self.n_feature, i, heuristic)
                holder = np.min(split_values,axis=0)[1]
                holder_at = np.argmin(split_values, axis=0)[1]
                l,r = self.split_holder_data(data[:,i], split_values[holder_at][0])
                if((holder < best_gain) and (len(l) != 0) and (len(r) != 0)):
                    best_gain = holder
                    best_at = i
                    best_thr = split_values[holder_at][0]
        
        elif(heuristic == 'info_gain'):
            best_gain = -np.infty
            for i in range(data.shape[1]):
                split_values = calculate_split_values(data, label, self.n_feature, i, heuristic)
                holder = np.max(split_values,axis=0)[1]
                holder_at = np.argmax(split_values, axis=0)[1]
                l,r = self.split_holder_data(data[:,i], split_values[holder_at][0])
                if((holder > best_gain) and (len(l) != 0) and (len(r) != 0)):
                    best_gain = holder
                    best_at = i
                    best_thr = split_values[holder_at][0]
        return best_at, best_thr

    def most_common_class(self,class_label):
        test_list = list(class_label)
        return max(set(test_list), key = test_list.count)

    def leaf_selector(self, data_point, current_node):
        if(current_node.isLeaf()):
            return current_node.value
        if(data_point[current_node.feature] <= current_node.threshold):
            return self.leaf_selector(data_point, current_node.l_child)
        return self.leaf_selector(data_point, current_node.r_child)
    
    def render_tree(self):  
        dot = Digraph(comment= self.title)  
        tree = self.traverse(dot,self.parent,1)
        tree.render('tree.pdf')  

    def traverse(self,tree_holder,root_node,label_idx):
        tree_holder.node(str(label_idx), str(root_node.feature) + "  " + str(root_node.threshold))
        if (not root_node.l_child.isLeaf()):
            l_child = tree_holder.node(str(2*label_idx), str(root_node.l_child.feature) + " " + str(root_node.l_child.threshold))
            l_hold = self.traverse(l_child,root_node.l_child,2*label_idx)
            tree_holder.edge(str(label_idx),str(2*label_idx))
        if (not root_node.r_child.isLeaf()):
            r_child = tree_holder.node(str(2*label_idx+1), str(root_node.r_child.feature) + " " + str(root_node.r_child.threshold))
            r_hold = self.traverse(r_child, root_node, 2*label_idx+1)
            tree_holder.edge(str(label_idx),str(2*label_idx + 1))
        return tree_holder


if __name__ == '__main__':
    train_data = np.load('hw3_data/iris/train_data.npy')
    train_labels = np.load('hw3_data/iris/train_labels.npy')
    test_data = np.load('hw3_data/iris/test_data.npy')
    test_labels = np.load('hw3_data/iris/test_labels.npy')
    n_test = len(test_labels)
    
    #default cases
    print("Average gini index with no pre-pruning")
    classifier = ID3('avg_gini_index',False)
    classifier.fit(train_data, train_labels)
    predicter = classifier.predict(test_data)
    print("Labelled correctly:  " + str(np.sum(predicter == test_labels)))
    print("Accuracy for avg_gini ==> "+ str(np.sum(predicter == test_labels)/len(test_labels)))
    
    print("Information Gain with no pre-pruning")
    classifier = ID3('info_gain', False)
    classifier.fit(train_data, train_labels)
    predicter = classifier.predict(test_data)
    print("Labelled correctly:  " + str(np.sum(predicter == test_labels)))
    print("Accuracy for info_gain ==> "+ str(np.sum(predicter == test_labels)/len(test_labels)))

    # pre-prunned cases 90% confidence
    print("Average gini index with with 90% confidence level chi-square test")
    classifier = ID3('avg_gini_index',True,0)
    classifier.fit(train_data, train_labels)
    predicter = classifier.predict(test_data)
    print("Labelled correctly:  " + str(np.sum(predicter == test_labels)))
    print("Accuracy for avg_gini with 90% confidence pre-prunning ==> "+ str(np.sum(predicter == test_labels)/len(test_labels)))

    print("Information Gain with 90% confidence level chi-square test")
    classifier = ID3('info_gain', True, 0)
    classifier.fit(train_data, train_labels)
    predicter = classifier.predict(test_data)
    print("Labelled correctly:  " + str(np.sum(predicter == test_labels)))
    print("Accuracy for info_gain with 90% confidence pre-prunning ==> "+ str(np.sum(predicter == test_labels)/len(test_labels)))

    # pre-prunned cases 95% confidence
    print("Average gini index with with 95% confidence level chi-square test")
    classifier = ID3('avg_gini_index',True,1)
    classifier.fit(train_data, train_labels)
    predicter = classifier.predict(test_data)
    print("Labelled correctly:  " + str(np.sum(predicter == test_labels)))
    print("Accuracy for avg_gini with 95% confidence pre-prunning ==> "+ str(np.sum(predicter == test_labels)/len(test_labels)))

    print("Information Gain with 95% confidence level chi-square test")
    classifier = ID3('info_gain', True, 1)
    classifier.fit(train_data, train_labels)
    predicter = classifier.predict(test_data)
    print("Labelled correctly:  " + str(np.sum(predicter == test_labels)))
    print("Accuracy for info_gain with 95% confidence pre-prunning ==> " + str(np.sum(predicter == test_labels)/len(test_labels)))
    
    # pre-prunned cases 75% confidence
    print("Average gini index with with 75% confidence level chi-square test")
    classifier = ID3('avg_gini_index',True,2)
    classifier.fit(train_data, train_labels)
    predicter = classifier.predict(test_data)
    print("Labelled correctly:  " + str(np.sum(predicter == test_labels)))
    print("Accuracy for avg_gini with 75% confidence pre-prunning ==> "+ str(np.sum(predicter == test_labels)/len(test_labels)))

    print("Information Gain with 75% confidence level chi-square test")
    classifier = ID3('info_gain', True, 2)
    classifier.fit(train_data, train_labels)
    predicter = classifier.predict(test_data)
    print("Labelled correctly:  " + str(np.sum(predicter == test_labels)))
    print("Accuracy for info_gain with 75% confidence pre-prunning ==> "+ str(np.sum(predicter == test_labels)/len(test_labels)))