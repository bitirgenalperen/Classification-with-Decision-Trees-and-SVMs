Decision tree:
	type "python dt.py"  ==> used packages: {numpy, math.log2, Graphviz}
	---	Info Gain
	--- Average Gini Index
	--- Preprunin Info Gain ---	Confidence 0.75
							---	Confidence 0.90
							---	Confidence 0.95
	--- Preprunin Average Gini Index ---	Confidence 0.75
									 ---	Confidence 0.90
									 ---	Confidence 0.95

	You can observer all the outputs. If you want to see only one, you can simply comment down the others under the 'main' part.

SVM:
	type "python svm.py x"  ==> used packages: {sys, copy, random, numpy, sklearn.svm, sklearn.model_selection, draw}
							==> x part can have the following values for each part: {1 : linsep, 2 : nonlinsep, 3 : fashion, 4 : imbalanced}
	--- C value tunning in linear kernel
	--- Kernel tunning with constant c = 1
	--- Gamma, c, degree, kernel tunning for a balanced data-set
	--- Information metrics analysis on an imbalanced data-set

Compiled version of the latex source file named 'report.pdf' included.