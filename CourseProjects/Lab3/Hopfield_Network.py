# -*- coding: utf-8 -*-
# Hopfiedl Network
# Author: Jun Zhang <junzha@kth.se>, Balint
# Feb 16th, 2019
# 

import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.cm as cm
import random


# input: an array
def sign(x):
    signals = np.sign(x)
    signals[signals == 0] = 1
    return signals

# input: targets and outputs are 2-D arraies. 
def accuracy(outputs, targets):
    return 1.0 - np.count_nonzero(outputs - targets, axis=1)/outputs.shape[1]

# input: targets and outputs are 1-D arraies. 
def accuracy_1(outputs, targets):
    return 1.0 - np.count_nonzero(outputs - targets)/(len(outputs))

def energy(weights, pat):
	sum = 0
	for i in range(len(pat)):
		for j in range(len(pat)):
			sum += weights[i][j] * pat[i] * pat[j] 
	return -1*sum

class DHNN_bidirectional:
	def __init__(self):
		self.weights = None


	#input: an 1-D or 2-D numpy array. Each row is a pattern
	def fit(self, patterns, update_mode = "asynchronous"):
		self.weights  = patterns.T @ patterns
		if(update_mode == "asynchronous"):
			np.fill_diagonal(self.weights, 0)
		else:
			np.fill_diagonal(self.weights, 0)
		return self.weights
		# print("weights:")
		# print(self.weights)

	#input: an 1-D numpy array. Each row is a pattern
	#output: curre_pred is the prediction/association of the pattern. 
	#Epochs_count counts the number it needs to achieve convergence. 
	def recall(self, pattern, update_mode = "asynchronous"):
		if(update_mode == "asynchronous"):
			prev_pred = [ i for i in pattern]
			for i in range(len(pattern)):
				sum = self.weights[i] @ prev_pred 
				if sum >= 0:
					prev_pred[i] = 1
				else: prev_pred[i] = -1
			IsConvergent = False
			Epochs_Count = 0
			while not IsConvergent:
				curr_pred = [ i for i in prev_pred]
				for i in range(len(pattern)):
					sum = self.weights[i] @ prev_pred
					if sum >= 0:
						curr_pred[i] = 1
					else: curr_pred[i] = -1
				IsConvergent = np.array_equal(prev_pred, curr_pred)
				Epochs_Count += 1
				prev_pred = curr_pred
		else:
			prev_pred = sign(self.weights @ pattern )
			IsConvergent = False
			Epochs_Count = 0
			while not IsConvergent:
				curr_pred = sign(self.weights @ prev_pred)
				IsConvergent = np.array_equal(prev_pred, curr_pred)
				Epochs_Count += 1
				prev_pred = curr_pred
		return curr_pred, Epochs_Count

	# It can recall a list a patterns
	def recall_list(self, patterns, update_mode = "asynchronous"):
		predictions = np.apply_along_axis(lambda pattern: self.recall(pattern,update_mode), 1, patterns)
		associations = np.vstack(predictions[:, 0])
		epochs = np.array(predictions[:, 1])
		return associations, epochs		

class DHNN_binary:
	def __init__(self):
		self.weights = None

	def fit(self, patterns, p):
	    N = patterns.shape[1]
	    P = patterns.shape[0]
	    mat = [0]*N
	    returnMat = []
	    for i in range(N):
	        m = mat[:]
	        returnMat.append(m)
	    for i in range(N):
	        for j in range(N):
	            sum = 0
	            for u in range(P):
	                sum += (patterns[u][i] -p) * (patterns[u][j] - p)
	            returnMat[i][j] = sum
	    self.weights = np.array(returnMat)
	    np.fill_diagonal(self.weights, 0)
	    print("Cal weights:")
	    print(self.weights)
	    return self.weights


	#input: an 1-D numpy array. Each row is a pattern
	#output: curre_pred is the prediction/association of the pattern. 
	#Epochs_count counts the number it needs to achieve convergence. 
	def recall(self, pattern, bias, update_mode = "asynchronous"):
		if(update_mode == "asynchronous"):
			prev_pred = [ i for i in pattern]
			for i in range(len(pattern)):
				# sum = 0
				# for j in range(len(pattern)):
					# sum = sum + (self.weights[i][j] * prev_pred[j])
				sum = self.weights[i] @ prev_pred - bias
				if sum >= 0:
					prev_pred[i] = 1
				else: prev_pred[i] = 0
			IsConvergent = False
			Epochs_Count = 0
			while not IsConvergent:
				curr_pred = [ i for i in prev_pred]
				for i in range(len(pattern)):
					# sum = 0
					# for j in range(len(pattern)):
						# sum = sum + (self.weights[i][j] * prev_pred[j])
					sum = self.weights[i] @ prev_pred -bias
					if sum >= 0:
						curr_pred[i] = 1
					else: curr_pred[i] = 0
				IsConvergent = np.array_equal(prev_pred, curr_pred)
				Epochs_Count += 1
				prev_pred = curr_pred
		else:
			prev_pred = 0.5 + sign(self.weights @ pattern - bias)/2
			IsConvergent = False
			Epochs_Count = 0
			while not IsConvergent:
				curr_pred = 0.5 + sign(self.weights @ prev_pred - bias)/2
				IsConvergent = np.array_equal(prev_pred, curr_pred)
				Epochs_Count += 1
				prev_pred = curr_pred
		return curr_pred, Epochs_Count

	# It can recall a list a patterns
	def recall_list(self, patterns, bias, update_mode = "asynchronous"):
		predictions = np.apply_along_axis(lambda pattern: self.recall(pattern,bias, update_mode), 1, patterns)
		associations = np.vstack(predictions[:, 0])
		epochs = np.array(predictions[:, 1])
		return associations, epochs		

# section 3.1 & 3.2
x1 = np.array([-1,	-1,	 1,	-1,	 1, -1,	-1,		1])
x2 = np.array([-1,	-1,	-1,	-1,	-1,  1,	-1,	   -1])
x3 = np.array([-1,	 1,	 1,	-1,	-1,  1,	-1,     1])
x1d = np.array([1,	-1,	 1,	-1,	  1, -1,	-1,	    1])
x2d = np.array([1,	 1,	-1, -1,	 -1,  1,	-1,	   -1])
x3d = np.array([1,	 1,	 1,	-1,	  1,  1,	-1,	    1])
ex = np.array([1,	 1,	-1,	 1,	 -1, -1,	 1,	   -1])

training_patterns = np.concatenate([[x1],[x2],[x3]])
test_patterns	  = np.concatenate([[x1d],[x2d],[x3d]])

DHNN = DHNN_bidirectional()
global_weights = DHNN.fit(training_patterns)


print("Recall patterns from themselves")
predictions, epochs = DHNN.recall_list(training_patterns)
print(predictions)
print("Accuracy:  ", accuracy(training_patterns,predictions))
print("Epochs:    ", epochs)
print("\n")

print("Recall patterns from test patterns")
predictions, epochs = DHNN.recall_list(test_patterns)
print(predictions)
print("Accuracy:  ", accuracy(training_patterns,predictions))
print("Epochs:    ", epochs)
print("\n")

print("Recall patterns from more distorted pattern")
predictions, epochs = DHNN.recall(ex)
print(predictions)
print("Accuracy:  ", accuracy(training_patterns,predictions))
print("Epochs:    ", epochs)
print("\n")

index 		   = []
attractors_set = []
epoches_set    = []
energy_list    = []
counter 	   = 0
print("global_weights:",global_weights)
print("Recall from all possible patterns and find all attractors:")
for i in itertools.product([1,-1],repeat = 8):
	i = np.array(i)
	energy_list.append(energy(global_weights,i))
	if(energy(global_weights,i) < -40):
		print("minimum: " + str(i) + " value: " + str(energy(global_weights,i)))
	prediction, epoch = DHNN.recall(np.array(i))
	if prediction not in attractors_set:
		attractors_set.append(prediction)
	if epoch not in epoches_set:
		epoches_set.append(epoch)
	if( np.array_equal(i,x1)):
		index.append('x1')
	elif( np.array_equal(i,x2)):
		index.append('x2')
	elif( np.array_equal(i,x3)):
		index.append('x3')
	else:
		index.append(" ")
	counter += 1
print("attractors_set:")
for i in attractors_set:
	print(i)
print("epoches_set:",epoches_set)
plt.figure(0)
plt.title("all possible patterns and their energies")
plt.ylabel('energy')
plt.xlabel('index')
plt.plot(range(256),energy_list)

minimas = np.array([[ 1,  1,  1,  1,  1, -1,  1,  1] 
 ,[ 1,  1, -1,  1,  1, -1,  1, -1] 
 ,[ 1,  1, -1,  1, -1,  1,  1, -1] 
 ,[ 1, -1, -1,  1,  1, -1,  1, -1] 
 ,[-1,  1,  1, -1, -1,  1, -1,  1] 
 ,[-1, -1,  1, -1,  1, -1, -1,  1] 
 ,[-1, -1,  1, -1, -1,  1, -1,  1] 
 ,[-1, -1, -1, -1, -1,  1, -1, -1]])
print("Recall patterns from all found minimas")
predictions, epochs = DHNN.recall_list(minimas)
print("predictions:")
print(predictions)
for i in range(len(minimas)):
	print("Accuracy:  ", accuracy(np.array(attractors_set),predictions[i]))
	print("Epochs:    ", epochs[i])
print("\n")

st1 = np.array([ 1,  1, -1,  1,  1,  1,  1,  1])
st2 = np.array([ 1,  1, -1,  1, -1, -1,  1,  1])
print("<Energy of st1: ",energy(global_weights,st1))
print("<Energy of st2: ",energy(global_weights,st2))

#Conclusion: There are 6 attractors in total, x1, x2, x3 and their opposites.

#3.2

pic_data = np.genfromtxt(fname = "pict.dat", delimiter = ",")
pic_data = pic_data.reshape(11,1024)
plt.figure(1)
for i in range(11):
	plt.subplot(3,4,1 + i)
	pic1 = pic_data[i].reshape(32,32)
	plt.imshow(pic1, cmap=cm.gray)
	plt.title(("Picture"+str(i+1)))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.6)
# training_patterns = np.array([pic_data[0,:]])
training_patterns = pic_data[0:3,:]
DHNN_3_2 = DHNN_bidirectional()
DHNN_3_2.fit(training_patterns,update_mode = "asynchronous")

recall_from_image = 11
print("Recall patterns from picture" + str(recall_from_image) )
prediction, epochs = DHNN_3_2.recall(pic_data[recall_from_image - 1,:],update_mode = "asynchronous")
print("Accuracy:  ", accuracy(training_patterns,prediction))
print("Epochs:    ", epochs)
print("\n")
plt.figure(2)
prediction = np.array(prediction).reshape(32,32)
plt.imshow(prediction, cmap=cm.gray)
plt.title("Recall from picture" + str(recall_from_image))


# section 3.4
accuracy_list = []
epochs_list   = []

accuracy_list_1 = []
epochs_list_1   = []

accuracy_list_2 = []
epochs_list_2   = []

for i in range(1024):
	distorted_pic = np.copy(pic_data[0,:])
	distorted_pic_1 = np.copy(pic_data[1,:])
	distorted_pic_2 = np.copy(pic_data[2,:])

	index = random.sample(range(1024),i)
	for j in index:
		distorted_pic[j] = distorted_pic[j] *(-1)
		distorted_pic_1[j] = distorted_pic_1[j] *(-1)
		distorted_pic_2[j] = distorted_pic_2[j] *(-1)


	prediction, epoches = DHNN_3_2.recall(distorted_pic,update_mode = "synchronous")
	prediction_1, epoches_1 = DHNN_3_2.recall(distorted_pic_1,update_mode = "synchronous")
	prediction_2, epoches_2 = DHNN_3_2.recall(distorted_pic_2,update_mode = "synchronous")

	accuracy_list.append(accuracy_1(training_patterns[0,:],prediction))
	epochs_list.append(epoches)

	accuracy_list_1.append(accuracy_1(training_patterns[1,:],prediction_1))
	epochs_list_1.append(epoches_1)

	accuracy_list_2.append(accuracy_1(training_patterns[2,:],prediction_2))
	epochs_list_2.append(epoches_2)
plt.figure(3)
plt.suptitle("Distortion Resistance")

x_cord = [ i/1024 for i in range(1024)]
plt.subplot(3,1,1)
plt.plot(x_cord,accuracy_list, label = "attractor1")
plt.xlabel("noise portion")
plt.ylabel("accuracy")
plt.xticks(np.arange(0, 1, 0.1))
plt.legend()
plt.subplot(3,1,2)
plt.plot(x_cord,accuracy_list_1, label = "attractor2")
plt.xlabel("noise portion")
plt.ylabel("accuracy")
plt.xticks(np.arange(0, 1, 0.1))
plt.legend()
plt.subplot(3,1,3)
plt.plot(x_cord,accuracy_list_2, label = "attractor3")
plt.xlabel("noise portion")
plt.ylabel("accuracy")
plt.xticks(np.arange(0, 1, 0.1))
plt.legend()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.2)

plt.figure(4)
plt.title("Distortion Resistance")
plt.xlabel("epoches")
plt.ylabel("accuracy")
plt.xticks(np.arange(0, 1, 0.1))
plt.plot(x_cord,epochs_list)

## Section 3.6


def generate_patterns(num_patterns,num_neurons, activity):
	ones = np.ones(int(num_neurons*num_patterns*activity))
	zeros = np.zeros(num_patterns*num_neurons - int(num_neurons*num_patterns*activity))
	return_patterns = np.append(ones,zeros)
	np.random.shuffle(return_patterns)
	return_patterns = return_patterns.reshape(num_patterns,num_neurons)
	return return_patterns



num_patterns = 30

memory_ability_01 = []
memory_ability_005 = []
memory_ability_001 = []
memory_ability_all = [memory_ability_01,memory_ability_005,memory_ability_001]

average_activity = [0.1,0.05,0.01]
plt.figure(5)
plt.title("Sparse Patterns ")
cita_bias_list = [ i/10 for i in range(-50,155,5)]
for k in range(3):
	new_patterns = generate_patterns(num_patterns , 100, average_activity[k])
	DHNN_3_6 = DHNN_binary()
	weights = DHNN_3_6.fit(new_patterns, average_activity[k])

	for cita_bias in cita_bias_list:
		memory_ability = 0
		for i in range(num_patterns):
			cur_pat = np.copy(new_patterns[i,:])
			pre = 0.5 + sign(weights @ cur_pat - cita_bias)/2
			if(accuracy_1(cur_pat,pre) == 1.0):
				memory_ability += 1
		# print("memory_ability: ", memory_ability)
		memory_ability_all[k].append(memory_ability)
	plt.plot(cita_bias_list,memory_ability_all[k], label="average activity = "+ str(average_activity[k]))
plt.legend()
plt.xlabel('bias')
plt.ylabel('number of memorized patterns')







plt.show()
