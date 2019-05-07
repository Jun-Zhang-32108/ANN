import numpy as np
import itertools
def sign(x):
    signals = np.sign(x)
    signals[signals == 0] = 1
    return signals

def accuracy(outputs, targets):
    return 1.0 - np.count_nonzero(outputs - targets, axis=1)/outputs.shape[1]

class hopfield_network:
    def __init__(self):
        self.weights = None

    def fit(self, patterns):
        self.weights = patterns.T @ patterns
        # np.fill_diagonal(self.weights, 0)
        # self.weights[np.diag_indices(patterns.shape[1])] = 0
        w,v = np.linalg.eig(self.weights)
        print("weights:")
        print(self.weights)
        print("eigenvalues: ",w)
    #function to be called on each pattern
    def associate_pattern(self, pat):
        minima = False
        prev_pred = sign(self.weights @ pat)
        # print("prev_pred:",prev_pred)
        step_count = 0

        while not minima:
            curr_pred = sign(self.weights @ prev_pred)

            minima = np.array_equal(prev_pred, curr_pred)
            prev_pred = curr_pred

            step_count += 1
        return curr_pred, step_count

    #function to work on a list of patterns
    def associate(self, patterns):
        preds = np.apply_along_axis(lambda pat: self.associate_pattern(pat), 1, patterns)
        associations = np.vstack(preds[:, 0])
        epochs = np.array(preds[:, 1])
        return associations, epochs


patterns = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],
                     [-1, -1, -1, -1, -1, 1, -1, -1],
                     [-1, 1, 1, -1, -1, 1, -1, 1]])

hopfield_nn = hopfield_network()
hopfield_nn.fit(patterns)

#associate patterns with themselves
associations, epochs = hopfield_nn.associate(patterns)
print(associations)
print(accuracy(patterns, associations))
print(epochs)
print('\n')

#the network was able to store all the patterns


distorted_patterns = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
                               [1, 1, -1, -1, -1, 1, -1, -1],
                               [1, 1, 1, -1, 1, 1, -1, 1]])

#associate distorted patterns
associations, epochs = hopfield_nn.associate(distorted_patterns)
print(associations)
print(accuracy(patterns, associations))
print(epochs)
print('\n')

# all the pattern converged, however, the second one was not correct. It converged to the opposite of the first stored pattern.
# There are 6 attractors, the stored patterns and their opposites.

more_distorted_pattern = np.array([[1, 1, 1, 1, 1, 1, 1, 1]]) #5 changed from the first pattern
# x3d = np.array([[1, 1, 1, -1, 1, 1, -1, 1]]) #5 changed from the first pattern
#associate distorted patterns
associations, epochs = hopfield_nn.associate(more_distorted_pattern)
print(associations)
print(accuracy(patterns, associations))
print(epochs)

# here, we converged to the opposite of the 2nd pattern
# 
attractors_set = []
epoches_set    = []
print("Recall from all possible patterns and find all attractors:")
for i in itertools.product([1,-1],repeat = 8):
    prediction, epoch = hopfield_nn.associate_pattern(np.array(i))
    if list(prediction) not in attractors_set:
        attractors_set.append(list(prediction))
    if epoch not in epoches_set:
        epoches_set.append(epoch)
print("attractors_set:")
for i in attractors_set:
    print(i)
print("epoches_set:",epoches_set)