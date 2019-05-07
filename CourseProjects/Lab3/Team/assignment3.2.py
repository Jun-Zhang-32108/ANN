import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
        # self.weights = sign(self.weights)
        np.fill_diagonal(self.weights, 0)
        print(self.weights)

    def energy_func(self, pat):
        sum = 0
        for i in range(len(pat)):
            for j in range(len(pat)):
                sum += self.weights[i][j] * pat[i] * pat[j] 
        return -1*sum

    def associate_async(self, pat, iteration_count=8000):
        energy_list = []
        for ind in range(iteration_count):
            radnom_index = np.random.choice(pat.shape[0])
            update_val = self.weights[radnom_index] @ pat
            pat[radnom_index] = 1 if update_val >= 0 else -1
            if ind % 1000 == 0:
                energy_list.append((self.energy_func(pat)))
                plt.imsave('p' + str(ind) + '.png', pat.reshape(32, 32), cmap=cm.gray)
        index = [(i)* 1000 for i in range(8)]
        plt.figure(1)
        plt.xlabel("iteration")
        plt.ylabel("energy")
        plt.title("Asynchronous Update Mode - Random Update Neuron")
        plt.plot(index, energy_list)
        return pat

        # update the weights in a fixed order and plot some of the recovered figures.
        # for ind in range(iteration_count):
        #     for i in range(len(pat)):
        #         update_val = self.weights[i] @ pat
        #         pat[i] = 1 if update_val >= 0 else -1
        #     if ind % 10 == 0:
        #         plt.subplot(2,5,1+int(ind/10))
        #         plt.imshow(pat.reshape(32,32),cmap = cm.gray)
        #         plt.title(("Picture"+str(int(ind/10)+1)))
        #         plt.imsave('p' + str(ind) + '.png', pat.reshape(32, 32), cmap=cm.gray)
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
        #         wspace=None, hspace=0.6)
        # return pat

    def associate_sync(self, pat):
        minima = False
        prev_pred = sign(self.weights @ pat)
        step_count = 0

        while not minima:
            curr_pred = sign(self.weights @ prev_pred)
            minima = np.array_equal(prev_pred, curr_pred)
            prev_pred = curr_pred

            step_count += 1
        return curr_pred, step_count

    def associate(self, patterns, update_rule="sync"):
        if update_rule == "async":
            associations = np.apply_along_axis(lambda pat: self.associate_async(pat), 1, patterns)
            return associations, None
        elif update_rule == "sync":
            preds = np.apply_along_axis(lambda pat: self.associate_sync(pat), 1, patterns)
            associations = np.vstack(preds[:, 0])
            epochs = np.array(preds[:, 1])
            return associations, epochs


image_data = np.genfromtxt('pict.dat', delimiter=',')
image_data = np.reshape(image_data, (11, 1024))

patterns = image_data[:4]
for i in range(len(patterns)):
    plt.imsave('p' + str(i) + '.png', patterns[i].reshape(32,32), cmap=cm.gray)

p10 = image_data[[9]]
p11 = image_data[[10]]


hopfield_nn = hopfield_network()
hopfield_nn.fit(patterns)

associations, epochs = hopfield_nn.associate(p10,update_rule="async")

#should be similar to the first image
print(accuracy(patterns[[0]], associations))
print(epochs)
print('\n')


plt.imsave('p' + str(11) + '.png', image_data[10].reshape(32,32), cmap=cm.gray)
plt.imsave('assoc.png', np.array(associations[0]).reshape(32,32), cmap=cm.gray)
plt.figure(2)
plt.imshow(np.array(associations[0]).reshape(32,32), cmap=cm.gray)

plt.show()

