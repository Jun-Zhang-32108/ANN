#Author: Balint Kovacs, Jun Zhang
#RBF network
import numpy as np


INTERVAL_START = 0.0
INTERVAL_END = 1.0

class rbf_network_with_som:
    def __init__(self):
        self.weights = None
        self.centers = None

    def fit(self, patterns, center_count, cl_epochs=20, centers_learning_rate=0.4):
        neighbourhood = 50
        self.init_centers(center_count, patterns.shape[1])
        for i in range(cl_epochs):
            shuffled_idxs = np.random.permutation(patterns.shape[0])
            for idx in shuffled_idxs:
                curr_pattern = patterns[idx]
                winning_ind = np.argmin(np.linalg.norm(self.centers-curr_pattern, axis=1))
                # upper_bound should + 1?
                lower_bound = winning_ind-neighbourhood if winning_ind-neighbourhood > 0 else 0
                upper_bound = winning_ind+neighbourhood if winning_ind+neighbourhood < self.centers.shape[0] else self.centers.shape[0]
                centers_to_update = self.centers[lower_bound:upper_bound]
                # here the update_vector/delta weight is the same for all neightbours. We can use other methods to update the nieghtbourhood as well which 
                # will give different updates to different neightbours according their distance to the winning center
                update_vector = centers_learning_rate*(curr_pattern - centers_to_update)
                centers_to_update += update_vector

            # weird update rule based on epochs
            if i < 4:
                neighbourhood -= 10
            elif i<9:
                neighbourhood = 5
            elif i<14:
                neighbourhood = 3
            else:
                neighbourhood = 1

    def order(self, patterns):
        closest_centers = dict()
        for idx, curr_pattern in enumerate(patterns):
            winning_center_idx = np.argmin(np.linalg.norm(self.centers-curr_pattern, axis=1))
            closest_centers[idx] = winning_center_idx
        # ordering dictionary based on keys
        return sorted(closest_centers, key=lambda k: closest_centers[k])

    def init_centers(self, center_count, pattern_dims):
        self.centers = np.random.uniform(low=INTERVAL_START, high=INTERVAL_END, size=(center_count,pattern_dims))


animal_names = np.loadtxt('animalnames.txt', dtype='object')
animal_attributes = np.loadtxt('animalattributes.txt', dtype='object')
animals_data = np.genfromtxt('animals.dat', delimiter=',')
animals_data = animals_data.reshape(animal_names.shape[0], animal_attributes.shape[0])


rbf_nn_cl = rbf_network_with_som()
rbf_nn_cl.fit(animals_data, center_count=100)
ordered_index = rbf_nn_cl.order(animals_data)
print(ordered_index)
print(animal_names[ordered_index])