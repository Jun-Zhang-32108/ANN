#Author: Balint Kovacs, Jun Zhang
#RBF Network
import numpy as np
import io
import matplotlib.pyplot as plt



INTERVAL_START = 0.0
INTERVAL_END = 1.0

class rbf_network_with_som:
    def __init__(self):
        self.weights = None
        self.centers = None

    def fit(self, patterns, center_count, cl_epochs=20, centers_learning_rate=0.2):
        neighbourhood = 2
        self.init_centers(center_count, patterns.shape[1])
        for i in range(cl_epochs):
            shuffled_idxs = np.random.permutation(patterns.shape[0])
            for idx in shuffled_idxs:
                curr_pattern = patterns[idx]
                winning_ind = np.argmin(np.linalg.norm(self.centers-curr_pattern, axis=1))

                #weird indexing to work in a cilcular way
                #why + 1 here?
                lower_bound = winning_ind-neighbourhood
                upper_bound = winning_ind+neighbourhood+1
                if upper_bound > center_count:
                    neighbourhood_indices = list(range(lower_bound, center_count))
                    neighbourhood_indices += list(range(upper_bound-center_count))
                else:
                    neighbourhood_indices = list(range(lower_bound, upper_bound))

                update_vector = centers_learning_rate*(curr_pattern - self.centers[neighbourhood_indices])
                self.centers[neighbourhood_indices] += update_vector

            if i < 7:
                neighbourhood = 2
            elif i < 17:
                neighbourhood = 1
            else:
                neighbourhood = 0

    def order(self, patterns):
        closest_centers = dict()
        for idx, curr_pattern in enumerate(patterns):
            winning_center_idx = np.argmin(np.linalg.norm(self.centers-curr_pattern, axis=1))
            closest_centers[idx] = winning_center_idx
        # ordering dictionary based on keys
        return sorted(closest_centers, key=lambda k: closest_centers[k])

    def init_centers(self, center_count, pattern_dims):
        self.centers = np.random.uniform(low=INTERVAL_START, high=INTERVAL_END, size=(center_count,pattern_dims))

#the file has both ; and , as delimiter, and it is difficult to read that
s = io.BytesIO(open('cities.dat', 'rb').read().replace(b';',b''))

cities_data = np.genfromtxt(s, dtype='float', skip_header=4, delimiter=",")

rbf_nn_cl = rbf_network_with_som()
rbf_nn_cl.fit(cities_data, center_count=10)
ordered_index = rbf_nn_cl.order(cities_data)
print(ordered_index)
print(cities_data)
route = cities_data[ordered_index]
print(route)

route = np.append(route, [route[0]], axis=0)
print(route)


plt.scatter(cities_data[:, 0], cities_data[:, 1])
plt.plot(route[:, 0], route[:, 1])
plt.show()

