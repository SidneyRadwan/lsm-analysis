import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D
from pylab import cm


class LSM:
    def __init__(self, dimensions):

        # Simulation constants
        self.time_step = 1 # (ms)
        self.stop_time = 1000 # (ms)

        self.conn_lambda = 2 # connectivity constant
        self.threshold_v = 15 # (mV)
        self.reset_v = 13.5 # (mV)
        self.t_memb = 30 # (ms)
        self.V_b = 13.5 # potential (mV) from nonspecific background current (13.5nA) * input resistance (1MOhm)

        self.x_dim, self.y_dim, self.z_dim = dimensions
        self.n_neurons = self.x_dim * self.y_dim * self.z_dim

        self.n_inhibitory = self.n_neurons // 5 # 20% inhibitory
        self.n_exhitatory = self.n_neurons - self.n_inhibitory

        # LSM neuron arrays and parameter arrays
        self.e_indices = None
        self.i_indices = None
        # access like liquid_v[self.e_indices]

        self.connections = np.zeros((self.n_neurons, self.n_neurons), dtype="int")
        self.U = np.zeros((self.n_neurons, self.n_neurons), dtype="float32")
        self.A = np.zeros((self.n_neurons, self.n_neurons), dtype="float32")
        self.t_rec = np.zeros((self.n_neurons, self.n_neurons), dtype="float32")
        self.t_fac = np.zeros((self.n_neurons, self.n_neurons), dtype="float32")

        # Data simulation will collect
        self.spikes_data = [] # 135*total_time_steps size array
        self.time_data = [] # 1*total_time_steps size array
        self.avg_v_data = [] # average membrane potential at time t
        self.spike_number = [] # vector indices of spikes
        self.spike_time_data = [] # time of spikes
        self.liquid_state_data = [] # the liquid state of the LSM at time t i.e X_M(t)
        self.conn_distances = []

        # Liquid neuron's supporting arrays -> set to initial values
        self.refractory_time = np.zeros((self.n_neurons, 1), dtype="float32") # reset time of liquid neurons
        self.spike_times = np.full((self.n_neurons, 1), np.NINF, dtype="float32") # time of last spike
        self.R = np.full((self.n_neurons, self.n_neurons), 1, dtype="float32") # available synaptic efficacy
        self.u = self.U # running value of utilisation of synaptic efficacy (~ likelihood of synaptic fire)
        self.liquid_v = np.zeros((self.n_neurons, 1), dtype="float32") # membrane voltage of liquid neurons
        # Sample liquid_v from uniform dist within [13.5mV, 15.0mV]
        for i in range(self.n_neurons):
            self.liquid_v[i] = random.uniform(self.reset_v, self.threshold_v)

        # Input neuron's supporting arrays -> set to initial values
        self.input_A = np.zeros((self.n_neurons, 1), dtype="float32") # scaling parameter for input neurons
        self.input_A[self.e_indices] = np.random.gamma(18, 18) # draw input_A from gamma distribution (nA * MOhm => mV) i.e * input resistance
        self.input_A[self.i_indices] = np.random.gamma(9, 9) # draw input_A from gamma distribution (nA * MOhm => mV) i.e * input resistance
        self.input_spike_times = np.full((self.n_neurons, 1), np.NINF, dtype="float32") # time of last spike
        self.input_v = np.zeros((self.n_neurons, 1), dtype="float32") # membrane voltage of input neurons
        self.t_syn = np.zeros((self.n_neurons, 1), dtype="float32") # input post synaptic current decay rate
        self.t_syn[self.e_indices] = 3 # (ms)
        self.t_syn[self.i_indices] = 6 # (ms)


    def index_to_3D_coords(self, k):
        """
        Returns vector index converted to Eucliedean coordinates
        """
        x = k % self.x_dim
        y = (k // self.x_dim) % self.y_dim
        z = (k // (self.x_dim * self.y_dim)) % self.z_dim
        return (x, y, z)

    def distance(self, a, b) -> float:
        """
        Returns euclidean distance between two 3D points a and b.
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

    def initialise_lsm(self):
        """
        Initialises neurons, connections and parameter arrays A, U, t_rec & t_fac.
        """

        # Initialise neuron types (exhitatory or inhibitory)
        shuffled_neurons = [i for i in range(0, self.n_neurons)]
        random.shuffle(shuffled_neurons)
        self.i_indices = shuffled_neurons[:self.n_inhibitory]
        self.e_indices = [i for i in range(0, self.n_neurons) if i not in self.i_indices]

        max_dist = np.NINF

        # Iterate over every pair of neurons i & j
        for i in range(0, self.n_neurons):
            for j in range(0, self.n_neurons):

                # Set parameter values for the connection types
                if i in self.e_indices and j in self.e_indices: # (ee)
                     C = 0.3
                     U_mean, U_SD = 0.5, 0.25
                     t_rec_mean, t_rec_SD = 1100, 550
                     t_fac_mean, t_fac_SD = 50, 25
                     A_mean, A_SD = 30, 15
                elif i in self.e_indices and j in self.i_indices: # (ei)
                     C = 0.2
                     U_mean, U_SD = 0.05, 0.025
                     t_rec_mean, t_rec_SD = 125, 62.5
                     t_fac_mean, t_fac_SD = 1200, 600
                     A_mean, A_SD = 60, 30
                elif i in self.i_indices and j in self.e_indices: # (ie)
                     C = 0.4
                     U_mean, U_SD = 0.25, 0.125
                     t_rec_mean, t_rec_SD = 700, 350
                     t_fac_mean, t_fac_SD = 20, 10
                     A_mean, A_SD = -19, 9.5
                elif i in self.i_indices and j in self.i_indices: # (ii)
                     C = 0.1
                     U_mean, U_SD = 0.32, 0.16
                     t_rec_mean, t_rec_SD = 144, 72
                     t_fac_mean, t_fac_SD = 60, 30
                     A_mean, A_SD = -19, 9.5

                # Compute and store random U from normal (Gaussian) distribution, if < 0 set to mean
                U_value = np.random.normal(U_mean, U_SD)
                self.U[i][j] = U_value if U_value > 0 else U_mean

                # Repeat for t_rec
                t_rec_value = np.random.normal(t_rec_mean, t_rec_SD)
                self.t_rec[i][j] = t_rec_value if t_rec_value > 0 else t_rec_mean

                # Repeat for t_fac
                t_fac_value = np.random.normal(t_fac_mean, t_fac_SD)
                self.t_fac[i][j] = t_fac_value if t_fac_value > 0 else t_fac_mean

                # Repeat for A
                A_value = np.random.normal(A_mean, A_SD)
                self.A[i][j] = A_value if A_value > 0 else A_mean

                # Convert vector indices of neurons i & j to Euclidian coordinates
                coords_i = self.index_to_3D_coords(i)
                coords_j = self.index_to_3D_coords(j)

                # Compute and store if there is a random connection between neurons i & j
                conn_prob = C * np.exp(- (self.distance(coords_i, coords_j) / self.conn_lambda) ** 2)
                if random.random() < conn_prob:
                    self.connections[i][j] = 1
                    self.conn_distances.append(self.distance(coords_i, coords_j))
                    if self.distance(coords_i, coords_j) > max_dist: max_dist = self.distance(coords_i, coords_j)

        # Set running variable u to updated U
        self.u = self.U
        print("Max connection distance: %.3f " % max_dist)


    def reset_simulation(self):
        """
        Resets data collection arrays and supporting simulation arrays.
        """
        self.spikes_data = []
        self.time_data = []
        self.avg_v_data = []
        self.spike_number = []
        self.spike_time_data = []
        self.liquid_state_data = []
        self.refractory_time = np.zeros((self.n_neurons, 1), dtype="float32")
        self.R = np.full((self.n_neurons, self.n_neurons), 1, dtype="float32")
        self.spike_times = np.full((self.n_neurons, 1), np.NINF, dtype="float32")
        self.input_spike_times = np.full((self.n_neurons, 1), np.NINF, dtype="float32")
        self.input_v = np.zeros((self.n_neurons, 1), dtype="float32")
        self.u = self.U
        for i in range(self.n_neurons):
            self.liquid_v[i] = random.uniform(self.reset_v, self.threshold_v)


    def run_simulation(self, spike_trains=None, plot=True, reset=True):
        """
        Runs simulation.
        """

        if reset == True:
            self.reset_simulation()

        t = 0 # (ms)
        print("Simulation Running")
        while t < self.stop_time:

            # Add spike train input at time t if present
            if spike_trains is not None:
                self.input_spike_times[spike_trains[t]] = t


            #print(np.amax(self.u), np.amin(self.u))

            # Compute updates for synaptic dynamics
            if t > 0:

                self.input_v = self.input_A * np.exp( - (t - self.input_spike_times) / self.t_syn)

                self.u = self.u * np.exp( - (t - self.spike_times) / self.t_fac) + self.U * ( 1 - self.u * np.exp( - (t - self.spike_times) / self.t_fac))

                self.R = self.R * (1 - self.u) * np.exp( - (t - self.spike_times) / self.t_rec) + 1 - np.exp( - (t - self.spike_times) / self.t_rec)

                # EPSP's sum (across rows) to create a combined potential + voltage from background current & input neurons
                EPSPs = np.sum( self.A * self.R * self.u * self.connections* np.exp( - (t - self.spike_times) / self.t_memb), axis=1, keepdims=True)

                #self.liquid_v = EPSPs - self.liquid_v * np.exp( - (t - self.spike_times) / self.t_memb)
                self.liquid_v = EPSPs  + self.V_b + self.input_v

            # Create empty spikes array to store the spikes this iteration
            spikes = np.zeros((self.n_neurons, 1), dtype="int")

            for i in range(self.n_neurons):
                if self.liquid_v[i] >= self.threshold_v and t >= self.refractory_time[i]:
                    #print(f"Neuron: {i} ,fired at time: {t}")
                    self.spike_number.append(i)
                    self.spike_time_data.append(t)
                    spikes[i] = 1
                    self.spike_times[i] = t

                    self.liquid_v[i] = self.reset_v
                    if i in self.i_indices:
                        self.refractory_time[i] = t + 2
                    else:
                        self.refractory_time[i] = t + 3

            # Store this iteration's data
            self.spikes_data.append(spikes)
            self.liquid_state_data.append(self.liquid_v)
            self.avg_v_data.append(np.mean(self.liquid_v))
            self.time_data.append(t)

            t += self.time_step

        print("Simulation Complete")
        print(f"Total spikes: {np.sum(self.spikes_data)}")

        if plot == True:
            self.plot_simulation()



    def plot_simulation(self):
        """
        Plot simulation.
        """

        spikes_data = np.squeeze(self.spikes_data)

        plt.figure(figsize = (14,8))
        N_SUBPLOTS = 3
        gs = gridspec.GridSpec(N_SUBPLOTS, 1)

        plt.subplot(gs[0, 0])
        plt.title("Spikes Raster Plot")

        plt.scatter(self.spike_time_data, self.spike_number, marker="|", s=5.)
        #plt.scatter(spike_time_data, spike_number, s=5.)
        plt.xlim([0, self.stop_time])
        plt.ylim([0, self.n_neurons])
        #plt.xlabel("Connection Distance")
        plt.ylabel("Neuron Number")

        plt.subplot(gs[1, 0])
        plt.title("# Fires")
        plt.plot(self.time_data, sum(spikes_data.T))
        plt.xlim([0, self.stop_time])
        #plt.xlabel("Time (ms)")
        plt.ylabel("Number of Fires")

        plt.subplot(gs[2, 0])
        plt.title("AVG Membrane Voltage")
        plt.plot(self.time_data, self.avg_v_data)
        plt.xlim([0, self.stop_time])
        plt.xlabel("Time (ms)")
        plt.ylabel("Average Membrane Voltage (mV)")

        plt.show()


    def plot_connection_distances(self):
        """
        Plot distribution of neural connection distances.
        """
        unique, counts = np.unique(self.conn_distances, return_counts=True)
        plt.figure(figsize = (10,6))
        plt.title("Neural Connection Distance Distribution")
        plt.scatter((unique), (counts))
        plt.xlabel("Connection Distance")
        plt.ylabel("Frequency of Connection Distance")
        plt.show()


    def liquid_state_distance_test(self):

        num_iters = 200

        plt.figure(figsize = (14,8))
        plt.title(f"State Distance (100Hz) Iterations = {num_iters}")
        colours = ["r", "b", "g", "m", "y"]
        plt.xlabel("Time (ms)")
        plt.ylabel("State Distance")
        labels = ["d(u,v)=", "d(u,v)=", "d(u,v)=", "d(u,v)="]
        offsets = [0, 1, 2, 4]

        for v in range(4):

            difference_norms = []

            spike_train_1 = []
            spike_train_2 = []
            spike_1_times = []
            spike_2_times = []
            for t in range(self.stop_time):
                spikes_1 = []
                spikes_2 = []
                # Create 100Hz spike trains for first and last 10 neurons
                if t % 50 == 0: # (1/100Hz) * 1000ms = 10ms
                    spikes_1 = [40]
                    spike_1_times.append(t)
                if t % 50 == 0 + offsets[v]:
                    spikes_2 = [40]
                    spike_2_times.append(t)
                spike_train_1.append(spikes_1)
                spike_train_2.append(spikes_2)

            def gaussian(x , s):
                return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )
            s = 2.09

            myGaussian = np.fromiter( (gaussian( x , 1 ) for x in range( -3, 4, 1 ) ), "float" )

            conv_1 = np.convolve( spike_1_times, myGaussian, mode='same' )
            conv_2 = np.convolve( spike_2_times, myGaussian, mode='same' )

            diff = np.linalg.norm(conv_1 - conv_2)
            print()
            print()
            labels[v] += "%.2f" % diff
            print("%.2f" % diff)
            print()
            print()
            for n in range(num_iters):

                # Run simulation
                self.run_simulation(spike_train_1, plot=False, reset=True)
                liquid_state_data_1 = np.squeeze(self.liquid_state_data)

                # Re-initialise & reset LSM and run simulation again
                self.initialise_lsm()
                self.run_simulation(spike_train_2, plot=False, reset=True) # reset needed for second run
                liquid_state_data_2 = np.squeeze(self.liquid_state_data)

                # Compute Eucliedean norm
                difference_norm = np.linalg.norm(liquid_state_data_2 - liquid_state_data_1, axis=1)
                difference_norms.append(difference_norm)


            avg_difference_norms = np.average(difference_norms, axis=0)

            plt.plot(avg_difference_norms, colours[v], label=labels[v])

        plt.legend(loc="lower left", fontsize=10)
        plt.show()


    def lsm_slice_analysis(self, plane="xy", slice_index="0", mode="fires"):
        """
        Visual analysis of neuron fires or potentials in LSM network slices.
        """

        plt.ion() # make plot interactive (interactable-on) e.g pausable
        plt.figure(figsize = (8, 8))

        for t in range(self.stop_time):
            if mode == "fires":
                data_array = self.spikes_data[t]
            elif mode == "potentials":
                data_array = self.liquid_state_data[t]
                max_v = np.max(self.liquid_state_data)

            data_array = np.reshape(data_array, (self.x_dim, self.y_dim, self.z_dim))

            if plane == "xy":
                z_index = slice_index
                slice_array = data_array[:, :, z_index]
            elif plane == "xz":
                y_index = slice_index
                slice_array = data_array[:, y_index, :]
            elif plane == "yz":
                x_index = slice_index
                slice_array = data_array[x_index, :, :]

            if mode == "fires":
                plt.imshow(slice_array, cmap="plasma", interpolation='nearest')
                plt.title(f'Neural Fires Plane \'{plane}\', Slice Index = {slice_index}: t = {t}')
                plt.clim(0, 1)
            elif mode == "potentials":
                plt.imshow(slice_array, cmap="viridis", interpolation='nearest')
                plt.title(f'Membrane Potentials for Plane \'{plane}\', Slice Index = {slice_index}: t = {t}')
                plt.clim(0, max_v)
            plt.colorbar()
            plt.pause(0.0001)
            plt.clf()

        if mode == "fires":
            temp = np.reshape(self.spikes_data, (self.stop_time, self.x_dim, self.y_dim, self.z_dim))
        elif mode == "potentials":
            temp = np.reshape(self.liquid_state_data, (self.stop_time, self.x_dim, self.y_dim, self.z_dim))

        if plane == "xy":
            average_array = np.average(temp[:][:, :, z_index], axis=0)
        elif plane == "xz":
            average_array = np.average(temp[:][:, y_index, :], axis=0)
        elif plane == "yz":
            average_array = np.average(temp[:][x_index, :, :], axis=0)

        if mode == "fires":
            plt.imshow(average_array, cmap="plasma", interpolation='nearest')
            plt.title(f'Average # Neural Fires for Plane \'{plane}\', Slice Index = {slice_index}')
            plt.clim(0, 1)

        elif mode == "potentials":
            plt.imshow(average_array, cmap="viridis", interpolation='nearest')
            plt.title(f'Average Membrane Potential for Plane \'{plane}\', Slice Index = {slice_index}')
            plt.clim(0, max_v)

        plt.colorbar()
        plt.show(block=True)



    def lsm_3D_analysis(self, mode="fires"):
        """
        Visual analysis of neuron fires or potentials in 3D LSM network.
        """
        if mode == "fires":
            #data = np.reshape(self.spikes_data, (self.stop_time, self.x_dim, self.y_dim, self.z_dim))
            data = self.spikes_data
        elif mode == "potentials":
            data = self.liquid_state_data

        plt.ion()
        # creating 3d figures
        fig = plt.figure(figsize=(8, 8))

        for t in range(self.stop_time):
            data_array = data[t]
            x_array = []
            y_array = []
            z_array = []
            values = []

            for i in range(self.n_neurons):
                if mode == "fires" and data_array[i] == 1:
                    x, y, z = self.index_to_3D_coords(i)
                    x_array.append(x)
                    y_array.append(y)
                    z_array.append(z)
                    values.append(1)
                elif mode == "potentials":
                    x, y, z = self.index_to_3D_coords(i)
                    x_array.append(x)
                    y_array.append(y)
                    z_array.append(z)
                    values.append(data_array[i])

            ax = Axes3D(fig)
            img = ax.scatter(x_array, y_array, z_array, marker='s',
                             s=100, c=values, cmap='plasma')

            fig.colorbar(img, ax=ax, shrink=0.5)

            # adding title and labels
            ax.set_title(f"3D Heatmap for {mode.capitalize()}: t = {t}")
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_xlim(0, self.x_dim)
            ax.set_ylim(0, self.y_dim)
            ax.set_zlim(0, self.z_dim)

            # displaying plot
            plt.show()
            plt.pause(0.001)
            plt.clf()

        # Plot average
        x_array = []
        y_array = []
        z_array = []
        values = []

        for i in range(self.n_neurons):
            x, y, z = self.index_to_3D_coords(i)
            x_array.append(x)
            y_array.append(y)
            z_array.append(z)
            values.append(np.average(data[:][i]))

        ax = Axes3D(fig)

        img = ax.scatter(x_array, y_array, z_array, marker='s',
                         s=100, c=values, cmap='hot')

        fig.colorbar(img, ax=ax, shrink=0.5)

        # adding title and labels
        ax.set_title(f"3D Heatmap of Averages for {mode.capitalize()}: t = {t}")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_xlim(0, self.x_dim)
        ax.set_ylim(0, self.y_dim)
        ax.set_zlim(0, self.z_dim)

        # displaying plot
        plt.show(block=True)


    def lsm_field_analysis(self, slice_index=0):
        """
        Computes and plots electric fields in the LSM network.
        """
        plane = "xy" # does not change plane yet
        slice_index = 0 # changes z_index

        plt.ion() # make plot interactive (interactable-on) e.g pausable
        plt.figure(figsize = (8, 8))

        # Pre-convert and store vector indices of neurons i & j to Euclidian coordinates
        coords = []
        for i in range(self.n_neurons):
            coords.append(self.index_to_3D_coords(i))

        # Pre-compute and store distances between neurons
        distance = np.zeros((self.n_neurons, self.n_neurons), dtype="float32")
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                distance[i][j] = self.distance(coords[i], coords[j])

        thetas = []
        slices_E_x = []
        slices_E_y = []
        for t in range(self.stop_time):

            E = np.zeros((self.n_neurons, 3), dtype="float32") # (x, y, z) components
            liquid_state = self.liquid_state_data[t]

            # Iterate over all pairs or neurons
            for i in range(self.n_neurons):
                for j in range(self.n_neurons):
                    v_i = liquid_state[i]
                    v_j = liquid_state[j]
                    coord_difference = np.array(coords[i]) - np.array(coords[j])
                    if i == j:
                        E[i] += 0
                    else:
                        v_sum = (v_i - v_j) * (coord_difference /  np.linalg.norm(coord_difference)) # x, y, z components
                        E[i] += v_sum / distance[i][j] # x, y, z components

            E_x = E[:, 0]
            E_y = E[:, 1]
            E_z = E[:, 2]

            E_x = np.reshape(E_x, (self.x_dim, self.y_dim, self.z_dim))
            E_y = np.reshape(E_y, (self.x_dim, self.y_dim, self.z_dim))
            E_z = np.reshape(E_z, (self.x_dim, self.y_dim, self.z_dim))

            # Take a slice off xy plane at slice_index = z_index
            slice_E_x = E_x[:, :, slice_index]
            slice_E_y = E_y[:, :, slice_index]

            # Normalise
            non_zeros = np.where(slice_E_x**2 + slice_E_y**2 != 0)
            slice_E_x[non_zeros] = slice_E_x[non_zeros] / np.sqrt(slice_E_x[non_zeros]**2 + slice_E_y[non_zeros]**2)
            slice_E_y[non_zeros] = slice_E_y[non_zeros] / np.sqrt(slice_E_x[non_zeros]**2 + slice_E_y[non_zeros]**2)
            slices_E_x.append(slice_E_x)
            slices_E_y.append(slice_E_y)

            # Compute theta
            theta = np.arctan2(slice_E_y, slice_E_x)
            thetas.append(theta)

            plt.imshow(theta, cmap="hsv", interpolation='nearest')
            plt.title(f"Thetas for Slice Index = {slice_index} on Plane = \'{plane}\': t = {t}")
            plt.colorbar()
            plt.clim(0, np.pi)
            for y in range(theta.shape[0]):
                 for x in range(theta.shape[1]):
                     plt.arrow(x, y, slice_E_x[y][x]*0.4, slice_E_y[y][x]*0.4,
                                length_includes_head=True, head_width=0.08, head_length=0.00002)
            plt.show()
            plt.pause(0.0001)
            plt.clf()


        avg_theta = np.average(thetas, axis=0)
        avg_slices_E_x = np.average(slices_E_x, axis=0)
        avg_slices_E_y = np.average(slices_E_y, axis=0)
        plt.imshow(avg_theta, cmap="hsv", interpolation='nearest')
        plt.title(f"Average Thetas for Slice Index = {slice_index} on Plane = \'{plane}\': t = {t}")
        for y in range(theta.shape[0]):
             for x in range(theta.shape[1]):
                 plt.arrow(x, y, avg_slices_E_x[y][x]*0.4, avg_slices_E_y[y][x]*0.4,
                            length_includes_head=True, head_width=0.08, head_length=0.00002)
        plt.colorbar()
        plt.clim(0, np.pi)
        plt.show(block=True)






def generate_spike_trains(total_neurons):
    # Booleans to create spike trains
    STRand = False # Set true to include randomised spike train
    ST20 = False # Set true to include 20Hz spike train
    ST40 = False # Set true to include 40Hz spike train
    ST100 = False # Set true to include 100Hz spike train
    STConst = True # Set true to include constant spike train
    STTrigger = True # Set true to include single spike train at t=0 (recommended -> True)

    # Create spike trains for simulation
    stop_time = 1000 # (ms)
    spike_trains = []
    possible_neurons = [i for i in range(total_neurons)]
    chosen_indices = random.sample(possible_neurons, k=random.randrange(total_neurons//5))
    for t in range(stop_time):
        spike_train = []

        # Add initial spike train trigger
        if STTrigger and t == 0:
            spike_train = [x for x in range(total_neurons)]
        # Add constant spike train
        elif STConst:
            #spike_train = [x for x in range(50)]
            spike_train = [i*15 for i in range(3*3)] # yz plane with x = 0
        # Add spike train of frequency 20Hz
        elif t % 50 == 0 and ST20: # (1/20Hz) * 1000ms = 50ms
            # spike_train = chosen_indices
            # spike_train = [x for x in range(total_neurons//10)]
            spike_train = [x for x in range(total_neurons)]
        # Add spike train of frequency 40Hz
        elif t % 25 == 0 and ST40: # (1/40Hz) * 1000ms = 25ms
            spike_train = [x for x in range(45) if x % 15 < 4]
        # Add spike train of frequency 100Hz
        elif t % 10 == 0 and ST100: # (1/100Hz) * 1000ms = 10ms
            spike_train = [x for x in range(50)]
        # Add randomised spike trains
        elif STRand:
            possible_neurons = [i for i in range(total_neurons)]
            chosen_indices = random.sample(possible_neurons, k=random.randrange(total_neurons//2))
            spike_train = chosen_indices

        spike_trains.append(spike_train)
    return spike_trains


def main():

    x_dim = 15 # array size in x dimension
    y_dim = 3 # array size in y dimension
    z_dim = 3 # array size in z dimension
    dimensions = (x_dim, y_dim, z_dim)
    total_neurons = x_dim * y_dim * z_dim

    spike_trains = generate_spike_trains(total_neurons)

    lsm = LSM(dimensions)

    OPTION = 5

    if OPTION == 1 or OPTION == 2 or OPTION == 5:
        # Run normal simulation
        lsm.initialise_lsm()
        print(f"Number of connections: {np.sum(lsm.connections)}")
        # Run simulation
        lsm.run_simulation(spike_trains, plot=True, reset=False) # reset not needed on first run


    if OPTION == 2:
        # Run slice analysis
        planes = ["xy", "xz", "yz"]
        plane = planes[0]
        slice_index = 1 # for plane = "xy" this would be 0 <= z-index <= z_dim
        modes = ["fires", "potentials"]
        mode = modes[0]
        print(f"Mode is:", mode, "Plane is:", plane)
        lsm.lsm_slice_analysis(plane, slice_index, mode)


    if OPTION == 3:
        # Run state distance test/plot
        lsm.initialise_lsm()
        print(f"Number of connections: {np.sum(lsm.connections)}")
        lsm.liquid_state_distance_test()

    if OPTION == 4:
        # Run simulation trials for average measurements
        avg_pot = []
        avg_num_spikes = []
        avg_sum = []
        conn_dists = []
        avg_pot_along_x = []
        avg_end_spikes = []
        for i in range(100):

            lsm.initialise_lsm()
            print(f"Number of connections: {np.sum(lsm.connections)}")
            # Run simulation
            lsm.run_simulation(spike_trains, plot=False, reset=True) # reset not needed on first run
            avg_pot.append(lsm.avg_v_data)
            avg_sum.append(np.sum(lsm.spikes_data))
            avg_num_spikes.append(np.sum(lsm.spikes_data, axis=1))
            conn_dists.append(lsm.conn_distances)
            avg_pot_along_x.append(np.average(np.reshape(lsm.liquid_v, (15,3,3)), axis=(1,2)))
            avg_end_spikes.append(np.sum(np.reshape(lsm.spikes_data, (1000,15,3,3))[:, 14, :, :]))
        print("Average potential", np.mean(avg_pot))
        print("Average sum", np.mean(avg_sum))
        print("Average num spikes", np.mean(avg_num_spikes))
        print("Average conn dist", np.mean(conn_dists))
        print("Max conn dist", np.max(conn_dists))
        print("Average Potential Along x", np.average(avg_pot_along_x, axis=0))
        print("Average End Spikes", np.mean(avg_end_spikes))

        # Average End Spikes 2431.20 with end input
        # Average End Spikes 2325.06 without end input

        plt.figure(figsize = (8, 8))
        plt.title("Average Potential Along x-axis")
        plt.plot(np.average(avg_pot_along_x, axis=0), axis=0)
        plt.show()


    if OPTION == 5:
        # Run 3D analysis
        modes = ["fires", "potentials"]
        mode = modes[1]
        print(f"Mode is:", mode)
        lsm.lsm_3D_analysis(mode)


    if OPTION == 6:
        # Run field analysis
        slice_index = 0 # for plane = "xy" this would be 0 <= z-index <= z_dim
        lsm.lsm_field_analysis(slice_index)





if __name__ == "__main__":
    main()
