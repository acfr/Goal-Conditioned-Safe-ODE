'''
Utility functions for data generation, training, and plotting.

Author: Dechuan Liu (April 2026)
'''

import numpy as np
import scipy.io

def map_obstacle(x, y):
    '''
    function to define the obstacle in the map (T with boundary)
    '''		
    # obstacle in the middle
    if y > 0 and y < 4 and abs(x) < 6:
        return True
    elif y < 0 and y > -4 and ((x < -4 and x > -6) or (x < 2 and x > -2) or (x < 6 and x > 4)):
        return True
    elif y < -4 and y >= -8 and (x < 2 and x > -2):
        return True
    
    # boundary
    if abs(x) > 7.9 or abs(y) > 7.9:
        return True

    return False

def	get_line_and_value(inner_radius = 0.0):
    '''
    mvoe the start points slightly into the inner region (traversable region)
    return a list of lines and a list of label for each line
    '''
    lines = [[[-8+inner_radius,-8+inner_radius], [-2-inner_radius, -8+inner_radius]], 
        [[-2-inner_radius, -8 + inner_radius], [-2-inner_radius, 0-inner_radius]], 
        [[-2-inner_radius, 0-inner_radius], [-4+inner_radius, 0-inner_radius]], 
        [[-4+inner_radius, 0-inner_radius], [-4+inner_radius,-4-inner_radius]], 
        [[-4+inner_radius,-4-inner_radius], [-6-inner_radius,-4-inner_radius]], 
        [[-6-inner_radius,-4-inner_radius], [-6-inner_radius,4+inner_radius]], 
        [[-6-inner_radius,4+inner_radius], [6+inner_radius,4+inner_radius]], 
        [[6+inner_radius,4+inner_radius], [6+inner_radius,-4-inner_radius]], 
        [[6+inner_radius,-4-inner_radius], [4-inner_radius,-4-inner_radius]], 
        [[4-inner_radius,-4-inner_radius], [4-inner_radius,0-inner_radius]], 
        [[4-inner_radius,0-inner_radius], [2+inner_radius,0-inner_radius]], 
        [[2+inner_radius,0-inner_radius], [2+inner_radius, -8+inner_radius]], 
        [[2+inner_radius, -8+inner_radius], [8-inner_radius,-8+inner_radius]], 
        [[8-inner_radius,-8+inner_radius], [8-inner_radius,8-inner_radius]], 
        [[8-inner_radius,8-inner_radius], [-8+inner_radius,8-inner_radius]], 
        [[-8+inner_radius,8-inner_radius], [-8+inner_radius,-8+inner_radius]]]
    values = [4,1,3,2,3,1,4,2,3,1,3,2,4,1,3,2]
    return lines, values


def sample_points_on_line(lines, values, n_samples):
    """
    Sample n points on the entire list of lines.
    
    Args:
        lines: List of [start_point, end_point] pairs representing line segments.
        values: List of values corresponding to each line segment.
        n_samples: Total number of points to sample.
        
    Returns:
        sample_points: List of sampled points.
        sample_labels: List of labels corresponding to sampled points.
    """
    # Calculate lengths of each line segment
    lengths = np.array([np.linalg.norm(np.array(end) - np.array(start)) for start, end in lines])
    
    # Calculate total length and the proportion of samples for each line
    total_length = np.sum(lengths)
    proportions = lengths / total_length
    samples_per_line = np.round(proportions * n_samples).astype(int)
    
    # Ensure total samples equal to n_samples by adjusting due to rounding
    while np.sum(samples_per_line) != n_samples:
        diff = n_samples - np.sum(samples_per_line)
        samples_per_line[np.argmax(proportions)] += diff
    
    sample_points = []
    sample_labels = []
    
    # Sample points on each line based on the proportion
    for (start, end), value, n_line_samples in zip(lines, values, samples_per_line):
        start = np.array(start)
        end = np.array(end)
        if n_line_samples > 0:
            # Linearly interpolate to get sample points
            t_vals = np.linspace(0, 1, n_line_samples, endpoint=False)
            line_samples = [(1 - t) * start + t * end for t in t_vals]
            sample_points.extend(line_samples)
            sample_labels.extend([value] * n_line_samples)
    
    return sample_points, sample_labels

def normalize_data(x, ori_range, new_range):
    min_ori, max_ori = ori_range
    min_new, max_new = new_range
    # Normalize data from [min_x, max_x] to [min_y, max_y]
    return (x - min_ori) / (max_ori - min_ori) * (max_new - min_new) + min_new


# draw boundary
import jax.numpy as jnp
def draw_boundary(ax, width=1.5, closed = True,
                data_range = (-8, 8), normalized_range = (0, 1),
                color = 'red'):
    '''
    draw the boundary of the map (T with boundary)
    '''
    lines, values = get_line_and_value()
    lines = jnp.array(lines)
    lines = normalize_data(lines, data_range, normalized_range)
    # print(lines)

    for line in lines:
        x0, y0 = line[0]
        x1, y1 = line[1]
        ax.plot([x0, x1], [y0, y1], linewidth=width, color=color)

    if closed:
        ax.plot([6/16, 10/16], [0.0, 0.0], linewidth=width, color='black', linestyle='--', solid_capstyle='round')


'''
Read the dataset and convert to 
    x: (nx2)
    y: (nx3) 
        - label
        - value
        - not in use
'''
class DataLoader_maze():
    def __init__(self, path, inf_replaced = 100):
        self.path = path
        self.load_data(inf_replaced = inf_replaced)

    def load_data(self, inf_replaced=100):
        """
        load data from the source.
        """
        mat_data = scipy.io.loadmat(self.path)

        self.x = np.vstack((mat_data['x_1'], mat_data['x_2']))
        self.y = np.vstack((mat_data['labels'], mat_data['y_1'], mat_data['y_2']))

        # get y range
        #  np.max(self.y[np.isfinite(self.y)])]
        self.y_range = [np.min(mat_data['y_1']), np.max(mat_data['y_1'][np.isfinite(mat_data['y_1'])])]
        
        # replace inf with some value
        self.y[np.isinf(self.y)] = inf_replaced

    def get_data(self, 
                 train_batches = 50,
                 test_batches = 3,
                 eval_batches = 3,
                 train_batch_size = 40,
                 eval_batch_size = 50,
                 test_batch_size = 50,
                 ):
        """
        split data and return
        """
        [data_dim, data_size] = np.shape(self.x)
        [data_dim_y, data_size] = np.shape(self.y)
        x = self.x.T
        y = self.y.T
        print(f'x is {x.shape} and y is {y.shape}')
        
        # random select data from the entire resource
        if (train_batch_size*train_batches+test_batch_size*test_batches+eval_batch_size*eval_batches) > data_size:
            raise ValueError("Too many data are required by train, test, and evaluate")
        
        # select train
        xtrain = np.zeros((train_batches* train_batch_size, data_dim))
        ytrain = np.zeros((train_batches* train_batch_size, data_dim_y))
        remaining_x = x
        remaining_y = y
        data_size_left = data_size
        # save
        indices_ = np.random.choice(data_size_left, size=train_batch_size*train_batches, replace=False)
        xtrain = remaining_x[indices_, :]
        ytrain = remaining_y[indices_, :]
        # shuffle it
        perm = np.random.permutation(xtrain.shape[0])
        xtrain = xtrain[perm]
        ytrain = ytrain[perm]

        # update
        remaining_indice = np.setdiff1d(np.arange(data_size_left), indices_)
        remaining_x = remaining_x[remaining_indice, :]
        remaining_y = remaining_y[remaining_indice, :]

        data_size_left -= train_batch_size*train_batches

        # test
        xtest = np.zeros((test_batches* test_batch_size, data_dim))
        ytest = np.zeros((test_batches* test_batch_size, data_dim_y))
        # save
        indices_ = np.random.choice(data_size_left, size=test_batch_size*test_batches, replace=False)
        xtest = remaining_x[indices_, :]
        ytest = remaining_y[indices_, :]

        # update
        remaining_indice = np.setdiff1d(np.arange(data_size_left), indices_)
        remaining_x = remaining_x[remaining_indice, :]
        remaining_y = remaining_y[remaining_indice, :]

        data_size_left -= test_batch_size*test_batches

        # eval
        xeval = np.zeros((eval_batches* eval_batch_size, data_dim))
        yeval = np.zeros((eval_batches* eval_batch_size, data_dim_y))
        # save
        indices_ = np.random.choice(data_size_left, size=eval_batch_size, replace=False)
        xeval = remaining_x[indices_, :]
        yeval = remaining_y[indices_, :]

        data = {
            "x": x,
            "y": y,
            "data_size": data_size,
            "xtrain": xtrain, 
            "ytrain": ytrain, 
            "xtest": xtest, 
            "ytest": ytest, 
            "xeval": xeval,
            "yeval": yeval,
            "train_batches": train_batches,
            "train_batch_size": train_batch_size,
            "test_batches": test_batches,
            "test_batch_size": test_batch_size,
            "eval_batches": eval_batches,
            "eval_batch_size": eval_batch_size,
            "data_dim": data_dim,
            "y_range": self.y_range,
        }

        return data