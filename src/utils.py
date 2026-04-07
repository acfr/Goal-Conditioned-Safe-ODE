#!/usr/bin/env python
'''
Utility functions for data generation, training, and plotting.

Author: Dechuan Liu (April 2026)
'''

import numpy as np
import scipy.io
from scipy.interpolate import griddata

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
    

########################################################################
# sampling methods for visualization and dataset generation
########################################################################
def generate_path_in_gmap_space(points, gmap_fn, zero_point, num_steps, step_size = 0.1):
    '''
    take points in x space (nx2)
    map them to y space (nx2)
    generate the path to zero point in y space (nxstepx2)
    return the path in m space (nxstepx2)

    run inverse and plot after that
    points_y = generate_path_in_gmap_space(points, 
                        lambda point: model_pl.apply(params_pl, point, method=model_pl.gmap), 
                        jnp.array(zero_point), num_steps)

    for path in points_y:
        # convert back to x space
        path = inverse_func(path)
        plot_gradient_descent_path(ax, path, xlim=x_range, ylim=y_range)
    '''
    g_opt_x = gmap_fn(zero_point)

    # g(x_0)-g(x_ox*pt) nx2
    points = gmap_fn(points) - g_opt_x

    # x by exp term nx(step)x2 
    points_exp = generate_sequence(points=points, step=num_steps, step_size=step_size)

    # g(x) in y space nx(step)x2 
    points_y = points_exp + g_opt_x

    return points_y

def generate_sequence(points, step, step_size = 0.1):
    """
    Generate sequences by multiplying each pair of points by exp(-t), 
    where t ranges from 0 to m.
    
    Args:
        points: An (n x 2) array of n points.
        m: An integer representing the upper limit for t.
    
    Returns:
        A (n x (m+1) x 2) array where each point is expanded into a sequence 
        of length (m+1) with shape mx2.
    """
    # Create a sequence of t values from 0 to m
    t_values = jnp.arange(step+1) * step_size  # Shape: (m+1)
    
    # Compute the exponential values exp(-t) for t in [0, m]
    exp_t = jnp.exp(-t_values)  # Shape: (m+1)
    
    # Reshape exp_t for broadcasting across points
    exp_t = exp_t[:, None]  # Shape: (m+1, 1)
    
    # Multiply each point by the exp(-t) sequence
    sequences = points[:, None, :] * exp_t  # Shape: (n, m+1, 2)
    
    return sequences

def monotone_uniform_map(x, output_range=(0, 1)):
	"""
	Monotone mapping that stretches input data so that the output is uniformly distributed.
	
	Args:
		x: 1D array-like of input values.
		output_range: Tuple (min, max) of target range.
		
	Returns:
		Array of same shape as x, mapped to target range.
	"""
	x = jnp.array(x)
	# Empirical CDF: rank-based monotone map
	ranks = jnp.argsort(jnp.argsort(x))
	ecdf = (ranks + 1) / len(x)  # scaled to (0, 1]
	
	# Map to output range
	out_min, out_max = output_range
	y = out_min + ecdf * (out_max - out_min)
	
	return y

def plot_gradient_descent_path( ax, path, title='Gradient Descent Path',
                               xlabel = 'x', ylabel='y',
                               xlim = [-np.pi, np.pi], ylim = [-8,8],
                               is_drawing_arrow = True,
                               label = 'Path', path_color = 'red',
                               arrowsize=1.0, arrow_start_ratio = 0.1, 
                               linewidth=0.7
                               ):
    """
    Plots the gradient descent path with error signs at each point.
    
    Args:
        path (jax.numpy.ndarray): Array of points representing the gradient descent path.
    """
    # Convert to NumPy for plotting and interpolation
    path = np.array(path)
    
    # Create the plot
    x_smooth, y_smooth = path[:, 0], path[:, 1]

    # Plot the smoothed path
    ax.plot(x_smooth, y_smooth, color=path_color, label=label, linewidth=linewidth)
    ax.plot(path[-1, 0], path[-1, 1], 'bo', label='Terminal', markersize=1)

    # draw an arrow from the manual starting point of the streamline to the end
    arrow_index = int(np.floor(len(x_smooth) * arrow_start_ratio))
    start = jnp.array([path[arrow_index, 0], path[arrow_index, 1]])
    end = jnp.array([path[arrow_index+1, 0], path[arrow_index+1, 1]])
    while arrow_index < len(x_smooth)-1 and np.linalg.norm(start - end) < 0.01/arrowsize:
        end = jnp.array([path[arrow_index+1, 0], path[arrow_index+1, 1]])
        arrow_index += 1

    ax.annotate(
        '', 
        xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle='simple',       # Sharp arrowhead
            color=path_color,
            mutation_scale=arrowsize       # Arrowhead size
        )
    )

    # Add plot labels and title
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(xlim)  # Set x-axis range from -1 to 1
    ax.set_ylim(ylim)  # Set y-axis range from -1 to 1

def sample_linear_on_sphere_boundary(rng, num_samples, center, radius):

    directions = jnp.linspace(0, 2 * jnp.pi, num_samples)
    points = center + jnp.stack((jnp.cos(directions), jnp.sin(directions)), axis=-1) * radius
    
    return points

def sample_linear_grid_points(x_range, y_range, num_points_x, num_points_y):
    """
    Samples linear grid points (evenly spaced points) within the specified x and y ranges
    and returns them as a list of (x, y) tuples.

    Parameters:
    - x_range: A tuple (x_min, x_max) specifying the range for x values.
    - y_range: A tuple (y_min, y_max) specifying the range for y values.
    - num_points_x: The number of points to sample along the x-axis.
    - num_points_y: The number of points to sample along the y-axis.

    Returns:
    - points: A JAX array of shape (num_points_x * num_points_y, 2) where each row is an (x, y) grid point.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Create evenly spaced values along the x and y axes
    x_values = jnp.linspace(x_min, x_max, num_points_x)
    y_values = jnp.linspace(y_min, y_max, num_points_y)
    
    # Generate all combinations of x and y values (i.e., a grid)
    x_grid, y_grid = jnp.meshgrid(x_values, y_values)
    
    # Stack the x and y values together into a single array
    points = jnp.stack((x_grid.flatten(), y_grid.flatten()), axis=-1)
    
    return points


def plot_value_contour( ax, x, y, title='Estimated Value Function', 
                       xlim = [-np.pi, np.pi], ylim = [-8,8], vlim = [-0.2,10],
                       grid_method = 'linear'
                       ):
    """
    plot the contour.
    """
    X = x[:,0]
    Y = x[:,1]
    Z = y
    x_unique = np.unique(X)
    y_unique = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_unique, y_unique)

    # Reshape Z to fit the grid
    Z_grid = griddata((X, Y), Z, (X_grid, Y_grid), method=grid_method)
    ax.set_title(title)
    ax.set_xlim(xlim)  # Set x-axis range from -1 to 1
    ax.set_ylim(ylim)  # Set y-axis range from -1 to 1
    contour = ax.contourf(X_grid, Y_grid, Z_grid, levels= np.linspace(vlim[0], vlim[1], 20), cmap='viridis', vmin=vlim[0], vmax=vlim[1], extend='both')
        
    