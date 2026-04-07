'''
Generate data for 2D corridor.

Author: Dechuan Liu (April 2026)
'''

import numpy as np
from scipy.spatial import KDTree
import random
from utils import map_obstacle, get_line_and_value, sample_points_on_line

"""
    A class that generate classification data for the maze env (0 for traservable points, 1-4 for points on the line)
    use inf to represent points on the edge
    use rrt to generate value for points on the path
    
    Features:
    - Generates random points in the maze and checks their validity (inside boundaries, outside obstacles).
    - Plots the maze, obstacles, and the points.
    
    Args:
    - boundary: tuple (xmin, xmax, ymin, ymax), the bounding box of the maze.
    - obstacles: list of functions (optional), each function takes (x, y) and returns True if (x, y) is inside the obstacle.
    
    Attributes:
    - boundary: The bounding box of the maze.
    - obstacles: The list of obstacles (functions that return True if a point is inside an obstacle).
    - points: A list that stores all points.
    - values: A list of classification for points (0,1)
"""
class MazeBarrierValue:
    def __init__(self, boundary, obstacles=None):
        self.boundary = boundary
        self.obstacles = obstacles if obstacles is not None else []
        self.tree = None
        self.path = None
        self.explored_nodes = []  # Store all explored nodes
        self.distances = {}  # Dictionary to store distances from the start to each explored node

    def is_valid(self, point):
        x, y = point
        xmin, xmax, ymin, ymax = self.boundary
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return False
        for obstacle in self.obstacles:
            if obstacle(x, y):
                return False
        return True

    def _random_point(self):
        xmin, xmax, ymin, ymax = self.boundary
        while True:
            point = (random.uniform(xmin, xmax), random.uniform(ymin, ymax))
            if self.is_valid(point):
                return point

    def _nearest(self, nodes, point):
        tree = KDTree(nodes)
        _, idx = tree.query(point)
        return nodes[idx]

    def _steer(self, from_point, to_point, step_size=0.1):
        direction = np.array(to_point) - np.array(from_point)
        length = np.linalg.norm(direction)
        if length == 0:
            return from_point
        direction = direction / length
        new_point = np.array(from_point) + step_size * direction
        return tuple(new_point)

    def solve_maze(self, start_point, end_point, max_iter=1000, step_size=0.1):
        start_point = np.array(start_point)
        end_point = np.array(end_point)

        nodes = [tuple(start_point)]  # Store as tuples
        parents = {tuple(start_point): None}
        self.distances[tuple(start_point)] = 0  # Distance from start to start is zero

        while len(self.explored_nodes) < max_iter:
            rand_point = self._random_point()
            nearest_point = self._nearest(nodes, rand_point)
            new_point = self._steer(nearest_point, rand_point, step_size)

            if self.is_valid(new_point):
                nodes.append(tuple(new_point))
                parents[tuple(new_point)] = nearest_point
                self.explored_nodes.append(tuple(new_point))  # Track explored node

                # Calculate the cumulative distance from the start
                distance_from_parent = np.linalg.norm(np.array(new_point) - np.array(nearest_point))
                self.distances[tuple(new_point)] = self.distances[nearest_point] + distance_from_parent

                if np.linalg.norm(np.array(new_point) - end_point) < step_size:
                    # Goal reached, backtrack to form the path
                    self.path = [end_point]
                    current = tuple(new_point)
                    while current is not None:
                        self.path.append(current)
                        current = parents[current]
                    self.path.reverse()
                    return self.path
        return None

    def generate_points_on_boundary(self, num_points, 
                                value_labels = None, 
                                use_line_label = False # use the label from get_line_and_value
                                ):
        
        # generate points on the boundary
        lines, values = get_line_and_value()

        if use_line_label:
            value_labels = values

        # put inf to values if not set
        if value_labels == None:
            value_labels = np.full_like(values, np.inf, dtype=np.float64)
        
        points, labels = sample_points_on_line(lines, value_labels, num_points)

        return points, labels
    
    def _random_in_boundary_point(self):
        xmin, xmax, ymin, ymax = self.boundary
        while True:
            point = (random.uniform(xmin, xmax), random.uniform(ymin, ymax))
            if not self.is_valid(point):
                return point

    def save_dataset_mat_with_label_and_in_boundary_point(self, 
                    filename, num_on_boundary_points = 0, # invalid point on the edge
                    num_in_boundary_points = 0):
        """
        Saves the dataset in .mat format with x, y, value, label lists.
                label: 	
                    0: rrt with value in value 
                    1-4: data on boundary line (represent direction )
                    5: data in the boundary
        Args:
        - filename: str, the name of the .mat file to save the dataset.
        """
        if self.distances is None:
            raise ValueError("Dataset has not been generated. Call inquire_values() first.")
        
        # add from rrt
        x_values, y_values = zip(*self.explored_nodes)

        # take the square here to change it to energy
        z_values = [self.distances[node]**2 for node in self.explored_nodes]
        label_values = [0]*len(x_values)

        # add points on the boundary
        points, labels= self.generate_points_on_boundary(num_on_boundary_points,
                                                use_line_label=True)
        
        x_values_on_bound, y_values_on_bound = zip(*points)
        z_values_on_bound = [np.inf]*len(x_values_on_bound)
        label_values_on_bound = labels

        # add points in the boundary
        if num_in_boundary_points > 0:
            points_in_boundary = []
            while len(points_in_boundary) < num_in_boundary_points:
                points_in_boundary.append(self._random_in_boundary_point())

            x_values_in_bound, y_values_in_bound = zip(*points_in_boundary)
            z_values_in_bound = [np.inf]*len(x_values_in_bound)
            label_values_in_bound = [5]*len(x_values_in_bound)
        else:
            x_values_in_bound, y_values_in_bound = [], []
            z_values_in_bound = []
            label_values_in_bound = []

        # Create a dictionary to hold the data
        '''
        x_1: x position
        x_2: y position
        y_1: value of rrt or inf for boundary
        y_2: not in use (=y_1)
        labels: label of output (0-5 represent path, on boundary, in boundary)
        '''
        data_dict = {
            'x_1': tuple(list(x_values) + list(x_values_on_bound) + list(x_values_in_bound)),
            'x_2': tuple(list(y_values) + list(y_values_on_bound) + list(y_values_in_bound)),
            'y_1': tuple(list(z_values) + list(z_values_on_bound) + list(z_values_in_bound)),
            'y_2': tuple(list(z_values) + list(z_values_on_bound) + list(z_values_in_bound)),
            'labels': tuple(list(label_values) + list(label_values_on_bound) + list(label_values_in_bound))
        }

        # Save the data to .mat file
        from scipy.io import savemat
        savemat(filename, data_dict)
        print(f"Dataset saved to {filename}")


# Generate dataset
if __name__ == "__main__":
    max_itr = 10000
    # Define the boundary and start/end points
    boundary = (-8, 8, -8, 8)
    start = (0, 7)
    end = (0,2)
    obstacles = [map_obstacle]
    gap = 0.1
    trajectory_num = 10

    generator = MazeBarrierValue(boundary, obstacles)
    generator.solve_maze(start, end, max_iter = int(0.5*max_itr))
    generator.save_dataset_mat_with_label_and_in_boundary_point(f'src/data/2D-corridor/map_{max_itr}_barrier_value.mat', num_on_boundary_points=int(max_itr/2))



