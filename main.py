import math
import numpy as np
from scipy import spatial
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time

file = ["SampleCoordinates.txt", "HungaryCities.txt", "GermanyCities.txt"]
radius = [0.08, 0.005, 0.0025]
start_city = [0, 311, 1573]
end_city = [5, 702, 10584]
R = 1   # normalized radii

file_to_run = int(input('Type 1,2 or 3 to select file to run\n 1: SampleCoordinates\n 2: HungaryCities\n 3:'
                        ' GermanyCities\n'))

if file_to_run == 1:
    filename = file[0]
    radius = radius[0]
    start_node = start_city[0]
    end_node = end_city[0]

elif file_to_run == 2:
    filename = file[1]
    radius = radius[1]
    start_node = start_city[1]
    end_node = end_city[1]

elif file_to_run == 3:
    filename = file[2]
    radius = radius[2]
    start_node = start_city[2]
    end_node = end_city[2]


# Task 1
def read_coordinate_file(filename):
    """
    read_coordinate_file
    takes a text-file as input and trims the data into the desired form. Then extracts coordinates in longitude and
    latitude form from each line in the data file and recalculates the coordinates to (x,y)-form and saves to a
    numpy array.

    :param filename:
    :return: A list of coordinates for cities
    """
    with open(filename, mode='r') as file:
        list_of_floats = []
        for line in file:
            strip_line = line.strip("{ } \n")   # strips {} and \n
            replace_line = strip_line.replace(" ", "")  # removes blank spaces
            split_line = replace_line.split(sep=",")    # Splits the by ,
            for item in split_line:
                list_of_floats.append(float(item))
    float_list = np.array(list_of_floats)
    reshaped_list = float_list.reshape(len(float_list) // 2, 2)     # Reshapes array into matrix with [x,y]
    x_coordinates = R * (reshaped_list[:, 1] * np.pi) / 180     # Mercator projector for x
    y_coordinates = R * np.log(np.tan((np.pi / 4 + (reshaped_list[:, 0] * np.pi) / 360)))   # Mercator projector for y
    coord_list = np.array([x_coordinates, y_coordinates]).T

    return coord_list

# Timing the first function and printing to command window, this will be repeated for every function.
start = time.time()
coord_list = read_coordinate_file(filename)
end = time.time()
print("read_coordinate_file: ", end - start)


# Task 2
def construct_graph_connections(coord_list, radius):
    """
    construct_graph_connections
    checks all coordinates from coord_list against each other to see if they are inside the given radius.

    :param coord_list: List of coordinates for all cities (x,y)-form
    :param radius: Minimum radius for coordinate point to be within
    :return: Two arrays one containing indices of connected cities and one that contains the distance between
    the connected cities.
    """
    distance_list = []
    first_index_list = []
    second_index_list = []
    for i, coord_i in enumerate(coord_list):
        for j in range(i + 1, len(coord_list)):    # Loop skipping the first element to prevent duplicates
            coord_j = coord_list[j]
            d = coord_i - coord_j
            distance = math.sqrt(d[0] ** 2 + d[1] ** 2)     # Calculates distance between two points

            if distance <= radius:
                distance_list.append(distance)
                first_index_list.append(i)
                second_index_list.append(j)
    distance_array = np.array(distance_list)
    index_array = np.array([first_index_list, second_index_list]).T
    return distance_array, index_array


def construct_fast_graph_connections(coord_list, radius):
    """
    construct_fast_graph_connections
    checks all coordinates from coord_list against each other to see if they are inside the given radius. This is
    essentially a faster way to do the previous function by using spatial algorithms.
    :param coord_list:
    :param radius:
    :return: Two arrays one containing indices of connected cities and one that contains the distance between
    the connected cities.
    """
    tree = spatial.cKDTree(coord_list)      # Scipy method for finding nearest neighbour
    index_array = tree.query_pairs(radius, output_type='ndarray')
    distances = []
    for pair in index_array:
        dist = spatial.distance.pdist(coord_list[pair])
        distances.extend(dist)

    distance_array = np.array(distances)

    return distance_array, index_array


start = time.time()
[distance, indices] = construct_graph_connections(coord_list, radius)
end = time.time()
print("construct_graph_connections: ", end - start)

start = time.time()
[distance, indices] = construct_fast_graph_connections(coord_list, radius)
end = time.time()
print("construct_fast_graph_connections: ", end - start)


N = len(coord_list)     # Number of cities


def construct_graph(indices, distance, N):
    """
    construct_graphs
    takes indices of connected cities and distances between them and stores in a compressed sparse row matrix
    (csr) which is a compact way to represent the data.
    :param indices:
    :param distance:
    :param N: number of cities
    :return: sparse graph of indices and distances of the cities
    """
    M = N   # M should be same length as N
    sparse_graph = csr_matrix((distance, (indices[:, 0], indices[:, 1])), shape=(M, N))     # Scipy method for csr
    return sparse_graph


start = time.time()
graph = construct_graph(indices, distance, N)
end = time.time()
print("construct_graph: ", end - start)


def find_shortest_path(graph, start_node, end_node):
    """
    find_shortest_path
    uses Scipy method shortest_path to determine the shortest path and its length. The function takes sparse graph as
    input and a start coordinate. The function then calculates the distance between every other coordinate and said coordinate
    including itself. The function also returns a predecessor array from which the shortest path can be extracted.
    :param graph: sparse graph
    :param start_node: start city
    :param end_node: end city
    :return: The shortest path and its length
    """
    path = [end_node]
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, indices=start_node,
                                              return_predecessors=True)
    path_length = dist_matrix[end_node]     # Takes the shortest path's length
    current_node = end_node

    while current_node != start_node:   # Creates the shortest path from predecessors by back tracking from the end node
        current_node = predecessors[current_node]
        path.append(current_node)

    path = path[::-1]      # Reverse the path into chronological order
    return path, path_length


start = time.time()
[path, path_length] = find_shortest_path(graph, start_node, end_node)
end = time.time()
print("find_shortest_path: ", end - start)


def plot_points(coord_list, indices, path):
    """
    plot_points
    plots all cities, the lines between them if they are within the given radius and the shortest path between
    start city and end city. Utilizing LineCollection to plot the lines more efficiently.
    :param coord_list:
    :param indices:
    :param path:
    """
    fig, ax = plt.subplots()
    ax.axis('equal')

    # Plots the cities represented as dots
    ax.plot(coord_list[:, 0], coord_list[:, 1], '.', color='red', markersize=5, zorder=0)

    # Plots the shortest path from start city to end city
    shortest_path = coord_list[path]
    ax.plot(shortest_path[:, 0], shortest_path[:, 1], '-', color='purple', markersize=3)

    # Plots the lines between neighbouring cities using LineCollection
    lines = coord_list[indices]
    line_segments = LineCollection(lines, linewidths=0.3, colors='grey')
    ax.add_collection(line_segments)

    # Adds title, legend and axis labels
    ax.legend(['Cities', 'Shortest Path'])
    plt.title('Shortest Path')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()


start = time.time()
plot_points(coord_list, indices, path)
end = time.time()
print("plot_points: ", start - end)

# Prints path and length
print(path)
print(path_length)