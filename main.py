# Necessary imports to run code
import math
import numpy as np
from scipy import spatial
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time

filename = "SampleCoordinates.txt"
radius = 0.08
start_node = 0
end_node = 5


def read_coordinate_file(filename):
    """
    Reads a file and trims the data into the desired form. Then extracts coordinates in longitude and latitude
    form from each line in the data file and recalculates the coordinates to (x,y)-form and saves to a
    numpy array.

    :param filename: The name of the file to be opened
    :return: A list of coordinates for cities
    """
    with open(filename, mode='r') as file:
        list_of_floats = []
        for line in file:
            strip_line = line.strip("{ } \n")
            replace_line = strip_line.replace(" ", "")
            split_line = replace_line.split(sep=",")
            for item in split_line:
                list_of_floats.append(float(item))
    float_list = np.array(list_of_floats)
    reshaped_list = float_list.reshape(len(float_list) // 2, 2)
    x_coordinates = (reshaped_list[:, 1] * np.pi) / 180
    y_coordinates = np.log(np.tan((np.pi / 4 + (reshaped_list[:, 0] * np.pi) / 360)))
    coord_list = np.array([x_coordinates, y_coordinates]).T

    return coord_list


start = time.time()
coord_list = read_coordinate_file(filename)
end = time.time()
print("read_coordinate_file: ", end - start)


def construct_graph_connections(coord_list, radius):
    """
    Compares the distance between all cities to all other cities and if a city is within the minimum radius of another
    city those indices are saved to an array and the distance between those cities are saved to another array.

    :param coord_list: List of coordinates for all cities (x,y)-form
    :param radius: Minimum radius for coordinate point to be within
    :return: Two arrays one containing indices of connected cities and one that contains the distance between
    the connected cities.
    """
    distance_list = []
    first_index_list = []
    second_index_list = []
    for i, coord_i in enumerate(coord_list):
        for j in range(i + 1, len(coord_list)):
            coord_j = coord_list[j]
            d = coord_i - coord_j
            distance = math.sqrt(d[0] ** 2 + d[1] ** 2)

            if distance <= radius:
                distance_list.append(distance)
                first_index_list.append(i)
                second_index_list.append(j)
    distance_array = np.array(distance_list)
    index_array = np.array([first_index_list, second_index_list]).T
    return distance_array, index_array


def construct_fast_graph_connections(coord_list, radius):
    tree = spatial.cKDTree(coord_list)
    index_list = tree.query_pair(coord_list, radius)
    print(index_list)

    return index_list


construct_fast_graph_connections(coord_list, radius)


start = time.time()
[distance, indices] = construct_graph_connections(coord_list, radius)
end = time.time()
print("construct_graph_connections: ", end - start)
print(indices)

N = len(coord_list)


def construct_graph(indices, distance, N):
    M = N
    sparse_graph = csr_matrix((distance, (indices[:, 0], indices[:, 1])), shape=(M, N))
    return sparse_graph


start = time.time()
graph = construct_graph(indices, distance, N)
end = time.time()


print("construct_graph: ", end - start)


def find_shortest_path(graph, start_node, end_node):
    path = [end_node]
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, indices=start_node,
                                              return_predecessors=True)
    path_length = dist_matrix[end_node]
    current_node = end_node

    while current_node != start_node:
        current_node = predecessors[current_node]
        path.append(current_node)

    path = path[::-1]
    print(path)
    return path, path_length


start = time.time()
[path, path_length] = find_shortest_path(graph, start_node, end_node)
end = time.time()


print("find_shortest_path: ", end - start)

def plot_points(coord_list, indices, path):
    fig, ax = plt.subplots()
    ax.axis('equal')
    lines = coord_list[indices]
    shortest_path = coord_list[path]
    ax.plot(coord_list[:, 0], coord_list[:, 1], '.', color='red', markersize=5, zorder=0)
    ax.plot(shortest_path[:, 0], shortest_path[:, 1], '-', color='purple', markersize=3)
    line_segments = LineCollection(lines, linewidths=0.3, colors='grey')
    ax.add_collection(line_segments)
    # Adds title and legend
    ax.legend(['Cities', 'Shortest Path'])
    plt.title('Shortest Path')
    plt.show()

#plot_points(coord_list, indices, path)
start = time.time()
end = time.time()
# print("plot_points: ", start - end)
