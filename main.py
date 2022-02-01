import numpy as np
import matplotlib.pyplot as plt

filename = "SampleCoordinates.txt"


def read_coordinate_file(filename):
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
    x_coordinates = (reshaped_list[:, 0] * np.pi) / 180
    y_coordinates = np.log(np.tan((np.pi / 4 + (reshaped_list[:, 1] * np.pi) / 360)))
    coord_list = np.array([x_coordinates, y_coordinates]).T

    return coord_list


coord_list = read_coordinate_file(filename)


def plot_points(coord_list):
    plt.figure()
    plt.plot(coord_list[:, 0], coord_list[:, 1], '.', color='red', markersize=3)
    plt.show()


plot_points(coord_list)





