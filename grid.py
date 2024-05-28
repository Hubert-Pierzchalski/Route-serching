import numpy as np
from math import ceil
import matplotlib.pyplot as plt


def calc_linear_function(x0, y0, x1, y1):
    a = (y0-y1) / (x0 - x1)
    b = y0 - a * x0
    return a, b


class Grid:
    def __init__(self, number):
        self.number_of_cities = number
        self.cities = np.zeros((2, self.number_of_cities))  # coordinates of cities
        self.grid = np.zeros((201, 201))  # map

    def set_grid(self):
        for i in range(self.number_of_cities): # i is a city
            x = np.random.randint(-100, 101)
            y = np.random.randint(-100, 101)
            self.cities[0, i] = x
            self.cities[1, i] = y
            self.grid[x + 100, y + 100] = 2  # marking locations of cities
            # matrix cannot start have negative values that's why adding 100
        # print(self.cities[0, 0], self.cities[1, 0])

    def roads(self):
        num = self.number_of_cities
        for i in range(num):
            x0 = self.cities[0, i]
            y0 = self.cities[1, i]
            for j in range(i+1,num):
                x1 = self.cities[0, j]
                y1 = self.cities[1, j]
                xdiff = int(abs(x0-x1))
                a, b = calc_linear_function(x0, y0, x1, y1)
                if (x1 < x0):
                    for x in range(xdiff):
                        y_grid = ceil(a * ((-1 * x) + x0) + b)
                        self.grid[int(-1 * x + x0 + 100), y_grid + 100] = 2
                elif (x1 > x0):
                    for x in range(xdiff):
                        y_grid = ceil(a * (x + x0) + b)
                        self.grid[int(x + x0 + 100), y_grid + 100] = 2

    def print_grid(self):

        plt.plot(self.grid)
        plt.show()
        # print(x, y)

grid = Grid(20)
grid.set_grid()
grid.roads()
grid.print_grid()
