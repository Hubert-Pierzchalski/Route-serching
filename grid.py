import numpy as np
from math import ceil, sqrt
import matplotlib.pyplot as plt
import networkx as nx


def calc_linear_function(x0, y0, x1, y1):
    a = (y0-y1) / (x0 - x1)
    b = y0 - a * x0
    return a, b


class Grid:
    def __init__(self, number):
        self.number_of_cities = number
        self.cities = np.zeros((2, self.number_of_cities))  # coordinates of cities
        self.grid = np.zeros((201, 201))  # map
        self.weighted_graph = np.zeros((number, number))
        self.G = nx.Graph()

    def set_grid(self):
        number = self.number_of_cities
        for i in range(number): # "i" is a city
            x = np.random.randint(-100, 101)
            y = np.random.randint(-100, 101)
            self.cities[0, i] = x
            self.cities[1, i] = y
            self.grid[x + 100, y + 100] = 1  # marking locations of cities
            self.G.add_node(i, pos=[x, y])
        for i in range(number):
            for j in range(number-i-1):
                j += i + 1
                weight = sqrt((self.cities[0, i] - self.cities[0, j]) ** 2 + (self.cities[1, i] - self.cities[1, j]) ** 2)
                self.weighted_graph[i, j] = weight
                if i != j:
                    self.G.add_edge(i, j, weight=weight)

        pos = nx.get_node_attributes(self.G, "pos")

        nx.draw_networkx_nodes(self.G, pos=pos)
        nx.draw_networkx_edges(self.G, pos=pos)

        #labels
        nx.draw_networkx_labels(self.G, pos, font_size=20, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        # nx.draw_networkx_edge_labels(self.G, pos, edge_labels)


    def roads(self):
        num = self.number_of_cities
        for i in range(num):
            x0 = self.cities[0, i]
            y0 = self.cities[1, i]
            for j in range(i+1, num):
                x1 = self.cities[0, j]
                y1 = self.cities[1, j]
                xdiff = int(abs(x0-x1))
                if x0 == x1:
                    pass
                else:
                    a, b = calc_linear_function(x0, y0, x1, y1)
                    if x1 < x0:
                        for x in range(xdiff):
                            y_grid = ceil(a * ((-1 * x) + x0) + b)
                            self.grid[int(-1 * x + x0 + 100), y_grid + 100] = 2
                            plt.plot(-1*x+x0, y_grid)
                    elif x1 > x0:
                        for x in range(xdiff):
                            y_grid = ceil(a * (x + x0) + b)
                            self.grid[int(x + x0 + 100), y_grid + 100] = 2
                            plt.plot(x+x0, y_grid)
#a

    def print_grid(self):
        plt.axis([-100, 100, -100, 100])
        plt.show()


grid = Grid(7)
grid.set_grid()
# grid.roads()
grid.print_grid()
