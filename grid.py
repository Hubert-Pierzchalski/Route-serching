import numpy as np
from math import ceil, sqrt
import matplotlib.pyplot as plt # used only for visualization
import networkx as nx # used only for visualization
from collections import deque

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

    def add_cities(self):
        number = self.number_of_cities
        for i in range(number):  # "i" is a city
            x = np.random.randint(-100, 101)
            y = np.random.randint(-100, 101)
            self.cities[0, i] = x
            self.cities[1, i] = y
            self.grid[x + 100, y + 100] = 1  # marking locations of cities
            self.G.add_node(i, pos=[x, y])

    def add_weighted_graph(self):
        number = self.number_of_cities
        for i in range(number):
            for j in range(number-i):
                j = i + j
                if i != j and self.weighted_graph[i, j] != np.inf:
                    weight = sqrt((self.cities[0, i] - self.cities[0, j]) ** 2 + (self.cities[1, i] - self.cities[1, j]) ** 2)
                    self.weighted_graph[i, j] = weight
                    self.weighted_graph[j, i] = weight
                    self.G.add_edge(i, j, weight=weight)
                else:
                    self.weighted_graph[i, j] = np.inf

        pos = nx.get_node_attributes(self.G, "pos")

        nx.draw_networkx_nodes(self.G, pos=pos)
        nx.draw_networkx_edges(self.G, pos=pos)

        # labels
        nx.draw_networkx_labels(self.G, pos, font_size=20, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels)

    def reduce_roads(self):
        number = self.number_of_cities
        new_roads = 0
        for i in range(self.number_of_cities):
            new_roads += i
        delete_cities = new_roads - ceil(new_roads * 0.8)
        while delete_cities > 0:
            city_a = np.random.randint(0, number)
            city_b = np.random.randint(0, number)
            if self.weighted_graph[city_a, city_b] != np.inf and city_a != city_b:
                self.weighted_graph[city_a, city_b] = np.inf
                self.weighted_graph[city_b, city_a] = np.inf
                delete_cities -= 1

    def BFS(self, starting_city):
        roads = self.weighted_graph.copy()
        visited_cities = []
        # roads[0, :] = np.inf
        # roads[:, 0] = np.inf
        queue = deque()
        visited_cities.append(starting_city)
        queue.append(starting_city)
        # first_object = 0
        while queue:
            # current_city = queue[first_object]
            current_city = queue.popleft()
            for city, length in enumerate(roads[current_city]):
                if length != np.inf:
                    queue.append(city)
                    visited_cities.append(city)
                    roads[city, :] = np.inf
                    roads[:, city] = np.inf
        print(visited_cities)

    def print_grid(self):
        print(self.weighted_graph)
        plt.axis([-100, 100, -100, 100])
        plt.show()


grid = Grid(4)
grid.add_cities()
# grid.reduce_roads()
grid.add_weighted_graph()
grid.BFS(0)

