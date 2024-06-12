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
        self.weighted_graph = np.zeros((number, number))
        self.G = nx.Graph()

    def add_cities(self):
        number = self.number_of_cities
        for i in range(number):  # "i" is a city
            x = np.random.randint(-100, 101)
            y = np.random.randint(-100, 101)
            self.cities[0, i] = x
            self.cities[1, i] = y
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
        delete_cities = new_roads - ceil(new_roads * 0.4)
        while delete_cities > 0:
            city_a = np.random.randint(0, number)
            city_b = np.random.randint(0, number)
            if self.weighted_graph[city_a, city_b] != np.inf and city_a != city_b:
                self.weighted_graph[city_a, city_b] = np.inf
                self.weighted_graph[city_b, city_a] = np.inf
                delete_cities -= 1

    def BFS(self, starting_city): # For TPS
        roads = self.weighted_graph.copy()
        visited_cities = []
        queue = deque()
        visited_cities.append(starting_city)
        queue.append(starting_city)
        # first_object = 0
        while queue:
            # current_city = queue[first_object]
            current_city = queue.popleft()
            for city, length in enumerate(roads[current_city]):
                if length != np.inf and city not in visited_cities:
                    queue.append(city)
                    visited_cities.append(city)
        return visited_cities

    def DFS(self, starting_city):
        roads = self.weighted_graph.copy()
        visited_cities = []
        stack = []
        stack.append(starting_city)
        while stack:
            current_city = stack.pop()
            visited_cities.append(current_city)
            for city, length in enumerate(roads[current_city]):
                if length != np.inf and city not in visited_cities:
                    stack.append(city)
        return visited_cities[0:self.number_of_cities]

    def minimum_spanning_tree(self):
        road = []
        roads = self.weighted_graph.copy()
        for i in range(self.number_of_cities-1):
            index = np.argmin(roads)
            cols = index % self.number_of_cities
            rows = index // self.number_of_cities
            road.append([rows, cols])
            roads[rows, cols] = np.inf
            roads[cols, rows] = np.inf
            for i in range(self.number_of_cities):
                if roads[rows, i] == np.inf:
                    roads[cols, i] = np.inf
                    roads[i, cols] = np.inf
            for i in range(self.number_of_cities):
                if roads[cols, i] == np.inf:
                    roads[rows, i] = np.inf
                    roads[i, rows] = np.inf
        print(road)

    def greedy_search(self, starting_city):
        roads = self.weighted_graph.copy()
        route = [starting_city]
        current_city = starting_city
        for i in range(self.number_of_cities-1):
            next_city = np.argmin(roads[current_city, :])
            route.append(next_city)
            roads[current_city, :] = np.inf
            roads[:, current_city] = np.inf
            current_city = next_city
        print(route)
        return route

    def route_distance(self, order_of_cities):
        distance = 0
        for i in range(len(order_of_cities)-1):
            # print(self.weighted_graph[order_of_cities[i], order_of_cities[i+1]])
            distance += self.weighted_graph[order_of_cities[i], order_of_cities[i+1]]
        return distance

    def bidirectional_search(self, starting_city, end_city):
        last_nodes_start = []
        current_city_start = starting_city
        current_city_end = end_city
        last_seen_start = deque()
        last_seen_end = deque()
        visited_cities_start = []
        visited_cities_end = []
        order_of_cities_start = [[None for x in range(self.number_of_cities)] for y in range(self.number_of_cities)]
        index = 0
        nodes_from_start = [np.inf] * self.number_of_cities
        for city in range(self.number_of_cities):
            if self.weighted_graph[current_city_start, city] != np.inf:
                nodes_from_start[city] = self.weighted_graph[current_city_start, city]
                order_of_cities_start[city][index] = current_city_start
                last_seen_start.append(city)
        visited_cities_start.append(current_city_start)

        while (nodes_from_start[end_city] == np.inf):
            index += 1
            current_city_start = last_seen_start.popleft()
            for city in range(self.number_of_cities):
                if self.weighted_graph[current_city_start, city] != np.inf and city not in visited_cities_start:
                    dist = self.weighted_graph[current_city_start, city] + nodes_from_start[current_city_start]
                    if dist < nodes_from_start[city]:
                        nodes_from_start[city] = dist
                        order_of_cities_start[city] = order_of_cities_start[current_city_start]
                        order_of_cities_start[city][index] = current_city_start
            visited_cities_start.append(current_city_start)
        order_of_cities_start[city][index + 1] = end_city
        print(order_of_cities_start[end_city], "\n", nodes_from_start[end_city])

        return order_of_cities_start[end_city][0:index+2]

        # while(nodes_from_start[end_city] == np.inf): #while(last_seen_start not in last_seen_end):
        # #for j in range(self.number_of_cities):
        #     index += 1
        #     last_nodes_start.clear()
        #     current_city_start = last_seen_start.popleft()
        #     for city in range(self.number_of_cities):
        #         if self.weighted_graph[current_city_start, city] != np.inf and city not in visited_cities:
        #             dist = self.weighted_graph[current_city_start, city] + nodes_from_start[current_city_start]
        #             last_nodes_start.append(city)
        #             if dist < nodes_from_start[city]:
        #                 nodes_from_start[city] = dist
        #                 order_of_cities[index] = current_city_start
        #             last_seen_start.append(city)
        #     visited_cities.append(current_city_start)
        # order_of_cities[index + 1] = end_city
        # print(nodes_from_start[end_city], "\n", order_of_cities)
        # return order_of_cities

    def print_grid(self):
        print(self.weighted_graph)
        plt.axis([-100, 100, -100, 100])
        plt.show()


grid = Grid(5)
grid.add_cities()
grid.reduce_roads()
grid.add_weighted_graph()
# order_of_cities = grid.BFS(0)
# print(order_of_cities)
# distance = grid.route_distance(order_of_cities)
# print(distance)
# order_of_cities = grid.DFS(0)
# print(order_of_cities)
# distance = grid.route_distance(order_of_cities)
# print(distance)
# grid.minimum_spanning_tree()
# order_of_cities = grid.greedy_search(0)
# distance = grid.route_distance(order_of_cities)
# print(distance)
# grid.print_grid()
order_of_cities = grid.bidirectional_search(0, 4)
print(order_of_cities)
distance = grid.route_distance(order_of_cities)
print(distance)
