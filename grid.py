import numpy as np
from math import ceil, sqrt
import matplotlib.pyplot as plt # used only for visualization
import networkx as nx # used only for visualization
from collections import deque


class Grid:
    def __init__(self, number):
        self.number_of_cities = number
        self.cities = np.zeros((2, self.number_of_cities))  # coordinates of cities
        self.weighted_graph = np.zeros((number, number))
        self.G = nx.Graph()
        self.DFS_arr = []

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
        delete_cities = new_roads - ceil(new_roads * 0.8)
        while delete_cities > 0:
            city_a = np.random.randint(0, number)
            city_b = np.random.randint(0, number)
            if self.weighted_graph[city_a, city_b] != np.inf and city_a != city_b:
                self.weighted_graph[city_a, city_b] = np.inf
                self.weighted_graph[city_b, city_a] = np.inf
                delete_cities -= 1

    def bfs(self, starting_city):
        possible_routes = []
        visited_cities = []
        queue = deque()
        queue.append(starting_city)
        current_route = [queue.popleft()]
        visited_cities = [current_route]
        current_city = starting_city
        for city, length in enumerate(self.weighted_graph[current_city]):
            if length != np.inf:
                auxiliary_route = current_route.copy()
                auxiliary_route.append(city)
                queue.append(auxiliary_route)

        while queue:
            visited_cities.clear()
            current_route = queue.popleft()
            visited_cities = current_route
            current_city = current_route[-1]
            for city, length in enumerate(self.weighted_graph[current_city]):
                if length != np.inf and city not in visited_cities:
                    auxiliary_route = current_route.copy()
                    auxiliary_route.append(city)
                    queue.append(auxiliary_route)
            if(len(current_route) == self.number_of_cities):
                possible_routes.append(current_route.copy())
        # print(possible_routes)

        lengths = []
        for routes in possible_routes:
            last_city = routes[-1]
            if self.weighted_graph[last_city, starting_city] != np.inf:
                routes.append(starting_city)
                lengths.append(self.route_distance(routes))

        result = lengths.index(min(lengths))
        # print("Theoreticall values\n", possible_routes[result], lengths[result], result)
        # print("all values\n", possible_routes, lengths)

        return possible_routes[result], lengths[result]

    def DFS(self, current_route):

        visited_cities = current_route
        possible_routes = []
        last_city = visited_cities[-1]
        for city, length in enumerate(self.weighted_graph[last_city]):
            if length != np.inf and city not in visited_cities:
                auxiliary = visited_cities.copy()
                auxiliary.append(city)
                if len(auxiliary) == self.number_of_cities:
                    if self.weighted_graph[city, visited_cities[0]] != np.inf:
                        auxiliary.append(visited_cities[0])
                        self.DFS_arr.append(auxiliary)
                        return auxiliary
                possible_routes.append(self.DFS(auxiliary))
        if possible_routes == []:
            None
        else:
            return possible_routes

    def DFS_start(self, starting_city):
        visited_cities = []
        possible_routes = []
        for city, length in enumerate(self.weighted_graph[starting_city]):
            if length != np.inf and city not in visited_cities:
                visited_cities.append(city)
                possible_routes.append(self.DFS([starting_city, city]))
        lengths = []
        for routes in self.DFS_arr:
            lengths.append(self.route_distance(routes))

        result = lengths.index(min(lengths))
        return self.DFS_arr[result], lengths[result]

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
        last_nodes_end = []
        current_city_start = starting_city
        current_city_end = end_city
        last_seen_start = deque()
        last_seen_end = deque()
        visited_cities_start = []
        visited_cities_end = []
        order_of_cities_start = [[None for x in range(self.number_of_cities)] for y in range(self.number_of_cities)]
        order_of_cities_end = [[None for x in range(self.number_of_cities)] for y in range(self.number_of_cities)]
        index = 0
        nodes_from_start = [np.inf] * self.number_of_cities
        nodes_from_end = [np.inf] * self.number_of_cities
        for city in range(self.number_of_cities):
            if self.weighted_graph[current_city_start, city] != np.inf:
                nodes_from_start[city] = self.weighted_graph[current_city_start, city]
                order_of_cities_start[city][index] = current_city_start
                last_seen_start.append(city)
                last_nodes_start.append(city)
            if self.weighted_graph[current_city_end, city] != np.inf:
                nodes_from_end[city] = self.weighted_graph[current_city_end, city]
                order_of_cities_end[city][index] = current_city_end
                last_seen_end.append(city)
                last_nodes_end.append(city)
        visited_cities_start.append(current_city_start)
        visited_cities_end.append(current_city_end)
        order_of_cities_start[end_city][index+1] = end_city

        if nodes_from_start[end_city] != np.inf:
            print(nodes_from_start[end_city], "\n", order_of_cities_start[end_city][0:index+2])
            return [starting_city, end_city]

        while (np.isin(last_nodes_start, last_nodes_end).any() == False):
            index += 1
            current_city_start = last_seen_start.popleft()
            current_city_end = last_seen_end.popleft()
            for city in range(self.number_of_cities):
                if self.weighted_graph[current_city_start, city] != np.inf and city not in visited_cities_start:
                    dist = self.weighted_graph[current_city_start, city] + nodes_from_start[current_city_start]
                    if dist < nodes_from_start[city]:
                        nodes_from_start[city] = dist
                        order_of_cities_start[city] = order_of_cities_start[current_city_start]
                        order_of_cities_start[city][index] = current_city_start
                if self.weighted_graph[current_city_end, city] != np.inf and city not in visited_cities_end:
                    dist = self.weighted_graph[current_city_end, city] + nodes_from_end[current_city_end]
                    if dist < nodes_from_end[city]:
                        nodes_from_end[city] = dist
                        order_of_cities_end[city] = order_of_cities_end[current_city_end]
                        order_of_cities_end[city][index] = current_city_end
            visited_cities_start.append(current_city_start)
            visited_cities_end.append(current_city_end)

        connecting_city = np.intersect1d(last_nodes_start, last_nodes_end)
        order_of_cities_end[connecting_city[0]].reverse()
        route = []
        for i in order_of_cities_start[connecting_city[0]]:
            if i != None:
                route.append(i)

        route.append(connecting_city[0])

        for i in order_of_cities_end[connecting_city[0]]:
            if i != None:
                route.append(i)

        distance = nodes_from_start[connecting_city[0] ]+ nodes_from_end[connecting_city[0]]
        return route, distance

    def print_grid(self):
        print(self.weighted_graph)
        plt.axis([-100, 100, -100, 100])
        plt.show()


grid = Grid(4)
grid.add_cities()
grid.reduce_roads()
grid.add_weighted_graph()
order_of_cities, distance = grid.bfs(0)
print(order_of_cities)
# distance = grid.route_distance(order_of_cities)
print(distance)
order_of_cities, distance = grid.DFS_start(0)
print(order_of_cities)
# distance = grid.route_distance(order_of_cities)
print(distance)
# grid.minimum_spanning_tree()
# order_of_cities = grid.greedy_search(0)
# distance = grid.route_distance(order_of_cities)
# print(distance)
# grid.print_grid()
# order_of_cities, distance_v1 = grid.bidirectional_search(0, 4)
# print(order_of_cities)
