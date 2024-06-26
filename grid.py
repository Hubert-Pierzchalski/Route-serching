import numpy as np
from math import ceil, sqrt
import matplotlib.pyplot as plt # used only for visualization
import networkx as nx # used only for visualization
from collections import deque


def is_connected(visited_cities, city_1, city_2):
    new_seen_cities = np.union1d(visited_cities[city_1], visited_cities[city_2])
    for city in visited_cities[city_1]:
        visited_cities[city] = new_seen_cities
    for city in visited_cities[city_2]:
        visited_cities[city] = new_seen_cities
    return visited_cities

def number_of_mins(visited_cities, lengths):
    possible_routes = []
    lowest_length = min(lengths)
    result = lengths.index(min(lengths))
    possible_routes.append(visited_cities[result])
    current_length = lowest_length
    del visited_cities[result]
    del lengths[result]
    while current_length == lowest_length:
        current_length = min(lengths)
        if current_length == lowest_length:
            result = lengths.index(min(lengths))
            possible_routes.append(visited_cities[result])
            del visited_cities[result]
            del lengths[result]
    print(possible_routes)

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
        # edge_labels = nx.get_edge_attributes(self.G, "weight")
        # nx.draw_networkx_edge_labels(self.G, pos, edge_labels)

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
        final_route = []
        for routes in possible_routes:
            last_city = routes[-1]
            if self.weighted_graph[last_city, starting_city] != np.inf:
                routes.append(starting_city)
                lengths.append(self.route_distance(routes))
                final_route.append(routes)

        result = lengths.index(min(lengths))
        # print("Theoreticall values\n", possible_routes[result], lengths[result], result)
        # print("all values\n", possible_routes, lengths)

        return final_route[result], lengths[result]

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

        # number_of_mins(self.DFS_arr, lengths)
        result = lengths.index(min(lengths))
        final_route = self.DFS_arr[result]
        self.DFS_arr.clear()
        return final_route, lengths[result]

    def mst(self):
        road = []
        roads = self.weighted_graph.copy()
        visited_cities = [[y] for y in range(self.number_of_cities)]
        while(len(road) != self.number_of_cities - 1):
            index = np.argmin(roads)
            city_1 = index % self.number_of_cities  #columns
            city_2 = index // self.number_of_cities  #rows
            roads[city_1, city_2] = np.inf
            roads[city_2, city_1] = np.inf
            if city_1 not in visited_cities[city_2]:
                road.append([city_1, city_2])
                np.append(visited_cities[city_1], (city_2))
                np.append(visited_cities[city_2], (city_1))
                # visited_cities[city_2].append(city_1)
                visited_cities = is_connected(visited_cities, city_1, city_2)
        roads = np.full((self.number_of_cities, self.number_of_cities), np.inf)
        for edge in road:
            roads[edge[0], edge[1]] = self.weighted_graph[edge[0], edge[1]]
            roads[edge[1], edge[0]] = self.weighted_graph[edge[1], edge[0]]
        # print(roads)
        # print(road)
        return road, roads

    def approx_mst(self, starting_city):
        road, roads = self.mst()
        possible_routes = []
        curr_longest = 0
        queue = deque()
        queue.append(starting_city)
        current_route = [queue.popleft()]
        visited_cities = [current_route]
        current_city = starting_city
        for city, length in enumerate(roads[current_city]):
            if length != np.inf:
                auxiliary_route = current_route.copy()
                auxiliary_route.append(city)
                queue.append(auxiliary_route)

        while queue:
            current_route = queue.popleft()
            visited_cities = current_route
            current_city = current_route[-1]
            for city, length in enumerate(roads[current_city]):
                if length != np.inf and city not in current_route:
                    auxiliary_route = current_route.copy()
                    auxiliary_route.append(city)
                    queue.append(auxiliary_route)
            if (len(current_route) == curr_longest):
                possible_routes.append(current_route.copy())
            elif (len(current_route) > curr_longest):
                curr_longest = len(current_route)
                possible_routes.clear()
                possible_routes.append(current_route.copy())

        if len(possible_routes[0]) == self.number_of_cities:
            # print("Everything fine") # indication while testing
            lengths = []
            final_routes = []
            for routes in possible_routes:
                last_city = routes[-1]
                if self.weighted_graph[last_city, starting_city] != np.inf:
                    routes.append(starting_city)
                    lengths.append(self.route_distance(routes))
                    final_routes.append(routes)

            if len(final_routes) != 0:
                result = lengths.index(min(lengths))
                print(final_routes[result], lengths[result])
                return final_routes[result]
            else:
                print("Not possible to find route")
                return None

        elif len(possible_routes[0]) == (self.number_of_cities-1):
            # print("Not enough by 1 city") # indication while testing
            dividend = np.arange(self.number_of_cities)
            lengths = []
            final_routes = []
            for routes in possible_routes:
                last_city = routes[-1]
                city_left = np.setdiff1d(dividend, routes)
                if self.weighted_graph[last_city, city_left[0]] != np.inf:
                    routes.append(city_left[0])
                    if self.weighted_graph[city_left[0], starting_city] != np.inf:
                        routes.append(starting_city)
                        lengths.append(self.route_distance(routes))
                        final_routes.append(routes)
            if len(final_routes) != 0:
                result = lengths.index(min(lengths))
                print(final_routes[result])
                return final_routes[result]
            else:
                print("Not possible to find route")
                return None

        else:
            print(f"Not possible to found correct route: {possible_routes}")
            return possible_routes

        # curr_longest = 0
        # queue = deque()
        # queue.append(starting_city)
        # current_route = [queue.popleft()]
        # visited_cities = [current_route]
        # while queue:
        #     visited_cities.clear()
        #     current_route = queue.popleft()
        #     visited_cities = current_route
        #     current_city = current_route[-1]
        #     for city, length in enumerate(self.weighted_graph[current_city]):
        #         if length != np.inf and city not in visited_cities:
        #             auxiliary_route = current_route.copy()
        #             auxiliary_route.append(city)
        #             queue.append(auxiliary_route)
        #     if(len(current_route) == self.number_of_cities):
        #         possible_routes.append(current_route.copy())
        # print(possible_routes)

    def greedy_search(self, starting_city):
        roads = self.weighted_graph.copy()
        route = [starting_city]
        current_city = starting_city
        for i in range(self.number_of_cities-1):
            if min(roads[current_city, :]) == np.inf:
                return False #Some times it's not possible to reache very city
            next_city = np.argmin(roads[current_city, :])
            route.append(next_city)
            roads[current_city, :] = np.inf
            roads[:, current_city] = np.inf
            current_city = next_city
        # i = 0
        # for j, city in enumerate(route):
        #     if self.weighted_graph[route[-1], starting_city] != np.inf:
        #         route.append(starting_city)
        #         return route
        #     else:
        #         route.append(route[-2-i-j])
        #         i += 1

        roads = self.weighted_graph.copy()
        current_city = route[-1]
        while(True):
            if self.weighted_graph[current_city, starting_city] != np.inf:
                route.append(starting_city)
                return route
            next_city = np.argmin(roads[current_city, :])
            route.append(next_city)
            roads[current_city, :] = np.inf
            roads[:, current_city] = np.inf
            current_city = next_city

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
            return [[starting_city, end_city], self.weighted_graph[starting_city, end_city]]

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
        # print(connecting_city)
        final_routes = []
        for middle_city in connecting_city:
            route = []
            order_of_cities_end[middle_city].reverse()
            for i in order_of_cities_start[middle_city]:
                if i != None:
                    route.append(i)

            route.append(middle_city)

            for i in order_of_cities_end[middle_city]:
                if i != None:
                    route.append(i)

            final_routes.append(route)
        length = []
        for route in final_routes:
            length.append(self.route_distance(route))

        # print("breakpoint", final_routes, length)
        index = np.argmin(length)
        return final_routes[index], length[index]

    def print_grid(self):
        # print(self.weighted_graph)
        plt.axis([-100, 100, -100, 100])
        plt.show()

number_of_cities = int(input("Enter the number of cities :"))
grid = Grid(number_of_cities)
grid.add_cities()
grid.reduce_roads()
grid.add_weighted_graph()
starting_city = int(input("Enter the starting city :"))
order_of_cities, distance_bfs = grid.bfs(starting_city)
print(f"BFS: {order_of_cities} \ndistance: {distance_bfs}")
order_of_cities, distance_dfs = grid.DFS_start(starting_city)
print(f"DFS: {order_of_cities} \ndistance: {distance_dfs}")
mst = grid.approx_mst(starting_city)
print(f"MST: {mst}")
order_of_cities = grid.greedy_search(starting_city)
print(f"Greedy: {order_of_cities}")
if order_of_cities:
    distance = grid.route_distance(order_of_cities)
    print(f"distance: {distance}")
grid.print_grid()
city_1 = int(input("Enter starting city: "))
city_2 = int(input("Enter ending city: "))
order_of_cities, distance_v1 = grid.bidirectional_search(city_1, city_2)
print(f"Bidirectional: {order_of_cities}\n distance: {distance_v1}")
