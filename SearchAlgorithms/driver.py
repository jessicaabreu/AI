#!/usr/bin/python
import sys, time, resource
from collections import OrderedDict, deque

goalState = '012345678'

class Tree(object):
    def __init__(self):
        self.data = None
        self.children = None
        self.parent = None
        self.mov = None
        self.depth = 0

    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

    def get_path_root(self):
        path_to_goal = []
        currentNode = self
        while currentNode.parent:
            path_to_goal.append(currentNode.mov)
            currentNode = currentNode.parent
        return path_to_goal

    def get_len_path(self):

        def generator_parent(node):
            while True:
                yield node
                node = node.parent

        depth = 0
        gen_parent = generator_parent(self)
        node = next(gen_parent)
        while node.parent:
            node = next(gen_parent)
            depth += 1
        return depth

def bfs(initialState, goalState = goalState):
    start_time = time.time()
    path_to_goal = []
    currentState = initialState[:]
    frontier = deque()
    frontier.append(currentState)
    exploredStates = set()
    root = Tree()
    root.data = initialState
    frontier_nodes = deque()
    frontier_nodes.append(root)
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_mem = 0
    allMovements = dictAllMovements()
    indexChange = {'Up':-3, 'Down':3, 'Left':-1, 'Right':1}
    compare_front = set()
    compare_front.add(initialState[:])
    compare_explored = set()

    while any(frontier):
        currentState = frontier.popleft()
        currentNode = frontier_nodes.popleft()
        compare_front.remove(currentState[:])
        delta_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - start_mem
        if delta_mem > max_mem:
            max_mem = delta_mem
        if currentState == goalState:
            path_to_goal = currentNode.get_path_root()[::-1]
            cost_of_path = len(path_to_goal)
            nodes_expanded = len(exploredStates)
            search_depth = len(path_to_goal)
            end_time = time.time()
            running_time = end_time - start_time
            max_ram_usage = max_mem * 10 ** (-6)
            max_search_depth = max(search_depth, frontier_nodes[-1].depth)
            create_text_file(path_to_goal, cost_of_path, nodes_expanded,
                             search_depth, max_search_depth, running_time,
                             max_ram_usage)
            return
        exploredStates.add(currentState)
        compare_explored.add(currentState)
        children = getChildren(currentState, allMovements, indexChange)
        for mov, child in children.iteritems():
            if child not in compare_explored and child not in compare_front:
                frontier.append(child)
                compare_front.add(child)
                node = Tree()
                node.data = child
                node.mov = mov
                node.parent = currentNode
                node.depth = currentNode.depth + 1
                frontier_nodes.append(node)

def dfs(initialState, goalState = goalState):
    start_time = time.time()
    currentState = initialState[:]
    frontier = deque()
    frontier.append(currentState)
    exploredStates = []
    root = Tree()
    root.data = initialState[:]
    frontier_nodes = deque()
    frontier_nodes.append(root)
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_mem = 0
    allMovements = dictAllMovements()
    indexChange = {'Up':-3, 'Down':3, 'Left':-1, 'Right':1}
    max_search_depth = 0
    compare_front = set()
    compare_front.add(initialState[:])
    compare_explored = set()
    d = 0

    while any(frontier):
        currentState = frontier.popleft()
        currentNode = frontier_nodes.popleft()
        compare_front.remove(currentState[:])
        max_search_depth = max(max_search_depth, currentNode.depth)

        delta_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - start_mem
        if delta_mem > max_mem:
            max_mem = delta_mem

        if currentNode.data == goalState:
            path_to_goal = currentNode.get_path_root()[::-1]
            cost_of_path = len(path_to_goal)
            nodes_expanded = len(exploredStates)
            search_depth = len(path_to_goal)
            end_time = time.time()
            running_time = end_time - start_time
            max_ram_usage = max_mem * 10 ** (-6)
            create_text_file(path_to_goal, cost_of_path, nodes_expanded,
                             search_depth, max_search_depth, running_time,
                             max_ram_usage)
            return
        exploredStates.append(currentState)
        compare_explored.add(currentState)
        children = getChildrenDfs(currentNode.data, allMovements, indexChange)
        for mov, child in children.iteritems():
            if child not in compare_explored and child not in compare_front:
                frontier.appendleft(child)
                compare_front.add(child)
                node = Tree()
                node.data = child
                node.mov = mov
                node.parent = currentNode
                node.depth = currentNode.depth + 1
                frontier_nodes.appendleft(node)

def ast(initialState, goalState = goalState):
    start_time = time.time()
    currentState = initialState[:]
    frontier = OrderedDict()
    frontier[initialState] = 0
    explored = {}
    root = Tree()
    root.data = initialState
    frontier_nodes = [root]
    path_to_goal = []
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_mem = 0
    allMovements = dictAllMovements()
    indexChange = {'Up':-3, 'Down':3, 'Left':-1, 'Right':1}
    dict_index_goal = {1:[0, 1], 2:[0, 2], 3:[1, 0], 4:[1, 1], 5:[1, 2], 
                       6:[2, 2], 7:[2, 1], 8:[2, 2]}
    compare_front = set()
    compare_front.add(initialState)
    compare_explored = set()
    explored_nodes = set()

    while any(frontier):
        min_cost = min(frontier.values())
        index_min = int(frontier.values().index(min_cost))
        currentState = str(frontier.keys()[index_min])
        frontier.pop(currentState)
        currentNode = frontier_nodes.pop(index_min)
        delta_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - start_mem
        if delta_mem > max_mem:
            max_mem = delta_mem
        if currentState == goalState:
            path_to_goal = currentNode.get_path_root()[::-1]
            cost_of_path = len(path_to_goal)
            nodes_expanded = len(compare_explored)
            search_depth = currentNode.depth
            max_search_depth = max(search_depth, max([node.depth for node in
                                              explored_nodes]))
            end_time = time.time()
            running_time = end_time - start_time
            max_ram_usage = max_mem * 10 ** (-6)
            create_text_file(path_to_goal, cost_of_path, nodes_expanded,
                             search_depth, max_search_depth, running_time,
                             max_ram_usage)
            return
        compare_front.remove(currentState)
        compare_explored.add(currentState)
        explored[currentState] = min_cost
        explored_nodes.add(currentNode)
        children = getChildrenAST(currentNode, initialState, currentState,
                                  allMovements, indexChange, dict_index_goal)
        # child is the state as a string
        for child, [cost, mov] in children.iteritems():
            if child not in compare_explored and child not in compare_front:
                node = Tree()
                node.data = child
                node.mov = mov
                node.parent = currentNode
                node.depth = currentNode.depth + 1
                frontier[child] = cost
                frontier_nodes.append(node)
                compare_front.add(child)
            elif child in compare_front:
                if cost < frontier[child]:
                    frontier[child] = cost
                    node = Tree()
                    node.data = child
                    node.mov = mov
                    node.parent = currentNode
                    node.depth = currentNode.depth + 1
                    index_child = frontier.keys().index(child)
                    frontier_nodes[index_child] = node
            elif child in compare_explored:
                if cost < explored[child]:
                    # if the node was already explored, but we found a cheaper
                    # way to go, we will update visited and re add the node to
                    # the front
                    explored[child] = cost
                    node_new = Tree()
                    node_new.data = child
                    node_new.mov = mov
                    node_new.parent = currentNode
                    node_new.depth = currentNode.depth + 1
                    frontier[child] = cost
                    frontier_nodes.append(node_new)
                    compare_front.add(child)
                    # if the children are already in the frontier for this
                    # case, than they must be updated costwise and nodewise
                    children_new = getChildrenAST(initialState, child, allMovements, indexChange)
                    for child_new, [cost_new, mov_new] in children_new.iteritems():
                        if child_new in compare_front:
                            frontier[child] = cost_new
                            node_child = Tree()
                            node_child.data = child_new
                            node_child.mov = mov_new
                            node_child.parent = node_new
                            node_child.depth = node_new.depth + 1
                            index_child = frontier.keys().index(child)
                            frontier_nodes[index_child] = node_child

def create_text_file(path_to_goal, cost_of_path, nodes_expanded, search_depth, 
                     max_search_depth, running_time, max_ram_usage):
    output = open("output.txt", "w+")
    output.write("path_to_goal: %s\n" % (path_to_goal))
    output.write("cost_of_path: %d\n" % (cost_of_path))
    output.write("nodes_expanded: %d\n" % (nodes_expanded))
    output.write("search_depth: %d\n" % (search_depth))
    output.write("max_search_depth: %d\n" % (max_search_depth))
    output.write("running_time: %.8f\n" % (running_time))
    output.write("max_ram_usage: %.8f\n" % (max_ram_usage))
    output.close()

def getChildrenAST(currentNode, original_state, state, allMovements,
                   indexChange, dict_index_goal):
    dicChildren = OrderedDict()
    movements = getPossibleMovements(state, allMovements)[:]
    indexZero = state.index('0')
    indexChange = {'Up':-3, 'Down':3, 'Left':-1, 'Right':1}
    list_cost = []
    list_state = []
    final_mov = []
    final_cost = []
    final_state = []
    g = currentNode.depth + 1

    for mov in movements:
        indexes = [indexZero, indexZero + indexChange[mov]]
        indexes.sort(key=int)
        [a, b] = indexes
        newState = state[0:a] + state[b] + state[a+1:b] + state[a] + state[b+1:]
        cost = g + getTotalManhatanDistance(newState, dict_index_goal)
        list_state.append(newState)
        list_cost.append(cost)

    while list_cost:
        index_min = list_cost.index(min(list_cost))
        final_cost.append(list_cost[index_min])
        list_cost.pop(index_min)
        final_state.append(list_state[index_min])
        list_state.pop(index_min)
        final_mov.append(movements[index_min])
        movements.pop(index_min)

    for i in range(len(final_cost)):
        dicChildren[final_state[i]] = [final_cost[i], final_mov[i]]

    return dicChildren

def getChildren(state, allMovements, indexChange):
    movChildren = OrderedDict()
    movements = getPossibleMovements(state, allMovements)
    indexZero = state.index('0')
    indexChange = {'Up':-3, 'Down':3, 'Left':-1, 'Right':1}
    for mov in movements:
        indexes = [indexZero, indexZero + indexChange[mov]]
        indexes.sort(key=int)
        [a, b] = indexes
        newState = state[0:a] + state[b] + state[a+1:b] + state[a] + state[b+1:]
        movChildren[mov] = newState
    return movChildren

def getChildrenDfs(state, allMovements, indexChange):
    movChildren = OrderedDict()
    movements = getPossibleMovements(state, allMovements)[::-1]
    indexZero = state.index('0')
    for mov in movements:
        indexes = [indexZero, indexZero + indexChange[mov]]
        indexes.sort(key=int)
        [a, b] = indexes
        newState = state[0:a] + state[b] + state[a+1:b] + state[a] + state[b+1:]
        movChildren[mov] = newState
    return movChildren

def getPossibleMovements(state, allMovements):
    indexZero = state.index('0')
    line = indexZero / 3
    column = indexZero % 3
    location = (line, column)

    return allMovements[location]

def dictAllMovements():
    from itertools import product

    dictMov = {}
    for i in product([0, 1, 2], repeat=2):
        dictMov[i] = []
        if i[0] != 0:
            dictMov[i].append('Up')
        if i[0] !=2:
            dictMov[i].append('Down')
        if i[1] != 0:
            dictMov[i].append('Left')
        if i[1] != 2:
            dictMov[i].append('Right')
    return dictMov

def getTotalManhatanDistance(state, dict_index_goal):

    def gen_dist_tile(state, n, dist, dict_index_goal):
        while True:
            index_n_state = state.index(str(n))
            line_column_state = [index_n_state / 3, index_n_state % 3]
            line_column_goal = dict_index_goal[n]
            h = abs(line_column_state[0] - line_column_goal[0]) + abs(line_column_state[1] - line_column_goal[1])
            n += 1
            dist += h
            yield n, dist
    dist = 0
    n = 1
    generator = gen_dist_tile(state, n, dist, dict_index_goal)
    while n <= 8:
        n, dist = next(generator)
    return dist

initialState = sys.argv[2].replace(',', '')
if sys.argv[1] == 'bfs':
    bfs(initialState)
elif sys.argv[1] == 'dfs':
    dfs(initialState)
elif sys.argv[1] == 'ast':
    ast(initialState)
