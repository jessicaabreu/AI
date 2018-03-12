#!/usr/bin/python
import sys, time, resource, copy
from collections import OrderedDict, deque

goalState = '012345678'

class Tree(object):
    def __init__(self):
        self.data = None
        self.children = None
        self.parent = None
        self.mov = None

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

    while any(frontier):
        currentState = frontier.popleft()
        currentNode = frontier_nodes.popleft()
        delta_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - start_mem
        if delta_mem > max_mem:
            max_mem = delta_mem
        if currentState == goalState:
            end_time = time.time()
            path_to_goal = currentNode.get_path_root()[::-1]
            cost_of_path = len(path_to_goal)
            nodes_expanded = len(exploredStates)
            search_depth = len(path_to_goal)
            max_search_depth = len(frontier_nodes[-1].get_path_root())
            running_time = end_time - start_time
            max_ram_usage = max_mem * 10 ** (-6)
            create_text_file(path_to_goal, cost_of_path, nodes_expanded,
                             search_depth, max_search_depth, running_time,
                             max_ram_usage)
            return
        children = getChildren(currentState, allMovements, indexChange)
        for mov, child in children.iteritems():
            if child not in frontier and child not in exploredStates:
                frontier.append(child)
                node = Tree()
                node.data = child
                node.mov = mov
                node.parent = currentNode
                frontier_nodes.append(node)
        exploredStates.add(currentState)

def dfs(initialState, goalState = goalState):
    start_time = time.time()
    currentState = initialState[:]
    frontier = deque()
    frontier.append(currentState)
    exploredStates = []
    root = Tree()
    root.data = initialState
    frontier_nodes = deque()
    frontier_nodes.append(root)
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_mem = 0
    allMovements = dictAllMovements()
    indexChange = {'Up':-3, 'Down':3, 'Left':-1, 'Right':1}

    while any(frontier):
        currentState = frontier.popleft()
        currentNode = frontier_nodes.popleft()
        delta_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - start_mem
        if delta_mem > max_mem:
            max_mem = delta_mem
        if currentState == goalState:
            end_time = time.time()
            path_to_goal = currentNode.get_path_root()[::-1]
            cost_of_path = len(path_to_goal)
            nodes_expanded = len(exploredStates)
            search_depth = len(path_to_goal)
            max_search_depth = max([len(x.get_path_root()) for x in
                                    frontier_nodes])
            running_time = end_time - start_time
            max_ram_usage = max_mem * 10 ** (-6)
            create_text_file(path_to_goal, cost_of_path, nodes_expanded,
                             search_depth, max_search_depth, running_time,
                             max_ram_usage)
            return
        children = getChildrenDfs(currentState, allMovements, indexChange)
        for mov, child in children.iteritems():
            if child not in frontier and child not in exploredStates:
                frontier.appendleft(child)
                node = Tree()
                node.data = child
                node.mov = mov
                node.parent = currentNode
                frontier_nodes.appendleft(node)
        exploredStates.append(currentState)

def ast(initialState, goalState = goalState):
    pass

def create_text_file(path_to_goal, cost_of_path, nodes_expanded, search_depth,
                     max_search_depth, running_time, max_ram_usage):
    output = open("output.txt", "w+")
    output.write("path_to_goal: %s\r\n" % (path_to_goal))
    output.write("cost_of_path: %d\r\n" % (cost_of_path))
    output.write("nodes_expanded: %d\r\n" % (nodes_expanded))
    output.write("search_depth: %d\r\n" % (search_depth))
    output.write("max_search_depth: %d\r\n" % (max_search_depth))
    output.write("running_time: %.8f\r\n" % (running_time))
    output.write("max_ram_usage: %.8f\r\n" % (max_ram_usage))
    output.close()

def getChildren(state, allMovements):
    '''
    Function provides a dictionary with movements and childrenStates
    '''
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
    '''
    Function provides a dictionary with movements and childrenStates
    '''
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
    '''
    Function receives the current state and returns all movements that can be
    done in lexical order.
    Inputs:
        state: list with current state of the board. As a list because
        assignment probably does not support classes like numpy.
        allMovements: dictionary with possible movements based on position.
        They keys are tuples with (line, column). The values are list of
        possible movements based on the keys.
    Outputs:
        possibleMov: list  of possible movements ('Up', 'Down', 'Left', 'Right')
    '''
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


initialState = sys.argv[2].replace(',', '')
if sys.argv[1] == 'bfs':
    bfs(initialState)
elif sys.argv[1] == 'dfs':
    dfs(initialState)
elif sys.argv[1] == 'ast':
    ast(initialState)
