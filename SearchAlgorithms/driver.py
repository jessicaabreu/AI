#!/usr/bin/python
import sys, time, resource, copy
from collections import OrderedDict

goalState = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

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
        currentNode = copy.copy(self)
        while currentNode.parent:
            path_to_goal.append(currentNode.mov)
            currentNode = copy.copy(currentNode.parent)
        return path_to_goal

def bfs(initialState, goalState = goalState):
    start_time = time.time()
    path_to_goal = []
    nodes_expanded = 0
    search_depth = 0
    max_search_depth = 0
    currentState = initialState[:]
    frontier = [currentState]
    exploredStates = []
    root = Tree()
    root.data = initialState
    frontier_nodes = [copy.copy(root)]
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_mem = 0

    while any(frontier):
        currentState = frontier.pop(0)
        currentNode = frontier_nodes.pop(0)
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
        children = getChildren(currentState)
        for mov, child in children.iteritems():
            if child not in frontier + exploredStates:
                frontier.append(child)
                node = Tree()
                node.data = child
                node.mov = mov
                node.parent = currentNode
                frontier_nodes.append(node)
        exploredStates.append(currentState)

def dfs(initialState, goalState = goalState):
    frontier = [initialState.index('0')]
    visited = []
    pass

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
    output.write("running_time: %f\r\n" % (running_time))
    output.write("max_ram_usage: %f\r\n" % (max_ram_usage))
    output.close()

def getChildren(state):
    '''
    Function provides a dictionary with movements and childrenStates
    '''
    movChildren = OrderedDict()
    movements = getPossibleMovements(state)
    indexZero = state.index('0')
    indexChange = {'Up':-3, 'Down':3, 'Left':-1, 'Right':1}
    for mov in movements:
        newState = state[:]
        newIndex = indexZero + indexChange[mov]
        newState[newIndex] = '0'
        newState[indexZero] = state[newIndex]
        movChildren[mov] = newState
    return movChildren

def getPossibleMovements(state):
    '''
    Function receives the current state and returns all movements that can be
    done in lexical order.
    Inputs:
        state: list with current state of the board. As a list because
        assignment probably does not support classes like numpy.
    Outputs:
        possibleMov: list  of possible movements ('Up', 'Down', 'Left', 'Right')
    '''
    possibleMov = []
    indexZero = state.index('0')
    line = indexZero / 3
    column = indexZero % 3
    if line != 0:
        possibleMov.append('Up')
    if line != 2:
        possibleMov.append('Down')
    if column != 0:
        possibleMov.append('Left')
    if column != 2:
        possibleMov.append('Right')
    return possibleMov



initialState = list(sys.argv[2].split(','))
if sys.argv[1] == 'bfs':
    bfs(initialState)
elif sys.argv[1] == 'dfs':
    dfs(initialState)
elif sys.argv[1] == 'ast':
    ast(initialState)
