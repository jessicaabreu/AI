from random import randint
import itertools
import numpy
from collections import deque
import math
import copy

from BaseAI import BaseAI
import time
import numpy as np
from Grid import Grid
import math

class PlayerAI(BaseAI):

    def getMove(self, grid):
        # moves = grid.getAvailableMoves()
        # return moves[randint(0, len(moves) - 1)] if moves else None
        root = TreePlay()
        root.grid = grid.clone()
        root.array = np.array(root.grid.map)
        max_depth = 4
        root.type = 'max'
        root.depth = 1
        nextMove = self.DFS_ab(root, max_depth)
        return nextMove

    def DFS_ab(self, root, max_depth):
        frontier = deque()
        frontier.append(root)
        while any(frontier):
            currentNode = frontier.popleft()
            if currentNode.parent:
                if currentNode.parent.a >= currentNode.parent.b:
                    # Branches are "prunned" if test is true
                    # Nodes down the branch are not explored
                    pass
                else:
                    # When node is visited, alpha and beta are updated
                    currentNode.a = float(currentNode.parent.a)
                    currentNode.b = float(currentNode.parent.b)
                    # Test if terminal or limit depth
                    # If true, evaluate and update previous nodes
                    # If false, get children and send them to frontier
                    if currentNode.depth == max_depth or \
                       currentNode.grid.canMove() == False:
                        currentNode.updateABLeaf()
                    else:
                        children = currentNode.getChildren()
                        # Appends children to the left
                        frontier.extendleft(children)
            else:
                # Will come here only once, in beginning
                children = currentNode.getChildren()
                frontier.extendleft(children)
        nextMove = root.bestChild.getNextMove()
        return nextMove

class TreePlay(object):

    def __init__(self):
        self.grid = None
        self.parent = None
        self.array = None
        self.type = None
        self.move = None
        self.depth = None
        self.bestChild = None
        self.children = []
	self.gridBestVal = float("nan")
        self.a = -float("inf")
        self.b = float("inf")

    def getGridValue(self):
        array = self.array
        np.seterr(divide='ignore')
        log = np.log2(array)
        np.seterr(divide='warn')
        log[np.isneginf(log)]=0

        numberAvailableTiles = array[array==0].shape[0]
        merge = self.merge(log)
        monotonicity = self.monotonicity(log)
        grad = self.getGrad(log)
        w_monotonocity = -1
        w_spaces = 0.5
        w_avg = 0.5
        w_grad = -0.4
        w_max = np.array([1.0, 0.4, 0.15])
        ravel_log = log.ravel()
        ravelled_ordered_indexes = np.argsort(ravel_log)
        max_values = np.array([ravel_log[ravelled_ordered_indexes[-1]],
                               ravel_log[ravelled_ordered_indexes[-2]],
                               ravel_log[ravelled_ordered_indexes[-3]]])

        if numberAvailableTiles == 0:
            numberAvailableTiles =1

        arrayValue = np.array([w_monotonocity * (monotonicity),
                               w_grad * grad,
                               w_spaces * numberAvailableTiles**2,
                               np.sum(w_max * max_values)])

        gridValue = np.sum(arrayValue)
	self.gridBestVal = gridValue

        return [gridValue, arrayValue]

    def getGrad(self, array_log):
        gradient_lines = np.gradient(array_log, axis=1)
        gradient_columns = np.gradient(array_log, axis=0)
        gradMetric = np.sum(np.abs(gradient_lines) + np.abs(gradient_columns))
        return gradMetric

    def getPathRoot(self):
        node = self
        path = []
        while node.parent:
            path.append(node.move)
            node = node.parent
        return path

    def merge(self, array_log):
        gradient_lines = np.gradient(array_log, axis=1)
        gradient_columns = np.gradient(array_log, axis=0)
        #mergeLine = gradient_lines[gradient_lines==0].shape[0]
        #mergeCol = gradient_columns[gradient_columns==0].shape[0]
        pairLine = gradient_lines==0
        pairCol = gradient_columns==0
        pairLine = pairLine.astype(int)
        pairCol = pairCol.astype(int)
        mergeLine = np.sum(pairLine * array_log)
        mergeCol = np.sum(pairCol * array_log)
        merge = mergeLine + mergeCol
        return merge

    def monotonicity(self, array_log):
        gradient_lines = np.gradient(array_log, axis=1)
        gradient_columns = np.gradient(array_log, axis=0)
        penalty = 0
        # Checking monotonocity in lines
        for l in range(gradient_lines.shape[0]):
            is_increasing = np.all(gradient_lines[l, :]>=0)
            is_decreasing = np.all(gradient_lines[l, :]<=0)
            if not is_decreasing:
                where_punish = gradient_lines[l, :]>= 0
                where = where_punish.astype(int)
                penalty += np.sum(array_log[l, :] * where)
        # Cheking that columns are decreasing from the top
        for c in range(gradient_columns.shape[1]):
            is_decreasing = np.all(gradient_columns[:, c]<=0)
            is_increasing = np.all(gradient_columns[:, c]>=0)
            if not is_decreasing:
                where_punish = gradient_columns[:, c]>= 0
                where = where_punish.astype(int)
                penalty += np.sum(array_log[:, c] * where)
        return penalty

    def getAvgMax(self):
        array = np.array(self.grid.map)
        indexes = ((1, 1), (1, 2), (2, 1), (2, 2))
        max_mean = 0
        for i in indexes:
            grid = array[i[0]-1:i[0]+2, i[1]-1:i[1]+2]
            mean = grid[grid>0].mean()
            if mean > max_mean:
                max_mean = mean
        return max_mean

    def getWeightedPairs(self):
        '''Allows us to get a metric that increases as pairs with larger
        numbers are more prevalent'''
        def updateMetric((l,c), grid, metric, investigatedPairs):
            to_check = set([(l, c-1), (l, c+1), (l-1, c), (l+1, c)])
            while to_check:
                n = to_check.pop()
                if (n, (l, c)) in investigatedPairs or -1 in n:
                    pass
                else:
                    try:
                        neighbor = grid[n[0], n[1]]
                        if neighbor == grid[l, c] and neighbor > metric:
                            metric = math.log(neighbor)
                            investigatedPairs.add((n, (l, c)))
                    except:
                        pass
            return metric, investigatedPairs

        grid = np.array(self.grid.map)
        (lines, columns) =  grid.shape
        metric = 0
        investigatedPairs = set()
        for l in range(lines):
            for c in range(columns):
                if grid[l, c] != 0:
                    metric, investigatedPairs = updateMetric((l, c), grid, 
                                                             metric, investigatedPairs)
        return metric

    def getChildren(self):
        if self.type == 'min':
            children = self.getMinChildren()
        elif self.type == 'max':
            children = self.getMaxChildren()
        return children

    def getMinChildren(self):
        availableCells = self.grid.getAvailableCells()
        children = deque()
        grid_map = np.ravel(self.array)
        grid_map_copy = np.reshape(grid_map, (4, 4))
        # for each available cell, there could came a 2 or a 4
        for cell in availableCells:
            for value in (2, 4):
                new_grid = Grid()
                new_grid.map = grid_map_copy.tolist()
                new_grid.setCellValue(cell, value)
                child = TreePlay()
                child.grid = new_grid
                child.array = np.array(new_grid.map)
                child.parent = self
                self.children.append(child)
                child.type = 'max'
                child.depth = self.depth + 1
                children.append(child)
        return children

    def getMaxChildren(self):
        availableMoves = self.getAvailableMoves()
        children = deque()
        grid_map = np.ravel(self.array)
        grid_map_copy = np.reshape(grid_map, (4, 4))
        # for each avaiable move, there will be a grid child
        for direction in availableMoves:
            new_grid = Grid()
            new_grid.map = grid_map_copy.tolist()
            new_grid.move(direction)
            child = TreePlay()
            child.grid = new_grid
            child.array = np.array(new_grid.map)
            child.parent = self
            self.children.append(child)
            child.type = 'min'
            child.depth = self.depth + 1
            child.move = direction
            children.append(child)
        return children

    def getNextMove(self):
        # Won't work on first node
        # Use only after alpha beta prunning is complete
        node = self
        while node.parent:
            move = node.move
            node = node.parent
        return move

    def updateABLeaf(self):
        gridValue = self.getGridValue()[0]
        self.bestChild = self
        self.gridBestVal = gridValue
        self.updateABUpwards()

    def updateABUpwards(self):
        node = self
        while node.parent:
            kids = [x.gridBestVal for x in node.parent.children]
            if node.type == 'min':
                max_grid = np.nanmax(kids)
                if max_grid > node.parent.a:
                    node.parent.a = float(max_grid)
                index_max_grid = kids.index(max_grid)
                bestChild = node.parent.children[index_max_grid].bestChild
                node.parent.bestChild = bestChild
                node.parent.gridBestVal = float(max_grid)
            elif node.type == 'max':
                min_grid = np.nanmin(kids)
                if min_grid < node.parent.b:
                    node.parent.b = float(min_grid)
                index_min_grid = kids.index(min_grid)
                bestChild = node.parent.children[index_min_grid].bestChild
                node.parent.bestChild = bestChild
                node.parent.gridBestVal = float(min_grid)
            node = node.parent

    def getAvailableMoves(self):
        dirs = [UP, DOWN, LEFT, RIGHT] = range(4)
        availableMoves = []
        array_map = self.array.ravel()
        mapCopy = np.reshape(array_map, (4, 4))

        for x in dirs:
            gridCopy = Grid()
            gridCopy.map = mapCopy.tolist()

            if gridCopy.move(x):
                availableMoves.append(x)

        return availableMoves

    def __str__(self, level=0):
        ret = "\t"*level+repr([self.move, self.gridBestVal])+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'
