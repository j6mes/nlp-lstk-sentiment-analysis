from __future__ import division

class Leaf:
    word = ""

    def __init__(self,word):
        self.word = word
    def __str__(self):
        return "Leaf node " + self.word

    def getWord(self):
        return self.word

class Node:
    def __init__(self):
        self.children = []

    def addChild(self,child):
        self.children.append(child)

    def getChildren(self):
        return self.children

    def numChildren(self):
        return len(self.children)

    def leaves(self,depth=0):
        leaves = []

        for child in self.children:
            if child.__class__ == Leaf:
                leaves.append([depth,child])
            else:
                leaves.extend(child.leaves(depth+1))
        return leaves


#Generate a tree from the training sample
def tree(model,ids,words):
    nodes = dict()

    word = 0
    node = 1

    for parent in ids.split("|"):
        parent = int(parent)

        if word<len(words.split(" ")):
            nodes[node] = Leaf(words.split(" ")[word])

        if not parent in nodes:
            nodes[parent] = Node()

        nodes[parent].addChild(nodes[node])

        word+=1
        node+=1
    return nodes[0]


def printChildren(node):
    if node.__class__ is Node:
        for child in node.getChildren():
            printChildren(child)
    else:
        print node

