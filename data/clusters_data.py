import random
random.seed(2018)
from random import random, shuffle
import math


# CREDIT
# https://stackoverflow.com/questions/44356063/how-to-generate-a-set-of-random-points-within-a-given-x-y-coordinates-in-an-x


def rand_cluster(n, c, r):
    """returns n random points in disk of radius r centered at c"""
    x, y = c
    points = []
    for i in range(n):
        theta = 2 * math.pi * random()
        s = r * random()
        points.append((x + s * math.cos(theta), y + s * math.sin(theta)))
    return points


def rand_clusters(k, n, r, a, b, c, d):
    """return k clusters of n points each in random disks of radius r
    where the centers of the disk are chosen randomly in [a,b]x[c,d]"""
    clusters = []
    target = []
    for i in range(k):
        x = a + (b - a) * random()
        y = c + (d - c) * random()

        clusters.extend(rand_cluster(n, (x, y), r))
        tmp = [0.0]*k
        for l in range(k):
            if l == i:
                tmp[l] = 1.0
        tmp = [tmp for _ in range(n)]

        target.extend(tmp)
    indexes = range(len(target))
    shuffle(indexes)
    clusters = [clusters[i] for i in indexes]
    target = [target[i] for i in indexes]
    return clusters,target

def rand_linear(n_points, x_max = 10 , x_min = -10, y_max = 10, y_min=-10):
    clusters = []
    target = []
    for _ in range(n_points):
        x = x_min + (x_max - x_min) * random()
        y = y_min + (y_max - y_min) * random()
        clusters.append([x,y])
        if x >= 0 :
            target.append([1])
        else:
            target.append([0])
    return clusters, target



