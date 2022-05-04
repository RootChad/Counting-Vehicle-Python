import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distanceBetweenTwoPoint(p1, p2):
        return math.sqrt(
            (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y- p1.y))




