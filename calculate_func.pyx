cdef extern from "<math.h>":
    float sqrt(float x)

def calculate_distance_4value(float p1x, float p1y, float p2x, float p2y):
    return sqrt((p1x - p2x) ** 2 + (p1y-p2y) ** 2)