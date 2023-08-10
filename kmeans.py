import random
import math

class KMeans:
    def __init__(self, k):
        self.k = k
        self.centroids=[]
    
    def train(self, data, num_iterations):
        random.seed(31)
        # data = 2d list of size row*cols
        rows, cols = len(data), len(data[0])
        assert self.k < rows
        random_indexes=set()
        while len(random_indexes)!=self.k:
            random_index = random.randint(0, rows-1)
            if random_index not in random_indexes:
                random_indexes.add(random_index)
        
        random_indexes = list(random_indexes)
        self.centroids=[data[index] for index in random_indexes]

        iterations=0
        while iterations<num_iterations:
            assignments = self.assign_clusters(data)
            self.centroids = self.compute_centroids(data, assignments)
            iterations+=1
    
    def test(self, data):
        assignments = self.assign_clusters(data)
        result = []
        for assign_index in range(len(assignments)):
            cluster = self.centroids[assignments[assign_index]]
            result.append(cluster[:])
        
        return result
    
    def assign_clusters(self, data):
        assignments=[0]*len(data)
        for data_index in range(len(data)):
            data_point = data[data_index]
            closest_cluster=-1
            closest_distance=float('inf')
            for cluster_index in range(len(self.centroids)):
                cur_distance = self.compute_distance(data_point, self.centroids[cluster_index])
                if cur_distance < closest_distance:
                    closest_distance = cur_distance
                    closest_cluster = cluster_index
            assignments[data_index] = closest_cluster
        
        return assignments
    
    def compute_centroids(self, data, assignments):
        centroids=[[0]*len(data[0]) for _ in range(self.k)]
        num_points=[0]*self.k
        for assign_index in range(len(assignments)):
            cluster_index = assignments[assign_index]
            num_points[cluster_index]+=1
            for col_index in range(len(data[0])):
                centroids[cluster_index][col_index] += data[assign_index][col_index]
        
        for index in range(self.k):
            for col_index in range(len(data[0])):
                centroids[index][col_index] = centroids[index][col_index]/num_points[index]

        return centroids
    
    def compute_distance(self, data1, data2):
        distance=0
        for index in range(len(data1)):
            distance += (data1[index]-data2[index])*(data1[index]-data2[index])
        
        return math.sqrt(distance)


    





        

