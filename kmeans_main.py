from kmeans import KMeans
import random

def run():
    model = KMeans(1)
    rows=10
    cols=3

    data = [[0]*cols for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            data[row][col] = row # random.randint(1, 10)

    model.train(data, 20)
    result = model.test(data)

    print(result)


if __name__=='__main__':
    run()