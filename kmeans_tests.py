import unittest
from kmeans import KMeans

class TestKMeans(unittest.TestCase):
    def test_train_method(self):
        model=KMeans(1)
        rows=2
        cols=3
        data = [[1, 1, 1], [1, 1, 1]]
        model.train(data, 10)
        result = model.test(data)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(result[0], [1, 1, 1])


if __name__=='__main__':
    unittest.main()
