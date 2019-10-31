import numpy as np
from util import generate_data, eval_rmse
import time

n = 10000
m = 10000
if __name__ == '__main__':
    train, test = generate_data()

    start = time.time()
    i = np.ones((n, 1))
    w = np.sqrt((train ** 2).dot(i))
    sim = np.dot(train, train.T) / w.dot(w.T)

    # indicate which movie was scored
    indicate_matrix = train > 0
    final_score = sim.dot(train) / sim.dot(indicate_matrix)

    RMSE = eval_rmse(test, final_score)

    print("time cost: ", time.time() - start, " s")
    print(RMSE)

