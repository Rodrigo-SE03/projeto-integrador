import numpy as np
from scipy import stats


class LinearRegression:

    def __init__(self, x, y):
        self.x = x
        self.y = y

        ones = np.ones(self.x.shape[0])
        self.x = np.column_stack((ones, self.x))
        self.b = np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y

        self.y_pred = self.x @ self.b        
        self.r2 = self.get_r_squared()
        self.adj_r2 = self.get_adjusted_r_squared()
        self.p = self.get_p_values()
        self.residuals = self.get_residuals()


    def get_r_squared(self):
        ss_res = np.sum((self.y - self.y_pred) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    
    def get_adjusted_r_squared(self):
        n = len(self.y)
        p = len(self.b) - 1
        r2 = self.get_r_squared()
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return adj_r2
    

    def get_p_values(self):
        mse = np.mean((self.y - self.y_pred)**2)
        cov_matrix = mse * np.linalg.inv(np.dot(self.x.T, self.x))
        T = self.b / np.sqrt(np.diag(cov_matrix))
        p = 2 * (1 - stats.t.cdf(np.abs(T), len(self.x) - 1))
        return p
        
        
    def get_residuals(self):
        return self.y - self.y_pred