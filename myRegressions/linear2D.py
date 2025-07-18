import numpy as np
import matplotlib.pyplot as plt

class FitResult:
    def __init__(self, slope, intercept, stderr, intercept_stderr, rvalue):
        self.slope = slope
        self.intercept = intercept
        self.stderr = stderr
        self.intercept_stderr = intercept_stderr
        self.rvalue = rvalue


class LinearRegression2D:
    def __init__(self, learning_rate = 0.01, decay = 0.00, tol = 1e-10, patience=10, stochastic = False, batch_size = 0) :
        self.lr = learning_rate
        self.tol = tol
        self.patience = patience
        self.decay = decay
        self.loss_history = []
        self.stochastic = stochastic
        self.batch_size = batch_size
        self.m = 0
        self.b = 0
        self.rvalue = 0
        self.SE_m = None
        self.SE_b = None

    def _loss_function(self, m, b, X, y):
        y_pred = m * X + b
        return np.mean((y - y_pred) ** 2)
    
    def _gradient_descent(self, m_now, b_now, X, y, current_lr):
        n = len(X)
        y_pred = m_now * X + b_now
        error = y - y_pred
        m_grad = -(2/n) * np.sum(X * error)
        b_grad = -(2/n) * np.sum(error)

        m = m_now - current_lr * m_grad
        b = b_now - current_lr * b_grad
        return m, b
    
    def _scale(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.x_min, self.x_max = X.min(), X.max()
        self.y_min, self.y_max = y.min(), y.max()
        X_scaled = (X - self.x_min) / (self.x_max - self.x_min)
        y_scaled = (y - self.y_min) / (self.y_max - self.y_min)
        return X_scaled, y_scaled
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        Xs, ys = self._scale(X, y)
        
        m_scaled = (ys[-1] - ys[0])/(Xs[-1] -Xs[0])
        b_scaled = ys[0] - m_scaled*Xs[0]
        counter = 0
        loss_last = 0
        i = 0

        
        if self.stochastic == False :
            while(i<10000) :
                current_lr = self.lr / (1 + self.decay * i)
                m_scaled, b_scaled = self._gradient_descent(m_scaled, b_scaled, Xs, ys, current_lr)
                
                loss_current = self._loss_function(m_scaled, b_scaled, Xs, ys)
                self.loss_history.append(loss_current)

                if i > 0 and abs(loss_current - loss_last)/loss_last < self.tol:
                    counter += 1
                    if counter >= self.patience:
                        # print(f"Early stopping on {i} epoch, no improvement in {self.patience} epochs.")
                        break
                else:
                    counter = 0
                loss_last = loss_current
                i += 1
        elif self.stochastic == True and self.batch_size >0 :

            permutation = np.random.permutation(len(Xs))
            Xs = Xs[permutation]
            ys = ys[permutation]
            X_batch = []
            y_batch = []
            for batch_start in range(0, len(Xs), self.batch_size):
                X_batch.append(Xs[batch_start:batch_start+self.batch_size])
                y_batch.append(ys[batch_start:batch_start+self.batch_size])
            
            num_of_batch = int(len(Xs)/self.batch_size)

            while(i<10000) :
                current_lr = self.lr / (1 + self.decay * i)
                Xs_batch = X_batch[i% num_of_batch]
                ys_batch = y_batch[i% num_of_batch]
                m_scaled, b_scaled = self._gradient_descent(m_scaled, b_scaled, Xs_batch, ys_batch, current_lr)
                
                loss_current = self._loss_function(m_scaled, b_scaled, Xs_batch, ys_batch)
                self.loss_history.append(loss_current)

                if i > 0 and abs(loss_current - loss_last)/loss_last < self.tol:
                    counter += 1
                    if counter >= self.patience:
                        # print(f"Early stopping on {i} epoch, no improvement in {self.patience} epochs.")
                        break
                else:
                    counter = 0
                loss_last = loss_current
                i += 1
        else :
            raise ValueError("batch_size must be greater than 0. for stochastic= True")


        self.m = m_scaled * (self.y_max - self.y_min) / (self.x_max - self.x_min)
        self.b = self.y_min + (self.y_max - self.y_min) * b_scaled - self.m * self.x_min
        m_err, b_err = self._calculate_std_errors(X, y)
        self.rvalue = self.r_value(X, y)
        return FitResult(slope=self.m, intercept=self.b, stderr=m_err, intercept_stderr=b_err, rvalue=self.rvalue)

    def predict(self, X):
        return self.m * X + self.b
    
    def _calculate_std_errors(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        residuals = y - y_pred
        n = len(X)
        sigma_squared = np.sum(residuals**2) / (n - 2)
        sigma = np.sqrt(sigma_squared)

        x_mean = np.mean(X)
        Sxx = np.sum((X - x_mean)**2)

        self.SE_m = sigma / np.sqrt(Sxx)
        self.SE_b = sigma * np.sqrt(1/n + x_mean**2 / Sxx)
        return self.SE_m, self.SE_b

    def r_value(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        correlation_matrix = np.corrcoef(y, y_pred)
        r = correlation_matrix[0, 1]
        return r
    
    def plot_regression(self, X, y):
        X = np.array(X)
        y = np.array(y)
        x_vals = np.linspace(X.min(), X.max(), 1000)
        y_vals = self.predict(x_vals)

        plt.figure(figsize=(8, 4))
        plt.scatter(X, y, label="data")
        plt.plot(x_vals, y_vals, color='red', label="linear regression")
        plt.legend()
        plt.title("Linear regression")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(16, 5))
        plt.subplot(1,2,1)
        plt.plot(np.arange(len(self.loss_history)), self.loss_history)
        plt.title("loss function")
        plt.xlabel("epochs")
        plt.ylabel("MSE")
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(np.arange(len(self.loss_history) - 1), np.diff(self.loss_history))
        plt.title("Δ loss")
        plt.xlabel("epochs")
        plt.ylabel("Δ MSE")
        plt.grid(True)
        plt.show()
    
    def print_summary(self):
        print(f"m (slope):     {self.m:.4f} ± {self.SE_m:.4f}")
        print(f"b (intercept): {self.b:.4f} ± {self.SE_b:.4f}")
        print(f"rvalue = {self.rvalue:.8f}")
