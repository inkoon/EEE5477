import numpy as np

# Initializing
data_w1 = np.array([
    [0.42, -0.2, 1.3, 0.39, -1.6, -0.029, -0.23, 0.27, -1.9, 0.87],
    [-0.087, -3.3, -0.32, 0.71, -5.3, 0.89, 1.9, -0.3, 0.76, -1.0],
    [0.58, -3.4, 1.7, 0.23, -0.15, -4.7, 2.2, -0.87, -2.1, -2.6]
])

data_w2 = np.array([
    [-0.4, -0.31, 0.38, -0.15, -0.35, 0.17, -0.011, -0.27, -0.065, -0.12],
    [0.58, 0.27, 0.055, 0.53, 0.47, 0.69, 0.55, 0.61, 0.49, 0.054],
    [0.089, -0.04, -0.035, 0.011, 0.034, 0.1, -0.18, 0.12, 0.0012, -0.063]
])

data_w3 = np.array([
    [0.83, 1.1, -0.44, 0.047, 0.28, -0.39, 0.34, -0.3, 1.1, 0.18],
    [1.6, 1.6, -0.41, -0.45, 0.35, -0.48, -0.079, -0.22, 1.2, -0.11],
    [-0.014, 0.48, 0.32, 1.4, 3.1, 0.11, 0.14, 2.2, -0.46, -0.49]
])


############################################
# 1.(a)
mu_hat = []
var_hat = []
for x in data_w1:
    mu_hat.append(np.mean(x))
    var_hat.append(np.var(x))

print('1.(a)')
print('## Maximum-likelihood mean ##')
for i, mu in enumerate(mu_hat):
    print(f'mu_{i+1} = {mu:.6f}')

print('## Maximum-likelihood variance ##')
for i, var in enumerate(var_hat):
    print(f'var_{i+1} = {var:.6f}')

############################################
# 1.(b)
mu_hat_2dim = []
var_hat_2dim = []

def dev(x):
    return np.mean(x) - x

def cov(x1, x2, n):
    return np.dot(x1,x2) / n

n = len(data_w1[0])
dev_x1 = dev(data_w1[0])
dev_x2 = dev(data_w1[1])
dev_x3 = dev(data_w1[2])

# feature 1 & 2
mu = np.zeros((2))
mu[0] = mu_hat[0]
mu[1] = mu_hat[1]
mu_hat_2dim.append(mu)
var = np.zeros((2,2))
var[0][0] = cov(dev_x1, dev_x1, n)
var[0][1] = cov(dev_x1, dev_x2, n)
var[1][0] = cov(dev_x2, dev_x1, n)
var[1][1] = cov(dev_x2, dev_x2, n)
var_hat_2dim.append(var)

# feature 1 & 3
mu = np.zeros((2))
mu[0] = mu_hat[0]
mu[1] = mu_hat[2]
mu_hat_2dim.append(mu)
var = np.zeros((2,2))
var[0][0] = cov(dev_x1, dev_x1, n)
var[0][1] = cov(dev_x1, dev_x3, n)
var[1][0] = cov(dev_x3, dev_x1, n)
var[1][1] = cov(dev_x3, dev_x3, n)
var_hat_2dim.append(var)

# feature 2 & 3
mu = np.zeros((2))
mu[0] = mu_hat[1]
mu[1] = mu_hat[2]
mu_hat_2dim.append(mu)
var = np.zeros((2,2))
var[0][0] = cov(dev_x2, dev_x2, n)
var[0][1] = cov(dev_x2, dev_x3, n)
var[1][0] = cov(dev_x3, dev_x2, n)
var[1][1] = cov(dev_x3, dev_x3, n)
var_hat_2dim.append(var)


print('\n\n1.(b)')
print('for feature 1 & 2')
print('## Maximum-likelihood mean ##')
print(mu_hat_2dim[0])
print('## Maximum-likelihood variance ##')
print(var_hat_2dim[0])

print('\nfor feature 1 & 3')
print('## Maximum-likelihood mean ##')
print(mu_hat_2dim[1])
print('## Maximum-likelihood variance ##')
print(var_hat_2dim[1])

print('\nfor feature 2 & 3')
print('## Maximum-likelihood mean ##')
print(mu_hat_2dim[2])
print('## Maximum-likelihood variance ##')
print(var_hat_2dim[2])

############################################
# 1.(c)
mu_hat_3dim = np.zeros((3))
mu_hat_3dim[0] = mu_hat[0]
mu_hat_3dim[1] = mu_hat[1]
mu_hat_3dim[2] = mu_hat[2]
var_hat_3dim = np.zeros((3,3))
var_hat_3dim[0][0] = cov(dev_x1, dev_x1, n)
var_hat_3dim[0][1] = cov(dev_x1, dev_x2, n)
var_hat_3dim[0][2] = cov(dev_x1, dev_x3, n)
var_hat_3dim[1][0] = cov(dev_x2, dev_x1, n)
var_hat_3dim[1][1] = cov(dev_x2, dev_x2, n)
var_hat_3dim[1][2] = cov(dev_x2, dev_x3, n)
var_hat_3dim[2][0] = cov(dev_x3, dev_x1, n)
var_hat_3dim[2][1] = cov(dev_x3, dev_x2, n)
var_hat_3dim[2][2] = cov(dev_x3, dev_x3, n)


print('\n\n1.(c)')
print('## Maximum-likelihood mean ##')
print(mu_hat_3dim)
print('## Maximum-likelihood variance ##')
print(var_hat_3dim)

############################################
# 1.(d)
mu_hat_2 = []
var_hat_2 = []
for x in data_w2:
    mu_hat_2.append(np.mean(x))
    var_hat_2.append(np.var(x))

mu_hat_diag = np.zeros((3))
mu_hat_diag[0] = mu_hat_2[0]
mu_hat_diag[1] = mu_hat_2[1]
mu_hat_diag[2] = mu_hat_2[2]
var_hat_diag = np.zeros((3, 3))
var_hat_diag[0][0] = var_hat_2[0]
var_hat_diag[1][1] = var_hat_2[1]
var_hat_diag[2][2] = var_hat_2[2]


print('\n\n1.(d)')
print('## Maximum-likelihood mean ##')
print(mu_hat_diag)
print('## Maximum-likelihood variance ##')
print(var_hat_diag)
