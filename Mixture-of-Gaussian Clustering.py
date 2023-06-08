import numpy as np
from sklearn.mixture import GaussianMixture
# read data
def read_training_data(file_path):
    # read width and hight
    data = np.loadtxt(file_path, delimiter=',')
    return data
# train
# my data file
file_path = r"C:\Users\Hasee\Desktop\11.csv"
train_data = read_training_data(file_path)
# Combine width and height into a set of data
data = np.array(train_data)
# Create GMM model
gmm = GaussianMixture(n_components=9)
# Fitting of data
gmm.fit(data)
# Obtain the mean and covariance of clustering
means = gmm.means_
covariances = gmm.covariances_
# Output Results
print("Result of clustering mean:")
print(means)
print("Result of clustering covariances:")
print(covariances)

