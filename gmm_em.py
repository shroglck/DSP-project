import numpy as np


class GMM:
    def __init__(self, K, ITER):
        # K := number of groups
        # ITER := maximum number of iterations
        self.K = K
        self.ITER = ITER

        # simple K length list containing 1 / K as each element
        self.fraction = [1/self.K for comp in range(self.K)]

    # Calculation of multiplicative normal
    def normal(self, test_samples, mean, covariance_matrix):

        dim = len(test_samples)     # diamention of samples
        dtrm1 = (2 * np.pi) ** dim 
        dtrm2 = np.linalg.det(covariance_matrix)
        up_trm = np.exp(-np.dot(np.dot((test_samples - mean).T, np.linalg.inv(covariance_matrix)), (test_samples - mean))/2)

        res = (dtrm1 * dtrm2) ** (-1 / 2)
        res = res * up_trm

        return res
        # return (2*np.pi)**(-len(test_samples)/2)*np.linalg.det(covariance_matrix)**(-1/2)*np.exp(-np.dot(np.dot((test_samples-mean).T, np.linalg.inv(covariance_matrix)), (test_samples-mean))/2)

    def em_fit(self, test_samples):
        # param test_samples := input 2-d data to be divided

        # spliting the dataset to new_test_samples
        new_test_samples = np.array_split(test_samples, self.K)

        # mean vector | mean of each column
        self.mean = [np.mean(x, axis=0) for x in new_test_samples]
        self.covariance_matrices = [
            np.cov(x.T) for x in new_test_samples]      # covariance matrix

        for iteration in range(self.ITER):

            '''
                resposibillity matrix
                r[n][k] := the probability of sample n to be part of group k
            '''
            self.responsibility = np.zeros((len(test_samples), self.K))

            for n in range(len(test_samples)):
                for k in range(self.K):
                    self.responsibility[n][k] = self.fraction[k] * self.normal(
                        test_samples[n], self.mean[k], self.covariance_matrices[k])
                    self.responsibility[n][k] /= sum([self.fraction[j]*self.normal(
                        test_samples[n], self.mean[j], self.covariance_matrices[j]) for j in range(self.K)])

            # calculating N
            N = np.sum(self.responsibility, axis=0)

            self.mean = np.zeros((self.K, len(test_samples[0])))        # initializing mean 

            for k in range(self.K):
                for n in range(len(test_samples)):
                    self.mean[k] += self.responsibility[n][k] * test_samples[n]             # updating mean
            
            # updating mean
            self.mean = [1 / N[k] * self.mean[k]
                         for k in range(self.K)]
            # now covariance matrices
            self.covariance_matrices = [
                np.zeros((len(test_samples[0]), len(test_samples[0]))) for k in range(self.K)]

            for k in range(self.K):
                self.covariance_matrices[k] = np.cov(
                    test_samples.T, aweights=(self.responsibility[:, k]), ddof=0)

            # updating covariances
            self.covariance_matrices = [
                1 / N[k] * self.covariance_matrices[k] for k in range(self.K)]

            # updating fraction
            self.fraction = [N[k] / len(test_samples) for k in range(self.K)]

    def predict(self, test_samples):
        probabilities = []
        for n in range(len(test_samples)):
            probabilities.append([self.normal(test_samples[n], self.mean[k], self.covariance_matrices[k])
                                  for k in range(self.K)])
        group = []
        for probability in probabilities:
            group.append(probability.index(max(probability)))
        return group
