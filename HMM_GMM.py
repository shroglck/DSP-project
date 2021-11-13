import numpy as np
from datetime import datetime
from scipy.stats import multivariate_normal as mvn

def random_normalized(d1,d2):
    x=np.random.random((d1,d2))
    return x/x.sum(axis=1,keepdims=True)

class HMM:
    def __init__(self,M,K):
        self.M=M
        self.K=K
    
    def fit(self,X,Z,max_iter):

        sequence_lengths=[]
        for x in X:
            sequence_lengths.append(len(x))
        Xc=np.concatenate(X)
        Z=np.concatenate(Z)
        T=len(Xc)
        start_positions=np.zeros(len(Xc),dtype=np.bool)
        end_positions=np.zeros(len(Xc),dtype=np.bool)
        start_position_values=[]
        last=0
        for length in sequence_lengths:
            start_position_values.append(last)
            start_positions[last]=1
            if last>0:
                end_positions[last-1]=1
            last+=length

        D=X[0].shape[1]

        #randomly initializing all parameters

        self.pi=np.ones(self.M)/self.M
        self.A=np.zeros((self.M,self.M))+1e-50
        self.R=np.ones((self.M,self.K))/self.K
        self.mu=np.zeros((self.M,self.K,D))+1e-50
        
        for i in range(self.M):
            for k in range(self.K):
                random_idx=np.random.choice(T)
                self.mu[i,k]=Xc[random_idx]
        
        self.sigma=np.zeros((self.M,self.K,D,D))+1e-50
        
        for j in range(self.M):
            for k in range(self.K):
                self.sigma[j,k]=np.eye(D)

        
        
        total_cont=np.zeros((self.M,1))+1e-50
        #computing the A matrix
        for t in range(T-1):
            self.A[Z[t],Z[t+1]]+=1
            total_cont[Z[t]]+=1
        self.A=self.A/total_cont
        self.pi=total_cont/T
        #EM algorithm starts
        for it in range(max_iter):
            
            print("it:", it)

            B = np.zeros((self.M, T))+1e-50
            component = np.zeros((self.M, self.K, T)) # we'll need these later
            for j in range(self.M):
                for k in range(self.K):
                    p = self.R[j,k] * mvn.pdf(Xc, self.mu[j,k], self.sigma[j,k])
                    print(p.min())
                    component[j,k,:] = p    
                    B[j,:] += p
            
            

            
            gamma = np.zeros((T, self.M, self.K))+1e-50
            for t in range(T):
                for j in range(self.M):
                    for k in range(self.K):
                        factor=1
                        if(Z[t]==j):
                            factor=1
                        else:
                            factor=1e-340
                        gamma[t,j,k] =  factor* component[j,k,t] / B[j,t]
            r_num = np.zeros((self.M, self.K))+1e-50
            r_den = np.zeros(self.M)+1e-50
            mu_num = np.zeros((self.M, self.K, D))+1e-50
            sigma_num = np.zeros((self.M, self.K, D, D))+1e-50

            r_num_n = np.zeros((self.M, self.K))+1e-50
            r_den_n = np.zeros(self.M)+1e-50
            for j in range(self.M):
                for k in range(self.K):
                    for t in range(T):
                        r_num_n[j,k] += gamma[t,j,k]
                        r_den_n[j] += gamma[t,j,k]
            r_num = r_num_n
            r_den = r_den_n

            mu_num_n = np.zeros((self.M, self.K, D))+1e-50
            sigma_num_n = np.zeros((self.M, self.K, D, D))+1e-50
            for j in range(self.M):
                for k in range(self.K):
                    for t in range(T):
                        # update means
                        mu_num_n[j,k] += gamma[t,j,k] * Xc[t]

                        # update covariances
                        sigma_num_n[j,k] += gamma[t,j,k] * np.outer(Xc[t] - self.mu[j,k], Xc[t] - self.mu[j,k])
            mu_num = mu_num_n
            sigma_num = sigma_num_n
           
            for j in range(self.M):
                for k in range(self.K):
                    self.R[j,k] = r_num[j,k] / r_den[j]
                    self.mu[j,k] = mu_num[j,k] / r_num[j,k]
                    self.sigma[j,k] = sigma_num[j,k] / r_num[j,k] + np.eye(D)
           
            assert(np.all(self.R <= 1))
            assert(np.all(self.A <= 1))
        print("A:", self.A)
        print("mu:", self.mu)
        print("sigma:", self.sigma)
        print("R:", self.R)
        print("pi:", self.pi)

    def save_model(self):
        dict_to_save={}
        dict_to_save["pi"]=self.pi
        dict_to_save["A"]=self.A
        dict_to_save["R"]=self.R
        dict_to_save["mu"]=self.mu
        dict_to_save["sigma"]=self.sigma
        dict_to_save["M"]=self.M
        dict_to_save["K"]=self.K

        np.save("HMM_model.npy",dict_to_save)
        


                


    def get_state_sequence(self, x):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = self.pi*self.B[:,x[0]]  ## replace B with normal 
        for t in range(1, T):
            for j in range(self.M):
                factor=1e-50
                for k in range(self.K):
                    factor+=mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                delta[t,j] = np.max(delta[t-1]*self.A[:,j]) * factor ## replace B with GMM model
                psi[t,j] = np.argmax(delta[t-1]*self.A[:,j])

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states


    