import numpy as np 

import cvxopt


from ml_algo_script.utils import kernels



class SupportVectorMachine:
    def __init__(self,c=1, kernal,power, gamma,coef ):
        self.c=c
        self.kernal=kernal
        self.power=power
        self.gamma=gamma
        self.coef=coef
        self.lagrange_multiplier=None
        self.support_vector=None
        self.support_vector_label=None
        self.intercept=None

    def fit (self, x_train, y_train):

        no_samples, no_feature= x_train.shape


        ##hme kernel_matrix baana hoga 
        kernel_maytrix= np.empty((no_samples,no_samples))
        for i in range(no_samples):
            for j in range(no_samples):
                kernel_maytrix[i,j]= self.kernal(x_train[i],x_train[j])


        ## ab hme lagrange multiplier nikalna hoga
        ## lagrange multiplier nikalne ke liye hme quadratic programming karna hoga
        ## quadratic programming ke liye hme cvxopt library ka use karna hoga
        ## cvxopt library ka use karne ke liye hme lagrange multiplier ki matrix banana hoga
        ## lagrange multiplier ki matrix banane ke liye hme cvxopt library ka use karna hoga
        p = cvxopt.matrix(np.outer(y_train, y_train)*kernel_maytrix,tc='d')
        q=cvxopt.matrix(np.one(no_samples)*-1)
        A= cvxopt.matrix(y_train,(1,no_samples),tc='d')
        b=cvxopt.matrix(0, tc='d')

        G_max = np.identity(no_samples) * -1
        G_min = np.identity(no_samples)
        G = cvxopt.matrix(np.vstack((G_max, G_min)))
        h_max = cvxopt.matrix(np.zeros(no_samples))
        h_min = cvxopt.matrix(np.ones(no_samples) * self.C)
        h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # solving the quadratic programming problem
        minimization = cvxopt.solver.qp(p,q,G,h,A,b)

        #lagrange multiplier in one dimension
        lagr_mult= np.ravel(minimization['x'])

        #get index of non zero lagrange multiplier
        index= lagr_mult > 1e-5
        self.lagrange_multiplier= lagr_mult[index]
        self.support_vector= x_train[index]
        self.support_vector_label= y_train[index]
        #get intercept

        self.intercept=self.support_vector_label[0]
        for i in range (len(self.lagrange_multiplier)):
            self.intercept -= self.lagrange_multiplier[i] * self.support_vector_label[i] * self.kernal(self.support_vector[i],self.support_vector[0])


        
        
    def prediction (self, x_test):
        y_pred=[]


        for sample in x_test:
            pred=0
            for i in range (len (self.lagrange_multiplier)):
                pred += self.lagrange_multiplier[i] * self.support_vector_label[i] * self.kernal(sample,self.support_vector[i])
                pred += self.intercept
                y_pred.append(np.sign(pred))
        return np.array(y_pred)