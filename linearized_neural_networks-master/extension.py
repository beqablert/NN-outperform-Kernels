import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sklearn.cluster
import sklearn.decomposition
import torch.nn as nn
import torch
from torch import optim
from torch import relu as relu
from jax import random
from neural_tangents import stax
import neural_tangents as nt
import scipy.sparse as ss

device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def make_linsep_GMM(dim, N, var,plot,mu_r_1=None,mu_r_2=None): 
  '''
  This Function generates the gaussian mixtrue models. Set plot = True to inspect the first four dimensions visually. 
  input:  dim     = dimension D
          N       = number of samples 
          var     = standard deviation (sigma) for all clusters 
          mu_r_1  = scaling facor for the distance of the cluster centers to the origin, default=None
          mu_r_2  = scaling facor for the distance of the cluster centers to the origin, default=None
  output: X       = data points of shape [N, dim]
          Y       = labels of shape [N]
          mus     = means of the 4 GMs of shape [4, dim]
  '''
  if mu_r_1==None:
    mu_r_1 = math.sqrt(dim)
  if mu_r_2 == None:
    mu_r_2 = math.sqrt(dim)

  
  # Cluster means of the 4 GMs in the first two dimensions. 
  # If mu_r is set to none, then the cluster centers will be (0,±1) and (±1, 0).

  mu1 = [0,                         mu_r_2/math.sqrt(dim)]
  mu2 = [0,                         (-1)*mu_r_2/math.sqrt(dim)]
  mu3 = [mu_r_1/math.sqrt(dim),     0]
  mu4 = [(-1)*mu_r_1/math.sqrt(dim),0]

  # Cluster means of the 4 GMs for the other D - 2 dimensions set to zero.
  if dim>2:
    mu1 = np.append(mu1, np.zeros((dim-2), dtype=int))
    mu2 = np.append(mu2, np.zeros((dim-2), dtype=int))
    mu3 = np.append(mu3, np.zeros((dim-2), dtype=int))
    mu4 = np.append(mu4, np.zeros((dim-2), dtype=int))

  # Shared diagonal coariance matrix.
  
  cov = np.eye(dim)* (var**2) 

  # Sampled datapoints from the 4 multivariate gaussians.

  cluster1 = np.random.multivariate_normal(mu1, cov, size = int(N/4) , check_valid='warn', tol=1e-8)
  cluster2 = np.random.multivariate_normal(mu2, cov, size = int(N/4) , check_valid='warn', tol=1e-8)
  cluster3 = np.random.multivariate_normal(mu3, cov, size = int(N/4) , check_valid='warn', tol=1e-8)
  cluster4 = np.random.multivariate_normal(mu4, cov, size = int(N/4) , check_valid='warn', tol=1e-8)

  # Labels for the 4 GMs according to the 2 clusters of an XOR distribution. 
  label1 = np.ones(int(N/4), dtype=int)*(0)
  label2 = np.ones(int(N/4), dtype=int)*(0)
  label3 = np.ones(int(N/4), dtype=int)*(1)
  label4 = np.ones(int(N/4), dtype=int)*(1)

  if plot==True:
    
    # This part visualizes the first four dimensions of the data. 
    
    plt.scatter(cluster1[:,0],cluster1[:,1] , color='red')
    plt.scatter(cluster2[:,0],cluster2[:,1] , color='blue')
    plt.scatter(cluster3[:,0],cluster3[:,1] , color='red')
    plt.scatter(cluster4[:,0],cluster4[:,1] , color='blue')
    plt.title('Input Space')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.gca().set_xticks([])
    plt.xticks([])
    plt.gca().set_yticks([])
    plt.yticks([])
    plt.xlim([-6,6])
    plt.ylim([-6,6])
    plt.show()

  return np.vstack((cluster1, cluster2, cluster3, cluster4)), np.hstack((label1, label2, label3, label4)) , np.vstack((mu1, mu2, mu3, mu4))

def oracle(X, mu, linsep):
  """
  This function implements the 'oracle' which is defined as a network "with knowledge of the means of 
  the mixture that assigns to each input the label of the nearest mean". 
  
  Input:  X       = data points of shape [N, dim]
          mu      = means of the 4 GMs of shape [4, dim]
  Output: labels  = assigned cluster to each datapoints of shape [N]
  """ 
  oracle = sklearn.cluster.KMeans(n_clusters=4, init=mu, n_init=1).fit(X)
  labels = oracle.labels_  

  ind1 = np.where(labels==0)[0]
  ind2 = np.where(labels==1)[0]
  ind3 = np.where(labels==2)[0]
  ind4 = np.where(labels==3)[0]

  cluster1 = np.hstack((ind1, ind2))
  cluster2 = np.hstack((ind3, ind4))

  if linsep == False:
    labels[labels==0] = 0
    labels[labels==1] = 0
    labels[labels==2] = 1
    labels[labels==3] = 1
  if linsep == True:
    labels[labels==0] = -1
    labels[labels==1] = 1
    labels[labels==2] = -1
    labels[labels==3] = 1

  return labels 

def make_splits(X, Y):
  '''
  input:  X       = data points of shape [N, dim]
          Y       = labels of shape      [N]
  output: X_train = 2/3 of the datapoints used for trainig of shape     [2N/3, dim]
          X_val   = 1/3 of the datapoints used for validation of shape  [N/3, dim]
          Y_train = 2/3 of the labels used for trainig of shape         [2N/3]
          Y_val   = 1/3 of the labels used for validation of shape      [N/3]
  '''
  N = np.shape(Y)[0]
  indices = np.arange(N)
  np.random.shuffle(indices)

  X = X[indices]
  Y = Y[indices]
  X_train = X[0:int(N*0.66),:]
  X_val   = X[int(N*0.66):,:]
  Y_train = Y[0:int(N*0.66)]
  Y_val   = Y[int(N*0.66):]

  return torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(Y_train), torch.from_numpy(Y_val)
  # return X_train, X_val, Y_train, Y_val

def plot_input_feature_spaces(dim, sigma, mu_r_1, mu_r_2):
  N   = 500
  X, Y, m     = make_GMM(dim , N , var = sigma, plot=True, mu_r_1=mu_r_1, mu_r_2=mu_r_2)
  F           = torch.randn((dim,dim*10))
  X_trafo     = transform_RF(X=X,F=F)
  fig         = plt.figure(figsize = (10, 7))
  plt.rcParams.update({'font.size': 16})
  ax          = fig.add_subplot(projection='3d')
  ax.scatter(X_trafo[:int(N/2),0],X_trafo[:int(N/2),1],X_trafo[:int(N/2),2],color='red')
  ax.scatter(X_trafo[int(N/2):int(N),0],X_trafo[int(N/2):int(N),1],X_trafo[int(N/2):int(N),2],color='blue')
  frame1      = plt.gca()
  frame1.axes.xaxis.set_ticklabels([])
  frame1.axes.yaxis.set_ticklabels([])
  frame1.axes.zaxis.set_ticklabels([])
  ax.set_xlabel('z1')
  ax.set_ylabel('z2')
  ax.set_zlabel('z3')
  plt.title('Feature Space of RF')
  plt.show()

class Student(nn.Module):
  """
  This is the 2-layerd neuronal network with K hidden neurons and 1 output neuron, used thoughtout this report. 
  """
  def __init__(self,K,N,weight_std_initial_layer=1):
      """ 
      Input:  K                         = number of hidden neurons
              N                         = number of samples 
              weight_std_initial_layer  = standard deviation for the weight initialization of the first
      """
      print("Creating a Student with InputDimension: %d, K: %d"%(N,K) )
      super(Student, self).__init__()
      
      self.N=N
      self.g=nn.ReLU()
      self.K=K
      self.loss=nn.MSELoss(reduction='mean')
      # Definition of the 2 layers 
      self.fc1 = nn.Linear(N, K, bias=False)
      self.fc2 = nn.Linear(K, 1, bias=False)

      ##For Figure 1 reproduction   
      #torch.nn.init.xavier_uniform_(self.fc2.weight)
      #torch.nn.init.xavier_uniform_(self.fc1.weight)
      nn.init.normal_(self.fc1.weight)
      nn.init.normal_(self.fc2.weight)

      ##For figure 4 reproduction
      #nn.init.normal_(self.fc1.weight,std=weight_std_initial_layer)
      #nn.init.normal_(self.fc2.weight,std=weight_std_initial_layer)


  def forward(self, x):
      # This is the input to the hidden layer. 
      x=self.fc1(x) /math.sqrt(self.N)
      x=self.g(x)
      x = self.fc2(x)
      return x

def HalfMSE(output, target): 
    loss = (0.5)*torch.mean((output - target)**2)
    return loss

def linear(x):
  return x

def centered_relu(x,var):
    a = math.sqrt(var)/math.sqrt(2*math.pi)
    return torch.relu(x)-a
    
def transform_RF(X,F):
    """
    This function tansforms the datapoints X into a feature space of P>>dim, with the 
    transform-matrix F. 
    Input:  X       = data points of shape [N, dim]
            F       = transformation matrix of shape [dim, P]
    Output: X_trafo = transformed datapoints in the feature space of shape [N, P]
    """
    D, P = F.shape
    X    = torch.from_numpy(X)
    X    = X.float()
    F   /= F.norm(dim=0).repeat(D, 1)
    F   *= math.sqrt(D)
    X_trafo = centered_relu((X@F) / math.sqrt(D),0)
    return X_trafo

class Student_RF(nn.Module):
  """
  This is the second layer for the Random Features, which takes the projected datapoints 
  and predcits the cluster labels via a linear model. 
  """
  def __init__(self,K,N,bias=False):
    """ 
    Input:  K                         = number of hidden neurons
            N                         = number of samples 
    """
    print("Creating a Student with InputDimension: %d, K: %d"%(N,K) )
    super(Student_RF, self).__init__()
    
    self.P=N
    self.g=linear
    self.K=1
    self.loss=nn.MSELoss(reduction='mean')
    self.fc1 = nn.Linear(self.P, K, bias)
    nn.init.normal_(self.fc1.weight,std=0.01)

  def forward(self, x):
    x = self.g(self.fc1(x)/math.sqrt(self.P))
    return x

class Student_NT(nn.Module):

    def __init__(self, K, D, std=0.01):
        """ initialisation of a student with:
        - K hidden nodes
        - N input dimensions """
        print("Creating a Neural Network ")
        super(Student_NT, self).__init__()
        self.a0 = (torch.randn(1,1,K)) #.cuda()
        a0 = torch.from_numpy(np.load('/content/gdrive/MyDrive/a0.npy'))
        self.a0 = a0
        fc1_w = torch.from_numpy(np.load('/content/gdrive/MyDrive/fc1.npy'))
        fc2_w = torch.from_numpy(np.load('/content/gdrive/MyDrive/fc2.npy'))
        G = torch.from_numpy(np.load('/content/gdrive/MyDrive/G.npy'))
        self.G = G
        self.g = nn.ReLU()
        self.K = K
        self.loss = nn.MSELoss()
        #First layer weights are fixed!
        # self.w = np.random.randn(D, K)
        # norm = np.linalg.norm(self.w,axis=0,keepdims=True)
        # self.w = self.w/(norm)
        # self.w = torch.from_numpy(self.w)
        self.w = fc1_w
        self.w = self.w.float()
        self.w = self.w #.cuda()
        # self.w = self.w.T
        self.fc1 = nn.Linear(D, K, bias=False)
        self.fc1.weight = nn.Parameter(self.w, requires_grad=False)  # Fix initialized weight
        self.fc2 = nn.Linear(K, 1, bias=True)
        nn.init.normal_(fc2_w, std=std)
        #torch.nn.init.xavier_uniform_(self.fc2.weight)
        #torch.nn.init.xavier_uniform_(self.fc1.weight)
        # self.G = (torch.randn(K, D)) #.cuda()
        # np.save('/content/gdrive/MyDrive/G.npy', (self.G).detach().numpy())
        # np.save('/content/gdrive/MyDrive/a0.npy', (self.a0).detach().numpy())
        # np.save('/content/gdrive/MyDrive/fc1.npy', (self.fc1.weight).detach().numpy())
        # np.save('/content/gdrive/MyDrive/fc2.npy', (self.fc2.weight).detach().numpy())

    def forward(self, x):
        # input to hidden
        x_ = x #/ torch.mean(torch.sqrt(torch.linalg.norm(x, axis=0, keepdims=True)))
        z = self.fc1(x)
        q = self.g(z)
        RF = self.fc2(q)
        zero_one_mat = torch.sign(z)
        # print(zero_one_mat.shape)
        zero_one_mat_exp = torch.unsqueeze(zero_one_mat, 2)
        # print(zero_one_mat_exp.shape)
        zero_one_mat_exp = zero_one_mat_exp.reshape((zero_one_mat_exp.shape[0], 1, self.K))
        U = torch.multiply(zero_one_mat_exp,self.a0)
        q2 = torch.tensordot(U,self.G,dims=([2],[0]))
        x_ = x_ #/ torch.linalg.norm(x_,axis=0,keepdims=True) # bigger norm input would cause the NT-matrix to dominate the output which lead to worse learning
        temp = torch.unsqueeze(x_, 2)
        aux_data = temp.reshape((temp.shape[0],1,temp.shape[1]))  # bs x 1 x d
        temp = torch.multiply(q2, aux_data)
        NT = temp.sum(2)  # bs x num_class
        x = NT + RF
        return x

def PCA(X, Y):
    # X     [N, dim]
    # U     [4, dim]
    # X_pca [N, 4]
    # Y_pca [N, 1]
    pca = sklearn.decomposition.PCA(n_components=4).fit(X)
    U   = pca.components_
    X_pca = np.matmul(X, U.T)
    Y_pca = Y
    return X_pca, Y_pca

N           = 500
dim         = 1000
sigma       = 1e-2
X, Y, mu    = make_linsep_GMM(dim = dim, N = N , var = sigma, plot = False)
X_pca,Y_pca = PCA(X,Y)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('first & second PC Dimensions')
plt.scatter( X_pca[:,0], X_pca[:,1], c= Y_pca)
plt.subplot(1, 3, 2)
plt.title('second & third PC Dimensions')
plt.scatter( X_pca[:,1], X_pca[:,2], c= Y_pca)
plt.subplot(1, 3, 3)
plt.title('third & fourth PC Dimensions')
plt.scatter( X_pca[:,2], X_pca[:,3], c= Y_pca)
# plt.show()

def log_sigmas(num_sigmas):
  """
  Defines the sigmas, that will be used to generate Figure 1.
  """
  sigma1 = np.logspace(-2, -1, num= int(num_sigmas/3))
  sigma2 = np.logspace(-1, 0,  num= int(num_sigmas/3))
  sigma3 = np.logspace(0 , 1,  num= int(num_sigmas/3))
  sigma  = np.round(np.append(sigma1, np.append(sigma2, sigma3)), 5)
  return sigma

######### definition of input parameters ############################# 
######################################################################
N           = 50000
dim         = 1000   # for actual run make this 1000
num_sigmas  = 15*3
sigma       = log_sigmas(num_sigmas)

reg_RF   = 0.0        # regulaization parameter
lr       = 0.1        # learning rate 
RF_error = np.zeros((num_sigmas))

######### initilize the second layer  for RF #########################
######################################################################
# student = Student_RF(N=4,K=1)
# params  = []
# params += [{'params': student.fc1.parameters(),'lr': lr,'weight_decay':reg_RF}]
# optimizer = optim.SGD(params, lr=lr, weight_decay=reg_RF)
# criterion = student.loss 

######## initilize the second layer  for NTK #########################
######################################################################
student = Student_NT(D=1000, K=12, std=0.1)
params  = []
params += [{'params': student.fc1.parameters(),'lr': lr,'weight_decay':reg_RF}]
optimizer = optim.SGD(params, lr=lr, weight_decay=reg_RF)
criterion = student.loss

######## initilize the second layer  for NTK #########################
######################################################################
init_fn, apply_fn, kernel_fn = stax.serial(stax.Dense(12), stax.Relu(), stax.Dense(1))

######### iterate over the sigmas  ###################################
######################################################################

for i in range(0,num_sigmas):
  X, Y, mu  = make_linsep_GMM(dim = dim, N=N , var = sigma[i], plot = False)
  # X, Y        = PCA(X_,Y_)                         

  X_train, X_val, Y_train, Y_val = make_splits(X, Y)
  X_val = (X_val).float()
  Y_val = (Y_val).float()

  ######### Prediction with NTK KRR ###################################
  ######################################################################

  X_train = (X_train).detach().numpy()
  Y_train = (Y_train).detach().numpy()
  X_val = (X_val).detach().numpy()
  Y_val = (Y_val).detach().numpy()
  n = len(Y_train)
  kernel = kernel_fn(X_train, X_train, 'ntk')
  # Y_train = np.reshape(Y_train, (3300,1))
  predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, Y_train)
  # print(X_train.shape)
  # print(Y_train.shape)
  preds = predict_fn(x_test=X_val, get='ntk')
  # kernel_test = kernel_fn(X_val, X_train, 'ntk')
  # # print(kernel_test)


  # cg = ss.linalg.cg(kernel, Y_train, maxiter=400, atol=1e-4, tol=1e-4)
  # x = np.copy(cg[0]).reshape((n, 1))
  # # # print(x)
  # preds = np.dot(kernel_test, x)
  # preds = preds - np.mean(preds)
  # print(preds)

  ######### Training the RF with online SGD on halfMSE #################
  ######################################################################

  ######### Training the NTK with batches SGD on halfMSE #################
  ######################################################################
  # student.train() 
  # for j in range(49): 
  # #   targets   = (Y_train[j]).float() 
  # #   inputs    = (X_train[j,:]).float()
  #   student.zero_grad()
  #   preds     = student(X_train[1000*j:(1000*(j+1)-1)].float())
  #   loss      = HalfMSE(preds, Y_train[1:1000].float()) 
  #   loss.backward()
  #   torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
  #   optimizer.step()

  ######### Evaluation of the training of RF on the classification error 
  ######################################################################
  # # student.eval() 
  # preds = torch.from_numpy(np.array(preds))
  # print(preds)
  # Y_val = torch.from_numpy(Y_val)
  # with torch.no_grad():
  #   # preds = student(X_val)
  #   # preds = preds[:,0]

  #   eg =  HalfMSE(preds, Y_val) 
  #   # calculate the classification error with the predictions
  #   eg_class = 1-torch.relu(torch.sign(preds).reshape(1, len(Y_val))*Y_val)
  #   print(Y_val)
  #   print(eg_class)
  #   eg_class = eg_class.sum()/float(preds.shape[0])
  #   #print("preds:{}, y_val:{}".format(preds,Y_val))
  #   RF_error[i] = eg_class 
  #   print("Test Data: Classification Error: {}; Variance: {}; halfMSE-Loss:{}".format(np.round(RF_error[i], 3),np.round(sigma[i], 3),eg))
  #   print("---------------------------------------------------------")

  ######### Evaluation of the training of NTK on the classification error 
  ######################################################################
  student.eval() 
  with torch.no_grad():
    preds = student(X_val)
    preds = preds[:,0]
    # print(preds.shape)
    # print(Y_val.shape)
    preds[preds==0] = -1
    Y_val[Y_val==0] = -1
    eg =  HalfMSE(preds, Y_val) 
    # calculate the classification error with the predictions
    eg_class = 1 - torch.relu(torch.sign(preds)*Y_val)
    # eg_class = (preds != Y_val)
    eg_class = eg_class.sum()/float(preds.shape[0])
    #print("preds:{}, y_val:{}".format(preds,Y_val))
    RF_error[i] = eg_class 
    print("Test Data: Classification Error: {}; Variance: {}; halfMSE-Loss:{}".format(np.round(RF_error[i], 3),np.round(sigma[i], 3),eg))
    print("---------------------------------------------------------")
