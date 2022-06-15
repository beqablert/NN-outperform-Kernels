# -*- coding: utf-8 -*-
"""practical_paper 2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ftmIntqT2TeicEEq-SGpz-JbNEMDV79w
"""

import numpy as np
import torch.nn as nn
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import math
import torch.optim as optim
import time
import torchvision
from torch.utils.data import Dataset, DataLoader
import scipy.sparse.linalg as ss
from jax import random
from neural_tangents import stax


"""#Define train module and Dataset"""

def train(model, loss_fn, train_data, val_data, epochs=750, device='cpu',model_name=None):

    print('train() called: model=%s, epochs=%d, device=%s\n' % \
          (type(model).__name__, epochs, device))
    if model_name==None:
        print("Choose NN, RF or NT! Abort Training...")
        return None
    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    history['plot_val'] = []

    start_time_sec = time.time()
    #l2_reg = np.logscale(-6,-2,20) #they used a grid for l2 but we keep it simple for now
    l2_reg_RF = 1e-1
    l2_reg_NT = 1e-2
    l2_reg_NN = 1e-3
    # regW = np.zeros(256)
    for epoch in range(1, epochs+1):
        if model_name == "NN":
            if epoch <=15:# "Warm up"
              train_dl = DataLoader(train_data, batch_size=500,shuffle=True)
              val_dl = DataLoader(val_data, batch_size=500,shuffle=True)
            else: # After "Warm up" increase batch size to 1000!
              train_dl = DataLoader(train_data, batch_size=1000,shuffle=True)
              val_dl = DataLoader(val_data, batch_size=1000,shuffle=True)
            lr_t = 1e-3 * np.max([1 + np.cos(epoch * np.pi / epochs), 1 / 15])
            optimizer = optim.SGD(model.parameters(), lr=lr_t, momentum=0.9, weight_decay=l2_reg_NN)
        elif model_name == "RF":
            train_dl = DataLoader(train_data, batch_size=10**4, shuffle=True)
            val_dl = DataLoader(val_data, batch_size=10**4, shuffle=True)
            lr_t = 1e-4 * np.max([1 + np.cos(epoch * np.pi / epochs), 1 / 15])
            optimizer = optim.Adam(model.parameters(), lr=lr_t, weight_decay=l2_reg_RF)
        elif model_name == "NT":
            train_dl = DataLoader(train_data, batch_size=10 ** 4, shuffle=True)
            val_dl = DataLoader(val_data, batch_size=10 ** 4, shuffle=True)
            lr_t = 1e-3 * np.max([1 + np.cos(epoch * np.pi / epochs), 1 / 15])
            optimizer = optim.Adam(model.parameters(), lr=lr_t, weight_decay=l2_reg_NT)
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for batch in train_dl:

            optimizer.zero_grad()

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            # print(yhat)
            # print(y)
            loss = loss_fn(yhat, y)
            
            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            # yhat_norm = (torch.linalg.norm(yhat, dim=0, ord=2))/(yhat.size(0))
            # y_norm = (torch.linalg.norm(y, dim=0, ord=2))/(y.size(0))
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            adjusted_labels = torch.sign(y - torch.mean(y))
            adjusted_predictions = torch.sign(yhat - torch.mean(yhat))
            # print(adjusted_labels[0])
            # print(adjusted_predictions[0])
            # yhat_norm = (torch.linalg.norm(yhat, dim=0, ord=2) ** 2) / len(val_dl.dataset)
            # y_norm = (torch.linalg.norm(y, dim=0, ord=2) ** 2)/ len(val_dl.dataset)
            num_val_correct  += torch.eq(adjusted_predictions, adjusted_labels).sum()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)        

        if epoch == 1 or epoch % 10 == 0: #show progress every 10 epochs
          print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                (epoch, epochs, train_loss, train_acc, val_loss, val_acc))
          print((torch.linalg.norm(yhat, dim=0, ord=2) ** 2)/len(val_dl.dataset))
          print((torch.linalg.norm(y, dim=0, ord=2) ** 2)/len(val_dl.dataset))
          print((val_loss - (torch.linalg.norm(y, dim=0, ord=2) ** 2)/len(val_dl.dataset))/((torch.linalg.norm(yhat, dim=0, ord=2) ** 2)/len(val_dl.dataset)))      

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        plot_val = (val_loss - (np.linalg.norm(torch.Tensor.cpu(y).detach().numpy(), 2) ** 2)/len(val_dl.dataset))/((np.linalg.norm(torch.Tensor.cpu(yhat).detach().numpy(), 2) ** 2)/len(val_dl.dataset))
        history['plot_val'].append(plot_val)

    # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history

#Dataset

class FashionDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""
    
    def __init__(self, data,label):
        """Method to initilaize variables.""" 
        
        self.labels = label
        self.images = data

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index,:]
        
        return image, label

    def __len__(self):
        return len(self.images)

class SynthDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""
    
    def __init__(self, X, Y):
        """Method to initilaize variables.""" 
        
        self.Y = Y
        self.X = X

    def __getitem__(self, index):
        Y = self.Y[index]
        X = self.X[index,:]
        
        return X, Y

    def __len__(self):
        return len(self.X)

"""#Define 2LNN, RF, NT
Neural Network (2 Layers)
"""

def RFK2(X, Z):
    """This function computes RF kernel for two-layer ReLU neural networks via
    an analytic formula.

    Input:
    X: d times n_1 matrix, where d is the feature dimension and n_i are # obs.
    Z: d times n_2 matrix, where d is the feature dimension and n_i are # obs.

    output:
    C: The kernel matrix of size n_1 times n_2.
    """
    pi = math.pi
    assert X.shape[0] == Z.shape[0]
    # X is sized d \times n
    nx = np.linalg.norm(X, axis=0, keepdims=True)
    nx = nx.T
    nz = np.linalg.norm(Z, axis=0, keepdims=True)

    C = np.dot(X.T, Z) #n_1 * n_2
    C = np.multiply(C, (nx ** -1))
    C = np.multiply(C, (nz ** -1))
    # Fixing numerical mistakes
    C = np.minimum(C, 1.0)
    C = np.maximum(C, -1.0)
    C = np.multiply(np.arcsin(C), C) / pi + C / 2.0 + np.sqrt(1 - np.power(C, 2)) / pi
    C = 0.5 * np.multiply(nx, np.multiply(C, nz))
    return C

def NTK2(X, Z):
	"""This function computes NTK kernel for two-layer ReLU neural networks via
	an analytic formula.

	Input:
	X: d times n_1 matrix, where d is the feature dimension and n_i are # obs.
	Z: d times n_2 matrix, where d is the feature dimension and n_i are # obs.

	output:
	C: The kernel matrix of size n_1 times n_2.
	"""
	pi = math.pi
	assert X.shape[0] == Z.shape[0]
	# X is sized d \times n
	nx = np.linalg.norm(X, axis=0, keepdims=True)
	nx = nx.T    
	nz = np.linalg.norm(Z, axis=0, keepdims=True)    

	C = np.dot(X.T, Z) #n_1 * n_2
	C = np.multiply(C, (nx ** -1))
	C = np.multiply(C, (nz ** -1))
	# Fixing numerical mistakes
	C = np.minimum(C, 1.0)
	C = np.maximum(C, -1.0)			

	C = np.multiply(1.0 - np.arccos(C) / pi, C) + np.sqrt(1 - np.power(C, 2)) / (2 * pi)
	C = np.multiply(nx, np.multiply(C, nz))
	return C

class NeuralNetwork(nn.Module):
  def __init__(self,K,p,std):
    """ initialisation of a student with:
    - K hidden nodes
    - N input dimensions
    -activation function act_function """
    print("Creating a Neural Network ")
    super(NeuralNetwork, self).__init__()
        
    self.g=nn.ReLU()
    self.soft = nn.Softmax(dim=1)
    self.K=K
    self.loss = square_loss
    self.drop = nn.Dropout(p=p)
    #change bias to true
    #self.fc1 = nn.utils.weight_norm(nn.Linear(N, K, bias=False))
    #self.fc2 = nn.utils.weight_norm(nn.Linear(K, 1, bias=False))
    self.fc1 = nn.Linear(256, K*256, bias=True)
    self.fc2 = nn.Linear(K*256, 1, bias=True)
    #torch.nn.init.xavier_uniform_(self.fc2.weight)
    #torch.nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.normal_(self.fc1.weight,std=std)
    nn.init.normal_(self.fc2.weight,std=std)
    #nn.init.normal_(self.fc1.weight,std=weight_std_initial_layer)
    #nn.init.normal_(self.fc2.weight,std=weight_std_initial_layer)


  def forward(self, x):
    # input to hidden
    x=self.fc1(x)/math.sqrt(self.K)
    x=self.g(x)
    x = self.fc2(x)
    x = self.drop(x)
    # x = self.soft(x)
    return x

def square_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2
     

class RF_Network(nn.Module):
    def __init__(self, K, std):
        """ initialisation of a student with:
        - K hidden nodes
        - N input dimensions """
        print("Creating a Neural Network ")
        super(RF_Network, self).__init__()

        self.g = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        self.K = K
        self.loss = square_loss
        #First layer weights are fixed!
        self.w = np.random.randn(256,K)
        norm = np.linalg.norm(self.w,axis=0,keepdims=True)
        self.w = self.w/norm
        self.w = torch.from_numpy(self.w)
        self.w = self.w.float()
        self.w = self.w.cuda()
        self.w = self.w.T
        self.fc1 = nn.Linear(256, K, bias=False)
        self.fc1.weight = nn.Parameter(self.w, requires_grad=False)  # Fix initialized weight
        self.fc2 = nn.Linear(K, 1, bias=True)
        nn.init.normal_(self.fc2.weight, std=std)
        self.wReg = self.fc2.weight


    def forward(self, x):
        # input to hidden
        x = self.fc1(x)
        x = self.g(x)
        x = self.fc2(x)
        # x = self.soft(x)
        return x


class NT_Network(nn.Module):
    def __init__(self, K, std):
        """ initialisation of a student with:
        - K hidden nodes
        - N input dimensions """
        print("Creating a Neural Network ")
        super(NT_Network, self).__init__()
        self.a0 = (torch.randn(1,1,K)).cuda()
        self.g = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        self.K = K
        self.loss = nn.MSELoss()
        #First layer weights are fixed!
        self.w = np.random.randn(256,K)
        norm = np.linalg.norm(self.w,axis=0,keepdims=True)
        print(norm)
        self.w = self.w/norm
        print(self.w)
        self.w = torch.from_numpy(self.w)
        self.w = self.w.float()
        self.w = self.w.cuda()
        self.w = self.w.T
        self.fc1 = nn.Linear(256, K, bias=False)
        self.fc1.weight = nn.Parameter(self.w, requires_grad=False)  # Fix initialized weight
        self.fc2 = nn.Linear(K, 1, bias=True)
        nn.init.normal_(self.fc2.weight, std=std)
        self.G = (torch.randn(K,256)).cuda()
        self.wReg = self.G

    def forward(self, x):
        # input to hidden
        x_ = x
        z = self.fc1(x)
        q = self.g(z)
        RF = self.fc2(q)
        zero_one_mat = 0.5 * (torch.sign(z) + 1.0)
        zero_one_mat_exp = torch.unsqueeze(zero_one_mat, 2)
        zero_one_mat_exp = zero_one_mat_exp.reshape((zero_one_mat_exp.shape[0], 1, self.K))
        U = torch.multiply(zero_one_mat_exp,self.a0)
        q2 = torch.tensordot(U,self.G,dims=([2],[0]))
        x_ = x_ / torch.linalg.norm(x_,axis=0,keepdims=True)
        temp = torch.unsqueeze(x_, 2)
        aux_data = temp.reshape((temp.shape[0],1,temp.shape[1]))  # bs x 1 x d
        temp = torch.multiply(q2, aux_data)
        NT = temp.sum(2)  # bs x num_class
        # print("////////////////////////////")
        # print(torch.norm(NT, 2))
        # print(torch.norm(RF, 2))
        # print(NT.shape)
        # print(RF.shape)
        # print('///////////////////////////')
        x = NT + RF
        #x = torch.tensor(x)
        #x = x.cuda()
        # x = self.soft(x)
        return x

"""# Generate FMNIST data with noise in high frequencies

Get Filter for adding noise to high frequencies
"""

def getFilter(dim):
  F = np.zeros((dim,dim))
  for i in range(dim):
    for j in range(dim):
      if ((dim -i)**2+(dim - j)**2) <= ((dim - 1)**2):
        F[i,j] = 1.0
  return F

"""Get data and add noise to higher frequencies according to the formulas in paper"""

def get_data_with_HF_noise(tau,x_train_,y_train):
  # implement 2D DCT
  def dct2(a):
      return dct(dct(a.T, norm='ortho').T, norm='ortho')

  # implement 2D IDCT
  def idct2(a):
      return idct(idct(a.T, norm='ortho').T, norm='ortho')

  # Generate plain fmnist data and preprocessing
  N = x_train_.shape[0] # nr of samples
  d = x_train_.shape[1]#Dimension of input image
  d_flatten = d**2 # flattened dimension of input image
  x_train  = x_train_.flatten().reshape(N, d**2)
  #x_test  = x_test.flatten().reshape(x_test.shape[0], x_test.shape[1]**2)

  #tau = np.linspace(0,3,num=15) # 15 points for different noises in their plot; Noise strength
  F = getFilter(d) # Filter which frequencies should get noise
  X_noisy = np.zeros((x_train.shape[0],d,d))
  for i in range(N): # for every image
    Z = np.random.randn(F.shape[0],F.shape[1])
    Z_tilde = np.multiply(Z,F) # Hadmard product => noise matrix
    img = x_train[i,:]
    img = img - np.mean(img) #Remove global mean to "center" data
    img = img.reshape((d, d))
    img_freq_space = dct2(img) # now we got the frequencies of the image. Next step add noise!
    img_noisy_freq = img_freq_space + tau* (np.linalg.norm(img_freq_space)/np.linalg.norm(Z_tilde))*Z_tilde #See 3. of appendix
    img_noisy = idct2(img_noisy_freq) # transform back to pixel space by inverse discrete fourier
    img_noisy = img_noisy /np.linalg.norm(img_noisy) * math.sqrt(d) # normalize to norm sqrt(d)
    X_noisy[i,:,:] = img_noisy
  
  X_noisy = (torch.from_numpy(X_noisy.flatten().reshape(N, d_flatten))).float()
  Y_train = torch.tensor(y_train)
  Y_train = Y_train.long()
  #Y_train = (torch.from_numpy(y_train)).long()
  return X_noisy,Y_train

def compute_accuracy(true_labels, preds):
    """This function computes the classification accuracy of the vector
    preds. """
    return np.mean(true_labels.numpy() == preds)

"""#Train Neural network"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


#Get and load data into dataloader

# (x_train_, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
history_NN_tau = []
history_RF_tau = []
history_NT_tau = []
history_RF_tau_val = []
history_NN_tau_val = []
history_NT_tau_val = []

noise_index = [2, 0]
# tau = np.linspace(0,3,num=15) # 15 points for different noises in their plot; Noise strength
# errors_RF = np.zeros((len(tau), 4)) #Train Loss, Train Accuracy, Test Loss, Test Accuracy

for i in range(len(noise_index)):
    criterion = nn.MSELoss()
    # print("Tau={}".format(tau[i]))
    print("Generate Data with noise in high frequencies....")
    # X_train,Y_train,= get_data_with_HF_noise(tau=tau[i],x_train_=x_train_,y_train=y_train)
    # X_test,Y_test,= get_data_with_HF_noise(tau=tau[i],x_train_=x_test,y_train=y_test)
    # train_data = FashionDataset(X_train,Y_train)
    # val_data = FashionDataset(X_test,Y_test)
    X = np.load('./datasets/synthetic/X_train_anisotropic_256_9_%d.npy'%(noise_index[i]))
    Y = np.load('./datasets/synthetic/y_train_anisotropic_256_9_%d.npy'%(noise_index[i]))	
    YT = np.load('./datasets/synthetic/y_test_anisotropic_256_9_%d.npy'%(noise_index[i]))
    XT = np.load('./datasets/synthetic/X_test_anisotropic_256_9_%d.npy'%(noise_index[i]))
    print(Y.shape[0])
    print(Y.shape[1])
    print(Y[0])
    train_data = SynthDataset(X, Y)
    val_data = SynthDataset(XT, YT)
    # net_NN = NeuralNetwork(K=6,p=0.2,std=1/math.sqrt(256)).to(device)
    # print("--------- Train Neural Network... ---------")
    # print(noise_index[i])
    # history_NN = train(
    #     model = net_NN,
    #     loss_fn = criterion,
    #     device=device,
    #     train_data = train_data,
    #     val_data = val_data,
    #     model_name= "NN")
    # history_NN_tau.append(history_NN["val_acc"])
    # history_NN_tau_val.append(history_NN["plot_val"])
    # print("---------- Calculate and Train RF Kernel... ---------")
    # print(noise_index[i])
    # net_RF = RF_Network(K=400,std=1/math.sqrt(256)).to(device)
    # history_RF = train(
    #     model = net_RF,
    #     loss_fn = criterion,
    #     device=device,
    #     train_data = train_data,
    #     val_data = val_data,
    #     model_name="RF")
    # history_RF_tau_val.append(history_RF["val_acc"])
    # history_RF_tau.append(history_RF["plot_val"])
    print("-------- Calculate NT Kernel.... ----------")
    print(noise_index[i])
    net_NT = NT_Network(K=160,std=1/math.sqrt(256)).to(device)
    history_NT = train(
        model = net_NT,
        loss_fn = criterion,
        device=device,
        train_data = train_data,
        val_data = val_data,
        model_name="NT")
    history_NT_tau.append(history_NT["val_acc"])
    history_NT_tau.append(history_NT["plot_val"])
    #print("Test Accuracy of Neural Network for tau = {} is {}".format(tau[i], history_NN["val_acc"][-1]))
    #print("Test Accuracy of Random Features for tau = {} is {}".format(tau[i], history_RF["val_acc"][-1]))
    #   print("Test Accuracy of Neural Network for tau = {} is {}".format(tau[i], history_NT["val_acc"][-1]))


#   K = NTK2(X_train.T,X_train.T)
#   KT = NTK2(X_test.T,X_train.T)
#   init_fn, apply_fn, kernel_fn = stax.serial(stax.Dense(512), stax.Relu(), stax.Dense(1))
#   n = X_train.shape[0]
#   kernel = np.zeros((n, n), dtype=np.float32)
#   m = n / 10
#   m = np.int(m)
#   kernel = kernel_fn(X_train)
# #   for i in range(10):
# #     for j in range(10):
# #         print('%d and %d'%(i, j))
# #         x1 = np.ndarray(X_train[i * m:(i + 1) * m, :], np.float32)
# #         x2 = np.ndarray(X_train[j * m:(j + 1) * m, :], np.float32)
# #         kernel[i * m:(i + 1) * m, j * m:(j + 1) * m] = kernel_fn(x1, x2, 'ntk')
#   KT = kernel_fn(X_test, X_train, 'ntk')
#   RK = K + 1e-4 * np.eye(len(Y_train), dtype=np.float32)
#   print(RK.shape)
#   cg = ss.cg(RK, Y_train, maxiter=400, atol=1e-4, tol=1e-4)
#   sol = np.copy(cg[0]).reshape((len(Y_train), 1))
#   yhat = np.dot(K, sol)
#   preds = np.dot(KT, sol)
#   print(preds.shape)
#   errors_RF[i, 0] = np.linalg.norm(Y_train - yhat) ** 2 / (len(Y_train) + 0.0)
#   errors_RF[i, 2] = np.linalg.norm(Y_test - preds) ** 2 / (len(Y_test) + 0.0)
#   errors_RF[i, 1] = compute_accuracy(Y_train, yhat)
#   errors_RF[i, 3] = compute_accuracy(Y_test, preds)
#   print('Training Error Random Features is'.format(errors_RF[i, 0]))
#   print('Test Error Random Features is'.format(errors_RF[i, 2]))
#   print('Training Accuracy Random Features is'.format(errors_RF[i, 1]))

# with open('/home/apdl008/Paper1/NN_val_acc_taus.txt', 'w') as f:
#     np.savetxt(f, history_NN_tau)
# with open('/home/apdl008/Paper1/RF_val_acc_taus.txt', 'w') as f:
#     np.savetxt(f, history_RF_tau)
# with open('/home/apdl008/Paper1/NT_val_acc_taus.txt', 'w') as f:
#         np.savetxt(f, history_NT_tau)

with open('./results/RF_results.txt', 'w') as f:
            np.savetxt(f, history_RF_tau)
            
with open('./results/RF_results_val_acc.txt', 'w') as f:
            np.savetxt(f, history_RF_tau_val)   

with open('./results/NN_results.txt', 'w') as f:
            np.savetxt(f, history_NN_tau)
            
with open('./results/NN_results_val_acc.txt', 'w') as f:
            np.savetxt(f, history_NN_tau_val)           

print("Results saved. Bye!")