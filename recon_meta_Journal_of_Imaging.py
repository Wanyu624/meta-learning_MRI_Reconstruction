"""
Created on Fri May 21 20:06:55 2021

@author: 啊咧啊咧
"""

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import Utils
import tensorflow.contrib.slim as slim
from time import time
from PIL import Image
import math
from skimage.measure import compare_ssim as ssim


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #0-1 1-3 2-0 3-2 

ckpt_model_number = 1000
EpochNum =1000
batch_size = 8
PhaseNumber = 11
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)   

mask10 = sio.loadmat('dataset/mask_10.mat')
Mask10 = np.expand_dims(mask10['mask_10'].astype(np.float32), axis=0)
mask20 = sio.loadmat('dataset/mask_20.mat')
Mask20 = np.expand_dims(mask20['mask_20'].astype(np.float32), axis=0)
mask30 = sio.loadmat('dataset/mask_30.mat')
Mask30 = np.expand_dims(mask30['mask_30'].astype(np.float32), axis=0)
mask40 = sio.loadmat('dataset/mask_40.mat')
Mask40 = np.expand_dims(mask40['mask_40'].astype(np.float32), axis=0)
#mask50 = sio.loadmat('dataset/mask_50.mat')
#Mask50 = np.expand_dims(mask50['mask_50'].astype(np.float32), axis=0)
Mask = np.vstack((Mask10, Mask20, Mask30, Mask40, Mask10, Mask20, Mask30, Mask40))
        
Kspace = tf.placeholder(tf.complex64, [None, 160, 180])#kspace_rec
Target = tf.placeholder(tf.float32, [None, 160, 180])#Target_rec
Target = tf.abs(Target)
Mask = tf.constant(Mask, dtype=tf.float32)#(batch_size, 160, 180) dtype=float32

def mriForwardOp(img, sampling_mask):
    # centered Fourier transform
    Fu = tf.fft2d(img)
    # apply sampling mask
    kspace = tf.complex(tf.real(Fu) * sampling_mask, tf.imag(Fu) * sampling_mask)
    return kspace

def mriAdjointOp(f, sampling_mask):
    # apply mask and perform inverse centered Fourier transform
    Finv = tf.ifft2d(tf.complex(tf.real(f) * sampling_mask, tf.imag(f) * sampling_mask))
    return Finv 
    
def nmse(img_t, img_s ):
    return np.sum((abs(img_s) - abs(img_t)) ** 2.) / np.sum(abs(img_t)**2)

def rmse(img_t, img_s):
    return np.linalg.norm( abs(img_t) - abs(img_s), 'fro' )/np.linalg.norm( abs(img_t), 'fro')

def psnr(img_t, img_s):
    mse = np.mean( ( abs(img_t) - abs(img_s) ) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = abs(img_t).max()
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def add_con2d_weight(w_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.keras.initializers.glorot_uniform(seed=None), name='Weights_%d' % order_no)
    return Weights

##############################################
# define our model
class Multi_modal_generator:

    def __init__(self, n):
        super(Multi_modal_generator,self).__init__()
        
        self.ddelta = 0.001
        self.coeff = 1.0 / (4.0 * self.ddelta) 
        self.coeff2 = 1.0 / (2.0 * self.ddelta)       

        self.gamma = 0.9
        self.sigma = 1.0
        
        self.k  = 3
        self.ch = 4
        
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.alpha1_r = tf.Variable(0.01, dtype=tf.float32, trainable=True)
            self.alpha1_i = tf.Variable(0.01, dtype=tf.float32, trainable=True)
            self.alpha1   = tf.complex(self.alpha1_r,self.alpha1_i)
            
            self.tau1_r = tf.Variable(0.01, dtype=tf.float32, trainable=True)
            self.tau1_i = tf.Variable(0.01, dtype=tf.float32, trainable=True)       
            self.tau1   = tf.complex(self.tau1_r,self.tau1_i)
            
            self.epsilon1 = tf.Variable(1E-8, dtype=tf.float32, trainable=True) 
    
            #regularization1
            self.g1Weights0_r = add_con2d_weight([self.k, self.k,    1,    self.ch], 40)
            self.g1Weights1_r = add_con2d_weight([self.k, self.k, self.ch, self.ch], 41)
            self.g1Weights2_r = add_con2d_weight([self.k, self.k, self.ch, self.ch], 42)
    
            self.g1Weights0_i = add_con2d_weight([self.k, self.k,    1,    self.ch], 50)
            self.g1Weights1_i = add_con2d_weight([self.k, self.k, self.ch, self.ch], 51)
            self.g1Weights2_i = add_con2d_weight([self.k, self.k, self.ch, self.ch], 52)
            
        with tf.variable_scope('validation', reuse=tf.AUTO_REUSE):
            self.w_1_r = tf.Variable(1, name='w_1r', dtype=tf.float32, trainable=True)
            self.w_2_r = tf.Variable(1, name='w_2r', dtype=tf.float32, trainable=True)
            self.w_3_r = tf.Variable(1, name='w_3r', dtype=tf.float32, trainable=True)
            self.w_4_r = tf.Variable(1, name='w_4r', dtype=tf.float32, trainable=True)
            
            self.w_1_i = tf.Variable(1, name='w_1i', dtype=tf.float32, trainable=True)
            self.w_2_i = tf.Variable(1, name='w_2i', dtype=tf.float32, trainable=True)
            self.w_3_i = tf.Variable(1, name='w_3i', dtype=tf.float32, trainable=True)
            self.w_4_i = tf.Variable(1, name='w_4i', dtype=tf.float32, trainable=True)
            
            self.w_1 = tf.complex(self.w_1_r, self.w_1_i)
            self.w_2 = tf.complex(self.w_2_r, self.w_2_i)
            self.w_3 = tf.complex(self.w_3_r, self.w_3_i)
            self.w_4 = tf.complex(self.w_4_r, self.w_4_i)
        
    def act(self, x_i): #sigma_activation
        x_i_relu  = tf.nn.relu(x_i)
        x_square  = x_i * x_i
        x_square *= self.coeff
        return tf.where(tf.abs(x_i) > self.ddelta, x_i_relu, x_square + 0.5*x_i + 0.25 * self.ddelta)
    
    def deri_act(self, x_i):#sigma_derivative 
        x_i_relu_deri = tf.where(x_i > 0, tf.ones_like(x_i), tf.zeros_like(x_i))
        return tf.where(tf.abs(x_i) > self.ddelta, x_i_relu_deri, self.coeff2 *x_i + 0.5)

##################################################################################################
        
    def g1_forward(self, x):
        
        x_r = tf.reshape(tf.real(x), shape=[batch_size, 160, 180, 1])
        x_i = tf.reshape(tf.imag(x), shape=[batch_size, 160, 180, 1])
               
        x1_r = tf.nn.conv2d(    x_r,        self.g1Weights0_r, strides=[1,1,1,1], padding='SAME') + tf.nn.conv2d(    x_i,        self.g1Weights0_i, strides=[1,1,1,1], padding='SAME')
        x1_i = tf.nn.conv2d(    x_r,        self.g1Weights0_i, strides=[1,1,1,1], padding='SAME') - tf.nn.conv2d(    x_i,        self.g1Weights0_r, strides=[1,1,1,1], padding='SAME')
        
        x2_r = tf.nn.conv2d(self.act(x1_r), self.g1Weights1_r, strides=[1,1,1,1], padding='SAME') + tf.nn.conv2d(self.act(x1_i), self.g1Weights1_i, strides=[1,1,1,1], padding='SAME')
        x2_i = tf.nn.conv2d(self.act(x1_r), self.g1Weights1_i, strides=[1,1,1,1], padding='SAME') - tf.nn.conv2d(self.act(x1_i), self.g1Weights1_r, strides=[1,1,1,1], padding='SAME')
        
        x3_r = tf.nn.conv2d(self.act(x2_r), self.g1Weights2_r, strides=[1,1,1,1], padding='SAME') + tf.nn.conv2d(self.act(x2_i), self.g1Weights2_i, strides=[1,1,1,1], padding='SAME')
        x3_i = tf.nn.conv2d(self.act(x2_r), self.g1Weights2_i, strides=[1,1,1,1], padding='SAME') - tf.nn.conv2d(self.act(x2_i), self.g1Weights2_r, strides=[1,1,1,1], padding='SAME')
        
        x1   = tf.complex(x1_r , x1_i)
        x2   = tf.complex(x2_r , x2_i)
        x3   = tf.complex(x3_r , x3_i)
        
        return [x1, x2, x3]

    def grad_g1(self, x): 
        g_forward_output = self.g1_forward(x)
        [x1, x2, x3] = g_forward_output
        
        x3_r = tf.reshape(tf.real(x3), shape=[batch_size, 160, 180, self.ch])
        x3_i = tf.reshape(tf.imag(x3), shape=[batch_size, 160, 180, self.ch])
        x2_r = tf.reshape(tf.real(x2), shape=[batch_size, 160, 180, self.ch])
        x2_i = tf.reshape(tf.imag(x2), shape=[batch_size, 160, 180, self.ch])
        x1_r = tf.reshape(tf.real(x1), shape=[batch_size, 160, 180, self.ch])
        x1_i = tf.reshape(tf.imag(x1), shape=[batch_size, 160, 180, self.ch])
        
        x3_r_dec      = tf.nn.conv2d_transpose(x3_r, self.g1Weights2_r, [batch_size, 160, 180, self.ch], [1, 1, 1, 1], padding='SAME') + tf.nn.conv2d_transpose(x3_i, self.g1Weights2_i, [batch_size, 160, 180, self.ch], [1, 1, 1, 1], padding='SAME')
        x3_i_dec      = tf.nn.conv2d_transpose(x3_r, self.g1Weights2_i, [batch_size, 160, 180, self.ch], [1, 1, 1, 1], padding='SAME') - tf.nn.conv2d_transpose(x3_i, self.g1Weights2_r, [batch_size, 160, 180, self.ch], [1, 1, 1, 1], padding='SAME')
        
        x2_r_deri_act = self.deri_act(x2_r)
        x2_i_deri_act = self.deri_act(x2_i)
        
        x2_r_dec      = tf.nn.conv2d_transpose((x2_r_deri_act * x3_r_dec), self.g1Weights1_r, [batch_size, 160, 180, self.ch], [1, 1, 1, 1], padding='SAME') + tf.nn.conv2d_transpose((x2_i_deri_act * x3_i_dec), self.g1Weights1_i, [batch_size, 160, 180, self.ch], [1, 1, 1, 1], padding='SAME')
        x2_i_dec      = tf.nn.conv2d_transpose((x2_r_deri_act * x3_r_dec), self.g1Weights1_i, [batch_size, 160, 180, self.ch], [1, 1, 1, 1], padding='SAME') - tf.nn.conv2d_transpose((x2_i_deri_act * x3_i_dec), self.g1Weights1_r, [batch_size, 160, 180, self.ch], [1, 1, 1, 1], padding='SAME')
        
        x1_r_deri_act = self.deri_act(x1_r)
        x1_i_deri_act = self.deri_act(x1_i)

        x1_r_dec      = tf.nn.conv2d_transpose((x1_r_deri_act * x2_r_dec), self.g1Weights0_r, [batch_size, 160, 180, 1], [1, 1, 1, 1], padding='SAME') + tf.nn.conv2d_transpose((x1_i_deri_act * x2_i_dec), self.g1Weights0_i, [batch_size, 160, 180, 1], [1, 1, 1, 1], padding='SAME')
        x1_i_dec      = tf.nn.conv2d_transpose((x1_r_deri_act * x2_r_dec), self.g1Weights0_i, [batch_size, 160, 180, 1], [1, 1, 1, 1], padding='SAME') - tf.nn.conv2d_transpose((x1_i_deri_act * x2_i_dec), self.g1Weights0_r, [batch_size, 160, 180, 1], [1, 1, 1, 1], padding='SAME')

        x1_r = tf.reshape(x1_r_dec,  shape=[batch_size, 160, 180])
        x1_i = tf.reshape(x1_i_dec,  shape=[batch_size, 160, 180])        
        x1_dec  = tf.complex(x1_r , x1_i) 
        return x1_dec
    
    def R_epsilon1(self, x, e1):
        x = tf.transpose(x, perm=[0,3,1,2])#(?,self.ch, 160, 180)
        x = tf.square(tf.real(x)) + tf.square(tf.imag(x))
        x = tf.reduce_sum(x, axis =1) + tf.square(e1)#(batch_size,self.ch)
        x = tf.reduce_sum(tf.sqrt(x) , axis =[-2,-1])
        x = x - e1
        return x
    
    def grad_R1(self, x, e1):        
        g_forward_output = self.g1_forward(x)#g_forward_output[-1](?, 160, 180,self.ch)
        grad_gx    = self.grad_g1(x)#(batch_size, 160, 180)
        
        de         = tf.transpose(g_forward_output[-1], perm=[0,3,1,2])#(?, 32, 160, 180)
        de         = tf.square(tf.real(de)) + tf.square(tf.imag(de))
        de         = tf.reduce_sum(de, axis =1) + tf.square(e1)#shape=(?, 160, 180), dtype=float32
        de         = tf.reduce_sum(tf.sqrt(de),axis=[-2,-1])
        x          = tf.reshape(tf.reciprocal(de),  shape=[-1, 1, 1])
        x          = tf.complex(x * tf.real(grad_gx), x * tf.imag(grad_gx))
        return x
    
    def l2_norm_square(self, x):
        x = tf.square(tf.real(x)) + tf.square(tf.imag(x))
        x = tf.reduce_sum(x , axis =[-2,-1]) #shape=(batch_size,), dtype=float32)
        return x
    
    def sigma_w(self, x):
        self.omega = tf.sigmoid(tf.stack([tf.abs(self.w_1), tf.abs(self.w_2), tf.abs(self.w_3), tf.abs(self.w_4), tf.abs(self.w_1), tf.abs(self.w_2), tf.abs(self.w_3), tf.abs(self.w_4)] , axis=0))
        return self.omega*x
    
    def grad_sigma_w(self, x):
        self.omega = tf.sigmoid(tf.stack([self.w_1, self.w_2, self.w_3, self.w_4, self.w_1, self.w_2, self.w_3, self.w_4] , axis=0))
        omega = tf.expand_dims( tf.expand_dims( self.omega,-1),-1)
        self.grad_omega = tf.tile(omega, (1,160,180))
        return self.grad_omega*x
    
####################################################################################################
    def block(self, layers1, alpha1, tau1, epsilon1):
        input_layers1 = tf.reshape(layers1, shape=[batch_size, 160, 180]) 
        self.a1 = 1E+3
        self.c1 = 1E+3
        self.b1 = tf.square(self.a1)*(self.c1)*tf.square(tf.abs(alpha1))
        
    ################################################################################################
        #Zs
        ATAx1 = mriAdjointOp(mriForwardOp(input_layers1, Mask), Mask)# FTPT(PFx1)
        ATf1  = mriAdjointOp(Kspace, Mask)#FTPT(f1)
        
        #x1 reconstruction        
        z1 = input_layers1 - alpha1 * (ATAx1 - ATf1)
        u1 = z1 - tf.scalar_mul(tau1, self.grad_R1(z1, epsilon1))
        x1 = u1
        phi_x1 = 0.5*self.l2_norm_square(mriForwardOp(input_layers1, Mask)- Kspace) + self.sigma_w(self.R_epsilon1(self.g1_forward(input_layers1)[-1], epsilon1))
        phi_u1 = 0.5*self.l2_norm_square(mriForwardOp(u1, Mask)- Kspace) + self.sigma_w(self.R_epsilon1(self.g1_forward(u1)[-1], epsilon1))#(batch_size, )
        phi_u1_x1 = phi_u1 - phi_x1
        u1_x1  = self.l2_norm_square(u1 - input_layers1)
        
        grad_phi_x1_ =  (ATAx1 - ATf1) + self.grad_sigma_w(self.grad_R1(input_layers1, epsilon1))
        norm_grad_phi_x1_ = tf.reduce_mean(tf.sqrt(self.l2_norm_square(grad_phi_x1_)))
        if_choose_u1 = tf.logical_and( (norm_grad_phi_x1_ <= self.a1 * tf.sqrt(u1_x1) ) , (phi_u1_x1 <= -0.5*self.b1 * u1_x1))
        
        if if_choose_u1==True:
            x1 = u1
        else:
            def condition(alpha1):
                v1 = input_layers1 - tf.scalar_mul(alpha1,  self.grad_sigma_w(self.grad_R1(input_layers1, epsilon1)) )  
                phi_v1 = 0.5*self.l2_norm_square(mriForwardOp(v1, Mask)- Kspace) + self.sigma_w(self.R_epsilon1(self.g1_forward(v1)[-1], epsilon1))#(batch_size, )
                phi_v1_x1 = phi_v1 - phi_x1
                v1_x1     = self.l2_norm_square(v1 - input_layers1)
                return [v1, phi_v1_x1, v1_x1]
            
            [v1, phi_v1_x1, v1_x1] = condition(alpha1)
    
            if_alpha1_decrease = tf.reshape(tf.reduce_mean(phi_v1_x1) <= tf.reduce_mean( -self.c1 * v1_x1), [])
            #check if alpha should decrease
            if if_alpha1_decrease==False:
                alpha1  = tf.where(if_alpha1_decrease, alpha1, tf.complex(0.9*tf.real(alpha1),0.9*tf.imag(alpha1)) )
                [v1, phi_v1_x1, v1_x1] = condition(alpha1)
            x1 = v1
                  
        #reduction criterion
        grad_phi_x1 = (ATAx1 - ATf1) + self.grad_sigma_w(self.grad_R1(x1, epsilon1))
        norm_grad_phi_x1 = tf.reduce_mean(tf.sqrt(self.l2_norm_square(grad_phi_x1)))
        epsilon1 = tf.cond( norm_grad_phi_x1 <= self.sigma * self.gamma * epsilon1, lambda: self.gamma * epsilon1, lambda: epsilon1)

        return [x1, alpha1, tau1, epsilon1]

    def forward(self, n):
        layers1 = []        
        x1_initial = mriAdjointOp(Kspace, Mask) #(?, 160, 180)
        layers1.append(x1_initial)

        a1 = []
        t1 = []
        e1 = []        
        a1.append(self.alpha1)
        t1.append(self.tau1)
        e1.append(self.epsilon1)
        
        for i in range(n):     
            [x1, alpha1, tau1, epsilon1] = self.block(layers1[-1], a1[-1], t1[-1], e1[-1])
            
            layers1.append(x1)
            a1.append(alpha1)
            t1.append(tau1)
            e1.append(epsilon1)     
                
        return layers1,e1

def l2norm_sq(xs):
    return tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in xs if t is not None])

def smooth_L1_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)

def MSE(y_true, y_pred):
    return tf.losses.mean_squared_error(y_true, y_pred, weights=1.0, scope=None)

    
def compute_cost(layers, Target):

    img_rec = tf.abs(layers[-1])

    #ssim
#    output_abs = tf.expand_dims(img_rec, -1)
#    target_abs = tf.expand_dims(Target, -1)
#    L = tf.reduce_max(target_abs, axis=(1, 2, 3)) - tf.reduce_min(target_abs, axis=(1, 2, 3))
#    ssim_tr1= Utils.ssim(output_abs[:1], target_abs[:1], L=L)
    
    cost_tr1 = MSE(Target[:1], img_rec[:1])
    cost_tr2 = MSE(Target[1:2], img_rec[1:2])
    cost_tr3 = MSE(Target[2:3], img_rec[2:3])
    cost_tr4 = MSE(Target[3:4], img_rec[3:4])
    cost_vl1 = MSE(Target[4:5], img_rec[4:5])
    cost_vl2 = MSE(Target[5:6], img_rec[5:6])
    cost_vl3 = MSE(Target[6:7], img_rec[6:7])
    cost_vl4 = MSE(Target[7:8], img_rec[7:8])
    
    return  cost_tr1, cost_tr2, cost_tr3, cost_tr4, cost_vl1, cost_vl2, cost_vl3, cost_vl4

model = Multi_modal_generator(PhaseNumber)
layers, e = model.forward(PhaseNumber)
cost_tr1, cost_tr2, cost_tr3, cost_tr4, cost_vl1, cost_vl2, cost_vl3, cost_vl4 = compute_cost(layers, Target)

loss_tr = cost_tr1+cost_tr2+cost_tr3+cost_tr4
loss_vl = cost_vl1+cost_vl2+cost_vl3+cost_vl4
      
theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'train')
omega = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'validation')

learning_rate = tf.train.exponential_decay(learning_rate = 0.001,
                                       global_step = global_step,
                                       decay_steps = 10000,
                                       decay_rate=0.9, staircase=False) 
Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam') 

lamda = tf.train.exponential_decay(learning_rate = 1e-5,
                                       global_step = global_step,
                                       decay_steps = 10000,
                                       decay_rate=1.001, staircase=False)
gdtht = tf.gradients(loss_tr, theta)
loss_trnorm = 0.5*lamda*l2norm_sq(gdtht)
loss = loss_vl + loss_trnorm 

stop_ct = l2norm_sq(tf.gradients(loss, theta))+l2norm_sq(tf.gradients(loss, omega))

trainer_tr = Optimizer.minimize(loss, global_step=global_step, var_list=theta)

trainer_vl = Optimizer.minimize(loss, global_step=global_step)#, var_list=omega

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())   

#logits = slim.get_variables_to_restore()
#init_Weights = 'Ratio_vali_8' + '/Saved_Model_107.ckpt' 
#init_, init_feeddic = slim.assign_from_checkpoint(init_Weights, logits, ignore_missing_vars = True)
#sess.run(init_, init_feeddic)

model_dir = 'Ratio_vali_8' 
log_file_name = "Log_output_%s.txt" % (model_dir)

#___________________________________________________________________________________Train

print("...................................")
print("Phase Number is %d" % (PhaseNumber)) 
print("...................................\n")
class Meta:

    def __init__(self, session, variables=tf.trainable_variables()):
        self.session = session

    def train(self):
        print('Start Loading Data...')
        Target_tr = sio.loadmat('dataset/train/cs/T1.mat')['T1'].astype(np.float32)
        Kspace_tr10 = sio.loadmat('dataset/train/cs/K1_10.mat')['K1_10'].astype(np.complex64)
        Kspace_tr20 = sio.loadmat('dataset/train/cs/K1_20.mat')['K1_20'].astype(np.complex64)
        Kspace_tr30 = sio.loadmat('dataset/train/cs/K1_30.mat')['K1_30'].astype(np.complex64)
        Kspace_tr40 = sio.loadmat('dataset/train/cs/K1_40.mat')['K1_40'].astype(np.complex64)
        #Kspace_tr50 = sio.loadmat('dataset/train/cs/K1_50.mat')['K1_50'].astype(np.complex64)
        
        Target_vl = sio.loadmat('dataset/validation/cs/T1.mat')['T1'].astype(np.float32)
        Kspace_vl10 = sio.loadmat('dataset/validation/cs/K1_10.mat')['K1_10'].astype(np.complex64)
        Kspace_vl20 = sio.loadmat('dataset/validation/cs/K1_20.mat')['K1_20'].astype(np.complex64)
        Kspace_vl30 = sio.loadmat('dataset/validation/cs/K1_30.mat')['K1_30'].astype(np.complex64)
        Kspace_vl40 = sio.loadmat('dataset/validation/cs/K1_40.mat')['K1_40'].astype(np.complex64)
        #Kspace_vl50 = sio.loadmat('dataset/validation/cs/K1_50.mat')['K1_50'].astype(np.complex64)

        print(Kspace_vl10.shape)
        ntrain = Target_tr.shape[0]
        nval = Target_vl.shape[0]
        print("each task contains %02d train data, %02d validation data" % (ntrain, nval))
      
        #update alpha        
        for epoch_p in range(0, EpochNum+1):          
            tr_randidx_all = range(ntrain) 
            #tr_randidx_k = range(Kspace_tr10.shape[0])
            vl_randidx_all = range(nval) 
            #vl_randidx_k = range(Kspace_vl10.shape[0])
          
            for batch_i in range(0, ntrain):
                ##################################################################################tr
                randidx = tr_randidx_all[batch_i:batch_i+1]
                #print(randidx) #range(0,2)
                
                target_tr = Target_tr[randidx, :, :]
                target_tr = np.tile(target_tr[0], (4, 1, 1))
                
                kspace10 = Kspace_tr10[randidx, :, :]
                kspace20 = Kspace_tr20[randidx, :, :]
                kspace30 = Kspace_tr30[randidx, :, :]
                kspace40 = Kspace_tr40[randidx, :, :]
                #kspace50 = Kspace_tr50[randidx, :, :]
                kspace_tr = np.vstack((kspace10, kspace20, kspace30, kspace40))#(5, 160, 180)
                #print(kspace.shape)
                
                ##################################################################################val                
                randidx_vl = vl_randidx_all[batch_i:batch_i+1]
                #print(randidx_vl)
                target_vl = Target_vl[randidx_vl, :, :]
                target_vl = np.tile(target_vl, (4, 1, 1))#(5, 160, 180)
                
                kspace10_vl = Kspace_vl10[randidx_vl, :, :]
                kspace20_vl = Kspace_vl20[randidx_vl, :, :]
                kspace30_vl = Kspace_vl30[randidx_vl, :, :]
                kspace40_vl = Kspace_vl40[randidx_vl, :, :]
                #kspace50_vl = Kspace_vl50[randidx_vl, :, :]
                kspace_vl = np.vstack((kspace10_vl, kspace20_vl, kspace30_vl, kspace40_vl))#(5, 160, 180)
                
                ####################################################################################cc
                target = np.vstack((target_tr, target_vl))
                kspace = np.vstack((kspace_tr, kspace_vl))##(15, 160, 180)
                feed_dict = {Target: target, Kspace: kspace}  
                
                self.session.run(trainer_tr, feed_dict=feed_dict)  
                self.session.run(trainer_tr, feed_dict=feed_dict) 
                self.session.run(trainer_tr, feed_dict=feed_dict) 
                self.session.run(trainer_tr, feed_dict=feed_dict) 
                self.session.run(trainer_tr, feed_dict=feed_dict)
            
                output_data = "Epoch [%02d/%02d],%02d, lamda: %.7f, stop_ct: %.14f, gradient: %.7f, train loss: %.7f, valid loss: %.7f, lr: %.7f \n" % ( epoch_p, EpochNum, batch_i,
                                     self.session.run(lamda), self.session.run(stop_ct, feed_dict=feed_dict), self.session.run(loss_trnorm, feed_dict=feed_dict),
                                     self.session.run(loss_tr, feed_dict=feed_dict), self.session.run(loss_vl, feed_dict=feed_dict), 
                                     self.session.run(learning_rate))
                print(output_data)
                
                self.session.run(trainer_vl, feed_dict=feed_dict)                 
            output_data = "Epoch [%02d/%02d], lamda: %.7f, e: %.7f, stop_ct: %.7f, gradient: %.7f, train loss: %.7f, valid loss: %.7f, lr: %.7f \n" % ( epoch_p, EpochNum, 
                                 self.session.run(lamda), self.session.run(e[-1], feed_dict=feed_dict), self.session.run(stop_ct, feed_dict=feed_dict), self.session.run(loss_trnorm, feed_dict=feed_dict),
                                 self.session.run(loss_tr, feed_dict=feed_dict), self.session.run(loss_vl, feed_dict=feed_dict), 
                                 self.session.run(learning_rate))
            print(output_data)
            #print(self.session.run(self.omega, feed_dict=feed_dict_val))
            print("-----------------------------------------------------------")
            output_file = open(log_file_name, 'a')
            output_file.write(output_data)
            output_file.close()
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if epoch_p % 1 == 0:
                saver.save(self.session, './%s/Saved_Model_%d.ckpt' % (model_dir, epoch_p), write_meta_graph=False) 
                       
meta = Meta(sess)
meta.train()
print("Training Finished")   

#___________________________________________________________________________________Test
print('Load Test Data...')
Target_test = sio.loadmat('dataset/test/cs/T1.mat')['T1'].astype(np.float32)
Kspace_te10 = sio.loadmat('dataset/test/cs/K1_10.mat')['K1_10'].astype(np.complex64)
Kspace_te20 = sio.loadmat('dataset/test/cs/K1_20.mat')['K1_20'].astype(np.complex64)
Kspace_te30 = sio.loadmat('dataset/test/cs/K1_30.mat')['K1_30'].astype(np.complex64)
Kspace_te40 = sio.loadmat('dataset/test/cs/K1_40.mat')['K1_40'].astype(np.complex64)
Kspace_te50 = sio.loadmat('dataset/test/cs/K1_50.mat')['K1_50'].astype(np.complex64)
ntest = 60#Target_test.shape[0]

PSNR1_All = []
SSIM1_All = []
NMSE1_All = []
PSNR2_All = []
SSIM2_All = []
NMSE2_All = []
PSNR3_All = []
SSIM3_All = []
NMSE3_All = []
PSNR4_All = []
SSIM4_All = []
NMSE4_All = []
PSNR5_All = []
SSIM5_All = []
NMSE5_All = []
TIME_All  = np.zeros([1, ntest], dtype=np.float32)

saver.restore(sess, './%s/Saved_Model_%d.ckpt' % (model_dir, ckpt_model_number))

result_file_name = "Rec_Results.txt"
    
idx_all = np.arange(ntest)
for imag_no in range(ntest):
    randidx = idx_all[imag_no:imag_no+1]
    target_te = Target_test[randidx, :, :]
    target_te = np.tile(target_te[0], (5, 1, 1))
    
    kspace10 = Kspace_te10[randidx, :, :]
    kspace20 = Kspace_te20[randidx, :, :]
    kspace30 = Kspace_te30[randidx, :, :]
    kspace40 = Kspace_te40[randidx, :, :]
    kspace50 = Kspace_te50[randidx, :, :]
    kspace_te = np.vstack((kspace10, kspace20, kspace30, kspace40, kspace50))#(5, 160, 180)

    u = target_te[randidx, :, :]
    k = kspace_te[randidx, :, :]
    
    feed_dict = {Target: u, Kspace: k} 
    
    start = time()
    Prediction_value = sess.run(np.abs(layers[-1]), feed_dict=feed_dict) 
    end = time()
    
    rec = np.reshape(Prediction_value, (5,160,180))
    rec = rec.astype(np.float32)
    reference = np.reshape(u, (5,160,180))
    reference = reference.astype(np.float32)
    
    PSNR1 =  psnr(reference[:1], rec[:1])
    SSIM1 =  ssim(reference[:1], rec[:1])    
    NMSE1 =  nmse(reference[:1], rec[:1])
    PSNR2 =  psnr(reference[1:2], rec[1:2])
    SSIM2 =  ssim(reference[1:2], rec[1:2])    
    NMSE2 =  nmse(reference[1:2], rec[1:2])
    PSNR3 =  psnr(reference[2:3], rec[2:3])
    SSIM3 =  ssim(reference[2:3], rec[2:3])    
    NMSE3 =  nmse(reference[2:3], rec[2:3])
    PSNR4 =  psnr(reference[3:4], rec[3:4])
    SSIM4 =  ssim(reference[3:4], rec[3:4])    
    NMSE4 =  nmse(reference[3:4], rec[3:4])
    PSNR5 =  psnr(reference[4:5], rec[4:5])
    SSIM5 =  ssim(reference[4:5], rec[4:5])    
    NMSE5 =  nmse(reference[4:5], rec[4:5])
    
    result1 = "Run time for %s:%.4f, PSNR1:%.4f, SSIM1:%.4f, NMSE1:%.4f. \n" % (imag_no+1, (end - start), PSNR1, SSIM1, NMSE1)
    result2 = "Run time for %s:%.4f, PSNR2:%.4f, SSIM2:%.4f, NMSE2:%.4f. \n" % (imag_no+1, (end - start), PSNR2, SSIM2, NMSE2)
    result3 = "Run time for %s:%.4f, PSNR3:%.4f, SSIM3:%.4f, NMSE3:%.4f. \n" % (imag_no+1, (end - start), PSNR3, SSIM3, NMSE3)
    result4 = "Run time for %s:%.4f, PSNR4:%.4f, SSIM4:%.4f, NMSE4:%.4f. \n" % (imag_no+1, (end - start), PSNR4, SSIM4, NMSE4)
    result5 = "Run time for %s:%.4f, PSNR5:%.4f, SSIM5:%.4f, NMSE5:%.4f. \n" % (imag_no+1, (end - start), PSNR5, SSIM5, NMSE5)
    print(result1)
    print(result2)
    print(result3)
    print(result4)
    print(result5)
    
    im_rec_name = "%s_rec_%d.mat" % (imag_no+1, ckpt_model_number)  
    
    # save mat file
    #Utils.saveAsMat(rec, im_rec_name, 'result',  mat_dict=None)
    
    PSNR1_All[0, imag_no] = PSNR1
    SSIM1_All[0, imag_no] = SSIM1
    NMSE1_All[0, imag_no] = NMSE1

    PSNR2_All[0, imag_no] = PSNR2
    SSIM2_All[0, imag_no] = SSIM2
    NMSE2_All[0, imag_no] = NMSE2

    PSNR3_All[0, imag_no] = PSNR3
    SSIM3_All[0, imag_no] = SSIM3
    NMSE3_All[0, imag_no] = NMSE3

    PSNR4_All[0, imag_no] = PSNR4
    SSIM4_All[0, imag_no] = SSIM4
    NMSE4_All[0, imag_no] = NMSE4

    PSNR5_All[0, imag_no] = PSNR5
    SSIM5_All[0, imag_no] = SSIM5
    NMSE5_All[0, imag_no] = NMSE5    
output_data = "ratio10% ckpt NO. is %d, Avg REC PSNR is %.4f dB, SSIM is %.4f, NMSE is %.4f\n" % (ckpt_model_number, np.mean(PSNR1_All), np.mean(SSIM1_All), np.mean(NMSE1_All))
output_data = "ratio20% ckpt NO. is %d, Avg REC PSNR is %.4f dB, SSIM is %.4f, NMSE is %.4f\n" % (ckpt_model_number, np.mean(PSNR2_All), np.mean(SSIM2_All), np.mean(NMSE2_All))
output_data = "ratio30% ckpt NO. is %d, Avg REC PSNR is %.4f dB, SSIM is %.4f, NMSE is %.4f\n" % (ckpt_model_number, np.mean(PSNR3_All), np.mean(SSIM3_All), np.mean(NMSE3_All))
output_data = "ratio40% ckpt NO. is %d, Avg REC PSNR is %.4f dB, SSIM is %.4f, NMSE is %.4f\n" % (ckpt_model_number, np.mean(PSNR4_All), np.mean(SSIM4_All), np.mean(NMSE4_All))
output_data = "ratio50% ckpt NO. is %d, Avg REC PSNR is %.4f dB, SSIM is %.4f, NMSE is %.4f\n" % (ckpt_model_number, np.mean(PSNR5_All), np.mean(SSIM5_All), np.mean(NMSE5_All))
print(output_data)

output_file = open(result_file_name, 'a')
output_file.write(output_data)
output_file.close()
sess.close()

print("Reconstruction READY")

