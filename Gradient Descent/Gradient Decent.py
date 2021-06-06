import numpy as np
import matplotlib.pyplot as plt

def draw(x1,x2):
    ln = plt.plot(x1,x2,'-')
    plt.pause(0.00000000000000000000000001) #delay in plotting new line
    ln[0].remove() #after updating remove the previously plotted line

def sigmoid(score):
   return 1/(1+ np.exp(-score))

def calc_error(line_parameter, points, y ):
   m= points.shape[0]
   p = sigmoid(points * line_parameter)
   cross_entropy = -(1/m)*(np.log(p).T * y + np.log(1-p).T * (1-y)) #loss function
   return cross_entropy

def gradient_descent(line_parameter, points ,y, alpha):
   m = points.shape[0]
   for i in range(5000):
      p = sigmoid(points * line_parameter) #sigmoid_activation
      gradient = (points.T * (p-y)) * (alpha/m)
      line_parameter = line_parameter - gradient
      w1= line_parameter.item(0)
      w2= line_parameter.item(1)
      b= line_parameter.item(2)
      x1= np.array([bottom_reg[0:, 0].min(), top_reg[0:, 0].max()])
      x2 = -b/w2  + x1 * (-w1/w2)
      draw(x1,x2)
      print('iteration:',i,'loss:',calc_error(line_parameter,all_points,y),"w1:",w1,"w2:",w2,"bias:",b)

n_pts = 100
np.random.seed(0)
bias = np.ones(n_pts)
top_reg = np.array([np.random.normal(10, 2, n_pts),np.random.normal(12, 2, n_pts),bias]).T
bottom_reg = np.array([np.random.normal(5, 2, n_pts),np.random.normal(6, 2, n_pts),bias]).T
all_points = np.vstack((top_reg,bottom_reg))
#print(all_points)

line_parameter = np.matrix([np.zeros(3)]).T

y= np.array([np.zeros(n_pts),np.ones(n_pts)]).reshape(n_pts*2, 1)

_,axis = plt.subplots(figsize=(4, 4))
axis.scatter(top_reg[:, 0], top_reg[0:, 1], color = 'r')
axis.scatter(bottom_reg[:, 0], bottom_reg[0:, 1], color = 'b')
gradient_descent(line_parameter, all_points,y,alpha= 0.06)
plt.show()