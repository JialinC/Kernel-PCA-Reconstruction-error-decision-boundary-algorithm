from kpcabound import kpcabound
import matplotlib.pyplot as plt
import numpy as np

#plt.rc('font', family='serif')
######################## get butterfly contour ##########################
image = plt.imread('resized_image.jpg')
#fig0 = plt.figure(figsize=(10,10))
#plt.subplot(221)
#plt.imshow(image[45:195,121:-29])
#print(np.shape(image[45:195,121:-29]))

combine_RGB = []
jrange = np.arange(np.shape(image)[0]) #199
irange = np.arange(np.shape(image)[1]) #300
s = (len(irange),len(jrange))
grad_hold = np.sqrt(6000)
############################### R kpcabound ###################################
R = np.zeros(s)
for i in irange:
    for j in jrange:
        R[i,j] = image[len(jrange)-1-j,i][0]
        
image_cooked_R = []           
gx,gy = np.gradient(R)
grad = np.sqrt(gx**2 + gy**2)

for i in irange:
    for j in jrange:
        if grad[i,j] > grad_hold:
            a = [i,j]
            image_cooked_R.append(a)
            combine_RGB.append(a)
image_cooked_R = np.array(image_cooked_R)

G = np.zeros(s)
for i in irange:
    for j in jrange:
        G[i,j] = image[len(jrange)-1-j,i][1]
            
image_cooked_G = []         
gx,gy = np.gradient(G)
grad = np.sqrt(gx**2 + gy**2)

for i in irange:
    for j in jrange:
        if grad[i,j] > grad_hold:
            a = [i,j]
            image_cooked_G.append(a)
            combine_RGB.append(a)
image_cooked_G = np.array(image_cooked_G)

B = np.zeros(s)
for i in irange:
    for j in jrange:
        B[i,j] = image[len(jrange)-1-j,i][2]
            
image_cooked_B = []         
gx,gy = np.gradient(B)
grad = np.sqrt(gx**2 + gy**2)

for i in irange:
    for j in jrange:
        if grad[i,j] > grad_hold:
            a = [i,j]
            image_cooked_B.append(a)
            combine_RGB.append(a)
image_cooked_B = np.array(image_cooked_B)
combine_RGB = np.array(combine_RGB)
'''
fig = plt.figure(figsize=(10,10))
plt.subplot(221)
plt.plot(image_cooked_R[:,0],image_cooked_R[:,1],'r.')
plt.title("(a) R channel",loc = 'left')

plt.subplot(222)
plt.plot(image_cooked_G[:,0],image_cooked_G[:,1],'g.')
plt.title("(b) G channel",loc = 'left')

plt.subplot(223)
plt.plot(image_cooked_B[:,0],image_cooked_B[:,1],'b.')
plt.title("(c) B channel",loc = 'left')

plt.subplot(224)
plt.plot(combine_RGB[:,0],combine_RGB[:,1],'k.')
plt.title("(d) combine channel",loc = 'left')
'''
sigma = 1.0 
numev = 120
outlier = 0
fig = plt.figure(figsize=(10,10))
plt.subplot(221)
plt.imshow(image[45:195,121:-29])
plt.title("(a) Original figure",loc = 'left')
plt.tick_params(direction = 'in')

plt.subplot(222)
data = image_cooked_R
kpcabound(data,sigma,numev,outlier,'r.')
plt.title(")b) Bound for R",loc = 'left')
plt.tick_params(direction = 'in')

plt.subplot(223)
data = image_cooked_G
kpcabound(data,sigma,numev,outlier,'g.')
plt.title("(c) Bound for G",loc = 'left')
plt.tick_params(direction = 'in')

plt.subplot(224)
data = combine_RGB
kpcabound(data,sigma,numev,outlier,'k.')
plt.title("(d) Bound for R&G&B",loc = 'left')
plt.tick_params(direction = 'in')
'''
print('sigma ='+str(sigma))
print('numev ='+str(numev))
name = 'sigma ='+str(sigma) + 'numev ='+str(numev)
plt.savefig(name+".pdf")
'''
plt.savefig("reconstruction_error.pdf")
############################### R pre-image ###################################
