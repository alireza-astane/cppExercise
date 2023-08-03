import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import matplotlib.animation as animation


tempTrajectory  = list(torch.load("/home/alireza-astane/dcgan/tempTrajectory.t").parameters())[0].numpy()
energyTrajectory = list(torch.load("/home/alireza-astane/dcgan/energyTrajectory.t").parameters())[0].numpy()
leftSideTrajectory = list(torch.load("/home/alireza-astane/dcgan/leftSideTrajectory.t").parameters())[0].numpy()
pressureTrajectory = list(torch.load("/home/alireza-astane/dcgan/pressureTrajectory.t").parameters())[0].numpy()
uTrajectory = list(torch.load("/home/alireza-astane/dcgan/uTrajectory.t").parameters())[0].numpy()
vTrajectory = list(torch.load("/home/alireza-astane/dcgan/vTrajectory.t").parameters())[0].numpy()
xTrajectory = list(torch.load("/home/alireza-astane/dcgan/xTrajectory.t").parameters())[0].numpy()


N =100
sigma = 1
epsilon = 1
dt = 0.00001
bigSteps = 1000
smallSteps = 100


fig, ax = plt.subplots()
images = []
jump = 1
name = "try1"
length = 50
plt.title("Animating md Model for bigSteps={0} size={1}".format(len(xTrajectory),length))
plt.xlim((0,length))
plt.ylim((0,length))
plt.xlabel("X")
plt.ylabel("Y")
ims = []
for i in range(len(xTrajectory)//jump):
    im = ax.scatter(xTrajectory[i*jump,:].real,xTrajectory[i*jump,:].imag,c="blue",s=1)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
ani.save("movie.mp4")
plt.show()


plt.figure(figsize=(10,10))
plt.title("the num of particles on the left side over steps")
plt.xlabel("steps")
plt.ylabel("the num of particles on the left side")
plt.plot(np.arange(bigSteps),leftSideTrajectory)
plt.savefig("leftSideTrajectory.png")
plt.show()


plt.figure(figsize=(10,10))
plt.title("the mechanical energy of the system over steps")
plt.xlabel("steps")
plt.ylabel("E")
plt.plot(np.arange(bigSteps),energyTrajectory)
plt.savefig("energyTrajectory.png")
plt.show()


plt.figure(figsize=(10,10))
plt.title("the pressure of the system over steps")
plt.xlabel("steps")
plt.ylabel("P")
plt.plot(np.arange(bigSteps),pressureTrajectory[:bigSteps])
plt.savefig("pressureTrajectory.png")
plt.show()

plt.figure(figsize=(10,10))
plt.title("the Temperature of the system over steps")
plt.xlabel("steps")
plt.ylabel("T")
plt.plot(np.arange(bigSteps),tempTrajectory[:bigSteps]/(2*N))
plt.savefig("pressureTrajectory.png")
plt.show()

print("T = ",np.mean(tempTrajectory[:bigSteps])/(2*N))
print("P = ",np.mean(pressureTrajectory[:bigSteps]))
