import matplotlib.pyplot as plt
import numpy as np

angle = np.arange(0, 180, 1)
softmax = np.cos(angle/180*np.pi)

m = 0.2
amsoftmax = np.cos(angle/180*np.pi) - m

m = 0.3
arcsoftmax = np.cos(angle/180*np.pi + m)

m = 4.0
l = 10.0
angle1 = np.arange(0, 180/4, 1)
a1 = l / (1+l) * np.cos(angle1/180*np.pi) + 1 / (1+l) * np.cos(4 * angle1/180*np.pi)
angle2 = np.arange(180/4, 180/2, 1)
a2 = l / (1+l) * np.cos(angle2/180*np.pi) + 1 / (1+l) * (-np.cos(4 * angle2/180*np.pi) - 2)
angle3 = np.arange(180/2, 180*3/4, 1)
a3 = l / (1+l) * np.cos(angle3/180*np.pi) + 1 / (1+l) * (np.cos(4 * angle3/180*np.pi) - 4)
angle4 = np.arange(180*3/4, 180, 1)
a4 = l / (1+l) * np.cos(angle4/180*np.pi) + 1 / (1+l) * (-np.cos(4 * angle4/180*np.pi) - 6)
angle_new = np.concatenate([angle1, angle2, angle3, angle4], axis=0)
asoftmax = np.concatenate([a1, a2, a3, a4], axis=0)

l = 0
a1 = l / (1+l) * np.cos(angle1/180*np.pi) + 1 / (1+l) * np.cos(4 * angle1/180*np.pi)
a2 = l / (1+l) * np.cos(angle2/180*np.pi) + 1 / (1+l) * (-np.cos(4 * angle2/180*np.pi) - 2)
a3 = l / (1+l) * np.cos(angle3/180*np.pi) + 1 / (1+l) * (np.cos(4 * angle3/180*np.pi) - 4)
a4 = l / (1+l) * np.cos(angle4/180*np.pi) + 1 / (1+l) * (-np.cos(4 * angle4/180*np.pi) - 6)
asoftmax_nolambda = np.concatenate([a1, a2, a3, a4], axis=0)

m = 1.20
asoftmax_new = np.cos(m * angle / 180 * np.pi)

plt.figure(1)
plt.plot(angle, softmax, 'b', label='Softmax')
plt.plot(angle_new, asoftmax_nolambda, 'r', label='ASoftmax ($m_1=4$, $\lambda=0$)')
plt.plot(angle_new, asoftmax, 'r', label='ASoftmax ($m_1=4$, $\lambda=10$)')
plt.plot(angle, arcsoftmax, 'c', label='ArcSoftmax ($m_2=0.30$)')
plt.plot(angle, amsoftmax, 'm', label='AMSoftmax ($m_3=0.20$)')
plt.xlabel(r'$\theta$', fontsize='x-large')
plt.ylabel(r'$\psi(\theta)$', fontsize='x-large')
plt.xlim((10, 120))
plt.ylim((-1.0, 1.0))
plt.legend(loc='lower left', fontsize='medium')
plt.savefig('target_logit_curve.pdf', format='pdf')
plt.show()
