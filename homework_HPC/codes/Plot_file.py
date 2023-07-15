import os
import sys
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

inverse_spacing = [  5.00000000e+00,   5.00000000e+01,   5.00000000e+02,   5.00000000e+03,
   5.00000000e+04,   5.00000000e+05,   5.00000000e+06,   5.00000000e+07]

error_16 = [0.0139964406751467152,
1.24398681400350597e-06,
1.22136967206643021e-10,
7.97584220890712459e-13,
2.48689957516035065e-12,
1.43884903991420288e-12,
1.80833126250945497e-12,
5.89750470680883154e-13
]

error_8 = [0.012429617633205936,
1.22695923110427429e-06,
1.23208110380801372e-10,
2.38919994899333687e-12,
8.06821276455593761e-12,
2.31104024805972585e-12,
2.16893170090770582e-12,
1.11022302462515654e-12
]

error_4 = [0.0122369721008883658,
1.22381402434257325e-06,
1.24687815628021781e-10,
5.25446353094594087e-12,
4.79314365975369583e-11,
4.9666937229631003e-12,
5.00932628710870631e-12,
2.27018404075352009e-12
]

error_2 = [0.0122224197180642591,
1.22368301092024012e-06,
1.27029053942351311e-10,
1.056754683759209e-11,
1.80193637788761407e-11,
9.96713822587480536e-12,
9.95292737115960335e-12,
4.60609328456484945e-12
]


error_1 = [0.0122223731145147951,
1.22368842170317293e-06,
1.31532118530230946e-10,
2.11333173183447798e-11,
1.10663478380956803e-10,
1.9936052808589011e-11,
1.992894738123141e-11,
9.21573928280849941e-12]

plt.figure(1)    ###################### plot the truncation error with different processes#################
plt.loglog(inverse_spacing, error_16, linestyle="-", label="Truncation error with 16 cores")
plt.loglog(inverse_spacing, error_8, linestyle="-", label="Truncation error with 8 cores")
plt.loglog(inverse_spacing, error_4, linestyle="-", label="Truncation error with 4 cores")
plt.loglog(inverse_spacing, error_2, linestyle="-", label="Truncation error with 2 cores")
plt.loglog(inverse_spacing, error_1, linestyle="-", label="Truncation error with 1 cores")


plt.loglog(inverse_spacing, [x ** -1 for x in inverse_spacing], linestyle="-.", label='1st order')
plt.loglog(inverse_spacing, [x ** -2 for x in inverse_spacing], linestyle="-.", label='2nd order')
plt.loglog(inverse_spacing, [x ** -3 for x in inverse_spacing], linestyle="-.", label="3rd order")
plt.loglog(inverse_spacing, [x ** -4 for x in inverse_spacing], linestyle="-.", label="4th order")
plt.loglog(inverse_spacing, [x ** -5 for x in inverse_spacing], linestyle="-.", label="5th order")
plt.loglog(inverse_spacing, [x ** -6 for x in inverse_spacing], linestyle="-.", label='6th order')
plt.legend(loc='lower left')

plt.title("Truncation error of simpson-rule integral with different cores paralal ")
plt.xlabel("The inverse of the grid spacing")
plt.ylabel("The truncation error")
plt.savefig("Truncation error with different cores.png")
plt.show()





cores = [1, 2, 4, 8, 16]
local_time =[
585.093271970748901,
294.155380010604858,
146.844188928604126,
86.7914021015167236,
45.380141019821167
]
efficiency_50 = [0.5, 0.5, 0.5, 0.5, 0.5]
efficiency_75 = [0.75, 0.75, 0.75, 0.75, 0.75]
efficiency_100 = [1, 1, 1, 1, 1]
efficiency = np.zeros(5)

for i in range(0, 5):
    efficiency[i] = 585.093271970748901/(local_time[i]*cores[i])

plt.figure(2)      ###################### plot the strong scaling efficiency#################
plt.plot(cores, efficiency, '*-', label="Efficiency")
plt.plot(cores, efficiency_50, linestyle="-.", label="50% efficiency")
plt.plot(cores, efficiency_75, linestyle="-.", label="75% efficiency")
plt.plot(cores, efficiency_100, linestyle="-.", label="100% efficiency")
plt.axis([0, 18, 0, 1.2])
plt.legend(loc='lower left')
plt.title("Strong scaling efficiency of varying cores")
plt.xlabel("Number of cores")
plt.ylabel("Efficiency")
plt.savefig("Strong scaling with different cores.png")
plt.show()



cores = [1, 2, 4, 8, 16]
local_time =[
56.8446991443634033,
57.6200780868530273,
64.9311239719390869,
65.0036101341247559,
71.3357329368591309
]
efficiency_50 = [0.5, 0.5, 0.5, 0.5, 0.5]
efficiency_75 = [0.75, 0.75, 0.75, 0.75, 0.75]

efficiency_100 = [1, 1, 1, 1, 1]

efficiency = np.zeros(5)

for i in range(0, 5):
    efficiency[i] = local_time[0]/local_time[i]

plt.figure(3)    ###################### plot the weak scaling efficiency#################
plt.plot(cores, efficiency, '*-', label="Efficiency")
plt.plot(cores, efficiency_50, linestyle="-.", label="50% efficiency")
plt.plot(cores, efficiency_75, linestyle="-.", label="75% efficiency")
plt.plot(cores, efficiency_100, linestyle="-.", label="100% efficiency")
plt.axis([0, 18, 0, 1.2])
plt.legend(loc='lower left')

plt.title("Weak scaling efficiency of varying cores")
plt.xlabel("Number of cores")
plt.ylabel("Efficiency")
plt.savefig("Weak scaling with different cores.png")
plt.show()




cores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

error =[
2.11333173183447798e-11,
1.056754683759209e-11,
3.99804633843814372e-11,
5.25446353094594087e-12,
4.16733314523298759e-12,
1.5383250229206169e-12,
4.16715550954904757e-11,
2.38919994899333687e-12,
2.89368529138300801e-12,
2.34834374168713111e-12,
7.62110374807889457e-11,
9.74509362094977405e-12,
2.32152075341218733e-11,
3.84794418550882256e-11,
7.93676235844031908e-12,
7.97584220890712459e-13
]
average = np.zeros(16)
a = np. average(error)
for i in range (0,16):
    average[i] = a

plt.figure(4)    ###################### plot the truncation error with fixed number of points #################
plt.plot(cores, error, '*-', label="Truncation error",)
plt.plot(cores, average, '-.', label="Average error",)
plt.xticks( arange(18) )
plt.legend(loc='upper left')

plt.title("Fixed nodes truncation error with varying cores")
plt.xlabel("Number of cores")
plt.ylabel("Truncation error")
plt.savefig("Truncation error with fixed nodes.png")
plt.show()
