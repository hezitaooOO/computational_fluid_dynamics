import numpy as np
from matplotlib import pyplot as plt

pi_16 = np.float16(np.pi)
pi_32 = np.float32(np.pi)
pi_64 = np.float64(np.pi)


error_16 = []
error_32 = []
error_64 = []

times_16 = []
times_32 = []
times_64 = []

i=1
while i <= 100:
    pi_star_16 = np.float16(pi_16 ** (i + 1))
    pi_hat_16 = np.float16(pi_star_16 * np.float16(1 / pi_16) ** i)

    error_16.append(np.float64(np.abs(pi_16-pi_hat_16)))
    times_16.append(i)

    i += 1

i=1
while i <= 100:
    pi_star_32 = np.float32(pi_32 ** (i + 1))
    pi_hat_32 = np.float32(pi_star_32 * np.float32(1 / pi_32) ** i)

    error_32.append(np.float64(np.abs(pi_32-pi_hat_32)))
    times_32.append(i)

    i += 1

i=1
while i <= 100:
    pi_star_64 = np.float64(pi_64 ** (i + 1))
    pi_hat_64 = np.float64(pi_star_64 * np.float64(1 / pi_64) ** i)

    error_64.append(np.float64(np.abs(pi_64-pi_hat_64)))
    times_64.append(i)

    i += 1



plt.figure(1)
plt.plot(times_16, error_16, '*-', label = 'Truncation error with float16')
plt.title('Truncation error with float16')
plt.xlabel('n')
plt.ylabel('Truncation error')
plt.legend(loc = 'best')
plt.savefig('Problem 1_truncation error with float16')

plt.figure(2)
plt.plot(times_32, error_32, '*-', label = 'Truncation error with float32')
plt.title('Truncation error with float32')
plt.xlabel('n')
plt.ylabel('Truncation error')
plt.legend(loc = 'best')
plt.savefig('Problem 1_truncation error with float32')


plt.figure(3)

plt.plot(times_64, error_64, '*-', label = 'Truncation error with float64')
plt.title('Truncation error with float64')
plt.xlabel('n')
plt.ylabel('Truncation error')
plt.legend(loc = 'best')
plt.savefig('Problem 1_truncation error with float64')



