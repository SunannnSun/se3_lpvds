import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import skewnorm

# Parameters for the normal distribution
mu, sigma = 0, 3  # mean and standard deviation

# Generate data for the PDF plot

x = np.linspace(-10, 18, 1000)  # Range of x values



pdf = norm.pdf(x, mu, sigma)  # PDF values corresponding to the x values
pdf2 = norm.pdf(x, 8, 3)


# Plot the probability density function (PDF) of the normal distribution
plt.figure(figsize=(8, 6))

# plt.plot(x, pdf, color='b', label='PDF')
plt.fill_between(x, pdf, color='steelblue', alpha=0.6)  # Fill the area under the curve

data = np.random.normal(3, sigma, 1000)

plt.hist(data, bins=50, density=True, alpha=0.6, color='gray')


plt.fill_between(x, pdf2, color='coral', alpha=0.6)  # Fill the area under the curve

plt.ylim((0, 0.25))

plt.axis('off')

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)



plt.savefig('sampling.png', dpi=600, transparent=True)
plt.show()
