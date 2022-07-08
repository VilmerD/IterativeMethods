from linalg.twogrid import *
from matrices.matricies import *
import matplotlib.pyplot as plt

n = 63
tol = 10 ** -9
x = interval(n)
x0 = np.zeros(n, )
f = source(n)
res = np.linalg.norm(discrete_second(n).dot(x0) + f)
r = [res]
while res > tol:
    x0 = twogrid(lambda n: -discrete_second(n), x0, f)
    res = np.linalg.norm(discrete_second(n).dot(x0) + f)
    r.append(res)

fig, ax = plt.subplots()
plt.semilogy(r)
plt.show()
