import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def residuals():
    x1, r1, etas1, iterations = newton(99, 0.1)
    x2, r2, etas2, iterations = newton(999, 0.1)
    fig, ax = plt.subplots()
    plt.title("Residuals with different dx, and eta_max = 0.1")
    ax1 = plt.subplot(121)
    plt.semilogy(range(0, len(r1)), r1, '.r')
    plt.title("dx = 1/100")
    ax2 = plt.subplot(122)
    plt.semilogy(range(0, len(r2)), r2, '.r')
    plt.title("dx = 1/1000")

    for axis in [ax1.xaxis, ax2.xaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_xlabel('k')
    ax2.set_xlabel('k')
    ax1.set_ylabel('Residual')
    plt.show()


def bara_en():
    x, rs, etas, iterations = newton(99, 0.999)
    fig, ax = plt.subplots()
    plt.subplot(131)
    plt.plot(range(0, len(iterations)), iterations, '.')
    plt.subplot(132)
    plt.semilogy(range(0, len(rs)), rs, 'r')
    plt.subplot(133)
    plt.plot(range(0, len(etas)), etas, '.')
    plt.title("")
    plt.show()


bara_en()