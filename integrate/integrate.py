from newton.newton import JFNK

def implicit_euler(F, u0, tint, dt, rtol, M):
    tk, uk, U = tint[0], u0, [u0]
    R, N, E = [], [], []
    while tk < tint[1]:
        func = lambda u: F(tk, u, uk)
        sols, res, nits, etas = JFNK(func, uk, rtol=rtol, M=M)
        uk = sols[-1]

        for L, I in zip((R, N, E), (res, nits, etas)):
            L.append(I)

        tk += dt
        U.append(uk)
    
    return U, R, N, E