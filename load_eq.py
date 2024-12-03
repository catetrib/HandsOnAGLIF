def load():
    import sympy as sym
    t = sym.Symbol('t')
    alpha, beta, gamma, IaA0, IdA0, t0, V0 = sym.symbols('alpha,beta,gamma,IaA0,IdA0,t0,V0')


    V = (1 / 2) * sym.exp(1) ** ((-1) * t0 * (1 + (-1) * beta) ** (1 / 2) + (-1) * t * ((1 + (-1) * beta) ** (1 / 2) + gamma)) * (1 + (-1) * beta) ** (-3 / 2) * ((-1) + beta + gamma ** 2) ** (-1) * ((-2) * sym.exp(1) ** (t * (1 + (-1) * beta) ** (1 / 2) + t0 * ((1 + (-1) * beta) ** (1 / 2) + gamma)) * IdA0 * (1 + (-1) * beta) ** (3 / 2) * beta * ((-1) + gamma) + 2 * sym.exp(1) ** (t0 * (1 + (-1) * beta) ** (1 / 2) + t * ((1 + (-1) * beta) ** (1 / 2) + gamma)) * (1 + (-1) * beta) ** (1 / 2) * ((-1) + (-1) * alpha + beta) * ((-1) + beta + gamma ** 2) + sym.exp(1) ** (2 * t0 * (1 + (-1) * beta) ** (1 / 2) + t * gamma) * ((-1) * V0 * ((-1) + (1 + (-1) * beta) ** (1 / 2)) * ((-1) + beta) * ((-1) + beta + gamma ** 2) + alpha * ((-1) + (1 + (-1) * beta) ** (1 / 2) + beta) * ((-1) + beta + gamma ** 2) + (-1) * ((-1) + beta) * ((IaA0 + (-1) * IdA0) * beta ** 2 + ((-1) + (1 + (-1) * beta) ** (1 / 2)) * beta * (1 + IdA0 * ((-1) + gamma)) + ((-1) + (1 + (-1) * beta) ** (1 / 2)) * ((-1) + gamma ** 2) + IaA0 * beta * ((-1) + gamma ** 2))) + ( -1) * sym.exp(1) ** (t * (2 * (1 + (-1) * beta) ** (1 / 2) + gamma)) * ((-1) * alpha * (1 + (1 + (-1) * beta) ** (1 / 2) + (-1) * beta) * ((-1) + beta + gamma ** 2) + V0 * (1 + (1 + (-1) * beta) ** (1 / 2)) * ((-1) + beta) * ((-1) + beta + gamma ** 2) + ((-1) + beta) * (((-1) * IaA0 + IdA0) * beta ** 2 + (1 + (1 + (-1) * beta) ** (1 / 2)) * ((-1) + gamma ** 2) + beta * (IaA0 + (1 + (1 + (-1) * beta) ** (1 / 2)) * (1 + IdA0 * ((-1) + gamma)) + (-1) * IaA0 * gamma ** 2))));
    Iadap = (1 / 2) * sym.exp(1) ** ((-1) * t0 * (1 + (-1) * beta) ** (1 / 2) + (-1) * t * ((1 + (-1) * beta) ** (1 / 2) + gamma)) * (1 + (-1) * beta) ** (-3 / 2) * ((-1) + beta + gamma ** 2) ** (-1) * (2 * sym.exp(1) ** (t * (1 + (-1) * beta) ** (1 / 2) + t0 * ((1 + (-1) * beta) ** (1 / 2) + gamma)) * IdA0 * (1 + (-1) * beta) ** (3 / 2) * beta + (-2) * sym.exp(1) ** (t0 * (1 + (-1) * beta) ** (1 / 2) + t * ((1 + (-1) * beta) ** (1 / 2) + gamma)) * alpha * (1 + (-1) * beta) ** (1 / 2) * ((-1) + beta + gamma ** 2) + (-1) * sym.exp(1) ** (t * (2 * (1 + (-1) * beta) ** (1 / 2) + gamma)) * (1 + alpha * (1 + (-1) * beta) ** (1 / 2) + (-2) * beta + IdA0 * (1 + (-1) * beta) ** (1 / 2) * beta + (-1) * alpha * (1 + (-1) * beta) ** (1 / 2) * beta + beta ** 2 + (-1) * IdA0 * ( 1 + (-1) * beta) ** (1 / 2) * beta ** 2 + (-1) * IdA0 * beta * gamma + IdA0 * beta ** 2 * gamma + (-1) * gamma ** 2 + (-1) * alpha * (1 + (-1) * beta) ** (1 / 2) * gamma ** 2 + beta * gamma ** 2 + IaA0 * (( -1) + (1 + (-1) * beta) ** (1 / 2)) * ((-1) + beta) * ((-1) + beta + gamma ** 2) + V0 * (1 + beta ** 2 + (-1) * gamma ** 2 + beta * ((-2) + gamma ** 2))) + sym.exp(1) ** (2 * t0 * (1 + (-1) * beta) ** (1 / 2) + t * gamma) * (1 + (-1) * alpha * (1 + (-1) * beta) ** (1 / 2) + (-2) * beta + (-1) * IdA0 * (1 + (-1) * beta) ** (1 / 2) * beta + alpha * (1 + (-1) * beta) ** (1 / 2) * beta + beta ** 2 + IdA0 * (1 + (-1) * beta) ** (1 / 2) * beta ** 2 + (-1) * IdA0 * beta * gamma + IdA0 * beta ** 2 * gamma + (-1) * gamma ** 2 + alpha * (1 + (-1) * beta) ** (1 / 2) * gamma ** 2 + beta * gamma ** 2 + (-1) * IaA0 * (1 + (1 + (-1) * beta) ** (1 / 2)) * ((-1) + beta) * ((-1) + beta + gamma ** 2) + V0 * (1 + beta ** 2 + (-1) * gamma ** 2 + beta * ((-2) + gamma ** 2))));
    Idep = sym.exp(1) ** (((-1) * t + t0) * gamma) * IdA0;
    return [V,Iadap,Idep]#,t,alpha, beta, gamma, IaA0, IdA0, t0, V0]

def load_V_beta_1():
    import sympy as sym
    t = sym.Symbol('t')
    alpha, beta, gamma, IaA0, IdA0, t0, V0 = sym.symbols('alpha,beta,gamma,IaA0,IdA0,t0,V0')

    V = -t0 + t0*(IaA0 - V0) + V0 + t*(1 - IaA0 + V0) + 0.5*(t - t0)*(2 + t - t0)*alpha - ((-1 + sym.exp((-t + t0)*gamma))*IdA0*(-1+gamma)) / gamma**2 + (IdA0*(t - t0)) / gamma

    return V

def load_eq_gamma_1_beta_1():
    import sympy as sym
    t = sym.Symbol('t')
    alpha, beta, gamma, IaA0, IdA0, t0, V0 = sym.symbols('alpha,beta,gamma,IaA0,IdA0,t0,V0')

    V = IdA0*(t - t0) - t0 + t0*(IaA0 - V0) + V0 + t*(1 - IaA0 + V0) + 1 / 2*(t - t0)*(2 + t - t0)*alpha
    Iadap = IdA0 * (-1 + sym.exp(1) ** ((-1) * t + t0) + t - t0) + IaA0*(1 - t + t0) + 0.5 * (t - t0) * (2 + 2 * V0 + t * alpha - t0 * alpha)
    Idep = IdA0 * (sym.exp(1) ** ((-1) * t + t0))

    return [V,Iadap,Idep]


def load_time_gamma_1_beta_1():
    import sympy as sym
    t = sym.Symbol('t')
    alpha, beta, gamma, IaA0, IdA0, t0, V0,Vth = sym.symbols('alpha,beta,gamma,IaA0,IdA0,t0,V0,Vth')

    func_t=(-1+IaA0-IdA0-V0+(-1+t0)*alpha+sym.sqrt((1-IaA0+IdA0+V0)**2+2*(1-IaA0+IdA0+Vth)*alpha+alpha**2 ))/alpha
    return func_t


def load_v3():
    import sympy as sym
    t = sym.Symbol('t')
    delta,Psi,alpha, beta, gamma, IaA0, IdA0, t0, V0 = sym.symbols('delta,Psi,alpha,beta,gamma,IaA0,IdA0,t0,V0')

    V = (1 / 2) * (beta + (-1) * delta) ** (-1) * (beta ** 2 + ((-1) + beta) * delta) ** (-1) * (4 * beta + (- 1) * (1 + delta) ** 2) ** (-1) * Psi * (2 * sym.exp(1) ** (((-1) * t + t0) * beta) * IdA0 * ((-1) + beta) * beta * (beta + (-1) * delta) * Psi + (-2) * (alpha + (-1) * beta + delta) * (beta ** 2 + ((- 1) + beta) * delta) * Psi + sym.exp(1) ** ((1 / 2) * (t + (-1) * t0) * ((-1) + delta + (-1) * Psi)) * (IdA0 * beta * (beta + (-1) * delta) * ((-1) + (-1) * delta + beta * (3 + delta + (-1) * Psi) + Psi)+ (-1) * (beta ** 2 + (-1) * delta + beta * delta) * (alpha * (1 + (-2) * beta + delta + (-1) * Psi) + (beta + (-1) * delta) * ((-1) + 2 * IaA0 * beta + (-1) * delta + Psi + V0 * ((-1) + (-1) * delta + Psi))))+ sym.exp(1) ** ((1 / 2) * (t + (-1) * t0) * ((-1) + delta + Psi)) * ((-1) * IdA0 * beta * (beta+(-1) * delta) * ((-1) + (-1) * delta + (-1) * Psi + beta * (3 + delta + Psi)) + ( beta ** 2 + (-1) * delta+beta * delta) * (alpha * (1 + (-2) * beta + delta + Psi) + (beta + (-1) * delta) * ((-1) + 2 * IaA0 *  beta+(-1) * delta + (-1) * Psi + (-1) * V0 * (1 + delta + Psi)))))
    Iadap = (1 / 2) * sym.exp(1) ** (t0 + (-1) * t0 * delta + (-1 / 2) * t * ((-1) + 2 * beta + delta + Psi)) * (beta+(-1) * delta) ** (-1) * (beta ** 2 + ((-1) + beta) * delta) ** (-1) * ( 4 * beta + (-1) * ( 1 + delta) ** 2) ** (-1) * ( 2 * sym.exp(1) ** (t0 * ((-1) + beta + delta) + ( 1 / 2) * t * (( -1) + delta + Psi)) * IdA0 * beta * (beta + ( -1) * delta) * (4 * beta + (-1) * (1 + delta) ** 2) + (-2) * sym.exp(1) ** (t0 * ((-1) + delta) + (1 / 2) * t * ( (-1) + 2 * beta + delta + Psi)) * alpha * (beta ** 2 + ((-1) + beta) * delta) * ((-4) * beta + ( 1 + delta) ** 2) + sym.exp(1) ** ((1 / 2) * t0 * ((-1) + delta + (-1)* Psi) + t * ((-1) + beta + delta + Psi)) * ((-1) * IdA0 * beta * (beta + (-1) * delta) * ((-1) * (1 + delta) ** 2 + ((-1) + delta) * Psi + 2 * beta * (2 + Psi)) + (beta ** 2 + ((-1) + beta) * delta) * (alpha * (1+(-4) * beta + delta * ( 2 + delta + (-1) * Psi) + Psi) + (beta + (-1) * delta) * (4 * IaA0 * beta+(-2) * (1 + V0) * Psi + IaA0 * (1 + delta) * ((-1) + (-1) * delta + Psi))))+sym.exp(1) ** (t * ((-1) + beta + delta)+(1 / 2) * t0 * ((-1) + delta + Psi))*(IdA0 * beta * (beta + (-1) * delta) * ((1 + delta) ** 2 + 2 * beta * ((-2) + Psi) + ((-1) + delta) * Psi) + (beta ** 2 + ((-1) + beta) * delta)* (alpha * ((-4) * beta + (1 + delta) ** 2 + ((-1) + delta) * Psi) + (beta + (-1) * delta) * (4 * IaA0 * beta+2 * (1+V0) * Psi + (-1) * IaA0 * (1 + delta) * (1 + delta + Psi)))))
    Idep = sym.exp(1) ** (((-1) * t + t0) * beta) * IdA0

    return [V, Iadap, Idep]