from jax import lax ,jit
@jit
def calc(Un, U_matrix):
    Un_1 = U_matrix @ Un
    return Un_1, Un_1

def fast_calc(U_matrix, Un, n_steps):
    func= jit (lambda Un,_: calc(Un, U_matrix))
    _, result = lax.scan(func, Un, None, length=n_steps)
    return result
