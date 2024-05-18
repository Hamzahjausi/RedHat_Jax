import jax.numpy as jnp
from jax import config
config.update ("jax_enable_x64", True)


def forward_setup(system_matrix, dt, t_max):
    n_steps = int(t_max / dt) + 1
    U_matrix = jnp.eye(system_matrix.shape[0]) + dt * system_matrix
    return n_steps, U_matrix

def backward_setup(system_matrix, dt, t_max):
    n_steps = int(t_max / dt) + 1
    I = jnp.eye(system_matrix.shape[0])
    U_matrix = jnp.linalg.inv(I - dt * system_matrix)
    return n_steps, U_matrix

def backward_setup(system_matrix, dt, t_max):
    n_steps = int(t_max / dt) + 1
    I = jnp.eye(system_matrix.shape[0])
    U_matrix = jnp.linalg.inv(I - dt * system_matrix)
    return n_steps, U_matrix


