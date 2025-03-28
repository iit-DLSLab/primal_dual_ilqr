from jax import debug, grad, jit, lax, scipy, vmap
import jax
import jax.numpy as np

from functools import partial

from trajax.optimizers import evaluate, linearize, quadratize,vectorize

from .kkt_helpers import compute_search_direction_kkt, tvlqr_kkt

from .dual_tvlqr import dual_lqr, dual_lqr_backward, dual_lqr_gpu,dual_lqr_backward_constrained

from .linalg_helpers import (
    invert_symmetric_positive_definite_matrix,
    project_psd_cone,
)

from .primal_tvlqr import tvlqr, tvlqr_gpu, rollout, rollout_gpu,non_linear_rollout, tvlqr_gpu_constrained, rollout_gpu_constrained,tvlqr_constrained
import time

def linearize_scan(fun, argnums=3):
    """Gradient or Jacobian operator using scan.

    Args:
        fun: numpy scalar or vector function with signature fun(x, u, t, *args).
        argnums: number of leading arguments of fun to process.

    Returns:
        A function that evaluates Gradients or Jacobians with respect to states and
        controls along a trajectory.

        Example:
            dynamics_jacobians = linearize(dynamics)
            cost_gradients = linearize(cost)
            A, B = dynamics_jacobians(X, pad(U), timesteps)
            q, r = cost_gradients(X, pad(U), timesteps)

            where,
              X is [T+1, n] state trajectory,
              U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
              timesteps is typically np.arange(T+1)

              and A, B are Dynamics Jacobians wrt state (x) and control (u) of
              shape [T+1, n, n] and [T+1, n, m] respectively;

              and q, r are Cost Gradients wrt state (x) and control (u) of
              shape [T+1, n] and [T+1, m] respectively.

              Note: due to padding of U, last row of A, B, and r may be discarded.
    """
    jacobian_x = jax.jacobian(fun)
    jacobian_u = jax.jacobian(fun, argnums=1)

    def scan_fun(carry, inputs):
        args = (*carry, *inputs)
        A = jacobian_x(*args)
        B = jacobian_u(*args)
        return carry, (A, B)

    def linearizer(x, u, t, *args):
        inputs = (x, u, t)
        _, (A, B) = lax.scan(scan_fun, args, inputs)
        return A, B

    return linearizer
def linearize_obj_scan(fun, argnums=5):
    """Gradient or Jacobian operator using scan.

    Args:
        fun: numpy scalar or vector function with signature fun(x, u, t, *args).
        argnums: number of leading arguments of fun to process.

    Returns:
        A function that evaluates Gradients or Jacobians with respect to states and
        controls along a trajectory.

        Example:
            dynamics_jacobians = linearize(dynamics)
            cost_gradients = linearize(cost)
            A, B = dynamics_jacobians(X, pad(U), timesteps)
            q, r = cost_gradients(X, pad(U), timesteps)

            where,
              X is [T+1, n] state trajectory,
              U is [T, m] control sequence (pad(U) pads a 0 row for convenience),
              timesteps is typically np.arange(T+1)

              and A, B are Dynamics Jacobians wrt state (x) and control (u) of
              shape [T+1, n, n] and [T+1, n, m] respectively;

              and q, r are Cost Gradients wrt state (x) and control (u) of
              shape [T+1, n] and [T+1, m] respectively.

              Note: due to padding of U, last row of A, B, and r may be discarded.
    """
    jacobian_x = jax.jacobian(fun)
    jacobian_u = jax.jacobian(fun, argnums=1)

    def scan_fun(carry, inputs):
        args = (*carry, *inputs)
        A = jacobian_x(*args)
        B = jacobian_u(*args)
        return carry, (A, B)

    def linearizer(x, u,  v, v1, t, *args):
        inputs = (x, u, v, v1,t)
        _, (A, B) = lax.scan(scan_fun, args, inputs)
        return A, B

    return linearizer
def lagrangian(cost, dynamics, x0):
    """Returns a function to evaluate the associated Lagrangian."""

    def fun(x, u, t, v, v_prev):
        c1 = cost(x, u, t)
        c2 = np.dot(v, dynamics(x, u, t))
        c3 = np.dot(v_prev, lax.select(t == 0, x0 - x, -x))
        return c1 + c2 + c3

    return fun
@partial(jit, static_argnums=(0, 1, 2, 3))
def compute_search_direction(
    cost,
    dynamics,
    hessian_approx,
    limited_memory,
    x0,
    X,
    U,
    V,
    c,
):
    """Computes the SQP search direction.

    Args:
      cost:          cost function with signature cost(x, u, t).
      dynamics:      dynamics function with signature dynamics(x, u, t).
      x0:            [n]           numpy array.
      X:             [T+1, n]      numpy array.
      U:             [T, m]        numpy array.
      V:             [T+1, n]      numpy array.
      c:             [T+1, n]      numpy array.
      make_psd:      whether to zero negative eigenvalues after quadratization.
      psd_delta:     the minimum eigenvalue post PSD cone projection.

    Returns:
      dX: [T+1, n] numpy array.
      dU: [T, m]   numpy array.
      q: [T+1, n]  numpy array.
      r: [T, m]    numpy array.
    """
    T = U.shape[0]

    pad = lambda A: np.pad(A, [[0, 1], [0, 0]])

    if hessian_approx is None:
        quadratizer = quadratize(lagrangian(cost, dynamics, x0), argnums=5)
        Q, R_pad, M_pad = quadratizer(X, pad(U), np.arange(T + 1), pad(V[1:]), V)
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx)(X, pad(U), np.arange(T + 1))
    
    R = R_pad[:-1]
    M = M_pad[:-1]   

    if limited_memory:
        linearizer = linearize_obj_scan(lagrangian(cost, dynamics, x0),argnums = 5)
        dynamics_linearizer = linearize_scan(dynamics)
    else :
        linearizer = linearize(lagrangian(cost, dynamics, x0),argnums = 5)
        dynamics_linearizer = linearize(dynamics)
    q, r_pad = linearizer(X, pad(U), np.arange(T + 1), pad(V[1:]), V)
    r = r_pad[:-1]

    A_pad, B_pad = dynamics_linearizer(X, pad(U), np.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]

    if limited_memory:
        K, k, P, p = tvlqr(Q, q, R, r, M, A, B, c[1:])
        dX, dU = rollout(K, k, c[0], A, B, c[1:])
    else:
        K, k, P, p = tvlqr_gpu(Q, q, R, r, M, A, B, c[1:])
        dX, dU = rollout_gpu(K, k, c[0], A, B, c[1:])
    
    dV = dual_lqr(dX, P, p)
    # dV = dual_lqr_backward(Q, q, M, A, dX, dU)
    # dV = dual_lqr_gpu(Q, q, M, A, dX, dU)

    # new_dX, new_dU, new_dV, LHS, rhs = tvlqr_kkt(Q, q, R, r, M, A, B, c[1:], c[0])

    # candidate_sol = np.concatenate([dX.flatten(), dU.flatten(), dV.flatten()])
    # candidate_sol = np.concatenate([new_dX.flatten(), new_dU.flatten(), new_dV.flatten()])
    # error = LHS @ candidate_sol - rhs
    # debug.print(f"error_norm={np.linalg.norm(error)}")

    # return new_dX, new_dU, new_dV, q, r

    return dX, dU, dV, q, r , K[0]
@jit
def merit_rho(c, dV):
    """Determines the merit function penalty parameter to be used.

    Args:
      c:             [T+1, n]  numpy array.
      dV:            [T+1, n]  numpy array.

    Returns:
        rho: the penalty parameter.
    """
    c2 = np.sum(c * c)
    dV2 = np.sum(dV * dV)
    return lax.select(c2 > 1e-12, 2.0 * np.sqrt(dV2 / c2), 1e-2)


@jit
def slope(dX, dU, dV, c, q, r, rho):
    """Determines the directional derivative of the merit function.

    Args:
      dX: [T+1, n] numpy array.
      dU: [T, m]   numpy array.
      dV: [T+1, n] numpy array.
      c:  [T+1, n] numpy array.
      q:  [T+1, n] numpy array.
      r:  [T, m] numpy array.
      rho: the penalty parameter of the merit function.

    Returns:
        dir_derivative: the directional derivative.
    """
    return np.sum(q * dX) + np.sum(r * dU) + 2*np.sum(dV * c) - rho * np.sum(c * c)

@partial(jit, static_argnums=(0, 1))
def line_search(
    merit_function,
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_merit,
    current_g,
    current_c,
    merit_slope,
    armijo_factor,
    alpha_0,
    alpha_mult,
    alpha_min,
):
    """Performs a primal-dual line search on an augmented Lagrangian merit function.

    Args:
      merit_function:  merit function mapping V, g, c to the merit scalar.
      X_in:            [T+1, n]      numpy array.
      U_in:            [T, m]        numpy array.
      V_in:            [T+1, n]      numpy array.
      dX:              [T+1, n]      numpy array.
      dU:              [T, m]        numpy array.
      dV:              [T+1, n]      numpy array.
      current_merit:   the merit function value at X, U, V.
      current_g:       the cost value at X, U, V.
      current_c:       the constraint values at X, U, V.
      merit_slope:     the directional derivative of the merit function.
      armijo_factor:   the Armijo parameter to be used in the line search.
      alpha_0:         initial line search value.
      alpha_mult:      a constant in (0, 1) that gets multiplied to alpha to update it.
      alpha_min:       minimum line search value.

    Returns:
      X: [T+1, n]     numpy array, representing the optimal state trajectory.
      U: [T, m]       numpy array, representing the optimal control trajectory.
      V: [T+1, n]     numpy array, representing the optimal multiplier trajectory.
      new_g:          the cost value at the new X, U, V.
      new_c:          the constraint values at the new X, U, V.
      no_errors:       whether no error occurred during the line search.
    """

    def continuation_criterion(inputs):
        _, _, _, _, _, new_merit, alpha = inputs
        # debug.print(f"{new_merit=}, {current_merit=}, {alpha=}, {merit_slope=}")\
        return np.logical_and(
            new_merit > current_merit + alpha * armijo_factor * merit_slope,
            alpha > alpha_min,
        )

    def body(inputs):
        _, _, _, _, _, _, alpha = inputs
        alpha *= alpha_mult
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        new_g, new_c = model_evaluator(X_new, U_new)
        new_merit = merit_function(V_new, new_g, new_c)
        new_merit = np.where(np.isnan(new_merit), current_merit, new_merit)
        return X_new, U_new, V_new, new_g, new_c, new_merit, alpha

    X, U, V, new_g, new_c, new_merit, alpha = lax.while_loop(
        continuation_criterion,
        body,
        (X_in, U_in, V_in, current_g, current_c, np.inf, alpha_0 / alpha_mult),
    )
    no_errors = alpha > alpha_min


    return X, U, V, new_g, new_c, no_errors

@partial(jit, static_argnums=(0, 1))
def parallel_line_search(
    merit_function,
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_merit,
    current_g,
    current_c,
    merit_slope,
    armijo_factor,
    
):
    """Performs a primal-dual line search on an augmented Lagrangian merit function in parralel fixing the number of steps.

    Args:
      merit_function:  merit function mapping V, g, c to the merit scalar.
      X_in:            [T+1, n]      numpy array.
      U_in:            [T, m]        numpy array.
      V_in:            [T+1, n]      numpy array.
      dX:              [T+1, n]      numpy array.
      dU:              [T, m]        numpy array.
      dV:              [T+1, n]      numpy array.
      current_merit:   the merit function value at X, U, V.
      current_g:       the cost value at X, U, V.
      current_c:       the constraint values at X, U, V.
      merit_slope:     the directional derivative of the merit function.
      armijo_factor:   the Armijo parameter to be used in the line search.
      alpha_0:         initial line search value.
      alpha_mult:      a constant in (0, 1) that gets multiplied to alpha to update it.
      alpha_min:       minimum line search value.

    Returns:
      X: [T+1, n]     numpy array, representing the optimal state trajectory.
      U: [T, m]       numpy array, representing the optimal control trajectory.
      V: [T+1, n]     numpy array, representing the optimal multiplier trajectory.
      new_g:          the cost value at the new X, U, V.
      new_c:          the constraint values at the new X, U, V.
      no_errors:       whether no error occurred during the line search.
    """
    def step_acceptance(merit,alpha):
        return merit > current_merit + alpha * armijo_factor * merit_slope
    alpha_values = np.exp2(-np.arange(11))
    def body(alpha):
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        new_g, new_c = model_evaluator(X_new, U_new) #this cam br probly avoided 
        new_merit = merit_function(V_new, new_g, new_c)
        new_merit = np.where(np.isnan(new_merit), current_merit, new_merit)
        return X_new, U_new, V_new, new_g, new_c, new_merit

    X, U, V, new_g, new_c, new_merit = vmap(body)(alpha_values)
    acceptance = vmap(step_acceptance)(new_merit,alpha_values)
    best_index = np.where(np.any(acceptance),np.argmin(acceptance),0)
    return X[best_index], U[best_index], V[best_index], new_g[best_index], new_c[best_index]

@partial(jit, static_argnums=(0))
def filter_line_search(
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_cost,
    current_c,
    q,
    r,
    # Hyperparameters
    alpha_min=1e-4,
    theta_max=1e-2,
    theta_min=1e-6,
    eta=1e-4,
    gamma_phi=1e-6,
    gamma_theta=1e-6,
    gamma_alpha=0.5,
):
    """Performs a backtracking line search.
    
    Args:
      X_in: [T+1, n] numpy array of current states.
      U_in: [T, m] numpy array of current controls.
      V_in: [T+1, n] numpy array of current multipliers.
      dX: [T+1, n] numpy array of state search direction.
      dU: [T, m] numpy array of control search direction.
      dV: [T+1, n] numpy array of multiplier search direction.
      current_cost: Current cost value (phi_k).
      current_c: Current constraint violation (theta_k).
      alpha_min: Minimum step size.
      theta_max: Maximum acceptable constraint violation.
      theta_min: Minimum constraint violation worth considering.
      eta: Armijo parameter for sufficient decrease.
      gamma_phi: Cost reduction parameter.
      gamma_theta: Constraint violation reduction parameter.
      gamma_alpha: Step size reduction factor.
      
    Returns:
      X: [T+1, n] numpy array of updated states.
      U: [T, m] numpy array of updated controls.
      V: [T+1, n] numpy array of updated multipliers.
      new_cost: Updated cost value.
      new_c: Updated constraint values.
      accepted: Whether a step was accepted.
    """
    # Initial values
    alpha = 1.0
    theta_k = np.sum(current_c * current_c)  # Constraint violation measure
    phi_k = current_cost
    slope = np.sum(q*dX) + np.sum(r*dU)

    def continuation_criterion(inputs):
        _, _, _, alpha, accepted = inputs
        return np.logical_and(np.logical_not(accepted), alpha > alpha_min)
    
    def body(inputs):
        _, _, _, alpha, _ = inputs
        
        # Compute trial point
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        
        # Evaluate at new point
        new_cost, new_c = model_evaluator(X_new, U_new)
        theta_new = np.sum(new_c * new_c)  # Constraint violation measure
        phi_new = new_cost
        
        # Case 1: Large constraint violation but improving
        case1 = np.logical_and(
            theta_new > theta_max,
            theta_new < (1 - gamma_theta) * theta_k
        )
        
        # Case 2: Small constraint violations and cost is decreasing
        case2 = np.logical_and(
            np.logical_and(
                np.maximum(theta_new, theta_k) < theta_min,
                slope < 0
            ),
            phi_new < phi_k + eta * alpha * slope
        )
        
        # Case 3: Either cost or constraint violation is significantly reduced
        case3 = np.logical_or(
            phi_new < phi_k - gamma_phi * theta_k,
            theta_new < (1 - gamma_theta) * theta_k
        )
        
        # Accept if any case is satisfied
        new_accepted = np.logical_or(np.logical_or(case1, case2), case3)
        
        # If not accepted, reduce alpha
        alpha = np.where(new_accepted, alpha, gamma_alpha * alpha)
        
        return X_new, U_new, V_new, alpha, new_accepted
    
    # Run the backtracking loop
    X, U, V, alpha, accepted = lax.while_loop(
        continuation_criterion,
        body,
        (X_in, U_in, V_in, alpha, False)
    )
    
    return X, U, V
@partial(jit, static_argnums=(0, 1))
def model_evaluator_helper(cost, dynamics,reference,parameter,x0, X, U):
    """Evaluates the costs and constraints based on the provided primal variables.

    Args:
      cost:            cost function with signature cost(x, u, t).
      dynamics:        dynamics function with signature dynamics(x, u, t).
      x0:              [n]           numpy array.
      X:               [T+1, n]      numpy array.
      U:               [T, m]        numpy array.

    Returns:
      g: the cost value (a scalar).
      c: the constraint values (a [T+1, n] numpy array).
    """
    T = U.shape[0]

    costs = partial(evaluate, cost)
    g = np.sum(costs(X, np.pad(U, [[0, 1], [0, 0]])))

    residual_fn = lambda t: dynamics(X[t], U[t], t) - X[t + 1]
    c = np.vstack([x0 - X[0], vmap(residual_fn)(np.arange(T))])

    return g, c
@partial(jit, static_argnums=(0,1,2,3))
def mpc(
    cost,
    dynamics,
    hessian_approx,
    limited_mempory,
    reference,
    parameter,
    W,
    x0,
    X_in,
    U_in,
    V_in,
    ):

    _cost = partial(cost,W,reference)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx,W,reference)
    else:
        _hessian_approx = None
    _dynamics = partial(dynamics,parameter=parameter)
    model_evaluator = partial(model_evaluator_helper, _cost, _dynamics,reference,parameter,x0)
    g, c = model_evaluator(X_in, U_in)
    dX,dU, dV, q, r, K = compute_search_direction(
            _cost,
            _dynamics,
            _hessian_approx,
            limited_mempory,
            x0,
            X_in,
            U_in,
            V_in,
            c,
        )
    # @jit
    # def merit_function(V, g, c, rho):
    #     return g + np.sum((V + 0.5 * rho * c) * c)

    # dV2 = np.sum(dV * dV)
    # c2 = np.sum(c * c)
    # rho  = 2.0 * np.sqrt(dV2 / c2)
    # merit = merit_function(V_in, g, c, rho)

    # merit_slope = slope(
    #     dX,
    #     dU,
    #     dV,
    #     c,
    #     q,
    #     r,
    #     rho,
    # )
    # X_new, U_new, V_new, g_new, c_new = parallel_line_search(
    #         partial(merit_function, rho=rho),
    #         model_evaluator,
    #         X_in,
    #         U_in,
    #         V_in,
    #         dX,
    #         dU,
    #         dV,
    #         merit,
    #         g,
    #         c,
    #         merit_slope,
    #         armijo_factor=1e-4,
    #     )
    X_new, U_new, V_new = filter_line_search(
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    g,
    c,
    q,
    r,)

    return X_new, U_new, V_new, K
