import jax.numpy as jnp
import jax

def quaternion_product(q1, q2):
    w1 = q1[0]
    x1 = q1[1]
    y1 = q1[2]
    z1 = q1[3]
    w2 = q2[0]
    x2 = q2[1]
    y2 = q2[2]
    z2 = q2[3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return jnp.array([w,x, y, z])
def quaternion_integration(w, q, dt):
    # v = w[:3]*dt*0.5*jnp.sin(jnp.linalg.norm(w[:3])*dt/2)/(jnp.linalg.norm(w[:3])*dt/2)
    # q_dot = jnp.array([v[0], v[1], v[2], jnp.cos(jnp.linalg.norm(w[:3])*dt/2)])
    # q_next = quaternion_product(q_dot,q)
    # return jnp.where(jnp.linalg.norm(w[:3]) < 1e-4,q,q_next)
    temp = jnp.sin(jnp.linalg.norm(w[1:])*dt*0.5)*w[1:]*dt*0.5/jnp.linalg.norm(w[1:])
    qw = jnp.array([jnp.cos(jnp.linalg.norm(w)*dt/2),temp[0],temp[1],temp[2]])
    return quaternion_product(q,qw)
def rpy_intgegration(w, rpy, dt):
    roll, pitch, yaw = rpy
    conj_euler_rates = jnp.array([
        [jnp.float32(1), jnp.float32(0), -jnp.sin(pitch)],
        [jnp.float32(0), jnp.cos(roll), jnp.cos(pitch) * jnp.sin(roll)],
        [jnp.float32(0), -jnp.sin(roll), jnp.cos(pitch) * jnp.cos(roll)]
    ])
    inv_conj_euler_rates = jnp.linalg.inv(conj_euler_rates)
    return rpy + jnp.dot(inv_conj_euler_rates, w) * dt