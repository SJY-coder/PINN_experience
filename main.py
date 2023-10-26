import deepxde as dde
import numpy as np
import torch
import math

# physical constants
gamma = 1.4
r_inf = 1.0
p_inf = 1.0
M_inf = 2.0
v_inf = 0.0
u_inf = math.sqrt(gamma * p_inf / r_inf) * M_inf


# Geometry

def euler_equation(coords, state):
    x = coords[0]
    y = coords[1]

    r = state[:, 0:1]
    p = state[:, 1:2]
    u = state[:, 2:3]
    v = state[:, 3:4]

    rE = p / (gamma - 1) + 0.5 * r * (u ** 2 + v ** 2)

    f1 = r * u
    f2 = r * u * u + p
    f3 = r * u * v
    f4 = (rE + p) * u

    g1 = r * v
    g2 = r * v * u
    g3 = r * v * v + p
    g4 = (rE + p) * v

    f1_x = dde.grad.jacobian(f1, coords, i=0, j=0)
    f2_x = dde.grad.jacobian(f2, coords, i=0, j=0)
    f3_x = dde.grad.jacobian(f3, coords, i=0, j=0)
    f4_x = dde.grad.jacobian(f4, coords, i=0, j=0)

    g1_y = dde.grad.jacobian(g1, coords, i=0, j=1)
    g2_y = dde.grad.jacobian(g2, coords, i=0, j=1)
    g3_y = dde.grad.jacobian(g3, coords, i=0, j=1)
    g4_y = dde.grad.jacobian(g4, coords, i=0, j=1)

    r1 = f1_x + g1_y
    r2 = f2_x + g2_y
    r3 = f3_x + g3_y
    r4 = f4_x + g4_y

    return [r1, r2, r3, r4]


rectangle = dde.geometry.Rectangle(xmin=[-0.75, -1.3], xmax=[0.75, 1.3])
triangle = dde.geometry.Triangle([-0.5, 0], [0.75, 0.27], [0.75, -0.27])
spatial_domain = dde.geometry.csg.CSGDifference(rectangle, triangle)


def inlet(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], -0.75)


def outlet(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.75)


def top_wall(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 1.3)


def bottom_wall(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], -1.3)


def non_slip(x, on_boundary):
    return on_boundary and triangle.on_boundary(x)


r_inlet_bc = dde.icbc.DirichletBC(spatial_domain, lambda x: r_inf, inlet, component=0)
p_inlet_bc = dde.icbc.DirichletBC(spatial_domain, lambda x: p_inf, inlet, component=1)
u_inlet_bc = dde.icbc.DirichletBC(spatial_domain, lambda x: u_inf, inlet, component=2)
v_inlet_bc = dde.icbc.DirichletBC(spatial_domain, lambda x: v_inf, inlet, component=3)

u_non_slip_bc = dde.icbc.DirichletBC(triangle, lambda x: 0, non_slip, component=2)
v_non_slip_bc = dde.icbc.DirichletBC(triangle, lambda x: 0, non_slip, component=3)

# p_outlet_bc = dde.icbc.DirichletBC(spatial_domain, lambda x: 0, outlet, component=1)
data = dde.data.PDE(
    spatial_domain,
    euler_equation,
    [
        r_inlet_bc,
        p_inlet_bc,
        u_inlet_bc,
        v_inlet_bc,
        u_non_slip_bc,
        v_non_slip_bc,
    ],
    num_domain=2601,
    num_boundary=400,
    num_test=100000,
)
net = dde.nn.FNN([2] + 4 * [50] + [4], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(iterations=30000)
model.compile("L-BFGS")
losshistory, train_state = model.train()

# X = spatial_domain.random_points(100000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
