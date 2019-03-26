from os import path
from gusto import *
from firedrake import (as_vector, Constant, PeriodicIntervalMesh,
                       SpatialCoordinate, ExtrudedMesh, FunctionSpace, Function,
                       conditional, sqrt, FiniteElement, TensorProductElement, BrokenElement)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from netCDF4 import Dataset
import pytest

# This setup creates a sharp bubble of warm air in a vertical slice
# This bubble is then advected by a prescribed advection scheme
# If the limiter is working, the advection should have produced
# no new maxima or minima. Advection is a solid body rotation.

def run(setup):

    state = setup.state
    tmax = 10 * setup.tmax
    Ld = setup.Ld
    x, z = SpatialCoordinate(state.mesh)

    Vu = state.spaces("HDiv")
    Vr = state.spaces("DG")
    Vt = state.spaces("HDiv_v")
    VDG1 = FunctionSpace(state.mesh, "DG", 1)
    Vpsi = FunctionSpace(state.mesh, "CG", 2)

    u = state.fields("u", space=Vu, dump=True)
    rho = state.fields("rho", space=Vr, dump=True)
    theta = state.fields("theta", space=Vt, dump=True)

    x_lower = 2 * Ld / 5
    x_upper = 3 * Ld / 5
    z_lower = 6 * Ld / 10
    z_upper = 8 * Ld / 10
    bubble_expr_1 = conditional(x > x_lower,
                               conditional(x < x_upper,
                                           conditional(z > z_lower,
                                                       conditional(z < z_upper, 1.0, 0.0),
                                                       0.0),
                                           0.0),
                               0.0)

    bubble_expr_2 = conditional(x > z_lower,
                               conditional(x < z_upper,
                                           conditional(z > x_lower,
                                                       conditional(z < x_upper, 1.0, 0.0),
                                                       0.0),
                                           0.0),
                               0.0)

    rho.assign(1.0)
    theta.assign(280.)
    rho_pert_1 = Function(Vr).interpolate(bubble_expr_1)
    rho_pert_2 = Function(Vr).interpolate(bubble_expr_2)
    theta_pert_1 = Function(Vt).interpolate(bubble_expr_1)
    theta_pert_2 = Function(Vt).interpolate(bubble_expr_2)

    rho.assign(rho + rho_pert_1 + rho_pert_2)
    theta.assign(theta + theta_pert_1 + theta_pert_2)

    # set up solid body rotation for advection
    xc = Ld / 2
    zc = Ld / 2
    r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
    umax = 0.1
    omega = umax * sqrt(2) / Ld
    r_out = 9 * Ld / 20
    r_in = 2 * Ld / 5
    A = omega * r_in / (2 * (r_in - r_out))
    B = - omega * r_in * r_out / (r_in - r_out)
    C = omega * r_in ** 2 * r_out / (r_in - r_out) / 2
    psi_expr = conditional(r < r_in,
                           omega * r ** 2 / 2,
                           conditional(r < r_out,
                                       A * r ** 2 + B * r + C,
                                       A * r_out ** 2 + B * r_out + C))
    psi = Function(Vpsi).interpolate(psi_expr)

    gradperp = lambda v: as_vector([-v.dx(1), v.dx(0)])
    u.project(gradperp(psi))

    rho_eqn = AdvectionEquation(state, Vr, "rho")
    theta_eqn = AdvectionEquation(state, Vt, "theta")

    rho_opts = EmbeddedDGOptions(embedding_space=Vr)
    theta_opts = EmbeddedDGOptions()


    schemes = [SSPRK3(state, rho_eqn, limiter=VertexBasedLimiter(Vr), options=rho_opts),
               SSPRK3(state, theta_eqn, limiter=ThetaLimiter(Vt), options=theta_opts)]

    timestepper = PrescribedAdvectionTimestepper(state, schemes)
    timestepper.run(t=0, tmax=tmax)

    return


def test_limiters(tmpdir, moist_setup):

    setup = moist_setup(tmpdir, "normal", degree=1)
    run(setup)

    dirname = str(tmpdir)
    filename = path.join(dirname, "diagnostics.nc")
    data = Dataset(filename, "r")

    rho_data = data.groups["rho"]
    max_rho = rho_data.variables["max"]
    min_rho = rho_data.variables["min"]

    theta_data = data.groups["theta"]
    max_theta = theta_data.variables["max"]
    min_theta = theta_data.variables["min"]

    tolerance = 0.01

    # check that maxima and minima do not exceed previous maxima and minima
    # however provide a small amount of tolerance
    assert max_theta[-1] <= max_theta[0] + (max_theta[0] - min_theta[0]) * tolerance
    assert min_theta[-1] >= min_theta[0] - (max_theta[0] - min_theta[0]) * tolerance
    assert max_rho[-1] <= max_rho[0] + (max_rho[0] - min_rho[0]) * tolerance
    assert min_rho[-1] >= min_rho[0] - (max_rho[0] - min_rho[0]) * tolerance
