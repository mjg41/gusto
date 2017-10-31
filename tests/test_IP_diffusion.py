from gusto import *
from firedrake import SpatialCoordinate, VectorFunctionSpace, Constant,\
    exp, as_vector
import pytest


def setup_IPdiffusion(dirname, vector, DG):

    dt = 0.01
    L = 10.
    domain = VerticalSliceDomain(L, 10., 50, 50)

    fieldlist = ['u', 'D']
    timestepping = TimesteppingParameters(dt=dt)
    parameters = CompressibleParameters()
    output = OutputParameters(dirname=dirname)

    state = State(domain,
                  vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  parameters=parameters,
                  output=output,
                  fieldlist=fieldlist)

    mesh = domain.mesh
    x = SpatialCoordinate(mesh)
    if vector:
        kappa = Constant([[0.05, 0.], [0., 0.05]])
        if DG:
            Space = VectorFunctionSpace(mesh, "DG", 1)
        else:
            Space = state.spaces("HDiv")
        fexpr = as_vector([exp(-(L/2.-x[0])**2 - (L/2.-x[1])**2), 0.])
    else:
        kappa = 0.05
        if DG:
            Space = state.spaces("DG")
        else:
            Space = state.spaces("HDiv_v")
        fexpr = exp(-(L/2.-x[0])**2 - (L/2.-x[1])**2)

    f = state.fields("f", Space)
    try:
        f.interpolate(fexpr)
    except NotImplementedError:
        f.project(fexpr)

    mu = 5.
    f_diffusion = InteriorPenalty(state, f, kappa=kappa, mu=mu)
    diffused_fields = [("f", f_diffusion)]
    stepper = AdvectionDiffusion(state, diffused_fields=diffused_fields)
    return stepper


def run(dirname, vector, DG):

    stepper = setup_IPdiffusion(dirname, vector, DG)
    stepper.run(t=0., tmax=2.5)
    return stepper.state.fields("f")


@pytest.mark.parametrize("vector", [True, False])
@pytest.mark.parametrize("DG", [True, False])
def test_ipdiffusion(tmpdir, vector, DG):

    dirname = str(tmpdir)
    f = run(dirname, vector, DG)
    assert f.dat.data.max() < 0.7
