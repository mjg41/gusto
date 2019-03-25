from gusto import *
from firedrake import (as_vector, Constant, PeriodicIntervalMesh,
                       SpatialCoordinate, ExtrudedMesh, FunctionSpace,
                       Function, conditional, sqrt, BrokenElement,
                       VectorFunctionSpace, errornorm, cos, pi, norm)

# This tests the recovered space advection scheme
# A bubble of air in the rho, theta and u spaces is advected halfway across the domain
# The test passes if the advection errors are low

def run(setup):

    state = setup.state
    tmax = 10 * setup.tmax
    Ld = setup.Ld
    x, z = SpatialCoordinate(state.mesh)

    Vu = state.spaces("HDiv")
    Vr = state.spaces("DG")
    Vt = state.spaces("HDiv_v")
    VDG0 = FunctionSpace(state.mesh, "DG", 0)
    VDG1 = FunctionSpace(state.mesh, "DG", 1)
    VCG1 = FunctionSpace(state.mesh, "CG", 1)
    VuDG1 = VectorFunctionSpace(state.mesh, "DG", 1)
    VuCG1 = VectorFunctionSpace(state.mesh, "CG", 1)
    Vt_brok = FunctionSpace(state.mesh, BrokenElement(Vt.ufl_element()))
    Vu_brok = FunctionSpace(state.mesh, BrokenElement(Vu.ufl_element()))

    u = state.fields("u", space=Vu, dump=False)
    v = state.fields("v", space=Vu, dump=True)
    rho = state.fields("rho", space=Vr, dump=True)
    theta = state.fields("theta", space=Vt, dump=True)

    rho.assign(1.0)
    theta.assign(280.0)

    # set up initial and final conditions
    xc_i = Ld / 4.
    xc_f = 3. * Ld / 4.
    zc = Ld / 2.
    rc = Ld / 4.
    r_i = sqrt((x - xc_i) ** 2 + (z - zc) ** 2)
    r_f = sqrt((x - xc_f) ** 2 + (z - zc) ** 2)
    expr_i = conditional(r_i > rc, 0.0, cos(pi * r_i / (2 * rc)) ** 2)
    expr_f = conditional(r_f > rc, 0.0, cos(pi * r_f / (2 * rc)) ** 2)

    rho_pert_i = Function(Vr).interpolate(expr_i)
    rho_pert_f = Function(Vr).interpolate(expr_f)
    rho_f = Function(Vr).assign(rho + rho_pert_f)
    rho.assign(rho + rho_pert_i)

    theta_pert_i = Function(Vt).interpolate(expr_i)
    theta_pert_f = Function(Vt).interpolate(expr_f)
    theta_f = Function(Vt).assign(theta + theta_pert_f)
    theta.assign(theta + theta_pert_i)

    v.project(as_vector([expr_i, expr_i]))
    v_f = Function(Vu).project(as_vector([expr_f, expr_f]))

    u.project(as_vector([0.5, 0.0]))

    rho_opts = RecoveredOptions(embedding_space=VDG1,
                                recovered_space=VCG1,
                                broken_space=VDG0,
                                boundary_method="density")
    theta_opts = RecoveredOptions(embedding_space=VDG1,
                                  recovered_space=VCG1,
                                  broken_space=Vt_brok)
    v_opts = RecoveredOptions(embedding_space=VuDG1,
                              recovered_space=VuCG1,
                              broken_space=Vu_brok,
                              boundary_method="velocity")

    rho_eqn = AdvectionEquation(state, Vr, "rho")
    theta_eqn = AdvectionEquation(state, Vt, "theta")
    v_eqn = AdvectionEquation(state, Vu, "v")

    schemes = [SSPRK3(state, rho_eqn, options=rho_opts),
               SSPRK3(state, theta_eqn, options=theta_opts),
               SSPRK3(state, v_eqn, options=v_opts)]

    timestepper = PrescribedAdvectionTimestepper(state, schemes)
    timestepper.run(t=0, tmax=tmax)

    return (errornorm(rho, rho_f) / norm(rho_f),
            errornorm(theta, theta_f) / norm(theta_f),
            errornorm(v, v_f) / norm(v_f))

def test_precipitation(tmpdir, moist_setup):

    setup = moist_setup(tmpdir, "normal", degree=0)
    rho_error, theta_error, v_error = run(setup)
    tolerance = 0.25
    # errors from advection
    assert rho_error < tolerance
    assert theta_error < tolerance
    assert v_error < tolerance
