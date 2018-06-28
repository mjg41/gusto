from firedrake import VectorFunctionSpace, FunctionSpace, as_vector, Function
from gusto import *
import pytest


def run(state, advected_fields, tmax):

    timestepper = Timestepper(state, advected_fields)
    timestepper.run(0, tmax)
    return timestepper.state.fields


def check_errors(ans, error, end_fields, field_names):
    for fname in field_names:
        f = end_fields(fname)
        f -= ans
        assert(abs(f.dat.data.max()) < error)


@pytest.mark.unit
def test_advection_supg(geometry, error, state, f_init, tmax, f_end):
    """
    This tests the embedded DG advection scheme for scalar and vector fields
    in slice geometry.
    """
    s = "_"
    advected_fields = []

    cgspace = FunctionSpace(state.mesh, "CG", 1)
    fspace = state.spaces("HDiv_v")
    vcgspace = VectorFunctionSpace(state.mesh, "CG", 1)
    vspace = state.spaces("HDiv")

    # expression for vector initial and final conditions
    vec_expr = [0.]*state.mesh.geometric_dimension()
    vec_expr[0] = f_init
    vec_expr = as_vector(vec_expr)
    vec_end_expr = [0.]*state.mesh.geometric_dimension()
    vec_end_expr[0] = f_end
    vec_end_expr = as_vector(vec_end_expr)

    cg_end = Function(cgspace).interpolate(f_end)
    hdiv_v_end = Function(fspace).interpolate(f_end)
    vcg_end = Function(vcgspace).interpolate(vec_end_expr)
    hdiv_end = Function(vspace).project(vec_end_expr)

    # setup cg scalar fields
    cg_scalar_fields = []
    for equation_form in ["advective", "continuity"]:
        for time_discretisation in ["ssprk", "im"]:
            # create functions and initialise them
            fname = s.join(("f", equation_form, time_discretisation))
            f = state.fields(fname, cgspace)
            f.interpolate(f_init)
            cg_scalar_fields.append(fname)
            eqn = SUPGAdvection(state, cgspace, equation_form=equation_form)
            if time_discretisation == "ssprk":
                advected_fields.append((fname, SSPRK3(state, f, eqn)))
            elif time_discretisation == "im":
                advected_fields.append((fname, ThetaMethod(state, f, eqn)))

    # setup cg vector fields
    cg_vector_fields = []
    for equation_form in ["advective", "continuity"]:
        for time_discretisation in ["ssprk", "im"]:
            # create functions and initialise them
            fname = s.join(("fvec", equation_form, time_discretisation))
            f = state.fields(fname, vcgspace)
            f.interpolate(vec_expr)
            cg_vector_fields.append(fname)
            eqn = SUPGAdvection(state, vcgspace, equation_form=equation_form)
            if time_discretisation == "ssprk":
                advected_fields.append((fname, SSPRK3(state, f, eqn)))
            elif time_discretisation == "im":
                advected_fields.append((fname, ThetaMethod(state, f, eqn)))

    # setup HDiv_v fields
    hdiv_v_fields = []
    ibp = "twice"
    for equation_form in ["advective", "continuity"]:
        for time_discretisation in ["ssprk", "im"]:
            # create functions and initialise them
            fname = s.join(("f", ibp, equation_form, time_discretisation))
            f = state.fields(fname, fspace)
            f.interpolate(f_init)
            hdiv_v_fields.append(fname)
            eqn = SUPGAdvection(state, fspace, ibp=ibp, equation_form=equation_form, supg_params={"dg_direction": "horizontal"})
            if time_discretisation == "ssprk":
                advected_fields.append((fname, SSPRK3(state, f, eqn)))
            elif time_discretisation == "im":
                advected_fields.append((fname, ThetaMethod(state, f, eqn)))

    # setup HDiv fields
    hdiv_fields = []
    ibp = "twice"
    for equation_form in ["advective", "continuity"]:
        for time_discretisation in ["ssprk", "im"]:
            # create functions and initialise them
            fname = s.join(("fvec", ibp, equation_form, time_discretisation))
            f = state.fields(fname, vspace)
            f.project(vec_expr)
            hdiv_fields.append(fname)
            eqn = SUPGAdvection(state, vspace, ibp=ibp, equation_form=equation_form, supg_params={"dg_direction": "horizontal"})
            if time_discretisation == "ssprk":
                advected_fields.append((fname, SSPRK3(state, f, eqn)))
            elif time_discretisation == "im":
                advected_fields.append((fname, ThetaMethod(state, f, eqn)))

    end_fields = run(state, advected_fields, tmax)
    check_errors(cg_end, error, end_fields, cg_scalar_fields)
    check_errors(hdiv_v_end, error, end_fields, hdiv_v_fields)
    check_errors(vcg_end, error, end_fields, cg_vector_fields)
    check_errors(hdiv_end, error, end_fields, hdiv_fields)
