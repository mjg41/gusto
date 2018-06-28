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
def test_advection_dg(geometry, ibp, equation_form, scheme, error, state, uexpr,
                      f_init, tmax, f_end):
    """
    This tests the DG advection discretisation for both scalar and vector
    fields in 2D slice and spherical geometry.
    """

    # set up function spaces
    fspace = FunctionSpace(state.mesh, "DG", 1)
    vspace = VectorFunctionSpace(state.mesh, "DG", 1)
    cell = state.mesh.ufl_cell().cellname()
    V1_elt = FiniteElement("BDM", cell, 2)
    u_space = FunctionSpace(state.mesh, V1_elt)

    # expression for vector initial and final conditions
    vec_expr = [0.]*state.mesh.geometric_dimension()
    vec_expr[0] = f_init
    vec_expr = as_vector(vec_expr)
    vec_end_expr = [0.]*state.mesh.geometric_dimension()
    vec_end_expr[0] = f_end
    vec_end_expr = as_vector(vec_end_expr)

    # functions containing expected values at tmax
    f_end = Function(fspace).interpolate(f_end)
    vec_end = Function(vspace).interpolate(vec_end_expr)

    s = "_"
    advected_fields = []

    # setup fields
    scalar_fields = []
    vector_fields = []

    # create functions and initialise them
    for ibp_opt in ibp:
        for eqn_opt in equation_form:
            for scheme_opt in scheme:
                fname = s.join(("f", ibp_opt, eqn_opt, scheme_opt))
                scalar_fields.append(fname)
                eqn = AdvectionEquation(fspace, state, fname, u_space, uexpr,
                                        ibp=ibp_opt, equation_form=eqn_opt)
                f = state.fields(fname)
                f.interpolate(f_init)

                vname = s.join(("vecf", ibp_opt, eqn_opt, scheme_opt))
                vector_fields.append(vname)
                veqn = AdvectionEquation(vspace, state, vname, u_space, uexpr,
                                         ibp=ibp_opt, equation_form=eqn_opt)
                v = state.fields(vname)
                v.interpolate(vec_expr)

                if scheme_opt == "ssprk3":
                    advected_fields.append((fname, SSPRK3(state, f, eqn)))
                    advected_fields.append((vname, SSPRK3(state, v, veqn)))
                elif scheme_opt == "im":
                    advected_fields.append((fname, ThetaMethod(state, f, eqn)))
                    advected_fields.append((vname, ThetaMethod(state, v, veqn)))

    end_fields = run(state, advected_fields, tmax)

    check_errors(f_end, error, end_fields, scalar_fields)
    check_errors(vec_end, error, end_fields, vector_fields)
