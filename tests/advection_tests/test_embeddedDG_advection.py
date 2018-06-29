from firedrake import FunctionSpace, \
    Function, interval, FiniteElement, TensorProductElement, HDiv
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
def test_advection_embedded_dg(geometry, ibp, equation_form, error,
                               state, uexpr, f_init, tmax, f_end):
    """
    This tests the embedded DG advection scheme for scalar fields
    in slice geometry.
    """
    if geometry == "sphere":
        pytest.skip("unsupported configuration")
    cell = state.mesh._base_mesh.ufl_cell().cellname()
    S1 = FiniteElement("CG", cell, 2)
    S2 = FiniteElement("DG", cell, 1)
    T0 = FiniteElement("CG", interval, 2)
    T1 = FiniteElement("DG", interval, 1)
    V2t_elt = TensorProductElement(S2, T0)
    V2h_elt = HDiv(TensorProductElement(S1, T1))
    V2v_elt = HDiv(V2t_elt)
    V2_elt = V2h_elt + V2v_elt
    V3_elt = TensorProductElement(S2, T1)

    fspace = FunctionSpace(state.mesh, V2t_elt)
    Vdg = FunctionSpace(state.mesh, V3_elt)
    state.spaces("HDiv", state.mesh, V2_elt)

    f_end = Function(fspace).interpolate(f_end)

    s = "_"
    advected_fields = []

    # setup scalar fields
    scalar_fields = []
    for ibp_opt in ibp:
        for eqn_opt in equation_form:
            for broken in [False]:
                # create functions and initialise them
                fname = s.join(("f", ibp_opt, eqn_opt, str(broken)))
                f = state.fields(fname, fspace)
                f.interpolate(f_init)
                scalar_fields.append(fname)
                if broken:
                    eqn = AdvectionEquation(fspace, state,
                                            fname, uexpr=uexpr,
                                            discretisation_option="embedded_DG",
                                            ibp=ibp_opt,
                                            equation_form=eqn_opt)
                else:
                    eqn = AdvectionEquation(fspace, state,
                                            fname, uexpr=uexpr,
                                            discretisation_option="embedded_DG",
                                            ibp=ibp_opt,
                                            equation_form=eqn_opt,
                                            Vdg=Vdg)
                advected_fields.append((fname, SSPRK3(state, f, eqn)))

    end_fields = run(state, advected_fields, tmax)
    check_errors(f_end, error, end_fields, scalar_fields)
