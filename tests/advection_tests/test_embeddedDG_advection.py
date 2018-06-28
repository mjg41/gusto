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
def test_advection_embedded_dg(geometry, error, state, f_init, tmax, f_end):
    """
    This tests the embedded DG advection scheme for scalar fields
    in slice geometry.
    """
    fspace = state.spaces("HDiv_v")
    f_end = Function(fspace).interpolate(f_end)

    s = "_"
    advected_fields = []

    # setup scalar fields
    scalar_fields = []
    for ibp in ["once", "twice"]:
        for equation_form in ["advective", "continuity"]:
            for broken in [True, False]:
                # create functions and initialise them
                fname = s.join(("f", ibp, equation_form, str(broken)))
                f = state.fields(fname, fspace)
                f.interpolate(f_init)
                scalar_fields.append(fname)
                if broken:
                    eqn = EmbeddedDGAdvection(state, fspace, ibp=ibp, equation_form=equation_form)
                else:
                    eqn = EmbeddedDGAdvection(state, fspace, ibp=ibp, equation_form=equation_form, Vdg=state.spaces("DG"))
                advected_fields.append((fname, SSPRK3(state, f, eqn)))

    end_fields = run(state, advected_fields, tmax)
    check_errors(f_end, error, end_fields, scalar_fields)
