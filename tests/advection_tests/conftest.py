import pytest
from firedrake import IcosahedralSphereMesh, PeriodicIntervalMesh, \
    ExtrudedMesh, SpatialCoordinate, \
    as_vector, VectorFunctionSpace, sin, exp, Function, FunctionSpace
from gusto import *
from math import pi


@pytest.fixture
def state(tmpdir, geometry):
    """
    returns an instance of the State class, having set up either spherical
    geometry or 2D vertical slice geometry
    """

    output = OutputParameters(dirname=str(tmpdir), dumplist=["f"], dumpfreq=15)

    if geometry == "sphere":
        mesh = IcosahedralSphereMesh(radius=1,
                                     refinement_level=3,
                                     degree=1)
        x = SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
        dt = pi/3. * 0.01
        uexpr = as_vector([-x[1], x[0], 0.0])

    elif geometry == "slice":
        m = PeriodicIntervalMesh(15, 1.)
        mesh = ExtrudedMesh(m, layers=15, layer_height=1./15.)
        dt = 0.01
        x = SpatialCoordinate(mesh)
        uexpr = as_vector([1.0, 0.0])

    else:
        raise ValueError("Specified geometry is not recognised")

    timestepping = TimesteppingParameters(dt=dt)
    state = State(mesh,
                  timestepping=timestepping,
                  output=output)

    return state

@pytest.fixture
def uexpr(geometry, state):
    """
    returns expression for advecting velocity, depending on geometry
    """
    x = SpatialCoordinate(state.mesh)
    if geometry == "sphere":
        uexpr = as_vector([-x[1], x[0], 0.0])

    elif geometry == "slice":
        uexpr = as_vector([1.0, 0.0])

    else:
        raise ValueError("Specified geometry is not recognised")

    return uexpr


@pytest.fixture
def f_init(geometry, state):
    """
    returns an expression for the initial condition
    """
    x = SpatialCoordinate(state.mesh)
    if geometry == "sphere":
        fexpr = exp(-x[2]**2 - x[1]**2)

    elif geometry == "slice":
        fexpr = sin(2*pi*x[0])*sin(2*pi*x[1])

    return fexpr


@pytest.fixture
def f_end(geometry, state):
    """
    returns an expression for the expected final state
    """
    x = SpatialCoordinate(state.mesh)
    if geometry == "sphere":
        fexpr = exp(-x[2]**2 - x[0]**2)

    elif geometry == "slice":
        fexpr = sin(2*pi*(x[0]-0.5))*sin(2*pi*x[1])

    return fexpr


@pytest.fixture
def tmax(geometry):
    return {"slice": 2.5,
            "sphere": pi/2}[geometry]


@pytest.fixture
def error(geometry):
    """
    returns the max expected error (based on past runs)
    """
    return {"slice": 7e-2,
            "sphere": 2.5e-2}[geometry]


def pytest_addoption(parser):
    parser.addoption("--geometry", action="store")
    parser.addoption("--ibp", action="store")
    parser.addoption("--equation_form", action="store")
    parser.addoption("--scheme", action="store")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    geometry = metafunc.config.option.geometry
    if 'geometry' in metafunc.fixturenames and geometry is not None:
        metafunc.parametrize("geometry", [geometry])
    else:
        metafunc.parametrize("geometry", ["sphere", "slice"])
    ibp = metafunc.config.option.ibp
    if 'ibp' in metafunc.fixturenames and ibp is not None:
        metafunc.parametrize("ibp", [ibp])
    else:
        metafunc.parametrize("ibp", [["once", "twice"]])
    equation_form = metafunc.config.option.equation_form
    if 'equation_form' in metafunc.fixturenames and equation_form is not None:
        metafunc.parametrize("equation_form", [equation_form])
    else:
        metafunc.parametrize("equation_form", [["advective", "continuity"]])
    scheme = metafunc.config.option.scheme
    if 'scheme' in metafunc.fixturenames:
        if scheme is not None:
            metafunc.parametrize("scheme", [scheme])
        else:
            metafunc.parametrize("scheme", [["ssprk3", "im"]])
