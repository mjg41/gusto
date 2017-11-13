from gusto import *
from firedrake import SpatialCoordinate, Function


def setup_gw(dirname):
    nlayers = 10  # horizontal layers
    columns = 30  # number of columns
    L = 1.e5
    H = 1.0e4  # Height position of the model top

    parameters = CompressibleParameters()
    domain = VerticalSliceDomain(parameters=parameters,
                                 nx=columns, nlayers=nlayers, L=L, H=H)
    dt = 6.0

    fieldlist = ['u', 'p', 'b']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/gw_incompressible", dumplist=['u'])

    state = State(domain,
                  vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  fieldlist=fieldlist)

    # Initial conditions
    u0 = state.fields("u")
    p0 = state.fields("p")
    b0 = state.fields("b")

    # z.grad(bref) = N**2
    x, z = SpatialCoordinate(domain.mesh)
    N = parameters.N
    bref = z*(N**2)

    b_b = Function(b0.function_space()).interpolate(bref)
    b0.interpolate(b_b)
    incompressible_hydrostatic_balance(state, b0, p0)
    state.initialise([('u', u0),
                      ('p', p0),
                      ('b', b0)])

    # Set up forcing
    forcing = IncompressibleForcing(state)

    return state, forcing


def run_gw_incompressible(dirname):

    state, forcing = setup_gw(dirname)
    dt = state.timestepping.dt
    forcing.apply(dt, state.xn, state.xn, state.xn)
    u = state.xn.split()[0]
    w = Function(state.spaces("DG")).interpolate(u[1])
    return w


def test_gw(tmpdir):

    dirname = str(tmpdir)
    w = run_gw_incompressible(dirname)
    assert max(abs(w.dat.data.min()), w.dat.data.max()) < 4e-8
