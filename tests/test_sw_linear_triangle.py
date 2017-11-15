from os import path
from gusto import *
from firedrake import SpatialCoordinate, as_vector
from math import pi
from netCDF4 import Dataset


def setup_sw(dirname):

    R = 6371220.
    H = 2000.
    day = 24.*60.*60.

    parameters = ShallowWaterParameters(H=H)
    domain = SphericalDomain(parameters=parameters,
                             radius=R, refinement_level=3,
                             rotation_option="trad_f")

    fieldlist = ['u', 'D']
    timestepping = TimesteppingParameters(dt=3600.)
    output = OutputParameters(dirname=dirname+"/sw_linear_w2", steady_state_error_fields=['u', 'D'], dumpfreq=12)

    state = State(domain,
                  horizontal_degree=1,
                  family="BDM",
                  timestepping=timestepping,
                  output=output,
                  fieldlist=fieldlist)

    # interpolate initial conditions
    # Initial/current conditions
    u0 = state.fields("u")
    D0 = state.fields("D")
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    x = SpatialCoordinate(domain.mesh)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = parameters.g
    Omega = parameters.omega_rate
    Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
    u0.project(uexpr)
    D0.interpolate(Dexpr)
    state.initialise([('u', u0),
                      ('D', D0)])

    Deqn = LinearAdvection(state, D0.function_space(), parameters.H, ibp="once", equation_form="continuity")
    advected_fields = []
    advected_fields.append(("u", NoAdvection(state, u0, None)))
    advected_fields.append(("D", ForwardEuler(state, D0, Deqn)))

    linear_solver = ShallowWaterSolver(state)

    # Set up forcing
    sw_forcing = ShallowWaterForcing(state, linear=True)

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            sw_forcing)

    return stepper, 2*day


def run_sw(dirname):

    stepper, tmax = setup_sw(dirname)
    stepper.run(t=0, tmax=tmax)


def test_sw_linear(tmpdir):
    dirname = str(tmpdir)
    run_sw(dirname)
    filename = path.join(dirname, "sw_linear_w2/diagnostics.nc")
    data = Dataset(filename, "r")

    Derr = data.groups["D_error"]
    D = data.groups["D"]
    Dl2 = Derr["l2"][-1]/D["l2"][0]
    uerr = data.groups["u_error"]
    u = data.groups["u"]
    ul2 = uerr["l2"][-1]/u["l2"][0]

    assert Dl2 < 4.e-3
    assert ul2 < 6.e-2
