from gusto import *
from firedrake import SpatialCoordinate, as_vector
from math import pi
import sys

dt = 3600.
day = 24.*60.*60.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 5*day

R = 6371220.
H = 2000.

domain = SphericalDomain(radius=R, refinement_level=3, degree=3)

fieldlist = ['u', 'D']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='sw_linear_w2', steady_state_error_fields=['u', 'D'])
parameters = ShallowWaterParameters(H=H)

state = State(domain,
              horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              fieldlist=fieldlist)

# Initial/current conditions
x = SpatialCoordinate(domain.mesh)
u0 = state.fields("u")
D0 = state.fields("D")
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Omega = parameters.Omega
Dexpr = - ((R * Omega * u_max)*(x[2]*x[2]/(R*R)))/g
u0.project(uexpr)
D0.interpolate(Dexpr)
state.initialise([('u', u0),
                  ('D', D0)])

Deqn = LinearAdvection(state, D0.function_space(), state.parameters.H, ibp="once", equation_form="continuity")
advected_fields = []
advected_fields.append(("u", NoAdvection(state, u0, None)))
advected_fields.append(("D", ForwardEuler(state, D0, Deqn)))

linear_solver = ShallowWaterSolver(state)

# Set up forcing
sw_forcing = ShallowWaterForcing(state, linear=True)

# build time stepper
stepper = Timestepper(state, advected_fields, linear_solver,
                      sw_forcing)

stepper.run(t=0, tmax=tmax)
