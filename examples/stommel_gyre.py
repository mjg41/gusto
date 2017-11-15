from gusto import *
from firedrake import SpatialCoordinate, pi, sin, as_vector
import sys

dt = 1.0e3
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 60*60*24*60

L = 1.0e6
W = 1.0e6
H = 1.e3

parameters = ShallowWaterParameters(H=H)
domain = PlaneDomain(parameters=parameters,
                     nx=100, ny=100, L=L, W=W,
                     bc_ids=[1, 2, 3, 4])
fieldlist = ["u", "D"]
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='stommel_gyre', dumpfreq=100)
diagnostic_fields = [CourantNumber()]

state = State(domain,
              horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              diagnostic_fields=diagnostic_fields,
              fieldlist=fieldlist)

u0 = state.fields("u")
D0 = state.fields("D")
state.initialise([("D", H)])

ueqn = VectorInvariant(state, u0.function_space())
Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")
advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("D", SSPRK3(state, D0, Deqn)))

linear_solver = ShallowWaterSolver(state, hybridised=False)

x, y = SpatialCoordinate(domain.mesh)
A = 1.e-7
wind_stress = as_vector([A*sin(pi*(y-0.5*L)/L), 0.])
gamma = 5.e-7
linear_dissipation = -gamma*u0

# Set up forcing
sw_forcing = ShallowWaterForcing(state, euler_poincare=False, extra_terms=wind_stress+linear_dissipation)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        sw_forcing)

stepper.run(t=0, tmax=tmax)
