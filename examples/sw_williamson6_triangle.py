from gusto import *
from firedrake import cos, sin
import sys

dt = 900.
day = 24.*60.*60.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 14*day

R = 6371220.
H = 8000.

parameters = ShallowWaterParameters(H=H)
domain = SphericalDomain(parameters=parameters,
                         radius=R, refinement_level=4,
                         rotation_option="trad_f")

fieldlist = ['u', 'D']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='sw_rossby_wave_ll', dumpfreq=24, dumplist_latlon=['D'])
diagnostic_fields = [CourantNumber()]

state = State(domain,
              horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# Initial/current conditions
mesh = domain.mesh
u0 = state.fields("u")
D0 = state.fields("D")
omega = 7.848e-6  # note lower-case, not the same as Omega
K = 7.848e-6
g = parameters.g
Omega = parameters.omega_rate

theta, lamda = latlon_coords(mesh)

u_zonal = R*omega*cos(theta) + R*K*(cos(theta)**3)*(4*sin(theta)**2 - cos(theta)**2)*cos(4*lamda)
u_merid = -R*K*4*(cos(theta)**3)*sin(theta)*sin(4*lamda)

uexpr = sphere_to_cartesian(mesh, u_zonal, u_merid)


def Atheta(theta):
    return 0.5*omega*(2*Omega + omega)*cos(theta)**2 + 0.25*(K**2)*(cos(theta)**8)*(5*cos(theta)**2 + 26 - 32/(cos(theta)**2))


def Btheta(theta):
    return (2*(Omega + omega)*K/30)*(cos(theta)**4)*(26 - 25*cos(theta)**2)


def Ctheta(theta):
    return 0.25*(K**2)*(cos(theta)**8)*(5*cos(theta)**2 - 6)


Dexpr = H + (R**2)*(Atheta(theta) + Btheta(theta)*cos(4*lamda) + Ctheta(theta)*cos(8*lamda))/g

u0.project(uexpr, form_compiler_parameters={'quadrature_degree': 8})
D0.interpolate(Dexpr)

state.initialise([('u', u0),
                  ('D', D0)])

ueqn = EulerPoincare(state, u0.function_space())
Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")

advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("D", SSPRK3(state, D0, Deqn)))

linear_solver = ShallowWaterSolver(state)

# Set up forcing
sw_forcing = ShallowWaterForcing(state)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        sw_forcing)

stepper.run(t=0, tmax=tmax)
