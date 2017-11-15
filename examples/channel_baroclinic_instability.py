from gusto import *
from firedrake import SpatialCoordinate, sin, pi, cos, exp, ln, Function, solve, TestFunction, TrialFunction, dx
import sys

dt = 10.
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 15*24*60*60.

# set up domain
delta_z = 1.e3  # 1km vertical grid spacing
nlayers = 30  # horizontal layers
H = 3.e4  # height of domain 30km
delta_x = 1.e5  # 100km horizontal grid spacing
L = 4.e7  # zonal extent of domain 40000km
W = 6.e6  # meridional extent of domain 6000km
nx = int(L/delta_x)
ny = int(W/delta_x)

parameters = CompressibleParameters()
domain = ChannelDomain(parameters=parameters,
                       nx=nx, ny=ny, nlayers=nlayers, L=L, W=W, H=H,
                       rotation_option="beta_plane")

fieldlist = ['u', 'rho', 'theta']

timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(dirname='channel_baroclinic_instability', dumpfreq=1,
                          perturbation_fields=['rho', 'theta'])
diagnostic_fields = [CourantNumber()]

state = State(domain,
              vertical_degree=1, horizontal_degree=1,
              family="RTCF",
              timestepping=timestepping,
              output=output,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
N = parameters.N
p_0 = parameters.p_0
p_s = p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa
f0 = parameters.f0
beta = parameters.beta

x, y, z = SpatialCoordinate(domain.mesh)

Vdg = state.spaces("DG")
eta = Function(Vdg).assign(1.e-7)
u_0 = 35.
T0 = 288
Gamma = 0.005
b = 2.
y0 = W/2.


def Phi_bar(eta):
    return T0*g/Gamma*(1 - eta**(R_d*Gamma/g))


def Phi_prime(eta):
    return 0.5*u_0*((f0 - beta*y0)*(y - 0.5*W*(1 + sin(2*pi*y/W)/pi)) + 0.5*beta*(y**2 - W*(y*sin(2*pi*y/W)/pi + 0.5*W/(pi**2)*cos(2*pi*y/W) - W/3. - 0.5*W/(pi**2))))


def Phi(eta):
    return Phi_bar(eta) + Phi_prime(eta)*ln(eta)*exp(-(ln(eta)/b)**2)


def T_bar(eta):
    return T0*eta**(R_d*Gamma/g)


def T_prime(eta):
    return Phi_prime(eta)/R_d*(2/(b**2)*(ln(eta)**2) - 1)*exp(-(ln(eta)/b)**2)


def T(eta):
    return T_bar(eta) + T_prime(eta)


v = TestFunction(Vdg)
tri = TrialFunction(Vdg)
F = (-g*z + Phi(eta))*v*dx(degree=8)
solve(F == 0, eta)

p = p_0*eta
rho_b = Function(Vr).interpolate(p/(R_d*T(eta)))
theta_b = Function(Vt).interpolate(T(eta)*pow(p_0/p, R_d/c_p))

# Calculate hydrostatic Pi
compressible_hydrostatic_balance(state, theta_b, rho_b, solve_for_rho=True)
theta0.assign(theta_b)
rho0.assign(rho_b)
u = -u_0*sin(pi*y/W)**2*ln(eta)*eta**(-ln(eta)/b**2)
u0.project(as_vector([u, 0.0, 0.0]))

state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up advection schemes
ueqn = VectorInvariant(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
thetaeqn = SUPGAdvection(state, Vt)
advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))

# Set up linear solver
linear_solver = CompressibleSolver(state)

# Set up forcing
compressible_forcing = CompressibleForcing(state, euler_poincare=False)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing)

stepper.run(t=0, tmax=tmax)
