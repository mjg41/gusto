from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, conditional, cos, pi, sqrt, \
    TestFunction, dx, TrialFunction, Constant, Function, \
    LinearVariationalProblem, LinearVariationalSolver, DirichletBC, \
    BrokenElement, FunctionSpace, VectorFunctionSpace, \
    NonlinearVariationalProblem, NonlinearVariationalSolver, exp, \
    MixedFunctionSpace, split, TestFunctions
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import sys

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 10.
    deltax = 1000.
else:
    deltax = 30.
    tmax = 600.

L = 3600.
H = 2400.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
diffusion = False
recovered = True
degree = 0 if recovered else 1

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname='grabowski_clark_rain_newsetup', dumpfreq=1, dumplist=['u', 'rho', 'theta'], perturbation_fields=['rho', 'theta', 'water_v'], log_level='INFO')
params = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [RelativeHumidity(), Precipitation(), Temperature(), ExnerPi()]

state = State(mesh, vertical_degree=degree, horizontal_degree=degree,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=params,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")
water_v0 = state.fields("water_v", theta0.function_space())
water_c0 = state.fields("water_c", theta0.function_space())
rain0 = state.fields("rain", theta0.function_space())
moisture = ["water_v", "water_c", "rain"]

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()
x, z = SpatialCoordinate(mesh)
quadrature_degree = (5, 5)
dxp = dx(degree=(quadrature_degree))

if recovered:
    VDG1 = FunctionSpace(mesh, "DG", 1)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
    Vu_DG1 = VectorFunctionSpace(mesh, "DG", 1)
    Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)
    Vu_brok = FunctionSpace(mesh, BrokenElement(Vu.ufl_element()))

    u_spaces = (Vu_DG1, Vu_CG1, Vu_brok)
    rho_spaces = (VDG1, VCG1, Vr)
    theta_spaces = (VDG1, VCG1, Vt_brok)

# Define constant theta_e and water_t
Tsurf = 283.0
psurf = 85000.
humidity = 0.2
Pi_surf = (psurf / state.parameters.p_0) ** state.parameters.kappa
r_v_surf = thermodynamics.r_v(state.parameters, 0.2, Tsurf, psurf)
epsilon = state.parameters.R_d / state.parameters.R_v
theta_surf = thermodynamics.theta(state.parameters, Tsurf, psurf) * (1 + r_v_surf / epsilon)
S = 1.3e-5
Hum = Constant(humidity)
theta0.interpolate(theta_surf * exp(S*z))
RH = Function(Vt).interpolate(Hum)
                              
# Calculate hydrostatic fields
gc_balance(state, theta0, RH, pi_boundary=Constant(Pi_surf))

# make mean fields
theta_b = Function(Vt).assign(theta0)
rho_b = Function(Vr).assign(rho0)
water_vb = Function(Vt).assign(water_v0)
theta_d = Function(Vt).assign(theta0 / (1 + water_v0 / epsilon))

# define perturbation
xc = L / 2
zc = 800.
r1 = 300.
r2 = 200.
r = sqrt((x - xc) ** 2 + (z - zc) ** 2)

H_expr = conditional(r > r1, 0.0,
                     conditional(r > r2,
                                 (1 - Hum) * (cos(pi * (r - r2) / (2 * (r1 - r2)))) ** 2, 1 - Hum))
H_pert = Function(Vt).interpolate(H_expr)

RH = Function(Vt).assign(RH + H_pert)

# find perturbed values
Z = MixedFunctionSpace((Vr, Vt, Vt))
z = Function(Z)

psi, gamma, phi = TestFunctions(Z)
rho, theta_v, w_v = z.split()

rho.assign(rho0)
theta_v.assign(theta0)
w_v.assign(water_v0)

rho, theta_v, w_v = split(z)

Pi = thermodynamics.pi(state.parameters, rho0, theta0)

pi = thermodynamics.pi(state.parameters, rho, theta_v)
p = thermodynamics.p(state.parameters, pi)
T = thermodynamics.T(state.parameters, theta_v, pi, r_v=w_v)
rh = thermodynamics.RH(state.parameters, w_v, T, p)

F = (-psi * Pi * dxp
     + psi * pi * dxp
     - gamma * theta_v * dxp
     + gamma * theta_d * (1 + w_v / epsilon) * dxp
     - phi * RH * dxp
     + phi * rh * dxp)

w_problem = NonlinearVariationalProblem(F, z)
w_solver = NonlinearVariationalSolver(w_problem)
w_solver.solve()

rho, theta_v, w_v = z.split()

theta0.assign(theta_v)
water_v0.assign(w_v)
rho0.assign(rho)
water_c0.assign(0.0)
rain0.assign(0.0)

# initialise fields
state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0),
                  ('water_v', water_v0),
                  ('water_c', water_c0),
                  ('rain', rain0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b),
                              ('water_v', water_vb)])

# Set up advection schemes
if recovered:
    ueqn = EmbeddedDGAdvection(state, Vu, equation_form="advective", recovered_spaces=u_spaces)
    rhoeqn = EmbeddedDGAdvection(state, Vr, equation_form="continuity", recovered_spaces=rho_spaces)
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective", recovered_spaces=theta_spaces)
    limiter = VertexBasedLimiter(VDG1)
else:
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective")
    limiter = ThetaLimiter(thetaeqn)

advected_fields = [('rho', SSPRK3(state, rho0, rhoeqn)),
                   ('theta', SSPRK3(state, theta0, thetaeqn)),
                   ('water_v', SSPRK3(state, water_v0, thetaeqn)),
                   ('water_c', SSPRK3(state, water_c0, thetaeqn)),
                   ('rain', SSPRK3(state, rain0, thetaeqn, limiter=limiter))]
if recovered:
    advected_fields.append(('u', SSPRK3(state, u0, ueqn)))
else:
    advected_fields.append(('u', ThetaMethod(state, u0, ueqn)))

linear_solver = CompressibleSolver(state, moisture=moisture)

# Set up forcing
if recovered:
    compressible_forcing = CompressibleForcing(state, moisture=moisture, euler_poincare=False)
else:
    compressible_forcing = CompressibleForcing(state, moisture=moisture)

# diffusion
bcs = [DirichletBC(Vu, 0.0, "bottom"),
       DirichletBC(Vu, 0.0, "top")]

diffused_fields = []

mu = 5. if recovered else 10.
if diffusion:
    diffused_fields.append(('u', InteriorPenalty(state, Vu, kappa=Constant(60.),
                                                 mu=Constant(10./deltax), bcs=bcs)))

# define condensation
physics_list = [Condensation(state), Fallout(state, moments=0), Coalescence(state)]

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing, physics_list=physics_list,
                        diffused_fields=diffused_fields)

stepper.run(t=0, tmax=tmax)
