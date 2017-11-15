from gusto import *
from firedrake import CubedSphereMesh, ExtrudedMesh, Expression, \
    VectorFunctionSpace, SpatialCoordinate, as_vector, FunctionSpace, Function
from firedrake import exp, acos, cos, sin
import numpy as np
import sys

dt = 10.
if '--running-tests' in sys.argv:
    tmax = dt
    nlayers = 4  # Number of horizontal layers (was 20)
    refinements = 2  # number of horizontal cells = 20*(4^refinements) (was 5)
else:
    tmax = 100.
    nlayers = 20  # Number of horizontal layers (was 20)
    refinements = 4  # number of horizontal cells = 20*(4^refinements) (was 5)

# build surface mesh
a_ref = 6.37122e6
X = 125.0  # Reduced-size Earth reduction factor
a = a_ref/X

m = CubedSphereMesh(radius=a,
                    refinement_level=refinements,
                    degree=2)

# build volume mesh
z_top = 1.0e4  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=z_top/nlayers,
                    extrusion_type="radial")

parameters = CompressibleParameters()
domain = SphericalDomain(mesh=mesh, parameters=parameters)

# Space for initialising velocity (using this ensures things are in layers)
W_VectorCG1 = VectorFunctionSpace(mesh, "CG", 1)
W_CG1 = FunctionSpace(mesh, "CG", 1)

# Create polar coordinates
z = Function(W_CG1).interpolate(Expression("sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - a", a=a))  # Since we use a CG1 field, this is constant on layers
lat = Function(W_CG1).interpolate(Expression("asin(x[2]/sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))"))
lon = Function(W_CG1).interpolate(Expression("atan2(x[1], x[0])"))

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt)
output = OutputParameters(Verbose=True, dumpfreq=1, dirname='meanflow_ref',
                          perturbation_fields=['theta', 'rho'])

state = State(domain,
              vertical_degree=1, horizontal_degree=1,
              family="RTCF",
              timestepping=timestepping,
              output=output,
              fieldlist=fieldlist)

# Initial conditions
u0 = state.fields.u
theta0 = state.fields.theta
rho0 = state.fields.rho

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# Initial conditions with u0
# u = u0 * cos(lat)
x = SpatialCoordinate(mesh)
u_max = 20.
g = parameters.g
N = parameters.N  # Brunt-Vaisala frequency (1/s)
p_0 = parameters.p_0  # Reference pressure (Pa, not hPa)
c_p = parameters.cp  # SHC of dry air at constant pressure (J/kg/K)
R_d = parameters.R_d  # Gas constant for dry air (J/kg/K)
kappa = parameters.kappa  # R_d/c_p
T_eq = 300.0  # Isothermal atmospheric temperature (K)
p_eq = 1000.0 * 100.0  # Reference surface pressure at the equator
u_0 = 20.0  # Maximum amplitude of the zonal wind (m/s)

d = 5000.0  # Width parameter for Theta'
lamda_c = 2.0*np.pi/3.0  # Longitudinal centerpoint of Theta'
phi_c = 0.0  # Latitudinal centerpoint of Theta' (equator)
deltaTheta = 1.0  # Maximum amplitude of Theta' (K)
L_z = 20000.0  # Vertical wave length of the Theta' perturbation
uexpr = as_vector([-u_max*x[1]/a, u_max*x[0]/a, 0.0])
u0.project(uexpr)
# Surface temperature
G = g**2/(N**2*c_p)
# TS = bigG + (Teq-bigG)*exp( -(u0*N2/(4.d0*g*g))*(u0+2.d0*om*as)*(cos(2.d0*lat)-1.d0)    )
Ts = Function(W_CG1).interpolate(G + (T_eq-G)*exp(-(u_max*N**2/(4*g*g))*u_max*(cos(2.0*lat)-1.0)))

# surface pressure
# ps = peq*exp( (u0/(4.0*bigG*Rd))*(u0+2.0*Om*as)*(cos(2.0*lat)-1.0) ) * (TS/Teq)**(cp/Rd)
psexp = p_eq*exp((u_max/(4.0*G*R_d))*u_max*(cos(2.0*lat)-1.0))*(Ts/T_eq)**(1.0/kappa)

ps = Function(W_CG1).interpolate(psexp)

# Background pressure
pexp = ps*(1 + G/Ts*(exp(-N**2*z/g)-1))**(1.0/kappa)

p = Function(W_CG1).interpolate(pexp)

# Background temperature
# Tbexp = Ts*(p/ps)**kappa/(Ts/G*((p/ps)**kappa - 1) + 1)
Tbexp = G*(1 - exp(N**2*z/g)) + Ts*exp(N**2*z/g)

Tb = Function(W_CG1).interpolate(Tbexp)

# Background potential temperature
thetabexp = Tb*(p_0/p)**kappa

thetab = Function(W_CG1).interpolate(thetabexp)

theta_b = Function(theta0.function_space()).interpolate(thetab)
rho_b = Function(rho0.function_space())

sin_tmp = sin(lat) * sin(phi_c)
cos_tmp = cos(lat) * cos(phi_c)

r = a*acos(sin_tmp + cos_tmp*cos(lon-lamda_c))

s = (d**2)/(d**2 + r**2)

theta_pert = deltaTheta*s*sin(2*np.pi*z/L_z)

theta0.interpolate(theta_b)
# Compute the balanced density
compressible_hydrostatic_balance(state, theta_b, rho_b, top=False,
                                 pi_boundary=(p/p_0)**kappa)
theta0.interpolate(theta_pert)
theta0 += theta_b
rho0.assign(rho_b)

state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up advection schemes
ueqn = EulerPoincare(state, Vu)
rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
thetaeqn = EmbeddedDGAdvection(state, Vt,
                               equation_form="advective")
advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))

# Set up linear solver
schur_params = {'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'schur',
                'ksp_type': 'gmres',
                'ksp_monitor_true_residual': True,
                'ksp_max_it': 100,
                'ksp_gmres_restart': 50,
                'pc_fieldsplit_schur_fact_type': 'FULL',
                'pc_fieldsplit_schur_precondition': 'selfp',
                'fieldsplit_0_ksp_type': 'preonly',
                'fieldsplit_0_pc_type': 'bjacobi',
                'fieldsplit_0_sub_pc_type': 'ilu',
                'fieldsplit_1_ksp_type': 'preonly',
                "fieldsplit_1_ksp_monitor_true_residual": True,
                'fieldsplit_1_pc_type': 'gamg',
                'fieldsplit_1_pc_gamg_sym_graph': True,
                'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
                'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
                'fieldsplit_1_mg_levels_ksp_max_it': 5,
                'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}
linear_solver = CompressibleSolver(state)

# Set up forcing
compressible_forcing = CompressibleForcing(state)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing)

stepper.run(t=0, tmax=tmax)  # tmax was 3600.
