from gusto import *
from gusto import thermodynamics
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, conditional, cos, pi, sqrt, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, TestFunction, dx, TrialFunction, Constant, Function, \
    LinearVariationalProblem, LinearVariationalSolver, DirichletBC, grad, \
    FunctionSpace, BrokenElement, VectorFunctionSpace, as_vector, sin
import sys

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 10.
    deltax = 1000.
else:
    deltax = 200.
    tmax = 4000.


shear = False
grad_pos = True
stochastic = False
pert = True


pert_str = "pert" if pert else "nopert"
stoch_str = "stoc" if stochastic else "det"
shear_str = "shear" if shear else "noshear"
grad_str = "pos" if grad_pos else "neg"

filename = "stoch_bouss_"+str(shear_str)+"_"+str(stoch_str)+"_"+str(grad_str)+"_"+str(pert_str)

L = 10000.
H = 10000.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'p', 'b']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname=filename, dumpfreq=20, dumplist=['u', 'p', 'b'], perturbation_fields=['b'], log_level='INFO')
params = EadyParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [B_Grad(), U_Grad()]

state = State(mesh, vertical_degree=0, horizontal_degree=0,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=params,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# Initial conditions
u0 = state.fields("u")
p0 = state.fields("p")
b0 = state.fields("b")

# spaces
Vu = u0.function_space()
Vp = p0.function_space()
Vb = b0.function_space()
Vpsi = FunctionSpace(mesh, "CG", 2)

# make spaces for recovered scheme
VDG1 = FunctionSpace(mesh, "DG", 1)
VCG1 = FunctionSpace(mesh, "CG", 1)
Vb_broken = FunctionSpace(mesh, BrokenElement(Vb.ufl_element()))
Vu_DG1 = VectorFunctionSpace(mesh, "DG", 1)
Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)
Vu_broken = FunctionSpace(mesh, BrokenElement(Vu.ufl_element()))

u_spaces = (Vu_DG1, Vu_CG1, Vu_broken)
b_spaces = (VDG1, VCG1, Vb_broken)

x, z = SpatialCoordinate(mesh)

# make stochastic velocity field
Xi = state.fields("Xi", Vu)

n_functions = 12
xis = []
for i in range(n_functions):
    xis.append(Function(Vpsi))

# build stochastic basis functions
xis[0].interpolate(cos(2 * pi * x / L) * sin(pi * z / H))
xis[1].interpolate(sin(2 * pi * x / L) * sin(pi * z / H))
xis[2].interpolate(cos(2 * pi * x / L) * sin(2 * pi * z / H))
xis[3].interpolate(sin(2 * pi * x / L) * sin(2 * pi * z / H))
xis[4].interpolate(cos(2 * pi * x / L) * sin(3 * pi * z / H))
xis[5].interpolate(sin(2 * pi * x / L) * sin(3 * pi * z / H))
xis[6].interpolate(cos(4 * pi * x / L) * sin(pi * z / H))
xis[7].interpolate(sin(4 * pi * x / L) * sin(pi * z / H))
xis[8].interpolate(cos(4 * pi * x / L) * sin(2 * pi * z / H))
xis[9].interpolate(sin(4 * pi * x / L) * sin(2 * pi * z / H))
xis[10].interpolate(cos(4 * pi * x / L) * sin(3 * pi * z / H))
xis[11].interpolate(sin(4 * pi * x / L) * sin(3 * pi * z / H))

# set up background buoyancy
Nsq = params.Nsq

if grad_pos:
    bgrad = 1.5
else:
    bgrad = 0.5

# background buoyancy
bref = Constant(Nsq)
btop = Constant(Nsq * bgrad)
b_b = Function(Vb).interpolate(bref + (z / H) * (btop - bref))

# calculate hydrostatic balance
p_b = Function(Vp)
incompressible_hydrostatic_balance(state, b_b, p_b)

# define perturbation
xc = L / 2
zc = 2000.
rc = 2000.
bdash = 2.0
r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
if pert:
    b_pert = Function(Vb).interpolate(0.1 * Nsq * sin(2*pi*x/L) * sin(2*pi*z/H))
else:
    b_pert = Function(Vb).assign(0.0)

# define initial buoyancy
b0.assign(b_b + b_pert)
p0.assign(p_b)

U = 40.

if shear:
    u0.project(as_vector([U * (2 * z / H - 1), 0.0]))

# initialise fields
state.initialise([('u', u0),
                  ('p', p0),
                  ('b', b0)])
state.set_reference_profiles([('p', p_b),
                              ('b', b_b)])

ueqn = EmbeddedDGAdvection(state, Vu, equation_form="advective", recovered_spaces=u_spaces)
beqn = EmbeddedDGAdvection(state, Vb, equation_form="advective", recovered_spaces=b_spaces)

advected_fields = [('u', SSPRK3(state, u0, ueqn, stochastic=stochastic)),
                   ('b', SSPRK3(state, b0, beqn, stochastic=stochastic))]

linear_solver = IncompressibleSolver(state, L)

# Set up forcing
forcing = IncompressibleForcing(state, euler_poincare=False, stochastic=stochastic)


# diffusion
bcs = [DirichletBC(Vu, 0.0, "bottom"),
       DirichletBC(Vu, 0.0, "top")]

diffused_fields = []

if diffusion:
    diffused_fields.append(('u', InteriorPenalty(state, Vu, kappa=Constant(60.),
                                                 mu=Constant(10./deltax), bcs=bcs)))

# define condensation
if stochastic:
    physics_list = [UpdateNoise(state, xis)]
else:
    physics_list = []

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        forcing, physics_list=physics_list,
                        diffused_fields=diffused_fields)

stepper.run(t=0, tmax=tmax)
