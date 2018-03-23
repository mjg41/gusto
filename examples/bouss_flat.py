from gusto import *
from gusto import thermodynamics
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    SpatialCoordinate, conditional, cos, pi, sqrt, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, TestFunction, dx, TrialFunction, Constant, Function, \
    LinearVariationalProblem, LinearVariationalSolver, DirichletBC, grad, \
    FunctionSpace, BrokenElement, VectorFunctionSpace, IntervalMesh
import sys
import numpy as np

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 10.
    deltax = 1000.
else:
    deltax = 100.
    tmax = 400.

L = 10000.
H = 10000.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

fieldlist = ['u', 'p', 'b']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
output = OutputParameters(dirname='boussinesq_flat', dumpfreq=5, dumplist=['u', 'p', 'b'], perturbation_fields=[], log_level='INFO')
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

# set up background buoyancy
Nsq = params.Nsq

# background buoyancy
bref = Constant(Nsq)
b_b = Function(Vb).project(bref)

# calculate hydrostatic balance
p_b = Function(Vp)
incompressible_hydrostatic_balance(state, b_b, p_b)

# define perturbation
zt = 8000.
zl0 = 2000.
zl = zl0 + np.random.randn() * 1000.0 * x / L
bdash = 1.0

b_pert = Function(Vb).interpolate(bdash * (1 - 2 * z / H))
b_prime = Function(Vb).interpolate(0.25 * np.random.random() * cos(x * 2 * pi / L))


# define initial buoyancy
b0.assign(b_b + b_pert + b_prime)
p0.assign(p_b)

# initialise fields
state.initialise([('u', u0),
                  ('p', p0),
                  ('b', b0)])
state.set_reference_profiles([('p', p_b),
                              ('b', b_b)])

ueqn = EmbeddedDGAdvection(state, Vu, equation_form="advective", recovered_spaces=u_spaces)
beqn = EmbeddedDGAdvection(state, Vb, equation_form="advective", recovered_spaces=b_spaces)

advected_fields = [('u', SSPRK3(state, u0, ueqn)),
                   ('b', SSPRK3(state, b0, beqn))]

linear_solver = IncompressibleSolver(state, L)

# Set up forcing
forcing = IncompressibleForcing(state, euler_poincare=False)


# diffusion
bcs = [DirichletBC(Vu, 0.0, "bottom"),
       DirichletBC(Vu, 0.0, "top")]

diffused_fields = []

if diffusion:
    diffused_fields.append(('u', InteriorPenalty(state, Vu, kappa=Constant(60.),
                                                 mu=Constant(10./deltax), bcs=bcs)))

# define condensation
physics_list = []

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        forcing, physics_list=physics_list,
                        diffused_fields=diffused_fields)

stepper.run(t=0, tmax=tmax)
