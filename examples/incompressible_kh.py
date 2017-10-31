from gusto import *
from firedrake import as_vector, DirichletBC, Constant,\
    VectorFunctionSpace, SpatialCoordinate, Function, tanh, pi, cosh, cos
import numpy as np
import sys
import itertools

dt = 0.1
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 3600.

# set up domain
nlayers = 32  # horizontal layers
delta_x = 1/nlayers
kappa = 2.38434
L = 4*2*pi/kappa
columns = int(L/delta_x)  # number of columns
H = 1.0  # Height position of the model top
domain = VerticalSliceDomain(L, H, columns, nlayers)

fieldlist = ['u', 'p', 'b']

timestepping = TimesteppingParameters(dt=dt)

#points_x = np.linspace(0., L, 2)
#points_z = [H/2.]
#points = np.array([p for p in itertools.product(points_x, points_z)])
#print(L)
#print(points)
output = OutputParameters(dirname='incompressible_kh', dumpfreq=10,
                          dumplist=['u', 'b'], perturbation_fields=['b'])
                          #point_data=[('b', points)])
diagnostic_fields = [CourantNumber()]
parameters = CompressibleParameters()

# setup state, passing in the mesh, information on the required finite element
# function spaces, z, k, and the classes above
state = State(domain,
              vertical_degree=1, horizontal_degree=1,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

u0 = state.fields("u")
b0 = state.fields("b")
p0 = state.fields("p")

# spaces
Vu = u0.function_space()
Vb = b0.function_space()

mesh = domain.mesh
x, z = SpatialCoordinate(mesh)
# remember: b = g theta / theta_0 = g rho / rho_0
g = parameters.g
z0 = 0.5
db = g*0.01
du = 0.1808
dz_b = 0.1
dz_u = 0.1
bref = 0.5*db*tanh((z-z0)/dz_b)
uref = 0.5*du*tanh((z-z0)/dz_u)

b_b = Function(Vb).interpolate(bref)

incompressible_hydrostatic_balance(state, b_b, p0)

A = 1.e-8
# b_pert = A*db*cos(kappa*x)/(cosh((z-z0)/dz_b))**2
r = Function(b0.function_space()).assign(Constant(0.0))
r.dat.data[:] += np.random.uniform(low=-1., high=1., size=r.dof_dset.size)
b_pert = r*A/(cosh((z-z0)/dz_b))**2
b0.interpolate(b_b + b_pert)
u0.project(as_vector([uref, 0.]))

# pass these initial conditions to the state.initialise method
state.initialise([('u', u0),
                  ('b', b0)])

# set the background buoyancy
state.set_reference_profiles([('b', b_b)])

ueqn = VectorInvariant(state, Vu)
supg = True
if supg:
    beqn = SUPGAdvection(state, Vb,
                         equation_form="advective")
else:
    beqn = EmbeddedDGAdvection(state, Vb,
                               equation_form="advective")
advected_fields = []
advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
advected_fields.append(("b", SSPRK3(state, b0, beqn)))

# diffusion
diffused_fields = []
diffused_fields.append(('u', InteriorPenalty(state, Vu, kappa=.01,
                                             mu=10./delta_x,
                                             bc_ids=["top", "bottom"])))

linear_solver = IncompressibleSolver(state, L)

forcing = IncompressibleForcing(state, euler_poincare=False)

stepper = CrankNicolson(state, advected_fields, linear_solver,
                        forcing, diffused_fields)

stepper.run(t=0, tmax=tmax)

