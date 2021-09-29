from math import pi
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, cos, sin, acos, conditional,
                       Constant, exp, Function, File)

# parameters
R = 6371220.
day = 24.*60.*60.

# set up mesh
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=5, degree=3)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)

# lat lon co-ordinates
theta, lamda = latlon_coords(mesh)

timestepping = TimesteppingParameters(dt=720.)
dirname = 'williamson1_moisture'
fieldlist = ['u', 'D']

output = OutputParameters(dirname=dirname, dumpfreq=40)

diagnostics = Diagnostics("u", "m1", "m2")

state = State(mesh, horizontal_degree=1,
              family="BDM",
              timestepping=timestepping,
              output=output,
              fieldlist=fieldlist,
              diagnostics=diagnostics)

# interpolate initial conditions
u0 = state.fields("u")
D0 = state.fields("D")
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
alpha = pi/2
h_max = 0.032
lamda_c = 3*pi/2
theta_c = 0
a = R * 3
r = a * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda - lamda_c))

# uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
def uexpr():
    u_zonal = u_max*(cos(theta)*cos(alpha) + sin(theta)*cos(lamda)*sin(alpha))
    u_merid = -u_max*sin(lamda)*sin(alpha)
    return sphere_to_cartesian(mesh, u_zonal, u_merid)


m1expr = (h_max/2)*(1 + cos(pi*r/R))

u0.project(uexpr())

# set up advected variable in the same space as the height field
VD = D0.function_space()
m1 = state.fields("m1", space=VD)
m2 = state.fields("m2", space=VD)

# set up constants and temperature and saturation profile
Gamma = Constant(-37.5604) # lapse rate - this could be lat-dependent
T0 = Constant(300)   # temperature at equator
T = Gamma * abs(theta) + T0   # temperature profile
e1 = Constant(0.98)  # level of saturation
ms = 3.8e-3 * exp((18 * T - 4824)/(T - 30))   # saturation profile
msout = Function(VD).interpolate(ms)
outfile = File('ms.pvd')
outfile.write(msout)

# set up a temperature field for viewing the temperature profile
temp_field = state.fields("temp", space=VD)
temp_field.interpolate(T)

# set up a field to visualise latitude
lat_field = state.fields("lat", space=VD)
lat_field.interpolate(theta)

mtot = state.fields("mtot", space=VD)

# initialise m1 as the height field in W1
m1.interpolate(conditional(r < R, m1expr, 0))
# initialise m1 as in Zerroukat and Allen 2020
# m1.interpolate(conditional(1 > theta/e2, (1 - theta/e2) * e1 * ms, 0))
mtot.interpolate(m1 + m2)


m1eqn = AdvectionEquation(state, VD, equation_form="advective")
m2eqn = AdvectionEquation(state, VD, equation_form="advective")

advected_fields = []
advected_fields.append(("m1", SSPRK3(state, m1, m1eqn)))
advected_fields.append(("m2", SSPRK3(state, m2, m1eqn)))


class Moisture(Physics):
    def __init__(self, state, ms):
        super().__init__(state)
        V = state.fields("m1").function_space()
        self.dm1 = Function(V)
        self.dm2 = Function(V)
        self.ms = ms

    def apply(self):
        ms = self.ms
        m1 = state.fields("m1")
        m2 = state.fields("m2")
        gamma1 = Constant(1.0)
        dt = state.timestepping.dt
        self.dm1.interpolate(conditional(m1 - ms > 0, gamma1 * (m1 - ms), 0))
        self.dm2.interpolate(
            conditional(ms - m1 > 0,
                        conditional(ms - m1 < m2, gamma1 * (ms - m1), m2),
                        0))
        m1 += self.dm2 - self.dm1
        m2 += self.dm1 - self.dm2
        mtot.interpolate(m1 + m2)
        #m1.interpolate(conditional(m1 > 0, m1, 0))
        #m2.interpolate(conditional(m2 > 0, m2, 0))


moisture = Moisture(state, ms)

# build time stepper
timestepper = AdvectionDiffusion(state, advected_fields,
                                 physics_list=[moisture])
timestepper.run(t=0, tmax=12*day)
