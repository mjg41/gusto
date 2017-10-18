"""
Some simple tools for making model configuration nicer.
"""

from abc import ABCMeta, abstractproperty
from firedrake import sqrt, warning, PeriodicIntervalMesh, \
    PeriodicRectangleMesh, ExtrudedMesh, SpatialCoordinate, \
    IcosahedralSphereMesh, CellNormal, inner, cross, interpolate, \
    Constant, as_vector
from math import fabs


__all__ = ["TimesteppingParameters", "OutputParameters", "CompressibleParameters", "ShallowWaterParameters", "EadyParameters", "CompressibleEadyParameters", "SphericalDomain", "ChannelDomain", "VerticalSliceDomain"]


class Configuration(object):

    def __init__(self, **kwargs):

        for name, value in kwargs.items():
            self.__setattr__(name, value)

    def __setattr__(self, name, value):
        """Cause setting an unknown attribute to be an error"""
        if not hasattr(self, name):
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))
        object.__setattr__(self, name, value)


class TimesteppingParameters(Configuration):

    """
    Timestepping parameters for Gusto
    """
    dt = None
    alpha = 0.5
    maxk = 4
    maxi = 1


class OutputParameters(Configuration):

    """
    Output parameters for Gusto
    """

    Verbose = False
    dumpfreq = 1
    dumplist = None
    dumplist_latlon = []
    dump_diagnostics = True
    checkpoint = False
    dirname = None
    #: Should the output fields be interpolated or projected to
    #: a linear space?  Default is interpolation.
    project_fields = False
    #: List of fields to dump error fields for steady state simulation
    steady_state_error_fields = []
    #: List of fields for computing perturbations
    perturbation_fields = []
    #: List of ordered pairs (name, points) where name is the field
    # name and points is the points at which to dump them
    point_data = []


class CompressibleParameters(Configuration):

    """
    Physical parameters for Compressible Euler
    """
    g = 9.810616
    N = 0.01  # Brunt-Vaisala frequency (1/s)
    cp = 1004.5  # SHC of dry air at const. pressure (J/kg/K)
    R_d = 287.  # Gas constant for dry air (J/kg/K)
    kappa = 2.0/7.0  # R_d/c_p
    p_0 = 1000.0*100.0  # reference pressure (Pa, not hPa)
    cv = 717.  # SHC of dry air at const. volume (J/kg/K)
    c_pl = 4186.  # SHC of liq. wat. at const. pressure (J/kg/K)
    c_pv = 1885.  # SHC of wat. vap. at const. pressure (J/kg/K)
    c_vv = 1424.  # SHC of wat. vap. at const. pressure (J/kg/K)
    R_v = 461.  # gas constant of water vapour
    L_v0 = 2.5e6  # ref. value for latent heat of vap. (J/kg)
    T_0 = 273.15  # ref. temperature
    w_sat1 = 380.3  # first const. in Teten's formula (Pa)
    w_sat2 = -17.27  # second const. in Teten's formula (no units)
    w_sat3 = 35.86  # third const. in Teten's formula (K)
    w_sat4 = 610.9  # fourth const. in Teten's formula (Pa)


class ShallowWaterParameters(Configuration):

    """
    Physical parameters for 3d Compressible Euler
    """
    g = 9.80616
    Omega = 7.292e-5  # rotation rate
    H = None  # mean depth


class EadyParameters(Configuration):

    """
    Physical parameters for Incompressible Eady
    """
    Nsq = 2.5e-05  # squared Brunt-Vaisala frequency (1/s)
    dbdy = -1.0e-07
    H = None
    L = None
    f = 1.e-4
    deltax = None
    deltaz = None
    fourthorder = False


class CompressibleEadyParameters(CompressibleParameters, EadyParameters):

    """
    Physical parameters for Compressible Eady
    """
    g = 10.
    N = sqrt(EadyParameters.Nsq)
    theta_surf = 300.
    dthetady = theta_surf/g*EadyParameters.dbdy
    Pi0 = 0.0


class PhysicalDomain(object, metaclass=ABCMeta):

    def __init__(self, mesh, *, coriolis=None, rotation_vector=None,
                 is_extruded=True, is_3d=True, is_rotating=True,
                 on_sphere=True, boundary_ids=None):

        if not is_3d and not(hasattr(self, "perp")):
            raise ValueError("a perp function must be defined for 2D domains")

        self.mesh = mesh
        self.is_rotating = is_rotating
        if is_rotating:
            if coriolis:
                self.coriolis = coriolis
            elif rotation_vector:
                self.rotation_vector = rotation_vector
            else:
                raise ValueError("If your domain is rotating, you need to specify a rotation vector or a coriolis expression")

        self.is_extruded = is_extruded
        self.is_3d = is_3d
        self.on_sphere = on_sphere
        if boundary_ids is None:
            self.boundary_ids = []
        else:
            self.boundary_ids = boundary_ids

    @abstractproperty
    def vertical_normal(self):
        pass


class SphericalDomain(PhysicalDomain):

    def __init__(self, radius=None, refinement_level=None, degree=None,
                 nlayers=None, mesh=None, *, coriolis=None,
                 rotation_vector=None, is_rotating=True):

        if mesh is None and None in [radius, refinement_level]:
            raise ValueError("You must provide either a mesh or the parameters to enable a mesh to be constructed.")

        if mesh is None:
            if degree is None:
                degree = 3
            mesh = IcosahedralSphereMesh(radius=radius,
                                         refinement_level=refinement_level,
                                         degree=degree)
            x = SpatialCoordinate(mesh)
            mesh.init_cell_orientations(x)

        if nlayers is None:
            is_extruded = False
            is_3d = False
            outward_normals = CellNormal(mesh)
            self.perp = lambda u: cross(outward_normals, u)
        else:
            is_extruded = True
            is_3d = True

        if is_rotating:
            if is_3d:
                if rotation_vector is None:
                    rotation_vector = as_vector((0., 0., 0.5e-4))
                if coriolis is not None:
                    raise ValueError("Cannot specify coriolis parameter for a 3d domian")
            else:
                if coriolis is None:
                    x = SpatialCoordinate(mesh)
                    if radius is None:
                        radius = sqrt(inner(x, x))
                        if fabs(radius-6371220.) > 1.:
                            warning("default coriolis parameters are specified for Earth-sized sphere, which yours does not seem to be")
                    Omega = 7.292e-5  # rotation rate
                    coriolis = 2*Omega*x[2]/radius
                if rotation_vector is not None:
                    raise ValueError("Cannot specify rotation vector for a 3d domian")

        super().__init__(mesh, coriolis=coriolis,
                         rotation_vector=rotation_vector,
                         is_extruded=is_extruded,
                         is_3d=is_3d, is_rotating=is_rotating)

    @property
    def vertical_normal(self):
        x = SpatialCoordinate(self.mesh)
        R = sqrt(inner(x, x))
        k = interpolate(x/R, self.mesh.coordinates.function_space())
        return k


class ChannelDomain(PhysicalDomain):

    def __init__(self, L=None, W=None, H=None, columns=None, nlayers=None,
                 mesh=None, *, coriolis=None, rotation_vector=None,
                 is_3d=True, is_rotating=True):

        if mesh is None and None in [L, W, H, columns, nlayers]:
            raise ValueError("You must provide either a mesh or the parameters to enable a mesh to be constructed.")

        if mesh is None:
            nx, ny = columns
            m = PeriodicRectangleMesh(nx, ny, L, W, quadrilateral=True)
            mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

        if is_rotating and rotation_vector is None:
            rotation_vector = as_vector((0., 0., 0.5e-4))

        super().__init__(mesh,
                         coriolis=coriolis,
                         rotation_vector=rotation_vector,
                         is_3d=is_3d,
                         is_rotating=is_rotating,
                         on_sphere=False, boundary_ids=["top", "bottom"])

    @property
    def vertical_normal(self):
        dim = self.mesh.topological_dimension()
        kvec = [0.0]*dim
        kvec[dim-1] = 1.0
        return Constant(kvec)


class VerticalSliceDomain(ChannelDomain):

    def __init__(self, L=None, H=None, columns=None, nlayers=None, mesh=None,
                 *, coriolis=None, rotation_vector=None,
                 is_3d=False, is_rotating=False):

        if mesh is None and None in [L, H, columns, nlayers]:
            raise ValueError("You must provide either a mesh or the parameters to enable a mesh to be constructed.")

        if all([coriolis, rotation_vector]):
            raise ValueError("You cannot specify both coriolis parameter and rotation_vector")

        is_rotating = any([coriolis, rotation_vector, is_rotating])
        if is_rotating:
            if not is_3d:
                warning("creating a 3d, single element thick mesh as you have a rotating domain.")
                is_3d = True

        if not is_3d:
            self.perp = lambda u: as_vector([-u[1], u[0]])

        if mesh is None:
            if is_3d:
                super().__init__(L=L, W=1.e4, H=H, columns=(columns, 1),
                                 nlayers=nlayers,
                                 coriolis=coriolis,
                                 rotation_vector=rotation_vector,
                                 is_3d=is_3d, is_rotating=is_rotating)
            else:
                m = PeriodicIntervalMesh(columns, L)
                mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
                super().__init__(mesh=mesh,
                                 coriolis=coriolis,
                                 rotation_vector=rotation_vector,
                                 is_3d=is_3d, is_rotating=is_rotating)
        else:
            super().__init__(mesh=mesh,
                             coriolis=coriolis,
                             rotation_vector=rotation_vector,
                             is_3d=is_3d, is_rotating=is_rotating)
