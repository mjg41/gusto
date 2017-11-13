"""
Some simple tools for making model configuration nicer.
"""

from firedrake import sqrt, warning, RectangleMesh, PeriodicIntervalMesh, \
    PeriodicRectangleMesh, ExtrudedMesh, SpatialCoordinate, \
    IcosahedralSphereMesh, CellNormal, inner, cross, interpolate, \
    Constant, as_vector, CubedSphereMesh, IntervalMesh
from firedrake.mesh import ExtrudedMeshTopology


__all__ = ["TimesteppingParameters", "OutputParameters", "CompressibleParameters", "ShallowWaterParameters", "EadyParameters", "CompressibleEadyParameters", "PlaneDomain", "SphericalDomain", "ChannelDomain", "VerticalSliceDomain"]


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


class RotationParameters(Configuration):
    """
    Class containing permissable rotation parameters, along with
    commonly used, Earth-relevant values for each
    """
    Omega = as_vector((0., 0., 0.5e-4))  # rotation vector
    omega_rate = 7.292e-5  # rotation rate
    f0 = 1.e-4  # coriolis parameter
    beta = 2.e-11  # rate of change of coriolis parameter with latitude


class CompressibleParameters(RotationParameters):

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


class ShallowWaterParameters(RotationParameters):

    """
    Physical parameters for 3d Compressible Euler
    """
    g = 9.80616
    H = None  # mean depth


class EadyParameters(RotationParameters):

    """
    Physical parameters for Incompressible Eady
    """
    Nsq = 2.5e-05  # squared Brunt-Vaisala frequency (1/s)
    dbdy = -1.0e-07
    H = None
    L = None
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


class PhysicalDomain(object):
    """Base class for defining a physical domain.
    Default parameters assume a 3D, rotating, spherical domain.
    """
    def __init__(self, mesh, *, parameters=None,
                 rotation_option=None,
                 is_extruded=True, is_3d=True,
                 on_sphere=True, bc_ids=None):

        if not is_3d and not(hasattr(self, "perp")):
            raise ValueError("a perp function must be defined for 2D domains")

        # store mesh and physical parameters class
        self.mesh = mesh
        self.parameters = parameters

        # check rotation option and set either Omega (the rotation vector) or
        # the coriolis parameter as required
        # the necessary constants are specified in the parameters class
        if rotation_option == "Omega":
            if not is_3d:
                raise ValueError("Cannot specify the rotation vector for a 2D domain")
            # save Omega as the rotation vector
            self.rotation_vector = parameters.Omega

        elif rotation_option == "f_plane":
            # set the coriolis parameter to be constant f
            self.coriolis = parameters.f0

        elif rotation_option == "beta_plane":
            # set the coriolis parameter to be f + beta*y
            x = SpatialCoordinate(mesh)
            f0 = parameters.f0
            beta = parameters.beta
            self.coriolis = f0 + beta*x[1]

        elif rotation_option == "trad_f":
            # set the coriolis parameter to be 2*Omega*sin(latitude)
            # where Omega is the rotation rate
            if not on_sphere:
                raise NotImplementedError("rotation_option %s not implemented non spherical domains" % rotation_option)
            omega_rate = parameters.omega_rate
            x = SpatialCoordinate(mesh)
            radius = sqrt(inner(x, x))
            self.coriolis = 2*omega_rate*x[2]/radius
        else:
            if rotation_option is not None:
                rotation_options = ["Omega", "f_plane", "beta_plane", "trad_f"]
                raise ValueError("rotation_option %s unrecognised. rotation_option must be one of: %s" % (rotation_option, [opt for opt in rotation_options]))

        # some booleans to mark whether the domain is extruded, 3D
        # and/or spherical
        self.is_extruded = is_extruded
        self.is_3d = is_3d
        self.on_sphere = on_sphere

        # store list of bc_ids for applying no normal flow boundary conditions
        if bc_ids is None:
            self.bc_ids = []
        else:
            self.bc_ids = bc_ids


class PlaneDomain(PhysicalDomain):

    """Class defining a 2D x-y plane domain.

    The user must either pass in a mesh, or the parameters required to
    construct one. The default behaviour is to construct a doubly
    periodic mesh and a nonrotating domain.

    :arg mesh: The mesh, if the user prefers to construct their own.
    :arg parameters: The :class:`Configuration` containing the physical
    parameters
    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the x direction
    :arg L: The length of the mesh in the x direction
    :arg W: The width of the mesh in the y direction
    :arg periodic_direction: May take the value ``"x"``, ``"y"`` or ``"both"``.
    Defaults to ``"both"``
    :arg rotation option: May take the value ``"Omega"``, ``"f_plane"``,
    ``"beta_plane"``, ``"trad_f"`` or ``"None"``. Defaults to ``"None"``,
    i.e. a nonrotating domain.
    :arg bc_ids: A list containing the boundary ids for those boundaries on
    which you wish to apply a no normal flow boundary condition.
    """
    def __init__(self, mesh=None, *, parameters=None,
                 nx=None, ny=None, L=None, W=None,
                 periodic_direction="both",
                 rotation_option=None,
                 bc_ids=None):

        if mesh is None and None in [L, W, nx, ny]:
            raise ValueError("You must provide either a mesh or the parameters to enable a mesh to be constructed.")

        # if the mesh has not been provided then we need to create one
        if mesh is None:
            if periodic_direction is not None:
                mesh = PeriodicRectangleMesh(nx, ny, L, W,
                                             direction=periodic_direction)
            else:
                mesh = RectangleMesh(nx, ny, L, W)

        # define a perp function for this domain
        self.perp = lambda u: as_vector([-u[1], u[0]])

        super().__init__(mesh, parameters=parameters,
                         rotation_option=rotation_option,
                         is_extruded=False, is_3d=False,
                         on_sphere=False,
                         bc_ids=bc_ids)


class SphericalDomain(PhysicalDomain):
    """Class defining a spherical domain.

    The user must either pass in a mesh, or the parameters required to
    construct one. The default behaviour is to construct a doubly
    periodic mesh and a nonrotating domain.

    :arg mesh: The mesh, if the user prefers to construct their own.
    :arg parameters: The :class:`Configuration` containing the physical
    parameters
    :arg radius: The radius of the sphere
    :arg refinement_level: The number of refinements
    :arg degree: (optional) polynomial degree of the coordinate space.
    Defaults to 3 if not provided.
    :arg ztop: The height position of the model top
    :arg nlayers: The number of layers in the vertical direction. Defaults
    to ``"None"``to give a 2D mesh.
    :arg rotation option: May take the value ``"Omega"``, ``"f_plane"``,
    ``"beta_plane"``, ``"trad_f"`` or ``"None"``. Defaults to ``"None"``,
    i.e. a nonrotating domain.
    """

    def __init__(self, mesh=None, *, parameters=None,
                 radius=None, refinement_level=None, degree=None,
                 z_top=None, nlayers=None,
                 rotation_option=None):

        if mesh is None and None in [radius, refinement_level]:
            raise ValueError("You must provide either a mesh or the parameters to enable a mesh to be constructed.")

        # if the mesh has not been provided then we need to create one
        if mesh is None:
            if nlayers is None:
                # if nlayers is not specified, construct a 2D icosahedral
                # mesh and define a perp function
                if degree is None:
                    degree = 3
                mesh = IcosahedralSphereMesh(radius=radius,
                                             refinement_level=refinement_level,
                                             degree=degree)
                x = SpatialCoordinate(mesh)
                mesh.init_cell_orientations(x)
                is_extruded = False
                is_3d = False
                outward_normals = CellNormal(mesh)
                self.perp = lambda u: cross(outward_normals, u)
            else:
                # if nlayers is specified, construct a 2D cubed sphere mesh
                # and extrude it
                if degree is None:
                    degree = 2
                m = CubedSphereMesh(radius=radius,
                                    refinement_level=refinement_level,
                                    degree=degree)
                mesh = ExtrudedMesh(m, layers=nlayers,
                                    layer_height=z_top/nlayers,
                                    extrusion_type="radial")
                is_extruded = True
                is_3d = True
        else:
            # check if mesh is extruded
            is_extruded = (isinstance(mesh.topology, ExtrudedMeshTopology))
            if not is_extruded:
                warning("some things might not work")
            # check if mesh is 3d
            is_3d = (mesh.geometric_dimension() == 3)

        super().__init__(mesh, parameters=parameters,
                         rotation_option=rotation_option,
                         is_extruded=is_extruded,
                         is_3d=is_3d)

    @property
    def vertical_normal(self):
        x = SpatialCoordinate(self.mesh)
        R = sqrt(inner(x, x))
        k = interpolate(x/R, self.mesh.coordinates.function_space())
        return k


class ChannelDomain(PhysicalDomain):
    """Class defining a 3D channel domain.

    The user must either pass in a mesh, or the parameters required to
    construct one. The default behaviour is to construct a doubly
    periodic mesh and a nonrotating domain.

    :arg mesh: The mesh, if the user prefers to construct their own.
    :arg parameters: The :class:`Configuration` containing the physical
    parameters
    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nlayers: The number of layers
    :arg L: The length of the mesh in the x direction
    :arg W: The width of the mesh in the y direction
    :arg H: The height of the domain
    :arg periodic_direction: May take the value ``"x"``, ``"y"`` or ``"both"``.
    Defaults to ``"both"``
    :arg rotation option: May take the value ``"Omega"``, ``"f_plane"``,
    ``"beta_plane"``, ``"trad_f"`` or ``"None"``. Defaults to ``"None"``,
    i.e. a nonrotating domain.
    :arg is_3d; boolean, True if domain is 3d, False in the special case
    that we have a 2d vertical slice.
    :arg bc_ids: A list containing the boundary ids for those boundaries on
    which you wish to apply a no normal flow boundary condition.
    """

    def __init__(self, mesh=None, *, parameters=None,
                 nx=None, ny=None, nlayers=None, L=None, W=None, H=None,
                 periodic_direction="both",
                 rotation_option=None,
                 is_3d=True, bc_ids=None):

        if mesh is None and None in [L, W, H, nx, ny, nlayers]:
            raise ValueError("You must provide either a mesh or all the parameters to enable a mesh to be constructed.")

        # if the mesh has not been provided then we need to create one
        if mesh is None:
            if periodic_direction is not None:
                m = PeriodicRectangleMesh(nx, ny, L, W,
                                          direction=periodic_direction,
                                          quadrilateral=True)
            else:
                m = RectangleMesh(nx, ny, L, W, quadrilateral=True)
                # warn user that constructed mesh is not periodic and
                # they haven't provided ids to apply no normal flow
                # boundary conditions
                if bc_ids is None:
                    warning("your mesh is not periodic and you have not provided boundary condition ids")
            mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
            is_extruded = True
        else:
            # check if mesh is extruded
            is_extruded = (isinstance(mesh.topology, ExtrudedMeshTopology))
            if not is_extruded:
                warning("some things might not work")

        # we always apply the no normal flow boundary condition at the
        # top and bottom - if the user has specified other boundary
        # ids then add them in without duplicating
        default_bc_ids = ["top", "bottom"]
        if bc_ids is not None:
            bc_ids = list(set(default_bc_ids).union(bc_ids))
        else:
            bc_ids = default_bc_ids

        super().__init__(mesh, parameters=parameters,
                         rotation_option=rotation_option,
                         is_3d=is_3d, is_extruded=is_extruded,
                         on_sphere=False, bc_ids=bc_ids)

    @property
    def vertical_normal(self):
        dim = self.mesh.topological_dimension()
        kvec = [0.0]*dim
        kvec[dim-1] = 1.0
        return Constant(kvec)


class VerticalSliceDomain(ChannelDomain):
    """Class defining a 2D (or pseudo-2D) x-z domain.

    The user must either pass in a mesh, or the parameters required to
    construct one. The default behaviour is to construct a
    mesh periodic in the x (and, if rotating, y) direction and a
    nonrotating domain.

    :arg mesh: The mesh, if the user prefers to construct their own.
    :arg parameters: The :class:`Configuration` containing the physical
    parameters
    :arg nx: The number of cells in the x direction
    :arg nlayers: The number of layers
    :arg L: The length of the mesh in the x direction
    :arg H: The height of the top of the mesh
    :arg periodic_direction: May take the value ``"x"`` or ``"None"``
    Defaults to ``"x"``
    :arg rotation option: May take the value ``"Omega"``, ``"f_plane"``,
    ``"beta_plane"``, ``"trad_f"`` or ``"None"``. Defaults to ``"None"``,
    i.e. a nonrotating domain.
    :arg is_3d: boolean, defaults to False. Set to True if you require a
    one-element thick vertical slice domain.
    :arg bc_ids: A list containing the boundary ids for those boundaries on
    which you wish to apply a no normal flow boundary condition.
    """

    def __init__(self, mesh=None, *, parameters=None,
                 nx=None, nlayers=None, L=None, H=None,
                 periodic_direction="x",
                 rotation_option=None, is_3d=False, bc_ids=None):

        if mesh is None and None in [L, H, nx, nlayers]:
            raise ValueError("You must provide either a mesh or the parameters to enable a mesh to be constructed.")

        # for a rotating vertical slice domain, we need to have a
        # component of the velocity in the y direction which requires
        # a 3D domain
        if rotation_option is not None and not is_3d:
            warning("creating a 3d, single element thick mesh as you have a rotating domain.")
            is_3d = True

        # if domain is not 3D then we need to specify the perp operator
        if not is_3d:
            self.perp = lambda u: as_vector([-u[1], u[0]])

        # if the mesh has not been provided then we need to create one
        if mesh is None:
            # if domain is 3d, pass parameters to parent class ChannelDomain
            # with ny=1 and W=H
            if is_3d:
                super().__init__(parameters=parameters,
                                 nx=nx, ny=1, nlayers=nlayers,
                                 L=L, W=H, H=H,
                                 rotation_option=rotation_option,
                                 is_3d=is_3d,
                                 bc_ids=bc_ids)
            else:
                # construct base 1d mesh and extrude, then pass to parent class
                if periodic_direction is not None:
                    m = PeriodicIntervalMesh(nx, L)
                else:
                    m = IntervalMesh(nx, L)
                mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
                super().__init__(mesh, parameters=parameters,
                                 rotation_option=rotation_option,
                                 is_3d=is_3d,
                                 bc_ids=bc_ids)
        else:
            super().__init__(mesh, parameters=parameters,
                             rotation_option=rotation_option,
                             is_3d=is_3d,
                             bc_ids=bc_ids)
