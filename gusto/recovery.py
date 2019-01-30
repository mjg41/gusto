"""
The recovery operators used for lowest-order advection schemes.
"""
from gusto.configuration import logger
from firedrake import (expression, function, Function, FunctionSpace, Projector,
                       VectorFunctionSpace, SpatialCoordinate, as_vector, Constant,
                       dx, Interpolator, quadrilateral, BrokenElement, interval,
                       TensorProductElement, FiniteElement, DirichletBC)
from firedrake.utils import cached_property
from firedrake.parloops import par_loop, READ, INC, RW
from pyop2 import ON_TOP, ON_BOTTOM
import ufl
import numpy as np

__all__ = ["Averager", "Boundary_Recoverer", "Recoverer"]


class Averager(object):
    """
    An object that 'recovers' a low order field (e.g. in DG0)
    into a higher order field (e.g. in CG1).
    The code is essentially that of the Firedrake Projector
    object, using the "average" method, and could possibly
    be replaced by it if it comes into the master branch.

    :arg v: the :class:`ufl.Expr` or
         :class:`.Function` to project.
    :arg v_out: :class:`.Function` to put the result in.
    """

    def __init__(self, v, v_out):

        if isinstance(v, expression.Expression) or not isinstance(v, (ufl.core.expr.Expr, function.Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(v))

        # Check shape values
        if v.ufl_shape != v_out.ufl_shape:
            raise RuntimeError('Shape mismatch between source %s and target function spaces %s in project' % (v.ufl_shape, v_out.ufl_shape))

        self._same_fspace = (isinstance(v, function.Function) and v.function_space() == v_out.function_space())
        self.v = v
        self.v_out = v_out
        self.V = v_out.function_space()

        # Check the number of local dofs
        if self.v_out.function_space().finat_element.space_dimension() != self.v.function_space().finat_element.space_dimension():
            raise RuntimeError("Number of local dofs for each field must be equal.")

        # NOTE: Any bcs on the function self.v should just work.
        # Loop over node extent and dof extent
        self._shapes = (self.V.finat_element.space_dimension(), np.prod(self.V.shape))
        self._average_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        vo[i][j] += v[i][j]/w[i][j];
        }}""" % self._shapes

    @cached_property
    def _weighting(self):
        """
        Generates a weight function for computing a projection via averaging.
        """
        w = Function(self.V)
        weight_kernel = """
        for (int i=0; i<%d; ++i) {
        for (int j=0; j<%d; ++j) {
        w[i][j] += 1.0;
        }}""" % self._shapes

        par_loop(weight_kernel, ufl.dx, {"w": (w, INC)})
        return w

    def project(self):
        """
        Apply the recovery.
        """

        # Ensure that the function being populated is zeroed out
        self.v_out.dat.zero()
        par_loop(self._average_kernel, ufl.dx, {"vo": (self.v_out, INC),
                                                "w": (self._weighting, READ),
                                                "v": (self.v, READ)})
        return self.v_out


class Boundary_Recoverer(object):
    """
    An object that performs a `recovery` process at the domain
    boundaries that has second order accuracy. This is necessary
    because the :class:`Averager` object does not recover a field
    with sufficient accuracy at the boundaries.

    The strategy is to minimise the curvature of the function in
    the boundary cells, subject to the constraints of conserved
    mass and continuity on the interior facets. The quickest way
    to perform this is by using the analytic solution and a parloop.

    Currently this is only implemented for the (DG0, DG1, CG1)
    set of spaces, and only on a `PeriodicIntervalMesh` or
    'PeriodicUnitIntervalMesh` that has been extruded.

    :arg v1: the continuous function after the first recovery
             is performed. Should be in CG1. This is correct
             on the interior of the domain.
    :arg v_out: the function to be output. Should be in DG1.
    :arg v1_ext: a CG1 function denoting which CG1 DOFs are
                  internal and which are external. Only necessary
                  with dynamics method.
    :arg v0_ext: a DG0 function denoting the number of exterior DOFs
                  per cell in the original field (pre-recovery).
                  This argument is only necessary when used with the
                  dynamics method.
    :arg method: string giving the method used for the recovery.
             Valid options are 'dynamics' and 'physics'.
    """

    def __init__(self, v1, v_out, v1_ext=None, v0_ext=None, method='physics'):

        #import pdb; pdb.set_trace()

        self.v_out = v_out
        self.v1 = v1
        self.v1_ext = v1_ext
        self.v0_ext = v0_ext
        
        self.method = method
        mesh = v1.function_space().mesh()
        VDG0 = FunctionSpace(mesh, "DG", 0)
        VCG1 = FunctionSpace(mesh, "CG", 1)

        # # check function spaces of functions -- this only works for a particular set
        if self.method == 'dynamics':
            if v1.function_space() != FunctionSpace(mesh, "CG", 1):
                raise NotImplementedError("This boundary recovery method requires v1 to be in CG1.")
            if v_out.function_space() != FunctionSpace(mesh, "DG", 1):
                raise NotImplementedError("This boundary recovery method requires v_out to be in DG1.")
            if v1_ext.function_space() != FunctionSpace(mesh, "CG", 1):
                raise ValueError("The v1_ext field should be in CG1.")
            if v0_ext.function_space() != FunctionSpace(mesh, "DG", 0):
                raise ValueError("The v0_ext field should be in DG0.")
        elif self.method == 'physics':
            # base spaces
            cell = mesh._base_mesh.ufl_cell().cellname()
            w_hori = FiniteElement("DG", cell, 0)
            w_vert = FiniteElement("CG", interval, 1)
            # build element
            theta_element = TensorProductElement(w_hori, w_vert)
            # spaces
            Vtheta = FunctionSpace(mesh, theta_element)
            Vtheta_broken = FunctionSpace(mesh, BrokenElement(theta_element))
            if v1.function_space() != Vtheta:
                raise ValueError("This boundary recovery method requires v_in to be in DG0xCG1 TensorProductSpace.")
            if v_out.function_space() != Vtheta_broken:
                raise ValueError("This boundary recovery method requires v_out to be in the broken DG0xCG1 TensorProductSpace.")
            if v1_ext != None:
                raise ValueError("The physics boundary recovery method should have v1_ext = None.")
            if v0_ext != None:
                raise ValueError("The physics boundary recovery method should have v0_ext = None.")
        else:
            raise ValueError("Specified boundary_method % not valid" % self.method)

        VuDG1 = VectorFunctionSpace(VDG0.mesh(), "DG", 1)
        x = SpatialCoordinate(VDG0.mesh())
        self.coords = Function(VuDG1).project(x)
        self.interpolator = Interpolator(self.v1, self.v_out)

        # check that we're using quads on extruded mesh -- otherwise it will fail!
        if not VDG0.extruded and VDG0.ufl_element().cell() != quadrilateral:
            raise NotImplementedError("This code only works on extruded quadrilateral meshes.")

        logger.warning('This boundary recovery method is bespoke: it should only be used extruded meshes based on a periodic interval in 2D.')

        self.right = Function(VDG0)

        if self.method == 'density':

            # STRATEGY
            # obtain a coordinate field for all the nodes
            Vu_orig = VectorFunctionSpace(mesh, BrokenElement(v0.ufl_element()))
            orig_coords = Function(Vu_orig).project(x)
            # make a CG2 field that is 1 at exterior nodes and 0 for interior nodes by applying BC
            VCG2 = FunctionSpace(VDG0.mesh(), "CG", 2)
            exterior_CG2 = Function(VCG2)
            boundary_conditions = [DirichletBC(VCG2, Constant(1.0), "top", method="geometric"),
                                   DirichletBC(VCG2, Constant(1.0), "bottom", method="geometric"),
                                   DirichletBC(VCG2, Constant(1.0), "on_boundary", method="geometric")]
            for bc in boundary_conditions:
                bc.apply(exterior_CG2)
            # make source field that is 1 for exterior nodes and 0 for interior nodes by interpolating from CG2 field
            exterior_v0 = Function(v0.function_space()).interpolate(exterior_CG2)
            # make a CG1 field that is 1 for exterior nodes and 0 for interior nodes by applying BC
            exterior_v1 = Function(v1.function_space()).interpolate(exterior_CG2)
            # make a DG0 field that contains the number of exterior nodes per cell for source field
            # make a DG0 field that contains the number of exterior nodes per cell for CG1 field
            # make a vector CG1 field for the new coordinates
            # fill the new vector CG1 field with the location of the corrected coordinates
                # this will involve determining which situation each cell is (8 diff situations)
            # run through the recoverd CG1 field. For each cell on the boundary:
                # use Gaussian elimination to find the constants in a linear approx. using the new coordinates with old values
                # use the constants in the linear approx. to find the new values at the old coordinates
            
            # make DG0 field that is one in rightmost cells, but zero otherwise
            # this is done as the DOF numbering is different in the rightmost cells
            max_coord = Function(VDG0).interpolate(Constant(np.max(self.coords.dat.data[:, 0])))

            right_kernel = """
            if (fmax(COORDS[0][0], fmax(COORDS[1][0], COORDS[2][0])) == MAX[0][0])
            RIGHT[0][0] = 1.0;
            """
            par_loop(right_kernel, dx,
                     args={"COORDS": (self.coords, READ),
                           "MAX": (max_coord, READ),
                           "RIGHT": (self.right, RW)})

            self.bottom_kernel = """
            if (RIGHT[0][0] == 1.0)
            {
            float x = COORDS[2][0] - COORDS[0][0];
            float y = COORDS[1][1] - COORDS[0][1];
            float a = CG1[3][0];
            float b = CG1[1][0];
            float c = DG0[0][0];
            DG1[1][0] = a;
            DG1[3][0] = b;
            DG1[2][0] = (1.0 / (pow(x, 2.0) + 4.0 * pow(y, 2.0))) * (-3.0 * a * pow(y, 2.0) - b * pow(x, 2.0) - b * pow(y, 2.0) + 2.0 * c * pow(x, 2.0) + 8.0 * c * pow(y, 2.0));
            DG1[0][0] = 4.0 * c - b - a - DG1[2][0];
            }
            else
            {
            float x = COORDS[1][0] - COORDS[3][0];
            float y = COORDS[3][1] - COORDS[2][1];
            float a = CG1[1][0];
            float b = CG1[3][0];
            float c = DG0[0][0];
            DG1[3][0] = a;
            DG1[1][0] = b;
            DG1[0][0] = (1.0 / (pow(x, 2.0) + 4.0 * pow(y, 2.0))) * (-3.0 * a * pow(y, 2.0) - b * pow(x, 2.0) - b * pow(y, 2.0) + 2.0 * c * pow(x, 2.0) + 8.0 * c * pow(y, 2.0));
            DG1[2][0] = 4.0 * c - b - a - DG1[0][0];
            }
            """

            self.top_kernel = """
            if (RIGHT[0][0] == 1.0)
            {
            float x = COORDS[2][0] - COORDS[0][0];
            float y = COORDS[1][1] - COORDS[0][1];
            float a = CG1[2][0];
            float b = CG1[0][0];
            float c = DG0[0][0];
            DG1[2][0] = a;
            DG1[0][0] = b;
            DG1[3][0] = (1.0 / (pow(x, 2.0) + 4.0 * pow(y, 2.0))) * (-3.0 * a * pow(y, 2.0) - b * pow(x, 2.0) - b * pow(y, 2.0) + 2.0 * c * pow(x, 2.0) + 8.0 * c * pow(y, 2.0));
            DG1[1][0] = 4.0 * c - b - a - DG1[3][0];
            }
            else
            {
            float x = COORDS[0][0] - COORDS[2][0];
            float y = COORDS[3][1] - COORDS[2][1];
            float a = CG1[2][0];
            float b = CG1[0][0];
            float c = DG0[0][0];
            DG1[0][0] = a;
            DG1[2][0] = b;
            DG1[3][0] = (1.0 / (pow(x, 2.0) + 4.0 * pow(y, 2.0))) * (-3.0 * a * pow(y, 2.0) - b * pow(x, 2.0) - b * pow(y, 2.0) + 2.0 * c * pow(x, 2.0) + 8.0 * c * pow(y, 2.0));
            DG1[1][0] = 4.0 * c - b - a - DG1[3][0];
            }
            """
            
        elif self.method == 'physics':
            self.bottom_kernel = """
            DG1[0][0] = CG1[1][0] - 2 * (CG1[1][0] - CG1[0][0]);
            DG1[1][0] = CG1[1][0];
            """

            self.top_kernel = """
            DG1[1][0] = CG1[0][0] - 2 * (CG1[0][0] - CG1[1][0]);
            DG1[0][0] = CG1[0][0];
            """

    def apply(self):

        self.interpolator.interpolate()
        par_loop(self.bottom_kernel, dx,
                 args={"DG1": (self.v_out, RW),
                       "CG1": (self.v1, READ),
                       "DG0": (self.v0, READ),
                       "COORDS": (self.coords, READ),
                       "RIGHT": (self.right, READ)},
                 iterate=ON_BOTTOM)

        par_loop(self.top_kernel, dx,
                 args={"DG1": (self.v_out, RW),
                       "CG1": (self.v1, READ),
                       "DG0": (self.v0, READ),
                       "COORDS": (self.coords, READ),
                       "RIGHT": (self.right, READ)},
                 iterate=ON_TOP)


class Recoverer(object):
    """
    An object that 'recovers' a field from a low order space
    (e.g. DG0) into a higher order space (e.g. CG1). This encompasses
    the process of interpolating first to a the right space before
    using the :class:`Averager` object, and also automates the
    boundary recovery process. If no boundary method is specified,
    this simply performs the action of the :class: `Averager`.

    :arg v_in: the :class:`ufl.Expr` or
         :class:`.Function` to project. (e.g. a VDG0 function)
    :arg v_out: :class:`.Function` to put the result in. (e.g. a CG1 function)
    :arg VDG: optional :class:`.FunctionSpace`. If not None, v_in is interpolated
         to this space first before recovery happens.
    :arg boundary_method: a string defining which type of method needs to be
         used at the boundaries. Valid options are 'density', 'velocity' or 'physics'.
    """

    def __init__(self, v_in, v_out, VDG=None, boundary_method=None):

        # check if v_in is valid
        if isinstance(v_in, expression.Expression) or not isinstance(v_in, (ufl.core.expr.Expr, function.Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(v_in))

        self.v_in = v_in
        self.v_out = v_out
        self.V = v_out.function_space()
        if VDG is not None:
            self.v = Function(VDG)
            self.interpolator = Interpolator(v_in, self.v)
        else:
            self.v = v_in
            self.interpolator = None

        self.VDG = VDG
        self.boundary_method = boundary_method
        self.averager = Averager(self.v, self.v_out)

        # check boundary method options are valid
        if boundary_method is not None:
            if boundary_method != 'scalar' and boundary_method != 'vector' and boundary_method != 'physics':
                raise ValueError("Specified boundary_method % not valid" % boundary_method)
            if VDG is None:
                raise ValueError("If boundary_method is specified, VDG also needs specifying.")

            # now specify things that we'll need if we are doing boundary recovery
            if boundary_method == 'physics':
                # check dimensions
                if self.V.value_size != 1:
                    raise ValueError('This method only works for scalar functions.')
                self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v, method='physics')
            elif boundary_method == 'scalar':
                # check dimensions
                if self.V.value_size != 1:
                    raise ValueError('This method only works for scalar functions.')
                
                # make exterior CG2 field (we need this to interpolate for general temperature space)
                mesh = self.V.mesh()
                VDG0 = FunctionSpace(mesh, "DG", 0)
                VCG2 = FunctionSpace(mesh, "CG", 2)
                exterior_CG2 = Function(VCG2)
                boundary_conditions = [DirichletBC(VCG2, Constant(1.0), "top", method="geometric"),
                                       DirichletBC(VCG2, Constant(1.0), "bottom", method="geometric"),
                                       DirichletBC(VCG2, Constant(1.0), "on_boundary", method="geometric")]
                for bc in boundary_conditions:
                    bc.apply(exterior_CG2)

                v_in_ext = Function(v_in.function_space()).interpolate(exterior_CG2)
                v_out_ext = Function(self.V).interpolate(exterior_CG2)
                v_in_extnum = Function(VDG0)

                find_number_of_exterior_DOFs_per_cell(v_in_ext, v_in_extnum)
                
                self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v, v1_ext=v_out_ext, v0_ext=v_in_extnum, method='dynamics')
            elif boundary_method == 'vector':
                # check dimensions
                if self.V.value_size != 2:
                    raise NotImplementedError('This method only works for 2D vector functions.')
                # declare relevant spaces
                mesh = self.V.mesh()
                VDG0 = FunctionSpace(mesh, "DG", 0)
                VCG1 = FunctionSpace(mesh, "CG", 1)
                VDG1 = FunctionSpace(mesh, "DG", 1)
                VCG2 = FunctionSpace(mesh, "CG", 2)
                v_out_ext = Function(VCG1)
                Vv = self.v_in.function_space()
                bcs_CG1 = [DirichletBC(VCG1, Constant(1.0), "top", method="geometric"),
                           DirichletBC(VCG1, Constant(1.0), "bottom", method="geometric"),
                           DirichletBC(VCG1, Constant(1.0), "on_boundary", method="geometric")]
                for bc in bcs_CG1:
                    bc.apply(v_out_ext)

                # need to find num of exterior values per cell for each dimension of vector field
                # first make a field in Vv that has 1 on the boundaries
                ones_list = [1. for i in range(self.V.value_size)]
                ones = Function(Vv).project(as_vector(ones_list))
                exterior_Vv = Function(Vv)
                bcs_Vv = [DirichletBC(Vv, ones, "top", method="geometric"),
                          DirichletBC(Vv, ones, "bottom", method="geometric"),
                          DirichletBC(Vv, ones, "on_boundary", method="geometric")]
                for bc in bcs_Vv:
                    bc.apply(exterior_Vv)

                # now, for each component, convert to CG2 which should be bigger than Vv
                exterior_CG2 = []
                v_in_extnum = []
                v_scalars = []
                v_out_scalars = []
                self.boundary_recoverers = []
                self.project_to_scalars_CG = []
                self.extra_averagers = []
                for i in range(self.V.value_size):
                    exterior_CG2.append(Function(VCG2).interpolate(exterior_Vv[i]))
                    # do horrendous hack to ensure values are either 0 or 1
                    exterior_CG2[i].interpolate(conditional(exterior_CG2[i] > 0.75, 1.0, 0.0))
                    v_in_extnum.append(Function(VDG0))
                    find_number_of_exterior_DOFs_per_cell(exterior_CG2[i], v_in_extnum[i])
                    v_scalars.append(Function(VDG1))
                    v_out_scalars.append(Function(VCG1))
                    self.project_to_scalar_CG.append(Projector(self.v_out[i], self.v_out_scalar[i]))
                    self.boundary_recoverers.append(Boundary_Recoverer(v_out_scalars[i], v_scalars[i], v1_ext=v_out_ext, v0_ext=v_in_extnum[i], method='dynamics'))
                    # need an extra averager that works on the scalar fields rather than the vector one
                    self.extra_averagers.append(Averager(self.v_scalars[i], self.v_out_scalars[i]))

                # the boundary recoverer needs to be done on a scalar fields
                # so need to extract component and restore it after the boundary recovery is done
                self.project_to_vector = Projector(as_vector(v_out_scalars), self.v_out)


    def project(self):
        """
        Perform the fully specified recovery.
        """

        if self.interpolator is not None:
            self.interpolator.interpolate()
        self.averager.project()
        if self.boundary_method is not None:
            if self.boundary_method == 'velocity':
                for i in range(self.V.value_size):
                    self.project_to_scalars_CG[i].project()
                    self.boundary_recoverers[i].apply()
                    self.extra_averagers[i].project()
                self.restore_vector()
            elif self.boundary_method == 'density' or self.boundary_method == 'physics':
                self.boundary_recoverer.apply()
                self.averager.project()
        return self.v_out


def find_number_of_exterior_DOFs_per_cell(field, output):
    """
    Finds the number of DOFs on the domain exterior
    per cell and stores it in a DG0 field.
    
    :arg field: the input field, containing a 1 at each
                exterior DOF and a 0 at each interior DOF.
    :arg output: a DG0 field to be output to.
    """

    shapes = field.function_space().finat_element.space_dimension()
    kernel = """
    for (int i=0; i<%d; ++i) {
    DG0[0][0] += ext[0][i];}
    """ % shapes
    
    par_loop(kernel, dx,
             args={"DG0": (output, RW),
                   "ext": (field, READ)})
