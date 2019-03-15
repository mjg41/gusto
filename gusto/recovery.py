"""
The recovery operators used for lowest-order advection schemes.
"""
from gusto.configuration import logger
from firedrake import (expression, function, Function, FunctionSpace, Projector,
                       VectorFunctionSpace, SpatialCoordinate, as_vector, Constant,
                       dx, Interpolator, quadrilateral, BrokenElement, interval,
                       TensorProductElement, FiniteElement, DirichletBC, conditional,
                       interpolate, dS, dS_h, dS_v, TestFunction, NonlinearVariationalSolver,
                       NonlinearVariationalProblem)
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

    :arg v_CG1: the continuous function after the first recovery
             is performed. Should be in CG1. This is correct
             on the interior of the domain.
    :arg v_DG1: the function to be output. Should be in DG1.
    :arg ext_DG1: a field in DG1 which is 1 on exterior DOFs but
                  0 on interior nodes.
    :arg ext_V0_CG1: a field in CG1 that is 1 for the exterior DOFs of
                     the original space but 0 on interior nodes.
    :arg method: string giving the method used for the recovery.
             Valid options are 'dynamics' and 'physics'.
    """

    def __init__(self, v_CG1, v_DG1, ext_DG1=None, ext_V0_CG1=None, method='physics', new_ext_coords=None):

        self.v_DG1 = v_DG1
        self.v_CG1 = v_CG1
        self.ext_DG1 = ext_DG1
        self.ext_V0_CG1 = ext_V0_CG1

        if new_ext_coords is not None:
            new_kernel = True
        else:
            new_kernel = False


        self.method = method
        mesh = v_CG1.function_space().mesh()
        VDG0 = FunctionSpace(mesh, "DG", 0)
        VCG1 = FunctionSpace(mesh, "CG", 1)
        VDG1 = FunctionSpace(mesh, "DG", 1)

        # # check function spaces of functions -- this only works for a particular set
        if self.method == 'dynamics':
            if v_CG1.function_space() != VCG1:
                raise NotImplementedError("This boundary recovery method requires v1 to be in CG1.")
            if v_DG1.function_space() != FunctionSpace(mesh, "DG", 1):
                raise NotImplementedError("This boundary recovery method requires v_out to be in DG1.")
            # check whether mesh is valid
            if mesh.topological_dimension() == 2:
                # if mesh is extruded then we're fine, but if not needs to be quads
                if not VDG0.extruded and mesh.ufl_cell().cell_name() != 'quadrilateral':
                    raise NotImplementedError('For 2D meshes this recovery method requires that elements are quadrilaterals')
            elif mesh.topological_dimension() == 3:
                # assume that 3D mesh is extruded
                if mesh._base_mesh.ufl_cell().cellname() != 'quadrilateral':
                    raise NotImplementedError('For 3D extruded meshes this recovery method requires a base mesh with quadrilateral elements')
            else:
                raise NotImplementedError('This boundary recovery is implemented only on certain classes of mesh.')
            if self.ext_DG1 is None or self.ext_V0_CG1 is None:
                raise ValueError('Need external fields for dynamics boundary methods')

        elif self.method == 'physics':
            # check that mesh is valid -- must be an extruded mesh
            if not VDG0.extruded:
                raise NotImplementedError('The physics boundary method only works on extruded meshes')
            # base spaces
            cell = mesh._base_mesh.ufl_cell().cellname()
            w_hori = FiniteElement("DG", cell, 0)
            w_vert = FiniteElement("CG", interval, 1)
            # build element
            theta_element = TensorProductElement(w_hori, w_vert)
            # spaces
            Vtheta = FunctionSpace(mesh, theta_element)
            Vtheta_broken = FunctionSpace(mesh, BrokenElement(theta_element))
            if v_CG1.function_space() != Vtheta:
                raise ValueError("This boundary recovery method requires v_CG1 to be in DG0xCG1 TensorProductSpace.")
            if v_DG1.function_space() != Vtheta_broken:
                raise ValueError("This boundary recovery method requires v_DG1 to be in the broken DG0xCG1 TensorProductSpace.")
        else:
            raise ValueError("Specified boundary_method % not valid" % self.method)

        VuDG1 = VectorFunctionSpace(VDG0.mesh(), "DG", 1)
        x = SpatialCoordinate(VDG0.mesh())
        self.orig_coords = Function(VuDG1).project(x)
        self.interpolator = Interpolator(self.v_CG1, self.v_DG1)

        if self.method == 'dynamics':

            # STRATEGY
            # obtain a coordinate field for all the nodes
            VuCG1 = VectorFunctionSpace(mesh, "CG", 1)
            VuDG1 = VectorFunctionSpace(mesh, "DG", 1)
            self.orig_coords = Function(VuDG1).project(x)
            self.new_coords = Function(VuDG1).project(x)

            find_effective_coords_kernel = """
            int dim = %d;
            int n = %d;

            /* find num of DOFs to adjust in this cell, DG1 */
            int sum_V1_ext = 0;
            for (int i=0; i<%d; ++i) {
            sum_V1_ext += round(EXT_V1[i][0]);}

            /* only do adjustment in cells with at least one DOF to adjust */
            if (sum_V1_ext > 0) {

            /* find max dist */
            float max_dist = 0;
            for (int i=0; i<%d; ++i){
            for (int j=0; j<%d; ++j) {
            float dist = 0;
            for (int k=0; k<dim; ++k) {
            dist += pow(ACT_COORDS[i][k] - ACT_COORDS[j][k], 2);}
            dist = pow(dist, 0.5);
            max_dist = fmax(dist, max_dist);}}

            /* loop through DOFs in cell and find which ones to adjust */
            for (int i=0; i<%d; ++i) {
            if (round(EXT_V1[i][0]) == 1){

            /* find closest interior node */
            float min_dist = max_dist;
            int index = -1;
            for (int j=0; j<%d; ++j) {
            if (round(EXT_V1[j][0]) == 0) {
            float dist = 0;
            for (int k=0; k<dim; ++k) {
            dist += pow(ACT_COORDS[i][k] - ACT_COORDS[j][k], 2);}
            dist = pow(dist, 0.5);
            if (dist <= min_dist) {
            min_dist = dist;
            index = j;}}}

            /* adjust coordinate */
            for (int j=0; j<dim; ++j) {
            EFF_COORDS[i][j] = 0.5 * (ACT_COORDS[i][j] + ACT_COORDS[index][j]);
            }
            }
            }
            }

            /* else do nothing */

            """ % ((np.prod(VuDG1.shape),) * 1 + (self.v_DG1.function_space().finat_element.space_dimension(), ) * 6)

            coords_kernel_2d = """
            int nDOF_V1 = %d;
            int nDOF_V0 = %d;
            int dim = %d;

            /* find num of ext DOFs in this cell, DG1 */
            int sum_V1_ext = 0;
            for (int i=0; i<nDOF_V1; ++i) {
            sum_V1_ext += round(EXT_V1[i][0]);}

            /* find num of ext DOFs in this cell, CG1 */
            int sum_V0_ext = 0;
            for (int i=0; i<nDOF_V0; ++i) {
            sum_V0_ext += round(EXT_V0[i][0]);}

            if (sum_V1_ext == 0){
            /* do nothing for internal cells */
            }

            else if (sum_V1_ext == 2 && sum_V0_ext == 0){
            /* cells on edge from DG0 */
            for (int i=0; i<nDOF_V1; ++i){ /* loop through each node */
            if (round(EXT_V1[i][0]) == 1){
            /* if the node is external, we need to give it new coords */
            /* first find a max_dist to be starting point to find min_dist */
            float max_dist = 0;
            for (int j=0; j<nDOF_V1; ++j) {
            float dist = 0;
            for (int k=0; k<dim; ++k) {
            dist += pow(OLD_COORDS[i][k] - OLD_COORDS[j][k], 2);}
            dist = pow(dist, 0.5);
            max_dist = fmax(dist, max_dist);}
            /* find the index of the closest internal node to each external node */
            float min_dist = max_dist;
            int index = -1;
            for (int j=0; j<nDOF_V1; ++j) {
            if (round(EXT_V1[j][0]) == 0) {
            float dist = 0;
            for (int k=0; k<dim; ++k) {
            dist += pow(OLD_COORDS[i][k] - OLD_COORDS[j][k], 2);}
            dist = pow(dist, 0.5);
            if (dist <= min_dist) {
            min_dist = dist;
            index = j;}}}
            /* new coords are halfway between old coords and nearest internal node */
            NEW_COORDS[i][0] = 0.5*(OLD_COORDS[i][0] + OLD_COORDS[index][0]);
            NEW_COORDS[i][1] = 0.5*(OLD_COORDS[i][1] + OLD_COORDS[index][1]);
            }}}

            else if (sum_V1_ext == 2 && sum_V0_ext == 2){
            /* recovery from theta space with node on edge */
            /* no extra recovery necessary */}

            else if (sum_V1_ext == 3 && sum_V0_ext == 0){
            /* cells in corner from DG0 */
            for (int i=0; i<nDOF_V1; ++i){ /* loop through each node */
            if (round(EXT_V1[i][0]) == 1){
            /* if the node is external, we need to give it new coords */
            /* first the index of the only internal node */
            int index = -1;
            for (int j=0; j<nDOF_V1; ++j) {
            if (round(EXT_V1[j][0]) == 0) {
            index = j;}}
            /* new coords are halfway between old coords and nearest internal node */
            NEW_COORDS[i][0] = 0.5*(OLD_COORDS[i][0] + OLD_COORDS[index][0]);
            NEW_COORDS[i][1] = 0.5*(OLD_COORDS[i][1] + OLD_COORDS[index][1]);
            }}}

            else if (sum_V1_ext == 3 && sum_V0_ext == 2){
            /* recovery from theta space in the corner of the domain */
            /* plan for this one is to first identify each node */
            int internal_idx = 100; /* index of internal node */
            int opposite_idx = 100; /* index of the node opposite the internal node */
            int both_idx = 100; /* index of node which is external but recorded as internal for theta space */
            int other_idx = 100; /* index of the other node */

            /* first identify internal node and 'both' node */
            for (int i=0; i<nDOF_V1; ++i) { /* loop through nodes to find the internal node */
            if (round(EXT_V1[i][0]) == 0) {
            internal_idx = i;}
            else if (round(EXT_V1[i][0] == 1) && round(EXT_V0[i][0]) == 0) {
            both_idx = i;}}

            /* now identify other nodes in terms of their distance from the other nodes */
            for (int i=0; i<nDOF_V1; ++i) {
            if (i != internal_idx && i != both_idx) {
            float internal_dist = 0;
            float both_dist = 0;
            for (int k=0; k<dim; ++k) {
            internal_dist += pow(OLD_COORDS[i][k] - OLD_COORDS[internal_idx][k], 2);
            both_dist += pow(OLD_COORDS[i][k] - OLD_COORDS[both_idx][k], 2);}
            internal_dist = pow(internal_dist, 0.5);
            both_dist = pow(both_dist, 0.5);
            if (internal_dist > both_dist) {
            opposite_idx = i;}
            else if (internal_dist < both_dist) {
            other_idx = i;}
            else {
            fprintf(stderr, "Something has gone wrong with determining which node is which in 2d theta space case");}}}

            /* now declare the new coords */
            for (int k=0; k<dim; ++k) {
            NEW_COORDS[internal_idx][k] = OLD_COORDS[internal_idx][k];
            NEW_COORDS[both_idx][k] = 0.5 * (OLD_COORDS[both_idx][k] + OLD_COORDS[internal_idx][k]);
            NEW_COORDS[opposite_idx][k] = 0.5 * (OLD_COORDS[opposite_idx][k] + OLD_COORDS[other_idx][k]);
            NEW_COORDS[other_idx][k] = OLD_COORDS[other_idx][k];}}

            else {
            fprintf(stderr, "Wrong number of exterior coords found");}

            """ % (self.v_DG1.function_space().finat_element.space_dimension(),
                   self.v_CG1.function_space().finat_element.space_dimension(),
                   np.prod(VuDG1.shape))

            coords_kernel_3d = """
            int nDOF_V1 = %d;
            int nDOF_V0 = %d;
            int dim = %d;

            /* find num of ext DOFs in this cell, DG1 */
            int sum_V1_ext = 0;
            for (int i=0; i<nDOF_V1; ++i) {
            sum_V1_ext += round(EXT_V1[i][0]);}

            /* find num of ext DOFs in this cell, CG1 */
            int sum_V0_ext = 0;
            for (int i=0; i<nDOF_V0; ++i) {
            sum_V0_ext += round(EXT_V0[i][0]);}

            if (sum_V1_ext == 0){
            /* do nothing for internal cells */
            }

            else if (sum_V1_ext == 4 && sum_V0_ext == 0){
            /* cells on face from DG0 or theta space without touching boundary */
            for (int i=0; i<nDOF_V1; ++i){ /* loop through each node */
            if (round(EXT_V1[i][0]) == 1){
            /* if the node is external, we need to give it new coords */
            /* first find a max_dist to be starting point to find min_dist */
            float max_dist = 0;
            for (int j=0; j<nDOF_V1; ++j) {
            float dist = 0;
            for (int k=0; k<dim; ++k) {
            dist += pow(OLD_COORDS[i][k] - OLD_COORDS[j][k], 2);}
            dist = pow(dist, 0.5);
            max_dist = fmax(dist, max_dist);}
            /* find the index of the closest internal node to each external node */
            float min_dist = max_dist;
            int index = -1;
            for (int j=0; j<nDOF_V1; ++j) {
            if (round(EXT_V1[j][0]) == 0) {
            float dist = 0;
            for (int k=0; k<dim; ++k) {
            dist += pow(OLD_COORDS[i][k] - OLD_COORDS[j][k], 2);}
            dist = pow(dist, 0.5);
            if (dist <= min_dist) {
            min_dist = dist;
            index = j;}}}
            /* new coords are halfway between old coords and nearest internal node */
            NEW_COORDS[i][0] = 0.5*(OLD_COORDS[i][0] + OLD_COORDS[index][0]);
            NEW_COORDS[i][1] = 0.5*(OLD_COORDS[i][1] + OLD_COORDS[index][1]);
            NEW_COORDS[i][2] = 0.5*(OLD_COORDS[i][2] + OLD_COORDS[index][2]);
            }}}

            else if (sum_V1_ext == 4 && sum_V0_ext == 4){
            /* recovery from theta space with node on face */
            /* no extra recovery necessary */}


            else if (sum_V1_ext == 6 && sum_V0_ext == 0){
            /* recovery at edge from DG0 or with theta space with no exterior nodes */
            for (int i=0; i<nDOF_V1; ++i){ /* loop through each node */
            if (round(EXT_V1[i][0]) == 1){
            /* if the node is external, we need to give it new coords */
            /* first find a max_dist to be starting point to find min_dist */
            float max_dist = 0;
            for (int j=0; j<nDOF_V1; ++j) {
            float dist = 0;
            for (int k=0; k<dim; ++k) {
            dist += pow(OLD_COORDS[i][k] - OLD_COORDS[j][k], 2);}
            dist = pow(dist, 0.5);
            max_dist = fmax(dist, max_dist);}
            /* find the index of the closest internal node to each external node */
            float min_dist = max_dist;
            int index = -1;
            for (int j=0; j<nDOF_V1; ++j) {
            if (round(EXT_V1[j][0]) == 0) {
            float dist = 0;
            for (int k=0; k<dim; ++k) {
            dist += pow(OLD_COORDS[i][k] - OLD_COORDS[j][k], 2);}
            dist = pow(dist, 0.5);
            if (dist <= min_dist) {
            min_dist = dist;
            index = j;}}}
            /* new coords are halfway between old coords and nearest internal node */
            NEW_COORDS[i][0] = 0.5*(OLD_COORDS[i][0] + OLD_COORDS[index][0]);
            NEW_COORDS[i][1] = 0.5*(OLD_COORDS[i][1] + OLD_COORDS[index][1]);
            NEW_COORDS[i][2] = 0.5*(OLD_COORDS[i][2] + OLD_COORDS[index][2]);
            }}}

            else if (sum_V1_ext == 6 && sum_V0_ext == 4){
            /* recovery at edge from theta space with one exterior node */
            int internal_indices[2] = {100, 100};
            int both_indices[2] = {100, 100};
            int opposite_indices[2] = {100, 100};
            int other_indices[2] = {100, 100};

            /* identify both internal nodes and */
            /* identify external nodes which are internal for theta space */
            int index_for_internal_indices = 0;
            int index_for_both_indices = 0;
            for (int i=0; i<nDOF_V1; ++i) {
            if (round(EXT_V1[i][0]) == 0) {
            internal_indices[index_for_internal_indices] = i;
            index_for_internal_indices += 1;}
            else if (round(EXT_V0[i][0]) == 0) {
            both_indices[index_for_both_indices] = i;
            index_for_both_indices += 1;}}

            /* now loop through other nodes */
            int index_for_other_indices = 0;
            int index_for_opposite_indices = 0;
            for (int i=0; i<nDOF_V1; ++i) {
            /* look for the remaining node types */
            if (round(EXT_V1[i][0]) == 1 && round(EXT_V0[i][0]) == 1) {
            float dist_internal_0 = 0;
            float dist_both_0 = 0;
            float dist_internal_1 = 0;
            float dist_both_1 = 0;
            for (int k=0; k<dim; ++k) {
            dist_internal_0 += pow(OLD_COORDS[i][k] - OLD_COORDS[internal_indices[0]][k], 2);
            dist_internal_1 += pow(OLD_COORDS[i][k] - OLD_COORDS[internal_indices[1]][k], 2);
            dist_both_0 += pow(OLD_COORDS[i][k] - OLD_COORDS[both_indices[0]][k], 2);
            dist_both_1 += pow(OLD_COORDS[i][k] - OLD_COORDS[both_indices[1]][k], 2);}
            dist_internal_0 = pow(dist_internal_0, 0.5);
            dist_internal_1 = pow(dist_internal_1, 0.5);
            dist_both_0 = pow(dist_both_0, 0.5);
            dist_both_1 = pow(dist_both_1, 0.5);
            float min_internal_dist = fmin(dist_internal_0, dist_internal_1);
            float min_both_dist = fmin(dist_both_0, dist_both_1);

            /* if they are closer to the nearest internal node rather than the nearest both node, then they are other nodes */
            if (min_internal_dist < min_both_dist) {
            other_indices[index_for_other_indices] = i;
            index_for_other_indices += 1;}
            /* else they are opposite nodes */
            else {
            opposite_indices[index_for_opposite_indices] = i;
            index_for_opposite_indices += 1;}}
            }

            /* coords of all internal and other nodes stay the same */
            for (int k=0; k<dim; ++k) {
            for (int j=0; j<2; ++j) {
            NEW_COORDS[internal_indices[j]][k] = OLD_COORDS[internal_indices[j]][k];
            NEW_COORDS[other_indices[j]][k] = OLD_COORDS[other_indices[j]][k];
            }}

            /* coords of opposite nodes move towards those of the nearest other node */
            /* coords of both nodes move towrds those of the closest internal node */
            for (int j=0; j<2; ++j) {
            float dist_other_0 = 0;
            float dist_other_1 = 0;
            float dist_internal_0 = 0;
            float dist_internal_1 = 0;
            for (int k=0; k<dim; ++k) {
            dist_other_0 += pow(OLD_COORDS[opposite_indices[j]][k] - OLD_COORDS[other_indices[0]][k], 2);
            dist_other_1 += pow(OLD_COORDS[opposite_indices[j]][k] - OLD_COORDS[other_indices[1]][k], 2);
            dist_internal_0 += pow(OLD_COORDS[both_indices[j]][k] - OLD_COORDS[internal_indices[0]][k], 2);
            dist_internal_1 += pow(OLD_COORDS[both_indices[j]][k] - OLD_COORDS[internal_indices[1]][k], 2);}
            dist_other_0 = pow(dist_other_0, 0.5);
            dist_other_1 = pow(dist_other_1, 0.5);
            dist_internal_0 = pow(dist_internal_0, 0.5);
            dist_internal_1 = pow(dist_internal_1, 0.5);
            if (dist_other_0 < dist_other_1) {
            for (int k=0; k<dim; ++k) {
            NEW_COORDS[opposite_indices[j]][k] = 0.5 * (OLD_COORDS[opposite_indices[j]][k] + OLD_COORDS[other_indices[0]][k]);}}
            else {
            for (int k=0; k<dim; ++k) {
            NEW_COORDS[opposite_indices[j]][k] = 0.5 * (OLD_COORDS[opposite_indices[j]][k] + OLD_COORDS[other_indices[1]][k]);}}
            if (dist_internal_0 < dist_internal_1) {
            for (int k=0; k<dim; ++k) {
            NEW_COORDS[both_indices[j]][k] = 0.5 * (OLD_COORDS[both_indices[j]][k] + OLD_COORDS[internal_indices[0]][k]);}}
            else {
            for (int k=0; k<dim; ++k) {
            NEW_COORDS[both_indices[j]][k] = 0.5 * (OLD_COORDS[both_indices[j]][k] + OLD_COORDS[internal_indices[1]][k]);}}
            }}

            else if (sum_V1_ext == 7 && sum_V0_ext == 0){
            /* recovery at corner from DG0 */
            for (int i=0; i<nDOF_V1; ++i){ /* loop through each node */
            if (round(EXT_V1[i][0]) == 1){
            /* if the node is external, we need to give it new coords */
            /* first the index of the only internal node */
            int index = -1;
            for (int j=0; j<nDOF_V1; ++j) {
            if (round(EXT_V1[j][0]) == 0) {
            index = j;}}
            /* new coords are halfway between old coords and nearest internal node */
            NEW_COORDS[i][0] = 0.5*(OLD_COORDS[i][0] + OLD_COORDS[index][0]);
            NEW_COORDS[i][1] = 0.5*(OLD_COORDS[i][1] + OLD_COORDS[index][1]);
            NEW_COORDS[i][2] = 0.5*(OLD_COORDS[i][2] + OLD_COORDS[index][2]);
            }}}

            else if (sum_V1_ext == 7 && sum_V0_ext == 4){
            /* recovery at corner from theta space */

            int internal_index = 100;
            int opposite_indices[4] = {100, 100, 100, 100};
            int opp_other_indices[3] = {100, 100, 100};
            int both_indices[3] = {100, 100, 100};
            int other_index = 100;
            int index_for_both_indices = 0;
            int index_for_opp_other_indices = 0;
            int index_for_opposite_indices = 0;

            for (int i=0; i<nDOF_V1; ++i) {
            if (round(EXT_V1[i][0]) == 0) { /* identify which node is the internal node */
            internal_index = i;
            }
            else if (round(EXT_V1[i][0]) == 1 && round(EXT_V0[i][0]) == 0) {
            /* identify which nodes are external but Vt internal ("both nodes") */
            both_indices[index_for_both_indices] = i;
            index_for_both_indices += 1;
            }
            else {
            opposite_indices[index_for_opposite_indices] = i;
            index_for_opposite_indices += 1;
            }
            }

            /* find which of the opposite nodes is closest to the internal node */
            /* first need to find max distance */
            float max_dist = 0;
            for (int i=0; i<nDOF_V1; ++i) {
            float dist = 0;
            for (int k=0; k<dim; ++k) {
            dist += pow(OLD_COORDS[internal_index][k] - OLD_COORDS[i][k], 2);
            }
            dist = pow(dist, 0.5);
            if (dist > max_dist) {
            max_dist = dist;
            }
            }
            /* now find min distance */
            int index_of_nearest_internal[4] = {100, 100, 100, 100};
            int trial_index = 100;
            for (int j=0; j<4; ++j) {
            float min_dist = max_dist;
            for (int l=0; l<4; ++l) {
            float dist = 0;
            if (l < 3) {
            trial_index = both_indices[l];}
            else {trial_index = internal_index;}
            for (int k=0; k<dim; ++k) {
            dist += pow(OLD_COORDS[trial_index][k] - OLD_COORDS[opposite_indices[j]][k], 2);}
            dist = pow(dist, 0.5);
            if (dist <= min_dist) {
            min_dist = dist;
            index_of_nearest_internal[j] = trial_index;}
            /* this gives us our other index */
            }
            }
            for (int j=0; j<4; ++j){
            if (index_of_nearest_internal[j] == internal_index){
            other_index = opposite_indices[j];
            }
            }
            /* now to assign all remaining both opposite indices */
            for (int j=0; j<4; ++j) {
            if (opposite_indices[j] != other_index) {
            opp_other_indices[index_for_opp_other_indices] = opposite_indices[j];
            index_for_opp_other_indices += 1;
            }
            }

            /* the opposite nodes move closer to the internal node */
            /* the both opp nodes move closer to the other node */
            for (int k=0; k<dim; ++k) {
            NEW_COORDS[internal_index][k] = OLD_COORDS[internal_index][k];
            NEW_COORDS[other_index][k] = OLD_COORDS[other_index][k];
            for (int i=0; i<3; ++i) {
            NEW_COORDS[both_indices[i]][k] = 0.5 * (OLD_COORDS[both_indices[i]][k] + OLD_COORDS[internal_index][k]);
            NEW_COORDS[opp_other_indices[i]][k] = 0.5 * (OLD_COORDS[opp_other_indices[i]][k] + OLD_COORDS[other_index][k]);
            }
            }
            }

            else {
            fprintf(stderr, "Wrong number of exterior coords found");}

            """ % (self.v_DG1.function_space().finat_element.space_dimension(),
                   self.v_CG1.function_space().finat_element.space_dimension(),
                   np.prod(VuDG1.shape))

            if VuCG1.mesh().topological_dimension() == 2:
                self.coords_kernel = coords_kernel_2d
            elif VuCG1.mesh().topological_dimension() == 3:
                self.coords_kernel = coords_kernel_3d
            else:
                raise NotImplementedError('This is only implemented for 2d and 3d at the moment.')



            if new_kernel:
                par_loop(find_effective_coords_kernel, dx,
                         args={"EXT_V1": (new_ext_coords, READ),
                               "ACT_COORDS": (self.orig_coords, READ),
                               "EFF_COORDS": (self.new_coords, RW)})
            else:
                par_loop(self.coords_kernel, dx,
                         args={"EXT_V1": (ext_DG1, READ),
                               "EXT_V0": (ext_V0_CG1, READ),
                               "NEW_COORDS": (self.new_coords, RW),
                               "OLD_COORDS": (self.orig_coords, READ)})

            VDG0 = FunctionSpace(mesh, "DG", 0)
            DG0_V0_num = Function(VDG0)
            DG0_CG1_num = Function(VDG0)

            num_kernel = """
            DG0[0][0] = 0;
            for (int i=0; i<%d; i++) {
            DG0[0][0] += EXT_V0[i][0];}
            """ % self.ext_V0_CG1.function_space().finat_element.space_dimension()

            par_loop(num_kernel, dx,
                     args={"DG0": (DG0_V0_num, RW),
                           "EXT_V0": (ext_V0_CG1, READ)})

            par_loop(num_kernel, dx,
                     args={"DG0": (DG0_CG1_num, RW),
                           "EXT_V0": (ext_DG1, READ)})

            meshDG0 = VDG0.ufl_domain()
            WDG0 = VectorFunctionSpace(meshDG0, VDG0.ufl_element())
            coordsDG0 = interpolate(meshDG0.coordinates, WDG0)
            meshCG1 = VCG1.ufl_domain()
            WCG1 = VectorFunctionSpace(meshCG1, VCG1.ufl_element())
            coordsCG1 = interpolate(meshCG1.coordinates, WCG1)

            # print('Look at V0 exterior numbers')
            # for (coord, value0, value1) in zip(coordsDG0.dat.data[:], DG0_V0_num.dat.data[:], DG0_CG1_num.dat.data[:]):
            #     print('[%.2f, %.2f %.2f] %i %i ' % (coord[0], coord[1], coord[2], value0, value1))

            # print('Look at exterior values', len(self.orig_coords.dat.data[:]), len(ext_V0_CG1.dat.data[:]))
            # for (coord, value) in zip(self.orig_coords.dat.data[:], ext_V0_CG1.dat.data[:]):
            #     print('[%.2f, %.2f] %.2f' % (coord[0], coord[1], value))

            # for (i, j) in zip(self.orig_coords.dat.data[:], self.new_coords.dat.data[:]):
            #     if i[0] < 0.09:
            #         print('[%.2f, %.2f] [%.2f, %.2f]' % (i[0], i[1], j[0], j[1]))
            #     elif i[1] < 0.09:
            #         print('[%.2f, %.2f] [%.2f, %.2f]' % (i[0], i[1], j[0], j[1]))
            #     elif i[0] > 0.91:
            #         print('[%.2f, %.2f] [%.2f, %.2f]' % (i[0], i[1], j[0], j[1]))
            #     elif i[1] > 0.91:
            #         print('[%.2f, %.2f] [%.2f, %.2f]' % (i[0], i[1], j[0], j[1]))

            # for (i, j) in zip(self.orig_coords.dat.data[:], self.new_coords.dat.data[:]):
            #     if i[0] > 75 and i[1] > 7.5 and i[2] > 0.75:
            #         print('[%.2f, %.2f %.2f] [%.2f, %.2f %.2f]' % (i[0], i[1], i[2], j[0], j[1], j[2]))


            gaussian_elimination_kernel = """
            int n = %d;
            /* do gaussian elimination to find constants in linear expansion */
            /* trying to solve A*a = f for a, where A is a matrix */
            double A[%d][%d], a[%d], f[%d], c;
            double A_max, temp_A, temp_f;
            int i_max, i, j, k;

            /* find number of exterior nodes per cell */
            int sum_V1_ext = 0;
            for (int i=0; i<%d; i++) {
            sum_V1_ext += round(EXT_V1[i][0]);}

            /* ask if there are any exterior nodes */
            if (sum_V1_ext > 0) {

            /* fill A and f with their values */
            for (i=0; i<%d; i++) {
            f[i] = DG1[i][0];
            A[i][0] = 1.0;
            A[i][1] = NEW_COORDS[i][0];
            A[i][2] = NEW_COORDS[i][1];
            A[i][3] = NEW_COORDS[i][0] * NEW_COORDS[i][1];
            if (n == 8){
            A[i][4] = NEW_COORDS[i][2];
            A[i][5] = NEW_COORDS[i][0] * NEW_COORDS[i][2];
            A[i][6] = NEW_COORDS[i][1] * NEW_COORDS[i][2];
            A[i][7] = NEW_COORDS[i][0] * NEW_COORDS[i][1] * NEW_COORDS[i][2];}}

            /* do Gaussian elimination */
            for (i=0; i<%d-1; i++) {
            /* loop through rows and columns */
            A_max = fabs(A[i][i]);
            i_max = i;

            /* find max value in ith column */
            for (j=i+1; j<%d; j++){
            if (fabs(A[j][i]) > A_max) { /* loop through rows below ith row */
            A_max = fabs(A[j][i]);
            i_max = j;}}

            /* swap rows to get largest value in ith column on top */
            if (i_max != i){
            temp_f = f[i];
            f[i] = f[i_max];
            f[i_max] = temp_f;
            for (k=i; k<%d; k++) {
            temp_A = A[i][k];
            A[i][k] = A[i_max][k];
            A[i_max][k] = temp_A;}}

            /* now scale rows below to eliminate lower diagonal values */
            for (j=i+1; j<%d; j++) {
            c = -A[j][i] / A[i][i];
            for (k=i; k<%d; k++){
            A[j][k] += c * A[i][k];}
            f[j] += c * f[i];}}

            /* do back-substitution to acquire solution */
            for (i=0; i<%d; i++){
            j = n-i-1;
            a[j] = f[j];
            for(k=j+1; k<=%d; ++k) {
            a[j] -= A[j][k] * a[k];}
            a[j] = a[j] / A[j][j];
            }

            /* extrapolate solution using new coordinates */
            for (i=0; i<%d; i++) {
            if (n == 4){
            DG1[i][0] = a[0] + a[1]*OLD_COORDS[i][0] + a[2]*OLD_COORDS[i][1] + a[3]*OLD_COORDS[i][0]*OLD_COORDS[i][1];
            }
            else if (n == 8){
            DG1[i][0] = a[0] + a[1]*OLD_COORDS[i][0] + a[2]*OLD_COORDS[i][1] + a[4]*OLD_COORDS[i][2] + a[3]*OLD_COORDS[i][0]*OLD_COORDS[i][1] + a[5]*OLD_COORDS[i][0]*OLD_COORDS[i][2] + a[6]*OLD_COORDS[i][1]*OLD_COORDS[i][2] + a[7]*OLD_COORDS[i][0]*OLD_COORDS[i][1]*OLD_COORDS[i][2];
            }}
            }
            /* do nothing if there are no exterior nodes */
            """ % ((self.v_DG1.function_space().finat_element.space_dimension(), )*15 )

            if VuCG1.mesh().topological_dimension() == 2 or VuCG1.mesh().topological_dimension() == 3:
                self.boundary_kernel = gaussian_elimination_kernel
            else:
                raise NotImplementedError('This is only implemented for 2d and 3d at the moment.')

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
        if self.method == 'physics':
            par_loop(self.bottom_kernel, dx,
                     args={"DG1": (self.v_DG1, RW),
                           "CG1": (self.v_CG1, READ)},
                     iterate=ON_BOTTOM)

            par_loop(self.top_kernel, dx,
                     args={"DG1": (self.v_DG1, RW),
                           "CG1": (self.v_CG1, READ)},
                     iterate=ON_TOP)
        else:
            par_loop(self.boundary_kernel, dx,
                     args={"DG1": (self.v_DG1, RW),
                           "OLD_COORDS": (self.orig_coords, READ),
                           "NEW_COORDS": (self.new_coords, READ),
                           "EXT_V1" : (self.ext_DG1, READ)})

            # print('OUTPUT AFTER RECOVERY')
            # for (i, j, l, ext) in zip(self.orig_coords.dat.data[:], self.new_coords.dat.data[:], self.v_DG1.dat.data[:], self.ext_DG1.dat.data[:]):
            #     if (i[0] >= 0.89 or i[0] <= 0.11) and (i[1] >= 0.89 or i[1] <= 0.11):
            #         print('[%.2f, %.2f] [%.2f, %.2f] %.4f %.2f' % (i[0], i[1], j[0], j[1], l, ext))


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
                raise ValueError("Specified boundary_method %s not valid" % boundary_method)
            if VDG is None:
                raise ValueError("If boundary_method is specified, VDG also needs specifying.")

            # now specify things that we'll need if we are doing boundary recovery
            if boundary_method == 'physics':
                # check dimensions
                if self.V.value_size != 1:
                    raise ValueError('This method only works for scalar functions.')
                self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v, method='physics')
            else:
                # STRATEGY
                # We need to pass the boundary recoverer two fields denoting the location
                # of nodes on the boundary, which will be 1 on the exterior and 0 otherwise.
                # (a) an exterior field in DG1, the space the boundary recoverer recovers
                #     into. This is done by straightforwardly applying DirichletBCs.
                # (b) an exterior field in the space of the original field, which we will
                #     call V0. There are various steps involved in doing this:
                #     1. Obtain the exterior field in CG2 (scalar or vector) by applying
                #        the Dirichlet BCs. CG2 is required as it will contain all the DOFs
                #        of V0 directly.
                #     2. Project this field into V0 to get a representation of the field in
                #        V0. Ideally we would do interpolation here rather than projection,
                #        but interpolation is not supported into our velocity fields.
                #     3. We can now perfectly represent this exterior V0 field in CG1 (scalar
                #        or vector), by interpolation. CG1 should be larger than any field we
                #        need to recover.
                #     4. As there may continuity between basis functions in V0, the projection
                #        will not necessarily have preserved the values of 0 or 1. We now do a
                #        hack, modifying the points individually so that they are 0 or 1. We
                #        have found this works if the projected value is below or above 0.5.

                mesh = self.V.mesh()
                V0_brok = FunctionSpace(mesh, BrokenElement(self.v_in.ufl_element()))
                VDG1 = FunctionSpace(mesh, "DG", 1)
                VCG1 = FunctionSpace(mesh, "CG", 1)
                ext_DG1 = Function(VDG1)
                bcs = [DirichletBC(VDG1, Constant(1.0), "top", method="geometric"),
                       DirichletBC(VDG1, Constant(1.0), "bottom", method="geometric"),
                       DirichletBC(VDG1, Constant(1.0), "on_boundary", method="geometric")]
                for bc in bcs:
                    bc.apply(ext_DG1)

                if boundary_method == 'scalar':
                    # check dimensions
                    if self.V.value_size != 1:
                        raise ValueError('This method only works for scalar functions.')

                    # make exterior CG2 field (we need this to interpolate for general temperature space)
                    V0 = self.v_in.function_space()

                    VCG2 = FunctionSpace(mesh, "CG", 2)
                    exterior_CG2 = Function(VCG2)
                    bcs = [DirichletBC(VCG2, Constant(1.0), "top", method="geometric"),
                           DirichletBC(VCG2, Constant(1.0), "bottom", method="geometric"),
                           DirichletBC(VCG2, Constant(1.0), "on_boundary", method="geometric")]

                    for bc in bcs:
                        bc.apply(exterior_CG2)

                    exterior_V0 = Function(V0_brok).project(exterior_CG2)
                    ext_V0_CG1 = Function(VCG1).interpolate(exterior_V0)

                    for (i, p) in enumerate(ext_V0_CG1.dat.data[:]):
                        if p > 0.5:
                            ext_V0_CG1.dat.data[i] = 1
                        else:
                            ext_V0_CG1.dat.data[i] = 0

                    x = SpatialCoordinate(mesh)
                    VuDG1 = VectorFunctionSpace(mesh, "DG", 1)
                    coords_DG1 = Function(VuDG1).project(x)
                    new_ext_coords = find_coords_to_correct(V0_brok, VDG1, VCG1, VCG2, coords_DG1)

                    self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v,
                                                                 ext_DG1=ext_DG1, ext_V0_CG1=ext_V0_CG1,
                                                                 method='dynamics', new_ext_coords=new_ext_coords)
                elif boundary_method == 'vector':
                    # check dimensions
                    if self.V.value_size != 2 and self.V.value_size != 3:
                        raise NotImplementedError('This method only works for 2D or 3D vector functions.')

                    VuCG1 = VectorFunctionSpace(mesh, "CG", 1)
                    VuCG2 = VectorFunctionSpace(mesh, "CG", 2)
                    exterior_VuCG2 = Function(VuCG2)
                    bcs = [DirichletBC(VuCG2, Constant(1.0), "top", method="geometric"),
                           DirichletBC(VuCG2, Constant(1.0), "bottom", method="geometric"),
                           DirichletBC(VuCG2, Constant(1.0), "on_boundary", method="geometric")]

                    for bc in bcs:
                        bc.apply(exterior_VuCG2)

                    exterior_V0 = Function(V0_brok).project(exterior_VuCG2)
                    ext_V0_VuCG1 = Function(VuCG1).interpolate(exterior_V0)

                    for (i, point) in enumerate(ext_V0_VuCG1.dat.data[:]):
                        for (j, p) in enumerate(point):
                            if p > 0.5:
                                ext_V0_VuCG1.dat.data[i][j] = 1
                            else:
                                ext_V0_VuCG1.dat.data[i][j] = 0

                    x = SpatialCoordinate(mesh)
                    VuDG1 = VectorFunctionSpace(mesh, "DG", 1)
                    coords_DG1 = Function(VuDG1).project(x)
                    new_ext_coords_vec = find_coords_to_correct(V0_brok, VuDG1, VuCG1, VuCG2, coords_DG1)

                    # now, break the problem down into components
                    v_scalars = []
                    v_out_scalars = []
                    ext_V0_CG1s = []
                    self.boundary_recoverers = []
                    self.project_to_scalars_CG = []
                    self.extra_averagers = []
                    new_ext_coords_list = []
                    for i in range(self.V.value_size):
                        v_scalars.append(Function(VDG1))
                        v_out_scalars.append(Function(VCG1))
                        ext_V0_CG1s.append(Function(VCG1).project(ext_V0_VuCG1[i]))
                        new_ext_coords_list.append(Function(VDG1).project(new_ext_coords_vec[i]))
                        self.project_to_scalars_CG.append(Projector(self.v_out[i], v_out_scalars[i]))
                        self.boundary_recoverers.append(Boundary_Recoverer(v_out_scalars[i], v_scalars[i],
                                                                           ext_DG1=ext_DG1,
                                                                           ext_V0_CG1=ext_V0_CG1s[i],
                                                                           method='dynamics',
                                                                           new_ext_coords=new_ext_coords_list[i]))
                        # need an extra averager that works on the scalar fields rather than the vector one
                        self.extra_averagers.append(Averager(v_scalars[i], v_out_scalars[i]))

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
            if self.boundary_method == 'vector':
                for i in range(self.V.value_size):
                    self.project_to_scalars_CG[i].project()
                    self.boundary_recoverers[i].apply()
                    self.extra_averagers[i].project()
                self.project_to_vector.project()
            elif self.boundary_method == 'scalar' or self.boundary_method == 'physics':
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


def find_coords_to_correct(V0_brok, DG1, CG1, CG2, coords):

    # make DG1 field with 1 at all exterior coords
    all_ext_in_DG1 = Function(DG1)

    bcs = [DirichletBC(DG1, Constant(1.0), "top", method="geometric"),
           DirichletBC(DG1, Constant(1.0), "bottom", method="geometric"),
           DirichletBC(DG1, Constant(1.0), "on_boundary", method="geometric")]

    for bc in bcs:
        bc.apply(all_ext_in_DG1)

    # make DG1 field with 1 at coords surrounding exterior coords of V0
    # first do it in CG2, as this should contain all DOFs of V0 and DG1
    all_ext_in_CG2 = Function(CG2)
    bcs = [DirichletBC(CG2, Constant(1.0), "top", method="geometric"),
           DirichletBC(CG2, Constant(1.0), "bottom", method="geometric"),
           DirichletBC(CG2, Constant(1.0), "on_boundary", method="geometric")]

    for bc in bcs:
        bc.apply(all_ext_in_CG2)

    approx_ext_in_V0 = Function(V0_brok).project(all_ext_in_CG2)
    approx_V0_ext_in_CG1 = Function(CG1).interpolate(approx_ext_in_V0)#is this step necessary?

    # now do horrible hack to get back to 1s and 0s
    for (i, point) in enumerate(approx_V0_ext_in_CG1.dat.data[:]):
        try:
            for (j, p) in enumerate(point):
                if p > 0.5:
                    approx_V0_ext_in_CG1.dat.data[i][j] = 1
                else:
                    approx_V0_ext_in_CG1.dat.data[i][j] = 0
        except:
            if point > 0.5:
                approx_V0_ext_in_CG1.dat.data[i] = 1
            else:
                approx_V0_ext_in_CG1.dat.data[i] = 0

    approx_V0_ext_in_DG1 = Function(DG1).interpolate(approx_V0_ext_in_CG1)

    approx_V0_ext_in_DG1 = Function(DG1).interpolate(approx_ext_in_V0)#is this step necessary?

    # now do horrible hack to get back to 1s and 0s
    for (i, point) in enumerate(approx_V0_ext_in_DG1.dat.data[:]):
        try:# this try needs changing to something neater
            for (j, p) in enumerate(point):
                if p > 0.5:
                    approx_V0_ext_in_DG1.dat.data[i][j] = 1
                else:
                    approx_V0_ext_in_DG1.dat.data[i][j] = 0
        except:
            if point > 0.5:
                approx_V0_ext_in_DG1.dat.data[i] = 1
            else:
                approx_V0_ext_in_DG1.dat.data[i] = 0

    corners_in_DG1 = Function(DG1)
    if DG1.value_size == 1:
        ones = Function(DG1).interpolate(Constant(1.0))
    elif DG1.value_size == 2:
        ones = Function(DG1).interpolate(as_vector([Constant(1.0), Constant(1.0)]))
    elif DG1.value_size == 3:
        ones = Function(DG1).interpolate(as_vector([Constant(1.0), Constant(1.0), Constant(1.0)]))
    if DG1.mesh().topological_dimension() == 2:
        if DG1.extruded:
            DG1_ext_hori = Function(DG1)
            DG1_ext_vert = Function(DG1)
            hori_bcs = [DirichletBC(DG1, Constant(1.0), "top", method="geometric"),
                        DirichletBC(DG1, Constant(1.0), "bottom", method="geometric")]
            vert_bc = DirichletBC(DG1, Constant(1.0), "on_boundary", method="geometric")
            for bc in hori_bcs:
                bc.apply(DG1_ext_hori)

            vert_bc.apply(DG1_ext_vert)
            corners_in_DG1.assign(DG1_ext_hori + DG1_ext_vert - all_ext_in_DG1)

        else:
            # we don't know whether its periodic or in how many directions
            DG1_ext_x = Function(DG1)
            DG1_ext_y = Function(DG1)
            x_bcs = [DirichletBC(DG1, Constant(1.0), 1, method="geometric"),
                     DirichletBC(DG1, Constant(1.0), 2, method="geometric")]
            y_bcs = [DirichletBC(DG1, Constant(1.0), 3, method="geometric"),
                     DirichletBC(DG1, Constant(1.0), 4, method="geometric")]
            try:
                for bc in x_bcs:
                    bc.apply(DG1_ext_x)
            except:
                pass

            try:
                for bc in y_bcs:
                    bc.apply(DG1_ext_y)
            except:
                pass

            corners_in_DG1.assign(DG1_ext_x + DG1_ext_y - all_ext_in_DG1)

    elif DG1.mesh().topological_dimension() == 3:
        DG1_vert_x = Function(DG1)
        DG1_vert_y = Function(DG1)
        DG1_hori = Function(DG1)
        x_bcs = [DirichletBC(DG1, Constant(1.0), 1, method="geometric"),
                 DirichletBC(DG1, Constant(1.0), 2, method="geometric")]
        y_bcs = [DirichletBC(DG1, Constant(1.0), 3, method="geometric"),
                 DirichletBC(DG1, Constant(1.0), 4, method="geometric")]
        hori_bcs = [DirichletBC(DG1, Constant(1.0), "top", method="geometric"),
                    DirichletBC(DG1, Constant(1.0), "bottom", method="geometric")]
        for bc in hori_bcs:
            bc.apply(DG1_hori)

        for bc in x_bcs:
            bc.apply(DG1_vert_x)

        for bc in y_bcs:
            bc.apply(DG1_vert_y)

        try:
            for bc in x_bcs:
                bc.apply(DG1_vert_x)
        except:
            pass

        try:
            for bc in y_bcs:
                bc.apply(DG1_vert_y)
        except:
            pass

        corners_in_DG1.assign(DG1_vert_x + DG1_vert_y + DG1_hori - all_ext_in_DG1)

    coords_to_correct = Function(DG1).assign(corners_in_DG1 + all_ext_in_DG1 - approx_V0_ext_in_DG1)
    for (i, point) in enumerate(coords_to_correct.dat.data[:]):
        try:
            for (j, p) in enumerate(point):
                if p > 0.5:
                    coords_to_correct.dat.data[i][j] = 1
                else:
                    coords_to_correct.dat.data[i][j] = 0
        except:
            if point > 0.5:
                coords_to_correct.dat.data[i] = 1
            else:
                coords_to_correct.dat.data[i] = 0

    # for (coord, i) in zip(coords.dat.data[:], coords_to_correct.dat.data[:]):
    #     if DG1.mesh().topological_dimension() == 2:
    #         if DG1.value_size == 1:
    #             print('[%.2f %.2f] %.2f' % (coord[0], coord[1], i))
    #         elif DG1.value_size == 2:
    #             print('[%.2f %.2f] %.2f %.2f' % (coord[0], coord[1], i[0], i[1]))
    #     elif DG1.mesh().topological_dimension() == 3:
    #         if DG1.value_size == 1:
    #             print('[%.2f %.2f %.2f] %.2f' % (coord[0], coord[1], coord[2], i))
    #         elif DG1.value_size == 3:
    #             print('[%.2f %.2f %.2f] %.2f %.2f %.2f' % (coord[0], coord[1], coord[2], i[0], i[1], i[2]))

    return coords_to_correct
