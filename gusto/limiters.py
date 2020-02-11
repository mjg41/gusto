from firedrake import (dx, BrokenElement, Function, FunctionSpace,
                       FiniteElement, TensorProductElement, interval,
                       Interpolator)
from firedrake.parloops import par_loop, READ, WRITE
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter

__all__ = ["ThetaLimiter", "NoLimiter"]


class ThetaLimiter(object):
    """
    A vertex based limiter for fields in the DG1xCG2 space,
    i.e. temperature variables. This acts like the vertex-based
    limiter implemented in Firedrake, but in addition corrects
    the central nodes to prevent new maxima or minima forming.
    """

    def __init__(self, space):
        """
        Initialise limiter
        :param space: the space in which theta lies.
        It should be the DG1xCG2 space.
        """

        mesh = space.mesh()
        self.Vt = FunctionSpace(mesh, BrokenElement(space.ufl_element()))

        # check that the space is the DG1 x CG2 space
        if not self.Vt.extruded:
            raise ValueError('This is not the right limiter for this space.')
        cell = mesh._base_mesh.ufl_cell().cellname()
        w_hori = FiniteElement("DG", cell, 1)
        w_vert = FiniteElement("CG", interval, 2)
        theta_elt = TensorProductElement(w_hori, w_vert)
        true_Vt = FunctionSpace(mesh, theta_elt)
        if true_Vt != space:
            raise ValueError('This is not the right limiter for this space.')

        DG_hori = FiniteElement("DG", cell, 1, variant="equispaced")
        DG_vert = FiniteElement("DG", interval, 1, variant="equispaced")
        DG_elt = TensorProductElement(DG_hori, DG_vert)
        self.DG1 = FunctionSpace(self.Vt.mesh(), DG_elt)  # space with only vertex DOFs
        self.vertex_limiter = VertexBasedLimiter(self.DG1)
        self.theta_old = Function(self.Vt)
        self.theta_after = Function(self.Vt)
        self.theta_DG1 = Function(self.DG1)  # theta function with correct vertex DOFs
        self.DG1_vertex_interpolator = Interpolator(self.theta_old, self.theta_DG1)
        self.Vt_vertex_interpolator = Interpolator(self.theta_DG1, self.theta_after)

        shapes = {'nDOFs_base': int(self.Vt.finat_element.space_dimension() / 3)}
        theta_domain = "{{[i,j]: 0 <= i < {nDOFs_base} and 0 <= j < 2}}".format(**shapes)

        adjust_values_instrs = ("""
                                <float64> max_value = 0.0
                                <float64> min_value = 0.0
                                for i
                                    theta[i*3] = theta_aft[i*3]
                                    theta[i*3+2] = theta_aft[i*3+2]

                                    max_value = fmax(theta_aft[i*3], theta_aft[i*3+2])
                                    min_value = fmin(theta_aft[i*3], theta_aft[i*3+2])
                                    if theta_old[i*3+1] > max_value
                                        theta[i*3+1] = 0.5 * (theta_aft[i*3] + theta_aft[i*3+2])
                                    elif theta_old[i*3+1] < min_value
                                        theta[i*3+1] = 0.5 * (theta_aft[i*3] + theta_aft[i*3+2])
                                    else
                                        theta[i*3+1] = theta_old[i*3+1]
                                    end
                                end
                                """)
        self._adjust_values_kernel = (theta_domain, adjust_values_instrs)

    def adjust_values(self, field):
        """
        Copies the vertex values back from the DG1 space to
        the original temperature space, and checks that the
        midpoint values are within the minimum and maximum
        at the adjacent vertices.
        If outside of the minimum and maximum, correct the values
        to be the average.
        """
        par_loop(self._adjust_values_kernel, dx,
                 {"theta": (field, WRITE),
                  "theta_aft": (self.theta_after, READ),
                  "theta_old": (self.theta_old, READ)},
                 is_loopy_kernel=True)

    def apply(self, field):
        """
        The application of the limiter to the theta-space field.
        """
        assert field.function_space() == self.Vt, \
            'Given field does not belong to this objects function space'

        self.theta_old.assign(field)
        self.DG1_vertex_interpolator.interpolate()
        self.vertex_limiter.apply(self.theta_DG1)
        self.Vt_vertex_interpolator.interpolate()
        self.adjust_values(field)


class NoLimiter(object):
    """
    A blank limiter that does nothing.
    """

    def __init__(self):
        """
        Initialise the blank limiter.
        """
        pass

    def apply(self, field):
        """
        The application of the blank limiter.
        """
        pass
