from abc import ABCMeta, abstractmethod
from firedrake import FacetNormal, div, dx, inner, \
    dS, dS_h, dS_v, ds, ds_v, ds_t, ds_b


class Term(object, metaclass=ABCMeta):

    off_centering = 0.5

    def __init__(self, state, function_space):
        self.state = state
        self.function_space = function_space
        self.parameters = state.parameters
        self.n = FacetNormal(state.mesh)

    def is_cg(self, V):
        nvertex = V.ufl_domain().ufl_cell().num_vertices()
        entity_dofs = V.finat_element.entity_dofs()
        try:
            return sum(map(len, entity_dofs[0].values())) == nvertex
        except KeyError:
            return sum(map(len, entity_dofs[(0, 0)].values())) == nvertex

    @property
    def dS(self):
        if self.is_cg(self.function_space):
            return None
        else:
            if self.function_space.extruded:
                return (dS_h + dS_v)
            else:
                return dS

    @property
    def ds(self):
        if self.is_cg(self.function_space):
            return None
        else:
            if self.function_space.extruded:
                return (ds_v + ds_t + ds_b)
            else:
                return ds

    @abstractmethod
    def evaluate(self, test, q, fields):
        pass

    def __call__(self, test, q, fields):
        return self.evaluate(test, q, fields)


class ShallowWaterPressureGradientTerm(Term):

    def evaluate(self, test, q, fields):
        g = self.parameters.g
        D = fields("D")
        return g*div(test)*D*dx


class ShallowWaterCoriolisTerm(Term):

    def evaluate(self, test, q, fields):
        f = self.parameters.coriolis
        u = fields("u")
        return -f*inner(test, self.state.perp(u))*dx


class ShallowWaterTopographyTerm(Term):

    def evaluate(self, test, q, fields):
        g = self.parameters.g
        b = self.parameters("topography")
        return g*div(test)*b*dx
