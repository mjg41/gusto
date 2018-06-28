from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict
from firedrake import TestFunction, TrialFunction, FiniteElement, \
    MixedFunctionSpace, FunctionSpace, SpatialCoordinate, sqrt
from gusto.diagnostics import Diagnostics
from gusto.terms import *
from gusto.transport_equation import *


class Equation(object, metaclass=ABCMeta):

    def __init__(self, function_space):
        self.terms = OrderedDict()
        self.test = TestFunction(function_space)
        self.trial = TrialFunction(function_space)

    def mass_term(self, q):
        return inner(self.test, q)*dx

    def add_term(self, term):
        key = term.__class__.__name__
        self.terms[key] = term

    def __call__(self, q, fields):
        L = 0.
        for name, term in self.terms.items():
            L += term(self.test, q, fields)
        return L


class AdvectionEquation(Equation):

    def __init__(self, function_space, state,
                 name=None, u_space=None, uexpr=None,
                 **kwargs):

        super().__init__(function_space)
        if name:
            state.fields(name, function_space)
            if hasattr(state, "diagnostics"):
                state.diagnostics.register(name)
            else:
                state.diagnostics = Diagnostics(name)
        if not u_space:
            try:
                u_space = state.spaces("HDiv")
            except AttributeError:
                raise ValueError("Must specify function space for advective velocity if state does not have the usual compatible finite element function spaces setup.")
        state.fields('uadv', u_space)
        if uexpr:
            state.fields('uadv').project(uexpr)

        self.add_term(AdvectionTerm(state, **kwargs))


class ShallowWaterMomentumEquation(Equation):

    def __init__(self, function_space, state, opts):
        super().__init__(function_space)
        self.bcs = None
        self.add_term(ShallowWaterPressureGradientTerm(state))
        self.add_term(ShallowWaterCoriolisTerm(state))
        self.add_term(VectorInvariantTerm(state))


class ShallowWaterDepthEquation(AdvectionEquation):

    def __init__(self, function_space, state, opts):
        super().__init__(function_space, state, equation_form="continuity")
        self.bcs = None


class Equations(object):

    def __init__(self, state, family, degree):
        self._build_function_spaces(state.spaces, state.mesh, family, degree)
        state.fields(self.fieldlist, self.mixed_function_space)

        if hasattr(state, "diagnostics"):
            state.diagnostics.register(*self.fieldlist)
        else:
            state.diagnostics = Diagnostics(*self.fieldlist)

    @abstractproperty
    def fieldlist(self):
        pass

    @abstractproperty
    def equation_list(self):
        pass

    @abstractproperty
    def mixed_function_space(self):
        pass

    @abstractmethod
    def _build_function_spaces(self):
        pass

    @property
    def equations(self):
        return {field: eqn for (field, eqn) in zip(self.fieldlist, self.equation_list)}

    def __call__(self, field):
        return self.equations[field]


class ShallowWaterEquations(Equations):

    fieldlist = ['u', 'D']

    def __init__(self, state, family, degree, u_opts=None, D_opts=None,
                 topography_expr=None):
        super().__init__(state, family, degree)
        self.ueqn = ShallowWaterMomentumEquation(self.u_space, state, u_opts)
        self.Deqn = ShallowWaterDepthEquation(self.D_space, state, D_opts)
        V = FunctionSpace(state.mesh, "CG", 3)
        x = SpatialCoordinate(state.mesh)
        R = sqrt(inner(x, x))
        Omega = state.parameters.Omega
        state.parameters.add_field("coriolis", V, 2*Omega*x[2]/R)
        if topography_expr:
            state.parameters.add_field("topography", V, topography_expr)

    @abstractproperty
    def equation_list(self):
        return [self.ueqn, self.Deqn]

    @abstractproperty
    def mixed_function_space(self):
        return MixedFunctionSpace((self.u_space, self.D_space))

    @abstractmethod
    def _build_function_spaces(self, spaces, mesh, family, degree):
        cell = mesh.ufl_cell().cellname()
        V1_elt = FiniteElement(family, cell, degree+1)
        self.u_space = spaces("HDiv", mesh, V1_elt)
        self.D_space = spaces("DG", mesh, "DG", degree)
