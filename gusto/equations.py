from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict
from firedrake import TestFunction, TrialFunction, FiniteElement, \
    MixedFunctionSpace, FunctionSpace, SpatialCoordinate, sqrt, BrokenElement
from gusto.diagnostics import Diagnostics
from gusto.terms import *
from gusto.transport_equation import *


class Equation(object, metaclass=ABCMeta):

    def __init__(self, function_space):
        self.function_space = function_space
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
                 name=None, uexpr=None,
                 discretisation_option=None,
                 **kwargs):

        # save discretisation option for use in advection scheme
        self.discretisation_option = discretisation_option
        if discretisation_option == "embedded_DG":
            Vdg = kwargs.pop("Vdg", None)
            if Vdg is None:
                # Create broken space
                V_elt = BrokenElement(function_space.ufl_element())
                V = state.spaces("broken", state.mesh, V_elt)
            else:
                V = Vdg
        else:
            V = function_space

        super().__init__(V)

        # if the name of the field is provided then this field needs
        # to be created and added to the Diagnostics class
        if name:
            state.fields(name, function_space)
            if hasattr(state, "diagnostics"):
                state.diagnostics.register(name)
            else:
                state.diagnostics = Diagnostics(name)

        u_space = state.spaces("HDiv")
        state.fields('uadv', u_space)

        # if uexpr is provided, use it to set the advective velocity
        if uexpr:
            state.fields('u', u_space).project(uexpr)

        self.add_term(AdvectionTerm(state, V, **kwargs))


class ShallowWaterMomentumEquation(Equation):

    def __init__(self, function_space, state, opts):
        super().__init__(function_space)
        self.bcs = None
        self.add_term(ShallowWaterPressureGradientTerm(state, function_space))
        self.add_term(ShallowWaterCoriolisTerm(state, function_space))
        self.add_term(VectorInvariantTerm(state, function_space))


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
