from abc import ABCMeta, abstractproperty
import functools
import operator
from firedrake import (Function, TestFunction, inner, dx, div, action,
                       SpatialCoordinate, sqrt, FunctionSpace,
                       MixedFunctionSpace, TestFunctions, TrialFunctions)
from gusto.configuration import logger
from gusto.form_manipulation_labelling import (all_terms, advecting_velocity,
                                               subject, time_derivative,
                                               linearisation,
                                               drop, index, advection,
                                               replace_labelled,
                                               has_labels, Term)
from gusto.diffusion import interior_penalty_diffusion_form
from gusto.transport_equation import (vector_invariant_form,
                                      continuity_form, advection_form,
                                      linear_continuity_form,
                                      advection_equation_circulation_form,
                                      advection_vector_manifold_form,
                                      kinetic_energy_form)
from gusto.state import build_spaces


class PrognosticEquation(object, metaclass=ABCMeta):
    """
    Base class for prognostic equations

    :arg state: :class:`.State` object
    :arg function space: :class:`.FunctionSpace` object, the function
         space that the equation is defined on
    :arg field_names: name(s) of the prognostic field(s)

    The class sets up the fields in state and registers them with the
    diagnostics class. It defines a mass term, labelled with the
    time_derivative label. All remaining forms must be defined in the
    child class form method. Calling this class returns the form
    mass_term + form
    """
    def __init__(self, state, function_space, field_name):

        self.state = state
        self.function_space = function_space
        self.field_name = field_name

        # default is to dump the field unless user has specified
        # otherwise when setting up the output parameters
        dump = state.output.dumplist or True
        state.fields(field_name, space=function_space, dump=dump, pickup=True)

        state.diagnostics.register(field_name)

    def mass_term(self):
        """
        Returns the labelled form for the mass term for self.function_space
        with its subject labelled and the time_derivative label applied.
        """
        test = TestFunction(self.function_space)
        q = Function(self.function_space)
        return subject(time_derivative(inner(q, test)*dx), q)

    @abstractproperty
    def form(self):
        """
        Child classes should define the forms for the remaining terms
        in the equation.
        """
        pass

    def __call__(self):
        """
        Returns the form for this equation
        """
        return self.mass_term() + self.form()


class AdvectionEquation(PrognosticEquation):
    """
    Class defining the advection equation.

    :arg state: :class:`.State` object
    :arg function_space: :class:`.FunctionSpace` object, the function
    :arg field_name: name of the prognostic field
    :kwargs: any kwargs to be passed on to the advection_form
    """
    def __init__(self, state, function_space, field_name,
                 **kwargs):
        super().__init__(state, function_space, field_name)
        self.kwargs = kwargs

    def form(self):
        return advection_form(self.state, self.function_space, **self.kwargs)


class ContinuityEquation(PrognosticEquation):
    """
    Class defining the continuity equation.

    :arg state: :class:`.State` object
    :arg function_space: :class:`.FunctionSpace` object, the function
    :arg field_name: name of the prognostic field
    :kwargs: any kwargs to be passed on to the continuity_form
    """

    def __init__(self, state, function_space, field_name,
                 **kwargs):
        super().__init__(state, function_space, field_name)
        self.kwargs = kwargs

    def form(self):
        return continuity_form(self.state, self.function_space, **self.kwargs)


class DiffusionEquation(PrognosticEquation):
    """
    Class defining the diffusion equation.

    :arg state: :class:`.State` object
    :arg function_space: :class:`.FunctionSpace` object, the function
    :arg field_name: name of the prognostic field
    :kwargs: any kwargs to be passed on to the diffuson_form
    """

    def __init__(self, state, function_space, field_name, **kwargs):

        super().__init__(state, function_space, field_name)
        self.kwargs = kwargs

    def form(self):
        return interior_penalty_diffusion_form(self.state, self.function_space, **self.kwargs)


class AdvectionDiffusionEquation(PrognosticEquation):
    """
    Class defining the advection-diffusion equation.

    :arg state: :class:`.State` object
    :arg field_name: name of the prognostic field
    :arg function_space: :class:`.FunctionSpace` object, the function
    :kwargs: any kwargs to be passed on to the advection_form or diffusion_form
    """
    def __init__(self, state, function_space, field_name, **kwargs):
        super().__init__(state, function_space, field_name)
        self.diff_kwargs = {}
        for k in ["kappa", "mu"]:
            assert k in kwargs.keys(), "diffusion form requires %s kwarg " % k
            self.diff_kwargs[k] = kwargs.pop(k)
        self.adv_kwargs = kwargs

    def form(self):
        return advection_form(self.state, self.function_space, **self.adv_kwargs) + interior_penalty_diffusion_form(self.state, self.function_space, **self.diff_kwargs)


class PrognosticMixedEquation(PrognosticEquation):
    """
    Base class for the equation set defined on a mixed function space.
    Child classes must define their fields and solver parameters for
    the mixed system.

    :arg state: :class:`.State` object
    :arg function space: :class:`.FunctionSpace` object, the function
         space that the equations are defined on - this should be a
         mixed function space

    The class sets up the fields in state and registers them with the
    diagnostics class. It defines a mass term, labelled with the
    time_derivative label. All remaining forms must be defined in the
    child class form method. Calling this class returns the form
    mass_term + form
    """

    def __init__(self, state, function_space):

        self.state = state
        self.function_space = function_space

        assert len(function_space) == len(self.fields), "size of function space and number of fields should match"

        # default is to dump all fields unless user has specified
        # otherwise when setting up the output parameters
        dump = state.output.dumplist or self.fields
        state.fields(self.field_name, *self.fields, space=function_space,
                     dump=dump, pickup=True)

        state.diagnostics.register(*self.fields)

    def mass_term(self):
        """
        Returns the labelled form for the mass term for self.function_space
        with its subject labelled and the time_derivative label applied.
        Also labels each term with the index of the space it is defined on.
        """
        tests = TestFunctions(self.function_space)
        qs = Function(self.function_space)
        return functools.reduce(
            operator.add,
            (index(subject(
                time_derivative(inner(q, test)*dx), qs), tests.index(test))
             for q, test in zip(qs.split(), tests)))

    @abstractproperty
    def field_name(self):
        """
        Child classes must define a name to use to access the mixed
        prognostic fields
        """
        pass

    @abstractproperty
    def fields(self):
        """
        Child classes must define a list of their prognostic field names.
        """
        pass

    @abstractproperty
    def solver_parameters(self):
        """
        Child classes must define default solver parameters for the
        mixed system.
        """
        pass


class ShallowWaterEquations(PrognosticMixedEquation):
    """
    Class defining the shallow water equations.

    :arg state: :class:`.State` object
    :arg family: str, specifies the velocity space family to use
    :arg degree: int, specifies the degree for the depth space
    :kwargs: (optional) expressions for additional fields and discretisation
    options to be passed to the form

    Default behaviour:
    * velocity advection term in vector invariant form.
    * Coriolis term present and the Coriolis parameter takes the value for
    the Earth. Pass in fexpr=None for non-rotating shallow water.
    """
    field_name = "sw"

    fields = ['u', 'D']

    solver_parameters = {
        'ksp_type': 'preonly',
        'mat_type': 'matfree',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {'ksp_type': 'cg',
                          'pc_type': 'gamg',
                          'ksp_rtol': 1e-8,
                          'mg_levels': {'ksp_type': 'chebyshev',
                                        'ksp_max_it': 2,
                                        'pc_type': 'bjacobi',
                                        'sub_pc_type': 'ilu'}}
    }

    def __init__(self, state, family, degree, **kwargs):

        fexpr = kwargs.pop("fexpr", "default")
        bexpr = kwargs.pop("bexpr", None)
        self.u_advection_option = kwargs.pop("u_advection_option", "vector_invariant_form")
        if kwargs:
            raise ValueError("unexpected kwargs: %s" % list(kwargs.keys()))

        # define the function spaces
        Vu, VD = build_spaces(state, family, degree)
        W = MixedFunctionSpace((Vu, VD))

        super().__init__(state, W)

        # setup optional coriolis and topography terms, default is for
        # the Coriolis term to be that for the Earth.
        if fexpr:
            self.setup_coriolis(state, fexpr)
        if bexpr:
            self.setup_topography(state, bexpr)

    def setup_coriolis(self, state, fexpr):
        """
        Create state.fields("coriolis").
        """
        if fexpr == "default":
            Omega = state.parameters.Omega
            x = SpatialCoordinate(state.mesh)
            R = sqrt(inner(x, x))
            fexpr = 2*Omega*x[2]/R

        V = FunctionSpace(state.mesh, "CG", 1)
        f = state.fields("coriolis", space=V)
        f.interpolate(fexpr)

    def setup_topography(self, state, bexpr):
        """
        Create state.fields("topography").
        """
        b = state.fields("topography", space=state.fields("D").function_space())
        b.interpolate(bexpr)

    def form(self):

        state = self.state
        g = state.parameters.g
        H = state.parameters.H

        W = self.function_space
        w, phi = TestFunctions(W)
        trials = TrialFunctions(W)
        X = Function(W)
        u, D = X.split()

        # define velocity advection term
        if self.u_advection_option == "vector_invariant_form":
            u_adv = vector_invariant_form(state, W, 0)
        elif self.u_advection_option == "vector_advection":
            u_adv = advection_vector_manifold_form(state, W, 0)
        elif self.u_advection_option == "circulation_form":
            ke_form = kinetic_energy_form(state, W, 0)
            ke_form = advection.remove(ke_form)
            ke_form = ke_form.label_map(all_terms,
                                        replace_labelled(subject,
                                                         advecting_velocity))
            u_adv = advection_equation_circulation_form(state, W, 0) + ke_form
        else:
            raise ValueError("Invalid u_advection_option: %s" % self.u_advection_option)

        # define pressure gradient term and its linearisation
        pressure_gradient_term = subject(-g*div(w)*D*dx, X)
        linear_pg_term = pressure_gradient_term.label_map(
            all_terms, replace_labelled(trials, subject))

        # the base form for u contains the velocity advection term and
        # the pressure gradient term
        u_form = u_adv + linearisation(pressure_gradient_term, linear_pg_term)

        # add any additional terms
        for field_name in ["coriolis", "topography"]:
            try:
                field = state.fields(field_name)
                add_term = True
            except AttributeError:
                logger.info("field %s not present" % field_name)
                add_term = False

            if add_term:
                if field_name == "coriolis":
                    # define the coriolis term and its linearisation
                    coriolis_term = subject(field*inner(w, state.perp(u))*dx, X)
                    linear_coriolis_term = coriolis_term.label_map(
                        all_terms, replace_labelled(trials, subject))
                    # add on the coriolis term
                    u_form += linearisation(coriolis_term, linear_coriolis_term)

                elif field_name == "topography":
                    # add on the topography term - the linearisation
                    # is not defined as we don't usually make it part
                    # of the linear solver, However, this will have to
                    # be defined when we start using exponential
                    # integrators.
                    u_form += -g*div(w)*field*dx

        # define the depth continuity term and its linearisation
        Dadv = continuity_form(state, W, 1)
        Dadv_linear = linear_continuity_form(state, W, 1, qbar=H).label_map(
            all_terms, replace_labelled(trials, subject, advecting_velocity))
        D_form = linearisation(Dadv, Dadv_linear)

        return index(u_form, 0) + index(D_form, 1)


class LinearShallowWaterEquations(ShallowWaterEquations):
    """
    Class defining the linear shallow water equations.
    """

    def form(self):

        # get shallow water equation form
        sw_form = super().form()

        # grab the linearisation of each term (a bilinear form) and
        # apply to the term's subject to get the linear form
        linear_form = sw_form.label_map(
            has_labels(linearisation),
            lambda t: Term(action(t.get(linearisation).form, t.get(subject)),
                           t.labels),
            drop)

        return linear_form
