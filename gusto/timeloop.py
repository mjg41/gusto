from abc import ABCMeta, abstractmethod, abstractproperty
from pyop2.profiling import timed_stage
from gusto.configuration import logger
from gusto.forcing import Forcing
from gusto.state import FieldCreator
from firedrake import DirichletBC

__all__ = ["Timestepper", "CrankNicolson", "ImplicitMidpoint"]


class BaseTimestepper(object, metaclass=ABCMeta):
    """
    Base timestepping class for Gusto

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    """

    def __init__(self, state, schemes, physics_list=None):

        self.state = state
        state.xn = state.fields
        state.xnp1 = FieldCreator()
        for field, eqn in schemes:
            state.xnp1(field, state.fields(field).function_space())
        if physics_list is not None:
            self.physics_list = physics_list
        else:
            self.physics_list = []

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        if self.state.xn("u").function_space().extruded:
            unp1 = self.state.xnp1.split()[0]
            M = unp1.function_space()
            bcs = [DirichletBC(M, 0.0, "bottom"),
                   DirichletBC(M, 0.0, "top")]

            for bc in bcs:
                bc.apply(unp1)

    def setup_timeloop(self, t, tmax, pickup):
        """
        Setup the timeloop by setting up diagnostics, dumping the fields and
        picking up from a previous run, if required
        """
        self.state.setup_diagnostics()
        with timed_stage("Dump output"):
            self.state.setup_dump(tmax, pickup)
            t = self.state.dump(t, pickup)
        return t

    @abstractmethod
    def timestep():
        pass

    def run(self, t, tmax, pickup=False):
        """
        This is the timeloop. After completing the semi implicit step
        any passively advected fields are updated, implicit diffusion and
        physics updates are applied (if required).
        """

        t = self.setup_timeloop(t, tmax, pickup)

        state = self.state
        dt = state.timestepping.dt

        while t < tmax - 0.5*dt:
            logger.info("at start of timestep, t=%s, dt=%s" % (t, dt))

            t += dt
            state.t.assign(t)

            for field in state.xnp1:
                field.assign(state.xn(field.name()))

            self.timestep(state)
            for field in state.xnp1:
                state.xn(field.name()).assign(field)

            with timed_stage("Dump output"):
                state.dump(t, pickup=False)

        if state.output.checkpoint:
            state.chkpt.close()

        logger.info("TIMELOOP complete. t=%s, tmax=%s" % (t, tmax))


class Timestepper(BaseTimestepper):

    def __init__(self, state, schemes,
                 physics_list=None):
        super().__init__(state, schemes, physics_list)
        self.schemes = schemes

    def timestep(self, state):
        for field, scheme in self.schemes:
            scheme.apply(state.xn(field), state.xnp1(field))


class BaseSemiImplicitTimestepper(BaseTimestepper):

    def __init__(self, state, equations,
                 advected_fields=None, diffused_fields=None,
                 physics_list=None):
        super().__init__(state, advected_fields, physics_list)
        self.equations = equations

        state.xnp1(equations.fieldlist, equations.mixed_function_space)

        if advected_fields is None:
            self.advected_fields = ()
        else:
            self.advected_fields = tuple(advected_fields)

        for field, scheme in self.advected_fields:
            scheme.setup_labels(self.label)

        if diffused_fields is None:
            self.diffused_fields = ()
        else:
            self.diffused_fields = tuple(diffused_fields)

    @abstractproperty
    def label(self):
        pass

    @property
    def passive_advection(self):
        """list of fields that are passively advected (and possibly diffused)"""
        return [(name, scheme) for name, scheme in
                self.advected_fields if name not in self.equations.fieldlist]

    @abstractmethod
    def timestep(self, state):
        self.semi_implicit_step(state)

        for name, advection in self.passive_advection:
            field = getattr(state.fields, name)
            # first computes ubar from state.xn and state.xnp1
            advection.update_ubar(state.xn, state.xnp1, state.timestepping.alpha)
            # advects a field from xn and puts result in xnp1
            advection.apply(field, field)

            state.xn(name).assign(state.xnp1(name))

        with timed_stage("Diffusion"):
            for name, diffusion in self.diffused_fields:
                field = getattr(state.fields, name)
                diffusion.apply(field, field)

        with timed_stage("Physics"):
            for physics in self.physics_list:
                physics.apply()

    @abstractmethod
    def semi_implicit_step(self):
        """
        Implement the semi implicit step for the timestepping scheme.
        """
        pass


class CrankNicolson(BaseSemiImplicitTimestepper):
    """
    This class implements a Crank-Nicolson discretisation, with Strang
    splitting and auxilliary semi-Lagrangian advection.

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg forcing: a :class:`.Forcing` object
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    """

    def __init__(self, state, equations, advected_fields, linear_solver,
                 diffused_fields=None, physics_list=None):

        super().__init__(state, equations, advected_fields, physics_list)
        self.linear_solver = linear_solver

        state.xstar = FieldCreator()
        state.xstar(equations.fieldlist, equations.mixed_function_space)
        state.xp = FieldCreator()
        state.xp(equations.fieldlist, equations.mixed_function_space)

        # list of fields that are advected as part of the nonlinear iteration
        self.active_advection = [(name, scheme) for name, scheme in advected_fields if name in equations.fieldlist]

        self.forcing = Forcing(state, equations)

    @property
    def label(self):
        return "advection"

    def timestep(self, state):
        super().timestep(state)

    def semi_implicit_step(self, state):

        alpha = state.timestepping.alpha
        xrhs = state.xrhs('xfields')
        xnp1 = state.xnp1('xfields')

        with timed_stage("Apply forcing terms"):
            self.forcing.apply(state.xn, state.xn,
                               state.xstar, label="explicit")

        for k in range(state.timestepping.maxk):

            with timed_stage("Advection"):
                for name, advection in self.active_advection:
                    # first computes ubar from state.xn and state.xnp1
                    advection.update_ubar(state.xn, state.xnp1, alpha)
                    # advects a field from xstar and puts result in xp
                    advection.apply(state.xstar(name), state.xp(name))

            xrhs.assign(0.)  # xrhs is the residual which goes in the linear solve

            for i in range(state.timestepping.maxi):

                with timed_stage("Apply forcing terms"):
                    self.forcing.apply(state.xp, state.xnp1,
                                       state.xrhs, label="implicit")

                xrhs -= state.xnp1('xfields')

                with timed_stage("Implicit solve"):
                    self.linear_solver.solve()  # solves linear system and places result in state.dy

                xnp1 += state.dy('xfields')

            self._apply_bcs()

        state.xn('xfields').assign(state.xnp1('xfields'))


class ImplicitMidpoint(BaseSemiImplicitTimestepper):
    """
    This class implements a Crank-Nicolson discretisation, with Strang
    splitting and auxilliary semi-Lagrangian advection.

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg linear_solver: a :class:`.TimesteppingSolver` object
    :arg diffused_fields: optional iterable of ``(field_name, scheme)``
        pairs indictaing the fields to diffusion, and the
        :class:`~.Diffusion` to use.
    :arg physics_list: optional list of classes that implement `physics` schemes
    """

    def __init__(self, state, equations, advected_fields, linear_solver,
                 diffused_fields=None, physics_list=None):

        super().__init__(state, equations, advected_fields, physics_list)
        self.linear_solver = linear_solver

        # list of fields that are advected as part of the nonlinear iteration
        self.active_advection = [(name, scheme) for name, scheme in advected_fields if name in equations.fieldlist]

    @property
    def label(self):
        return "all"

    def timestep(self, state, **kwargs):
        super().timestep(state, **kwargs)

    def semi_implicit_step(self, state, maxk=4):

        alpha = state.timestepping.alpha
        xrhs = state.xrhs('xfields')
        xnp1 = state.xnp1('xfields')

        xrhs.assign(0.)

        for k in range(maxk):

            with timed_stage("Advection"):
                for name, advection in self.active_advection:
                    # first computes ubar from state.xn and state.xnp1
                    advection.update_xbar(state.xn, state.xnp1, alpha)
                    # advects a field from xn and puts result in xnp1
                    advection.apply(state.xn(name), state.xrhs(name))

            xrhs -= state.xnp1('xfields')

            with timed_stage("Implicit solve"):
                self.linear_solver.solve()  # solves linear system and places result in state.dy

            xnp1 += state.dy('xfields')

            self._apply_bcs()

        state.xn('xfields').assign(state.xnp1('xfields'))
