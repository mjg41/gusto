from firedrake import Function, split, TrialFunction, TestFunction, \
    FacetNormal, inner, dx, cross, div, jump, avg, dS_v, \
    LinearVariationalProblem, LinearVariationalSolver, \
    Constant, as_vector, SpatialCoordinate
from gusto.configuration import DEBUG
from gusto.state import FieldCreator
from gusto.transport_equation import TransportTerm
from gusto import thermodynamics


__all__ = ["CompressibleForcing", "IncompressibleForcing", "EadyForcing", "CompressibleEadyForcing"]


class Forcing(object):
    """
    Base class for forcing terms for Gusto.

    :arg state: x :class:`.State` object.
    """

    def __init__(self, state, equations):
        self.state = state
        self.fieldlist = equations.fieldlist

        self.x0 = FieldCreator()
        self.x0(equations.fieldlist, equations.mixed_function_space)

        # this is the function that contains the result of solving
        # <test, trial> = <test, F(x0)>, where F is the forcing term
        self.xF = FieldCreator()
        self.solvers = {"explicit": {}, "implicit": {}}

        for field in equations.fieldlist:
            if not all([isinstance(term, TransportTerm) for name, term in equations(field).terms.items()]):
                self.xF(field, state.fields(field).function_space())
                self._build_forcing_solver(field, equations(field))

    def coriolis_term(self):
        u0 = split(self.x0)[0]
        return -inner(self.test, cross(2*self.state.Omega, u0))*dx

    def sponge_term(self):
        u0 = split(self.x0)[0]
        return self.state.mu*inner(self.test, self.state.k)*inner(u0, self.state.k)*dx

    def euler_poincare_term(self):
        u0 = split(self.x0)[0]
        return -0.5*div(self.test)*inner(self.state.h_project(u0), u0)*dx

    def hydrostatic_term(self):
        u0 = split(self.x0)[0]
        return inner(u0, self.state.k)*inner(self.test, self.state.k)*dx

    def forcing_term(self):
        L = self.pressure_gradient_term()
        if self.extruded:
            L += self.gravity_term()
        if self.coriolis:
            L += self.coriolis_term()
        if self.euler_poincare:
            L += self.euler_poincare_term()
        if self.topography:
            L += self.topography_term()
        if self.extra_terms is not None:
            L += inner(self.test, self.extra_terms)*dx
        # scale L
        L = self.scaling * L
        # sponge term has a separate scaling factor as it is always implicit
        if self.sponge:
            L -= self.impl*self.state.timestepping.dt*self.sponge_term()
        # hydrostatic term has no scaling factor
        if self.hydrostatic:
            L += (2*self.impl-1)*self.hydrostatic_term()
        return L

    def _build_forcing_solver(self, field, equation):
        a = equation.mass_term(equation.trial)
        L_explicit = 0.
        L_implicit = 0.
        dt = self.state.timestepping.dt
        for name, term in equation.terms.items():
            if not isinstance(term, TransportTerm):
                L_explicit += dt * term.off_centering * term(equation.test, self.x0(field), self.x0)
                L_implicit += dt * (1. - term.off_centering) * term(equation.test, self.x0(field), self.x0)

        explicit_forcing_problem = LinearVariationalProblem(
            a, L_explicit, self.xF(field), bcs=equation.bcs
        )

        implicit_forcing_problem = LinearVariationalProblem(
            a, L_implicit, self.xF(field), bcs=equation.bcs
        )

        solver_parameters = {}
        if self.state.output.log_level == DEBUG:
            solver_parameters["ksp_monitor_true_residual"] = True

        self.solvers["explicit"][field] = LinearVariationalSolver(
            explicit_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix=field+"ExplicitForcingSolver"
        )
        self.solvers["implicit"][field] = LinearVariationalSolver(
            implicit_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix=field+"ImplicitForcingSolver"
        )

    def apply(self, x_in, x_nl, x_out, label):
        """
        Function takes x as input, computes F(x_nl) and returns
        x_out = x + scale*F(x_nl)
        as output.

        :arg x_in: :class:`.Function` object
        :arg x_nl: :class:`.Function` object
        :arg x_out: :class:`.Function` object
        :arg implicit: forcing stage for sponge and hydrostatic terms, if present
        """
        self.x0('xfields').assign(x_nl('xfields'))
        x_out('xfields').assign(x_in('xfields'))

        for field, solver in self.solvers[label].items():
            solver.solve()
            x = x_out(field)
            x += self.xF(field)


class CompressibleForcing(Forcing):
    """
    Forcing class for compressible Euler equations.
    """

    def pressure_gradient_term(self):

        u0, rho0, theta0 = split(self.x0)
        cp = self.state.parameters.cp
        n = FacetNormal(self.state.mesh)
        Vtheta = self.state.spaces("HDiv_v")

        # introduce new theta so it can be changed by moisture
        theta = theta0

        # add effect of density of water upon theta
        if self.moisture is not None:
            water_t = Function(Vtheta).assign(0.0)
            for water in self.moisture:
                water_t += self.state.fields(water)
            theta = theta / (1 + water_t)

        pi = thermodynamics.pi(self.state.parameters, rho0, theta0)

        L = (
            + cp*div(theta*self.test)*pi*dx
            - cp*jump(self.test*theta, n)*avg(pi)*dS_v
        )
        return L

    def gravity_term(self):

        g = self.state.parameters.g
        L = -g*inner(self.test, self.state.k)*dx

        return L

    def theta_forcing(self):

        cv = self.state.parameters.cv
        cp = self.state.parameters.cp
        c_vv = self.state.parameters.c_vv
        c_pv = self.state.parameters.c_pv
        c_pl = self.state.parameters.c_pl
        R_d = self.state.parameters.R_d
        R_v = self.state.parameters.R_v

        u0, _, theta0 = split(self.x0)
        water_v = self.state.fields('water_v')
        water_c = self.state.fields('water_c')

        c_vml = cv + water_v * c_vv + water_c * c_pl
        c_pml = cp + water_v * c_pv + water_c * c_pl
        R_m = R_d + water_v * R_v

        L = -theta0 * (R_m / c_vml - (R_d * c_pml) / (cp * c_vml)) * div(u0)

        return self.scaling * L

    def _build_forcing_solvers(self):

        super(CompressibleForcing, self)._build_forcing_solvers()
        # build forcing for theta equation
        if self.moisture is not None:
            _, _, theta0 = split(self.x0)
            Vt = self.state.spaces("HDiv_v")
            p = TrialFunction(Vt)
            q = TestFunction(Vt)
            self.thetaF = Function(Vt)

            a = p * q * dx
            L = self.theta_forcing()
            L = q * L * dx

            theta_problem = LinearVariationalProblem(a, L, self.thetaF)

            solver_parameters = {}
            if self.state.output.log_level == DEBUG:
                solver_parameters["ksp_monitor_true_residual"] = True
            self.theta_solver = LinearVariationalSolver(
                theta_problem,
                solver_parameters=solver_parameters,
                option_prefix="ThetaForcingSolver"
            )

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        super(CompressibleForcing, self).apply(scaling, x_in, x_nl, x_out, **kwargs)
        if self.moisture is not None:
            self.theta_solver.solve()
            _, _, theta_out = x_out.split()
            theta_out += self.thetaF


class IncompressibleForcing(Forcing):
    """
    Forcing class for incompressible Euler Boussinesq equations.
    """

    def pressure_gradient_term(self):
        _, p0, _ = split(self.x0)
        L = div(self.test)*p0*dx
        return L

    def gravity_term(self):
        _, _, b0 = split(self.x0)
        L = b0*inner(self.test, self.state.k)*dx
        return L

    def _build_forcing_solvers(self):

        super(IncompressibleForcing, self)._build_forcing_solvers()
        Vp = self.state.spaces("DG")
        p = TrialFunction(Vp)
        q = TestFunction(Vp)
        self.divu = Function(Vp)

        u0, _, _ = split(self.x0)
        a = p*q*dx
        L = q*div(u0)*dx

        divergence_problem = LinearVariationalProblem(
            a, L, self.divu)

        solver_parameters = {}
        if self.state.output.log_level == DEBUG:
            solver_parameters["ksp_monitor_true_residual"] = True
        self.divergence_solver = LinearVariationalSolver(
            divergence_problem,
            solver_parameters=solver_parameters,
            options_prefix="DivergenceSolver"
        )

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        super(IncompressibleForcing, self).apply(scaling, x_in, x_nl, x_out, **kwargs)
        if 'incompressible' in kwargs and kwargs['incompressible']:
            _, p_out, _ = x_out.split()
            self.divergence_solver.solve()
            p_out.assign(self.divu)


class EadyForcing(IncompressibleForcing):
    """
    Forcing class for Eady Boussinesq equations.
    """

    def forcing_term(self):

        L = Forcing.forcing_term(self)
        dbdy = self.state.parameters.dbdy
        H = self.state.parameters.H
        Vp = self.state.spaces("DG")
        _, _, z = SpatialCoordinate(self.state.mesh)
        eady_exp = Function(Vp).interpolate(z-H/2.)

        L -= self.scaling*dbdy*eady_exp*inner(self.test, as_vector([0., 1., 0.]))*dx
        return L

    def _build_forcing_solvers(self):

        super(EadyForcing, self)._build_forcing_solvers()

        # b_forcing
        dbdy = self.state.parameters.dbdy
        Vb = self.state.spaces("HDiv_v")
        F = TrialFunction(Vb)
        gamma = TestFunction(Vb)
        self.bF = Function(Vb)
        u0, _, b0 = split(self.x0)

        a = gamma*F*dx
        L = -self.scaling*gamma*(dbdy*inner(u0, as_vector([0., 1., 0.])))*dx

        b_forcing_problem = LinearVariationalProblem(
            a, L, self.bF
        )

        solver_parameters = {}
        if self.state.output.log_level == DEBUG:
            solver_parameters["ksp_monitor_true_residual"] = True
        self.b_forcing_solver = LinearVariationalSolver(
            b_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix="BForcingSolver"
        )

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        super(EadyForcing, self).apply(scaling, x_in, x_nl, x_out, **kwargs)
        self.b_forcing_solver.solve()  # places forcing in self.bF
        _, _, b_out = x_out.split()
        b_out += self.bF


class CompressibleEadyForcing(CompressibleForcing):
    """
    Forcing class for compressible Eady equations.
    """

    def forcing_term(self):

        # L = super(EadyForcing, self).forcing_term()
        L = Forcing.forcing_term(self)
        dthetady = self.state.parameters.dthetady
        Pi0 = self.state.parameters.Pi0
        cp = self.state.parameters.cp

        _, rho0, theta0 = split(self.x0)
        Pi = thermodynamics.pi(self.state.parameters, rho0, theta0)
        Pi_0 = Constant(Pi0)

        L += self.scaling*cp*dthetady*(Pi-Pi_0)*inner(self.test, as_vector([0., 1., 0.]))*dx  # Eady forcing
        return L

    def _build_forcing_solvers(self):

        super(CompressibleEadyForcing, self)._build_forcing_solvers()
        # theta_forcing
        dthetady = self.state.parameters.dthetady
        Vt = self.state.spaces("HDiv_v")
        F = TrialFunction(Vt)
        gamma = TestFunction(Vt)
        self.thetaF = Function(Vt)
        u0, _, _ = split(self.x0)

        a = gamma*F*dx
        L = -self.scaling*gamma*(dthetady*inner(u0, as_vector([0., 1., 0.])))*dx

        theta_forcing_problem = LinearVariationalProblem(
            a, L, self.thetaF
        )

        solver_parameters = {}
        if self.state.output.log_level == DEBUG:
            solver_parameters["ksp_monitor_true_residual"] = True
        self.theta_forcing_solver = LinearVariationalSolver(
            theta_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix="ThetaForcingSolver"
        )

    def apply(self, scaling, x_in, x_nl, x_out, **kwargs):

        Forcing.apply(self, scaling, x_in, x_nl, x_out, **kwargs)
        self.theta_forcing_solver.solve()  # places forcing in self.thetaF
        _, _, theta_out = x_out.split()
        theta_out += self.thetaF
