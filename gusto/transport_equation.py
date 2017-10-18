from abc import ABCMeta, abstractmethod
from firedrake import Function, TestFunction, TrialFunction, \
    FacetNormal, \
    dx, dot, grad, div, jump, avg, dS, dS_v, dS_h, inner, \
    outer, sign, cross, CellNormal, sqrt, \
    curl, BrokenElement, FunctionSpace, as_matrix, VectorElement


__all__ = ["LinearAdvection", "AdvectionEquation", "EmbeddedDGAdvection", "SUPGAdvection", "VectorInvariant", "EulerPoincare"]


class TransportEquation(object, metaclass=ABCMeta):
    """
    Base class for transport equations in Gusto.

    The equation is assumed to be in the form:

    q_t + L(q) = 0

    where q is the (scalar or vector) field to be solved for.

    :arg V: :class:`.FunctionSpace object. The function space that q lives in.
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """

    def __init__(self, state, V, *, ibp="once",
                 solver_params=None):

        self.V = V
        self.ibp = ibp

        # set up functions required for forms
        self.ubar = Function(state.spaces("HDiv"))
        self.test = TestFunction(V)
        self.trial = TrialFunction(V)

        # find out if we are CG
        if isinstance(V.ufl_element(), VectorElement):
            self.is_cg = all([ele.sobolev_space().name == "H1" for ele in V.ufl_element().sub_elements()])
        elif isinstance(V.ufl_element(), BrokenElement):
            self.is_cg = False
        else:
            self.is_cg = V.ufl_element().sobolev_space().name == "H1"

        # DG, embedded DG and hybrid SUPG methods need surface measures,
        # n and un
        if self.is_cg:
            self.dS = None
        else:
            if state.physical_domain.is_extruded:
                self.dS = (dS_h + dS_v)
            else:
                self.dS = dS
            self.n = FacetNormal(state.mesh)
            self.un = 0.5*(dot(self.ubar, self.n) + abs(dot(self.ubar, self.n)))

        if solver_params:
            self.solver_parameters = solver_params

        # default solver options
        else:
            self.solver_parameters = {'ksp_type': 'cg',
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'ilu'}

    def mass_term(self, q):
        return inner(self.test, q)*dx

    @abstractmethod
    def advection_term(self):
        pass


class LinearAdvection(TransportEquation):
    """
    Class for linear transport equation.

    :arg V: :class:`.FunctionSpace object. The function space that q lives in.
    :arg qbar: The reference function that the equation has been linearised
               about. It is assumed that the reference velocity is zero and
               the ubar below is the nonlinear advecting velocity
               0.5*(u'^(n+1) + u'(n)))
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    :arg equation_form: (optional) string, can take the values 'continuity',
                        which means the equation is in continuity form
                        L(q) = div(u'*qbar), or 'advective', which means the
                        equation is in advective form L(q) = u' dot grad(qbar).
                        Default is "advective"
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """

    def __init__(self, state, V, qbar, ibp=None,
                 equation_form="advective", solver_params=None):

        super().__init__(state, V, ibp=ibp,
                         solver_params=solver_params)

        if equation_form == "advective" or equation_form == "continuity":
            self.continuity = (equation_form == "continuity")
        else:
            raise ValueError("equation_form must be either 'advective' or 'continuity', not %s" % equation_form)

        self.vertical_normal = state.physical_domain.vertical_normal
        self.qbar = qbar

        # currently only used with the following option combinations:
        if self.continuity and ibp is not "once":
            raise NotImplementedError("If we are solving a linear continuity equation, we integrate by parts once")
        if not self.continuity and ibp is not None:
            raise NotImplementedError("If we are solving a linear advection equation, we do not integrate by parts.")

    def advection_term(self, q):

        if self.continuity:
            L = (-dot(grad(self.test), self.ubar)*self.qbar*dx +
                 jump(self.ubar*self.test, self.n)*avg(self.qbar)*self.dS)
        else:
            k = self.vertical_normal
            L = self.test*dot(self.ubar, k)*dot(k, grad(self.qbar))*dx
        return L


class AdvectionEquation(TransportEquation):
    """
    Class for discretisation of the transport equation.

    :arg V: :class:`.FunctionSpace object. The function space that q lives in.
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    :arg equation_form: (optional) string, can take the values 'continuity',
                        which means the equation is in continuity form
                        L(q) = div(u*q), or 'advective', which means the
                        equation is in advective form L(q) = u dot grad(q).
                        Default is "advective"
    :arg vector_manifold: Boolean. If true adds extra terms that are needed for
    advecting vector equations on manifolds.
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """
    def __init__(self, state, V, *, ibp="once",
                 equation_form="advective",
                 vector_manifold=False, solver_params=None):

        super().__init__(state, V=V, ibp=ibp,
                         solver_params=solver_params)

        if equation_form == "advective" or equation_form == "continuity":
            self.continuity = (equation_form == "continuity")
        else:
            raise ValueError("equation_form must be either 'advective' or 'continuity'")

        self.vector_manifold = vector_manifold

    def advection_term(self, q):

        if self.continuity:
            if self.ibp == "once":
                L = -inner(grad(self.test), outer(q, self.ubar))*dx
            else:
                L = inner(self.test, div(outer(q, self.ubar)))*dx
        else:
            if self.ibp == "once":
                L = -inner(div(outer(self.test, self.ubar)), q)*dx
            else:
                L = inner(outer(self.test, self.ubar), grad(q))*dx

        if self.dS is not None and self.ibp is not None:
            L += dot(jump(self.test), (self.un('+')*q('+')
                                       - self.un('-')*q('-')))*self.dS
            if self.ibp == "twice":
                L -= (inner(self.test('+'),
                            dot(self.ubar('+'), self.n('+'))*q('+'))
                      + inner(self.test('-'),
                              dot(self.ubar('-'), self.n('-'))*q('-')))*self.dS

        if self.vector_manifold:
            un = self.un
            w = self.test
            u = q
            n = self.n
            dS = self.dS
            L += un('+')*inner(w('-'), n('+')+n('-'))*inner(u('+'), n('+'))*dS
            L += un('-')*inner(w('+'), n('+')+n('-'))*inner(u('-'), n('-'))*dS

        return L


class EmbeddedDGAdvection(AdvectionEquation):
    """
    Class for the transport equation, using an embedded DG advection scheme.

    :arg V: :class:`.FunctionSpace object. The function space that q lives in.
    :arg ibp: (optional) string, stands for 'integrate by parts' and can take
              the value None, "once" or "twice". Defaults to "once".
    :arg equation_form: (optional) string, can take the values 'continuity',
                        which means the equation is in continuity form
                        L(q) = div(u*q), or 'advective', which means the
                        equation is in advective form L(q) = u dot grad(q).
                        Default is "advective"
    :arg vector_manifold: Boolean. If true adds extra terms that are needed for
    advecting vector equations on manifolds.
    :arg Vdg: (optional) :class:`.FunctionSpace object. The embedding function
              space. Defaults to None which means that a broken space is
              constructed for you.
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """

    def __init__(self, state, V, *, ibp="once",
                 equation_form="advective", vector_manifold=False,
                 Vdg=None, solver_params=None):

        if Vdg is None:
            # Create broken space, functions and projector
            V_elt = BrokenElement(V.ufl_element())
            self.space = FunctionSpace(state.mesh, V_elt)
        else:
            self.space = Vdg

        super().__init__(state,
                         V=self.space,
                         ibp=ibp,
                         equation_form=equation_form,
                         vector_manifold=vector_manifold,
                         solver_params=solver_params)


class SUPGAdvection(AdvectionEquation):
    """
    Class for the transport equation.

    :arg V: :class:`.FunctionSpace object. The function space that q lives in.
    :arg ibp: (optional) string, stands for 'integrate by parts' and can
              take the value None, "once" or "twice". Defaults to "twice"
              since we commonly use this scheme for parially continuous
              spaces, in which case we don't want to take a derivative of
              the test function. If using for a fully continuous space, we
              don't integrate by parts at all (so you can set ibp=None).
    :arg equation_form: (optional) string, can take the values 'continuity',
                        which means the equation is in continuity form
                        L(q) = div(u*q), or 'advective', which means the
                        equation is in advective form L(q) = u dot grad(q).
                        Default is "advective"
    :arg supg_params: (optional) dictionary of parameters for the SUPG method.
                      Can contain:
                      'ax', 'ay', 'az', which specify the coefficients in
                      the x, y, z directions respectively
                      Appropriate defaults are provided for these parameters,
                      in particular, the space is assumed to be continuous.
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """
    def __init__(self, state, V, *, ibp="twice",
                 equation_form="advective", tau_h=None, tau_v=None,
                 solver_params=None):

        if not solver_params:
            # SUPG method leads to asymmetric matrix (since the test function
            # is effectively modified), so don't use CG
            solver_params = {'ksp_type': 'gmres',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}

        super().__init__(state, V, ibp=ibp,
                         equation_form=equation_form,
                         solver_params=solver_params)

        # if using SUPG we either integrate by parts twice, or not at all
        if ibp == "once":
            raise ValueError("if using SUPG we don't integrate by parts once")
        if ibp is None and not self.is_cg:
            raise ValueError("are you very sure you don't need surface terms?")

        dim = state.mesh.geometric_dimension()
        default_tau = state.timestepping.dt/sqrt(15.)
        if tau_v is None:
            tau_v = default_tau

        if self.is_cg:
            if tau_h is None:
                tau_h = [default_tau]*(dim-1)
            if dim == 2:
                tau = as_matrix([[tau_h[0], 0.0],
                                 [0.0, tau_v]])
            elif dim == 3:
                tau = as_matrix([[tau_h[0], 0.0, 0.0],
                                 [0.0, tau_h[1], 0.0],
                                 [0.0, 0.0, tau_v]])
            dtest = dot(dot(self.ubar, tau), grad(self.test))
        elif V.ufl_element().sobolev_space()[0].name in ["L2", "HDiv"]:
            self.dS = dS_v
            dtest = tau_v * self.ubar[dim-1] * self.test.dx(dim-1)
        else:
            raise NotImplementedError("SUPG is not implemented for your function space")

        self.test += dtest


class VectorInvariant(TransportEquation):
    """
    Class defining the vector invariant form of the vector advection equation.

    :arg V: Function space
    :arg ibp: (optional) string, stands for 'integrate by parts' and can
              take the value None, "once" or "twice". Defaults to "once".
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """
    def __init__(self, state, V, *, ibp="once",
                 solver_params=None):

        super().__init__(state, V, ibp=ibp,
                         solver_params=solver_params)

        self.physical_domain = state.physical_domain
        if state.physical_domain.is_3d and ibp == "twice":
            raise NotImplementedError("ibp=twice is not implemented for 3d problems")

    def advection_term(self, q):

        Upwind = 0.5*(sign(dot(self.ubar, self.n))+1)
        if self.physical_domain.is_3d:
            # <w,curl(u) cross ubar + grad( u.ubar)>
            # =<curl(u),ubar cross w> - <div(w), u.ubar>
            # =<u,curl(ubar cross w)> -
            #      <<u_upwind, [[n cross(ubar cross w)cross]]>>

            both = lambda u: 2*avg(u)

            L = (
                inner(q, curl(cross(self.ubar, self.test)))*dx
                - inner(both(Upwind*q),
                        both(cross(self.n, cross(self.ubar, self.test))))*self.dS
            )

        else:

            perp = self.physical_domain.perp
            if self.physical_domain.is_extruded:
                perp_u_upwind = (
                    lambda q: Upwind('+')*perp(q('+')) + Upwind('-')*perp(q('-'))
                )
            elif self.physical_domain.on_sphere:
                outward_normals = CellNormal(self.physical_domain.mesh)
                perp_u_upwind = (
                    lambda q: Upwind('+')*cross(outward_normals('+'), q('+'))
                    + Upwind('-')*cross(outward_normals('-'), q('-'))
                )
            gradperp = lambda u: perp(grad(u))

            if self.ibp == "once":
                L = (
                    -inner(gradperp(inner(self.test, perp(self.ubar))), q)*dx
                    - inner(jump(inner(self.test, perp(self.ubar)), self.n),
                            perp_u_upwind(q))*self.dS
                )
            else:
                L = (
                    (-inner(self.test, div(perp(q))*perp(self.ubar)))*dx
                    - inner(jump(inner(self.test, perp(self.ubar)), self.n),
                            perp_u_upwind(q))*self.dS
                    + jump(inner(self.test,
                                 perp(self.ubar))*perp(q), self.n)*self.dS
                )

        L -= 0.5*div(self.test)*inner(q, self.ubar)*dx

        return L


class EulerPoincare(VectorInvariant):
    """
    Class defining the Euler-Poincare form of the vector advection equation.

    :arg physical_domain: :class:`.PhysicalDomain` object.
    :arg V: Function space
    :arg ibp: string, stands for 'integrate by parts' and can take the value
              None, "once" or "twice". Defaults to "once".
    :arg solver_params: (optional) dictionary of solver parameters to pass to the
                        linear solver.
    """

    def advection_term(self, q):
        L = super().advection_term(q)
        L -= 0.5*div(self.test)*inner(q, self.ubar)*dx
        return L
