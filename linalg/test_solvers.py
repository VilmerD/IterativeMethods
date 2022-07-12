import unittest

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from krylov import gmres

class TestGMRES(unittest.TestCase):
    m_list: list = None
    positive_definite_systems: list = None
    badly_conditioned_systems: list = None

    def setUp(self):
        # System sizes
        self.m_list = [1, 10, 100]

        # Generate a set of systems with positive definite rhs
        positive_definite_systems = []
        for m in self.m_list:
            # Generate SPD matrix A
            A = np.random.rand(m, m)
            A = (A + A.T)/2                                                 # Make it symmetric
            A -= np.eye(A.shape[0])*(sla.eigvalsh(A, lower=True)[0]-1)      # Shift eigenvalues

            # Generate b
            b = np.random.rand(m)

            positive_definite_systems.append((A, b))

        self.positive_definite_systems = positive_definite_systems

        # Generate badly conditioned systems
        cond = 1e16
        badly_conditioned_systems = []
        for m in self.m_list:
            # Generate SPD matrix A with large cond
            A = np.random.rand(m, m)
            A = (A + A.T)/2
            _, w = sla.eig(A)
            d = np.diag(np.logspace(0, np.log10(cond), m))
            A = w.T.dot(d.dot(w))

            # Generate b
            b = np.random.rand(m)

            badly_conditioned_systems.append((A, b))

        self.badly_conditioned_systems = badly_conditioned_systems

    def test_return_size(self):
        """
        Tests that the return size is the same as input
        """
        # Shapes to test
        for i, (A, b) in enumerate(self.positive_definite_systems):
            with self.subTest(i=i):
                x, _, _ = gmres(A, b)
                self.assertEqual(x.shape, b.shape)

    def test_convergance_steps(self):
        """
        Tests that GMRES converges after at most m steps, where m is the systems size
        """
        for i, (A, b) in enumerate(self.positive_definite_systems):
            with self.subTest(i=i):
                _, nits, _ = gmres(A, b, k_max=A.shape[0])
                self.assertLessEqual(nits, A.shape[0])

    def test_1_step_convergance(self):
        """
        Tests that GMRES converges directly given the solution as initial guess
        """
        for i, (A, b) in enumerate(self.positive_definite_systems):
            with self.subTest(i=i):
                x0 = sla.solve(A, b)
                _, nits, _ = gmres(A, b, x0=x0)
                self.assertGreaterEqual(1, nits)

    def test_tolerance(self):
        """
        Tests that the tolerance is fulfilled
        """
        tols = (1, 1e-3, 1e-9)
        A, b = self.positive_definite_systems[-1]
        for i, t in enumerate(tols):
            with self.subTest(i=i):
                x, _, _ = gmres(A, b, k_max=A.shape[0], tol=t)
                self.assertLessEqual(sla.norm(b - A.dot(x)), t*sla.norm(b))

    def test_linear_operator(self):
        """
        Tests that GMRES works with linear operator
        """
        for i, (A, b) in enumerate(self.positive_definite_systems):
            with self.subTest(i=i):
                A = spla.aslinearoperator(A)
                _, _, _ = gmres(A, b)

    # Tests with preconditioner
    def test_tolerance_with_preconditioner(self):
        """
        Tests that the tolerance is fulfilled with a preconditioner
        """
        tols = (1, 1e-3, 1e-9)
        A, b = self.positive_definite_systems[-1]
        for i, t in enumerate(tols):
            M = spla.LinearOperator(A.shape, matvec=lambda x: spla.spilu(A).solve(x))
            with self.subTest(i=i):
                x, _, _ = gmres(A, b, k_max=A.shape[0], tol=t, m_right=M)
                self.assertLessEqual(sla.norm(b - A.dot(x)), t*sla.norm(b))

    def test_perfect_preconditioner(self):
        """
        Tests that gmres converges in one step if preconditioner is the inverse
        """
        for i, (A, b) in enumerate(self.positive_definite_systems):
            M = spla.LinearOperator(A.shape, matvec=lambda x: spla.splu(A).solve(x))
            with self.subTest(i=i):
                _, nits, _ = gmres(A, b, k_max=A.shape[0], m_right=M)
                self.assertLessEqual(nits, 1)

    def test_1_step_convergance_preconditioned(self):
        """
        Tests that GMRES converges directly given the solution as initial guess
        """
        for i, (A, b) in enumerate(self.positive_definite_systems):
            M = np.random.rand(*A.shape)
            with self.subTest(i=i):
                x0 = sla.solve(A, b)
                _, nits, _ = gmres(A, b, x0=x0, m_right=M)
                self.assertGreaterEqual(1, nits)
    
    def test_hard_problem(self):
        """ Tests that no errors are found if it can't converge """
        for i, (A, b) in enumerate(self.badly_conditioned_systems):
            with self.subTest(i=i):
                _, nits, _ = gmres(A, b, k_max=A.shape[0])
                self.assertEqual(nits, A.shape[0])

if __name__=='__main__':
    unittest.main()