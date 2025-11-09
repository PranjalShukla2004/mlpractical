import numpy as np

seed = 22102017
rng = np.random.RandomState(seed)


class L1Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L1 norm.
    """

    def __init__(self, coefficient):
        """Create a new L1 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L1 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        p = np.asarray(parameter)
        # λ * Σ |p_i|
        return self.coefficient * np.sum(np.abs(p))

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        p = np.asarray(parameter)
        # Subgradient: λ * sign(p); np.sign(0) = 0 (valid subgradient at 0)
        return self.coefficient * np.sign(p)

    def __repr__(self):
        return 'L1Penalty({0})'.format(self.coefficient)


class L2Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L2 norm.
    """

    def __init__(self, coefficient):
        """Create a new L2 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L2 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        p = np.asarray(parameter)
        # (λ/2) * Σ p_i^2  -> gives gradient λ * p
        return 0.5 * self.coefficient * np.sum(p * p)

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        p = np.asarray(parameter)
        # ∇ = λ * p
        return self.coefficient * p

    def __repr__(self):
        return 'L2Penalty({0})'.format(self.coefficient)

class L1L2MixPenalty(object):
    """L1 & L2 mix penalty.
    """

    def __init__(self, coefficient, l1_ratio):
        """Create a new L1 & L2 mix penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        assert 0.0 <= l1_ratio <= 1.0
        self.l1_ratio = float(l1_ratio)
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L1 & L2 mix penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        p = np.asarray(parameter)
        l1 = np.sum(np.abs(p))
        l2_half = 0.5 * np.sum(p * p)    # ½ ||p||^2 so grad w.r.t. this term is just p
        return self.coefficient * (self.l1_ratio * l1 + (1.0 - self.l1_ratio) * l2_half)


    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        p = np.asarray(parameter)
        # ∂/∂p [ λ ( r ||p||_1 + (1-r) * ½ ||p||^2 ) ]
        #   = λ ( r * sign(p) + (1-r) * p )
        return self.coefficient * (self.l1_ratio * np.sign(p) + (1.0 - self.l1_ratio) * p)


    def __repr__(self):
        return 'L1L2MixPenalty({0})'.format(self.coefficient)
