# https://github.com/tensorflow/privacy

import numpy as np
from dp_accounting import dp_event, privacy_accountant
from dp_accounting.rdp import rdp_privacy_accountant

def _compute_rdp_from_event(orders, event, count):
  """Computes RDP from a DpEvent using RdpAccountant.
  Args:
    orders: An array (or a scalar) of RDP orders.
    event: A DpEvent to compute the RDP of.
    count: The number of self-compositions.
  Returns:
    The RDP at all orders. Can be `np.inf`.
  """
  orders_vec = np.atleast_1d(orders)

  if isinstance(event, dp_event.SampledWithoutReplacementDpEvent):
    neighboring_relation = privacy_accountant.NeighboringRelation.REPLACE_ONE
  elif isinstance(event, dp_event.SingleEpochTreeAggregationDpEvent):
    neighboring_relation = privacy_accountant.NeighboringRelation.REPLACE_SPECIAL
  else:
    neighboring_relation = privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE

  accountant = rdp_privacy_accountant.RdpAccountant(orders_vec,
                                                    neighboring_relation)
  accountant.compose(event, count)
  rdp = accountant._rdp  # pylint: disable=protected-access

  if np.isscalar(orders):
    return rdp[0]
  else:
    return rdp

def compute_rdp(q, noise_multiplier, steps, orders):
  """(Deprecated) Computes RDP of the Sampled Gaussian Mechanism.
  This function has been superseded by more general accounting mechanisms in
  Google's `differential_privacy` package. It may at some future date be
  removed.
  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
      to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  if q > 1.0:
    q=1.0
  event = dp_event.PoissonSampledDpEvent(
      q, dp_event.GaussianDpEvent(noise_multiplier))

  return _compute_rdp_from_event(orders, event, steps)

def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  """(Deprecated) Computes delta or eps from RDP values.
  This function has been superseded by more general accounting mechanisms in
  Google's `differential_privacy` package. It may at some future date be
  removed.
  Args:
    orders: An array (or a scalar) of RDP orders.
    rdp: An array of RDP values. Must be of the same length as the orders list.
    target_eps: If not `None`, the epsilon for which we compute the
      corresponding delta.
    target_delta: If not `None`, the delta for which we compute the
      corresponding epsilon. Exactly one of `target_eps` and `target_delta` must
      be `None`.
  Returns:
    A tuple of epsilon, delta, and the optimal order.
  Raises:
    ValueError: If target_eps and target_delta are messed up.
  """
  if target_eps is None and target_delta is None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (Both are).")

  if target_eps is not None and target_delta is not None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (None is).")

  accountant = rdp_privacy_accountant.RdpAccountant(orders)
  accountant._rdp = rdp  # pylint: disable=protected-access

  if target_eps is not None:
    delta, opt_order = accountant.get_delta_and_optimal_order(target_eps)
    return target_eps, delta, opt_order
  else:
    eps, opt_order = accountant.get_epsilon_and_optimal_order(target_delta)
    return eps, target_delta, opt_order