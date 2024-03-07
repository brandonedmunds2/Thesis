#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import jax.numpy as jnp
import jax.random as jnr

from jax import jit, jvp, vmap, nn
from jax.tree_util import tree_flatten
from tensorflow_privacy_dep import compute_rdp, get_privacy_spent
import math

def get_grad_jacobian_trace_func(grad_func, fisher_iters, fisher_batch_size):
    """
    Returns a function that computes the (square root of the) trace of the Jacobian
    of the parameters.
    """

    # last batch rounds up to a full batch
    @jit
    def grad_jacobian_trace(rng, params,reg_params,prune_mask, inputs,targets,grads_mean,grads_std):
        num_iters=fisher_iters
        num_batches=math.ceil(float(num_iters)/float(fisher_batch_size))
        trace = jnp.zeros(inputs.shape[0])
        for i in range(num_batches):
            full_trace=vmap(inner_grad_jacobian_trace, in_axes=(0,0,None,None,None,None,None,None,None))(
                jnr.split(rng,fisher_batch_size),
                jnp.arange(fisher_batch_size),
                params,reg_params,prune_mask,inputs,targets,grads_mean,grads_std)
            trace=trace+jnp.mean(full_trace,axis=0)
        # set nan values to 0 because gradient is 0
        return jnp.sqrt(jnp.nan_to_num(trace/(num_batches)) + 1e-7)

    @jit
    def inner_grad_jacobian_trace(rng,i, params,reg_params,prune_mask, inputs,targets,grads_mean,grads_std):

        inputs = jnp.expand_dims(inputs, 1)
        targets = jnp.expand_dims(targets, 1)
            
        flattened_shape = jnp.reshape(inputs, (inputs.shape[0], -1)).shape
        perex_grad = lambda x: vmap(grad_func, in_axes=(0,None,None,None, 0, 0,None,None,None))(
            jnr.split(rng,inputs.shape[0]),params,reg_params, prune_mask,x, targets,None,grads_mean,grads_std
        )[0]
        
        trace = jnp.zeros(inputs.shape[0])
        w=jnr.normal(rng,flattened_shape)
        w=jnp.reshape(w,inputs.shape)
        _, w = jvp(perex_grad, (inputs,), (w,))

        # compute norm of the JVP:
        w, _ = tree_flatten(w)
        w = [
            jnp.power(jnp.reshape(v, (v.shape[0], -1)), 2).sum(axis=1)
            for v in w
        ]
        trace = trace + sum(w)
        
        return trace

    # return the function:
    return grad_jacobian_trace


def get_dp_accounting_func(batch_size, sigma):
    """
    Returns the (eps, delta)-DP accountant if alpha=None,
    or the (alpha, eps)-RDP accountant otherwise.
    """
    
    def compute_epsilon(steps, num_examples, target_delta=1e-5, alpha=None):
        if num_examples * target_delta > 1.:
            warnings.warn('Your delta might be too high.')
        q = batch_size / float(num_examples)
        if alpha is None:
            orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
            rdp_const = compute_rdp(q, sigma, steps, orders)
            eps, _, _ = get_privacy_spent(orders, rdp_const, target_delta=target_delta)
        else:
            eps = compute_rdp(q, sigma, steps, alpha)
        return eps
    
    return compute_epsilon