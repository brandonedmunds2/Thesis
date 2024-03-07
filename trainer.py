#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp
import jax.random as jnr
from jax import jit, grad, vmap, nn
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from flatten_util import ravel_pytree
from prune import pruner
import math
from accountant import get_grad_jacobian_trace_func

# combine FIM prediction and grads
def get_add_pred2grad_func(prune_inds):
    @jit
    def add_pred2grad(grads,preds,prune_mask):
        flat,rev= ravel_pytree(grads)
        return rev(flat+jnp.zeros_like(flat).at[prune_inds].set(preds.ravel()))
    return add_pred2grad

# prepare inputs to the FIM model
def get_get_reg_inputs_func(prune_inds,cnn_reg,standardize):
    @jit
    def get_reg_inputs(rng,params,inputs,targets,grads,prune_mask,grads_mean,grads_std):
        tmp=jnp.concatenate(tree_flatten(params)[0],axis=None).at[prune_inds].get()
        if standardize:
            tmp=jnp.concatenate((tmp,(jnp.concatenate(tree_flatten(grads)[0],axis=None).at[prune_inds].get()-grads_mean.at[prune_inds].get())/(grads_std.at[prune_inds].get()+1e-7)))
        else:
            tmp=jnp.concatenate((tmp,(jnp.concatenate(tree_flatten(grads)[0],axis=None).at[prune_inds].get())))
        tmp=jnp.broadcast_to(tmp,(inputs.shape[0],tmp.shape[0]))
        tmp=jnp.concatenate((targets.reshape(targets.shape[0],-1),tmp),axis=1)
        if cnn_reg:
            return (inputs,tmp)
        else:
            return jnp.concatenate((inputs.reshape(inputs.shape[0],-1),tmp),axis=1)
    return get_reg_inputs

def get_loss_func(*,predict=None,loss_type=None,lam_pow=None,entropy_loss=None,regd_grad_func=None,size=None,cnn_reg=None,standardize=False,fisher_iters=None,fisher_batch_size=None):
    if loss_type=="entropy":
        @jit
        def entropy(rng,params, inputs, targets):
            predictions = nn.log_softmax(predict(params, inputs))
            if predictions.ndim == 1:
                return -jnp.sum(predictions * targets)
            return -jnp.mean(jnp.sum(predictions * targets, axis=-1))
        return entropy
    elif loss_type=="mse":
        @jit
        def mse(rng,params, inputs, targets):
            predictions = predict(params, inputs)
            return jnp.mean(jnp.sum(jnp.power(predictions - targets,2),axis=-1))
        return mse
    else:
        get_reg_inputs=get_get_reg_inputs_func(size,cnn_reg,standardize)
        fil_accountant = get_grad_jacobian_trace_func(
            regd_grad_func,
            fisher_iters=fisher_iters,
            fisher_batch_size=fisher_batch_size
        )
        @jit
        def reg_loss(rng, params,reg_params,prune_mask, inputs, targets,lam,grads_mean,grads_std):
            grads=grad(entropy_loss, argnums=1)(rng,params,inputs,targets)
            regd_res,_=regd_grad_func(rng,params,reg_params,prune_mask,inputs,targets,None,grads_mean,grads_std)
            flat,rev= ravel_pytree(regd_res)
            return jnp.mean(jnp.sum(jnp.power(predict(reg_params,get_reg_inputs(rng,params,inputs,targets,grads,prune_mask,grads_mean,grads_std)),2),axis=-1))+lam*jnp.mean(jnp.divide(jnp.power(fil_accountant(rng, params,reg_params,prune_mask, inputs, targets,grads_mean,grads_std),lam_pow),jnp.sum(jnp.power(flat,2),axis=-1)))
        return reg_loss

def get_grad_func(*,loss=None,argnum=None,norm_clip=0,soft_clip=True,do_fam=False,reg_predict=None,size=None,cnn_reg=None,standardize=False,ret_grads=False):
    # if using FIM on the grads
    if argnum==1 and do_fam:
        get_reg_inputs=get_get_reg_inputs_func(size,cnn_reg,standardize)
        add_pred2grad=get_add_pred2grad_func(size)
    @jit
    def clipped_grad(rng,params,reg_params,prune_mask,inputs,targets,lam,grads_mean,grads_std):
        base_grads=None
        pert=None
        if argnum==2:
            grads = grad(loss, argnums=argnum)(rng,params,reg_params,prune_mask,inputs,targets,lam,grads_mean,grads_std)
        else:
            base_grads=grad(loss, argnums=argnum)(rng,params,inputs,targets)
            if do_fam:
                pert=reg_predict(reg_params,get_reg_inputs(rng,params,inputs,targets,base_grads,prune_mask,grads_mean,grads_std))
                grads = add_pred2grad(base_grads,pert,prune_mask)
            else:
                grads=base_grads
            grads=pruner(grads,prune_mask)
        if norm_clip != 0:
            grads, tree_def = tree_flatten(grads)
            total_grad_norm = jnp.add(jnp.linalg.norm(
                jnp.asarray([jnp.sqrt(jnp.sum(jnp.power(neg.ravel(),2))+1e-7) for neg in grads])), 1e-7)
            if soft_clip:
                divisor = nn.gelu(total_grad_norm / norm_clip - 1) + 1
            else:
                divisor = jnp.maximum(total_grad_norm / norm_clip, 1.)
            grads = [g / divisor for g in grads]
            grads=tree_unflatten(tree_def, grads)
        if ret_grads:
            return (grads, (base_grads,pert))
        else:
            return (grads, (None,None))
    
    return clipped_grad

def get_update_func(*,get_params=None,reg_get_params=None,grad_func=None,opt_update=None,prune_only_grads=None,norm_clip=0,argnum=None,ret_grads=False):
    @jit
    def update(*,i, rng, opt_state, reg_opt_state,prune_mask, inputs,targets,reg_curr_lam, sigma, weight_decay,grads_mean,grads_std):
        inputs = jnp.expand_dims(inputs, 1)
        targets = jnp.expand_dims(targets, 1)
        if prune_only_grads:
            params = get_params(opt_state)
        else:
            params = pruner(get_params(opt_state),prune_mask)
        if reg_get_params != None:
            reg_params = reg_get_params(reg_opt_state)
        else:
            reg_params=None
        multiplier = 1 if norm_clip == 0 else norm_clip
        
        # compute parameter gradient:
        grads, (base_grads,pert) = vmap(grad_func, in_axes=(0,None,None,None, 0, 0,None,None,None))(jnr.split(rng,inputs.shape[0]),params,reg_params,prune_mask, inputs, targets,reg_curr_lam,grads_mean,grads_std)
        grads_flat_init, grads_treedef = tree_flatten(grads)

        # add noise
        grads = [g.sum(0) for g in grads_flat_init]
        rngs = jnr.split(rng, len(grads))
        grads = [
            (g + multiplier * sigma * jnr.normal(r, g.shape)) / len(targets)
            for r, g in zip(rngs, grads)
        ]
        
        # weight decay
        if argnum==2:
            params, _ = tree_flatten(reg_params)
        else:
            params, _ = tree_flatten(params)
        grads = [
            g + weight_decay * param
            for g, param in zip(grads, params)
        ]
        grads = tree_unflatten(grads_treedef, grads)
        if argnum==1:
            grads=pruner(grads,prune_mask)

        # perform parameter update:
        if argnum==2:
            update_opt_state=reg_opt_state
        else:
            update_opt_state=opt_state

        if ret_grads:
            return (grads_flat_init,tree_flatten(base_grads)[0],tree_flatten(pert)[0], opt_update(i, grads, update_opt_state))
        else:
            return (None,None,None, opt_update(i, grads, update_opt_state))

    return update