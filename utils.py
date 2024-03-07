#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp
from jax.example_libraries import stax
from jax.example_libraries import optimizers

import torch
import torchvision
import numpy as np
import lpips

import math


def _l2_normalize(x, eps=1e-7):
    return x * jax.lax.rsqrt((x ** 2).sum() + eps)

def standardize_data(train,test):
    train_mean=jnp.mean(train,axis=0)
    train_std=jnp.std(train,axis=0)
    train=(train-train_mean)/(train_std+1e-7)
    test=(test-train_mean)/(train_std+1e-7)
    return (train,test)

def calc_sub_fact(gelu_approx,sigma,norm_clip,delta,num_samples,batch_size):
    if sigma > 0 and norm_clip!=0:
        eps = math.sqrt(2 * math.log(1.25 / delta)) * 2 * gelu_approx / sigma
        q = float(batch_size) / num_samples
        subsampling_factor = q / (q + (1-q) * math.exp(-eps))
    elif sigma>0 and norm_clip==0:
        subsampling_factor = 1
    else:
        subsampling_factor=0
    multiplier = 1 if norm_clip==0 else norm_clip
    return subsampling_factor,multiplier

def batch_predict(predict, params, images, batch_size):
    num_images = images.shape[0]
    num_batches = int(math.ceil(float(num_images) / float(batch_size)))
    predictions = []
    for i in range(num_batches):
        lower = i * batch_size
        upper = min((i+1) * batch_size, num_images)
        predictions.append(predict(params, images[lower:upper]))
    return jnp.concatenate(predictions)

def estimate_spectral_norm(f, input_shape, seed=0, n_steps=20):
    input_shape = tuple([1] + [input_shape[i] for i in range(1, len(input_shape))])
    rng = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(rng, input_shape)
    v0 = jnp.zeros_like(f(u0))
    def fun(carry, _):
        u, v = carry
        v, f_vjp = jax.vjp(f, u)
        v = _l2_normalize(v)
        u, = f_vjp(v)
        u = _l2_normalize(u)
        return (u, v), None
    (u, v), _ = jax.lax.scan(fun, (u0, v0), xs=None, length=n_steps)
    return jnp.vdot(v, f(u))


def accuracy(predictions, targets):
    """
    Compute accuracy of `predictions` given the associated `targets`.
    """
    target_class = jnp.argmax(targets, axis=-1)
    predicted_class = jnp.argmax(predictions, axis=-1)
    return jnp.mean(predicted_class == target_class)

# calculate reconstruction error
def re(predictions, targets):
    return jnp.mean(jnp.mean(jnp.power(predictions - targets,2),axis=-1))

# preprocesses data for LPIPS and does LPIPS
def get_lpips_func(full_im_shape,torch_device):
    def lpips_func(predictions,targets):
        if len(full_im_shape) != 3:
            return jnp.mean(jnp.zeros((predictions.shape[0],)))
        predictions=jnp.reshape(predictions,(-1,)+full_im_shape)
        targets=jnp.reshape(targets,(-1,)+full_im_shape)
        max_val=max((jnp.max(predictions),jnp.max(targets)))
        min_val=min((jnp.min(predictions),jnp.min(targets)))
        predictions=2*(predictions-min_val)/(max_val-min_val)-1
        targets=2*(targets-min_val)/(max_val-min_val)-1
        if full_im_shape[2] != 3:
            predictions=jnp.repeat(predictions,3,3)
            targets=jnp.repeat(targets,3,3)
        predictions=torch.from_numpy(np.asarray(jnp.reshape(predictions,(predictions.shape[0],predictions.shape[3],predictions.shape[1],predictions.shape[2]))))
        targets=torch.from_numpy(np.asarray(jnp.reshape(targets,(targets.shape[0],targets.shape[3],targets.shape[1],targets.shape[2]))))
        predictions=torchvision.transforms.Resize((64,64))(predictions)
        targets=torchvision.transforms.Resize((64,64))(targets)
        return jnp.mean(jnp.squeeze(jnp.asarray(lpips.LPIPS().to(torch_device)(predictions.to(torch_device),targets.to(torch_device)).detach().cpu().numpy())))
    return lpips_func

def get_model(rng, model_name, input_shape, num_labels):
    """
    Returns model specified by `model_name`. Model is initialized using the
    specified random number generator `rng`.
    """

    if model_name == "cnn":
        init_random_params, predict = stax.serial(
            stax.Conv(16, (8, 8), padding="SAME", strides=(2, 2)),
            stax.Gelu,
            stax.AvgPool((2, 2), (1, 1)),
            stax.Conv(32, (4, 4), padding="VALID", strides=(2, 2)),
            stax.Gelu,
            stax.AvgPool((2, 2), (1, 1)),
            stax.Flatten,
            stax.Dense(32),
            stax.Gelu,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
    elif model_name == "cnn_tanh":
        init_random_params, predict = stax.serial(
            stax.Conv(16, (8, 8), padding="SAME", strides=(2, 2)),
            stax.Tanh,
            stax.AvgPool((2, 2), (1, 1)),
            stax.Conv(32, (4, 4), padding="VALID", strides=(2, 2)),
            stax.Tanh,
            stax.AvgPool((2, 2), (1, 1)),
            stax.Flatten,
            stax.Dense(32),
            stax.Tanh,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
    elif model_name == "cnn_cifar":
        init_random_params, predict = stax.serial(
            stax.Conv(32, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.Conv(32, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.AvgPool((2, 2), (2, 2)),
            stax.Conv(64, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.Conv(64, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.AvgPool((2, 2), (2, 2)),
            stax.Conv(128, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.Conv(128, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.AvgPool((2, 2), (2, 2)),
            stax.Flatten,
            stax.Dense(128),
            stax.Tanh,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
    elif model_name == "mlp_reg":
        init_random_params, predict = stax.serial(
            stax.Flatten,
            stax.BatchNorm(0),
            stax.Dense(512),
            stax.Gelu,
            stax.BatchNorm(0),
            stax.Dense(256),
            stax.Gelu,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
    elif model_name == "mlp_small_reg":
        init_random_params, predict = stax.serial(
            stax.Flatten,
            stax.BatchNorm(0),
            stax.Dense(256),
            stax.Gelu,
            stax.BatchNorm(0),
            stax.Dense(256),
            stax.Gelu,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
    elif model_name == "mlp_big_reg":
        init_random_params, predict = stax.serial(
            stax.Flatten,
            stax.BatchNorm(0),
            stax.Dense(1024),
            stax.Gelu,
            stax.BatchNorm(0),
            stax.Dense(1024),
            stax.Gelu,
            stax.BatchNorm(0),
            stax.Dense(512),
            stax.Gelu,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
    elif model_name == "mlp_reg_no_bn":
        init_random_params, predict = stax.serial(
            stax.Flatten,
            stax.Dense(512),
            stax.Gelu,
            stax.Dense(256),
            stax.Gelu,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
    elif model_name == "mlp":
        init_random_params, predict = stax.serial(
            stax.Flatten,
            stax.Dense(512),
            stax.Relu,
            stax.Dense(256),
            stax.Relu,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
    elif model_name == "cnn_big_aux_reg":
        conv=stax.serial(
            stax.Conv(32, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Gelu,
            stax.Conv(32, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Gelu,
            stax.AvgPool((2, 2), (2, 2)),
            stax.Conv(64, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Gelu,
            stax.Conv(64, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Gelu,
            stax.AvgPool((2, 2), (2, 2)),
            stax.Conv(128, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Gelu,
            stax.Conv(128, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Gelu,
            stax.AvgPool((2, 2), (2, 2)),
            stax.Flatten)
        init_random_params, predict = stax.serial(
            stax.parallel(conv,stax.Identity),
            stax.FanInConcat(),
            stax.BatchNorm(0),
            stax.Dense(256),
            stax.Gelu,
            stax.BatchNorm(0),
            stax.Dense(256),
            stax.Gelu,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
    elif model_name == "cnn_aux_reg":
        conv=stax.serial(
            stax.Conv(16, (8, 8), padding="SAME", strides=(2, 2)),
            stax.Gelu,
            stax.AvgPool((2, 2), (1, 1)),
            stax.Conv(32, (4, 4), padding="VALID", strides=(2, 2)),
            stax.Gelu,
            stax.AvgPool((2, 2), (1, 1)),
            stax.Flatten)
        init_random_params, predict = stax.serial(
            stax.parallel(conv,stax.Identity),
            stax.FanInConcat(),
            stax.BatchNorm(0),
            stax.Dense(256),
            stax.Gelu,
            stax.BatchNorm(0),
            stax.Dense(256),
            stax.Gelu,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)        
    elif model_name == "mlp_tanh":
        init_random_params, predict = stax.serial(
            stax.Flatten,
            stax.Dense(512),
            stax.Tanh,
            stax.Dense(256),
            stax.Tanh,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
    elif model_name == "linear":
        init_random_params, predict = stax.serial(stax.Flatten,stax.Dense(num_labels))
        _, init_params = init_random_params(rng, input_shape)

    else:
        raise ValueError(f"Unknown model: {model_name}")
    if num_labels==1:
        def predict_binary(params, inputs):
            logits = predict(params, inputs)
            return jnp.hstack([logits, jnp.zeros(logits.shape)])
        return init_params, predict_binary        
    # return initial model parameters and prediction function:
    return init_params, predict

# gets optimizer
def get_opt(opt,step_size,momentum):
    if opt == "sgd":
        opt_init, opt_update, get_params = optimizers.momentum(
            step_size, momentum
        )
    elif opt == "adam":
        opt_init, opt_update, get_params = optimizers.adam(step_size)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")
    return opt_init, opt_update, get_params