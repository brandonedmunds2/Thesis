#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import pickle

import jax
import jax.numpy as jnp
import jax.random as jnr
import hydra

from jax import vmap, nn
from jax import grad
from jax.example_libraries import optimizers
from jax.tree_util import tree_flatten, tree_unflatten
from prune import get_topk_mask_func, pruner
import plots

import torch
import math
import accountant
import datasets
import trainer
import utils
import time

# train FIM model
def reg_train(*,rng=None,verbosity=0,reg_update=None,reg_loss=None,reg_curr_lam=None,reg_get_params=None,reg_opt_state=None,get_params=None,opt_state=None,train_features=None,train_labels=None,test_features=None,test_labels=None,batch_size=None,num_epochs=None,weight_decay=None,prune_mask=None,prune_only_grads=None,loss=None,standardize=False):
    # calculate mean and std to statically standardize grads
    if standardize:
        grads=vmap(grad(loss, argnums=1),in_axes=(0,None,0,0))(jnr.split(rng,train_features.shape[0]),get_params(opt_state) if prune_only_grads else pruner(get_params(opt_state),prune_mask),jnp.expand_dims(train_features,1),jnp.expand_dims(train_labels,1))
        grads, _ = tree_flatten(grads)
        grads=jnp.concatenate([i.reshape(i.shape[0],-1) for i in grads],axis=1)
        grads_mean=jnp.mean(grads,axis=0)
        grads_std=jnp.std(grads,axis=0)
        grads_std=jnp.where(grads_std==0,1,grads_std)
        del grads
    else:
        grads_mean=None
        grads_std=None

    # get data
    data_stream, num_batches = datasets.get_datastream(
        train_features,train_labels, batch_size
    )
    batches=data_stream()

    reg_num_iters = 0
    for reg_epoch in range(num_epochs):
        train_round_reg_loss=0
        test_round_reg_loss=0
        for reg_batch_counter in range(num_batches):
            # get next batch:
            reg_num_iters += 1
            reg_inputs, reg_targets, reg_batch_idx = next(batches)
            rng, subrng = jnr.split(rng)
            _,_,_,reg_opt_state = reg_update(i=reg_num_iters, rng=subrng, opt_state=opt_state, reg_opt_state=reg_opt_state,prune_mask=prune_mask, inputs=reg_inputs,targets=reg_targets,reg_curr_lam=reg_curr_lam, sigma=0, weight_decay=weight_decay,grads_mean=grads_mean,grads_std=grads_std)
            rng, subrng = jnr.split(rng)
            train_round_reg_loss+=jnp.mean(vmap(reg_loss, in_axes=(0,None,None,None, 0, 0,None,None,None))(jnr.split(subrng,reg_inputs.shape[0]),get_params(opt_state) if prune_only_grads else pruner(get_params(opt_state),prune_mask), reg_get_params(reg_opt_state),prune_mask,jnp.expand_dims(reg_inputs, 1),jnp.expand_dims(reg_targets, 1),reg_curr_lam,grads_mean,grads_std))
            test_round_reg_loss+=jnp.mean(vmap(reg_loss, in_axes=(0,None,None,None, 0, 0,None,None,None))(jnr.split(subrng,test_features.shape[0]),get_params(opt_state) if prune_only_grads else pruner(get_params(opt_state),prune_mask), reg_get_params(reg_opt_state),prune_mask,jnp.expand_dims(test_features, 1),jnp.expand_dims(test_labels, 1),reg_curr_lam,grads_mean,grads_std))
        if verbosity > 0:
            logging.info(f'epoch {reg_epoch} train reg loss: {train_round_reg_loss}')
            logging.info(f'epoch {reg_epoch} test reg loss: {test_round_reg_loss}')
    return reg_opt_state,grads_mean,grads_std

def train(*,rng=None,verbosity=0,metric_funcs=[],plot_grads=False,predict=None,update=None,get_params=None,opt_state=None,train_features=None,train_labels=None,test_features=None,test_labels=None,batch_size=None,num_epochs=None,weight_decay=None,prune_mask=None,prune_only_grads=None,sigma=0,delta=None,do_accounting=False,subsampling_factor=None,multiplier=None,fil_accountant=None,dp_accountant=None,do_fam=False,reg_update=None,reg_loss=None,reg_lam=None,reg_lam_decay=None,reg_min_lam=None,reg_get_params=None,reg_opt_state=None,reg_train_features=None,reg_train_labels=None,reg_batch_size=None,reg_num_epochs=None,reg_epoch_decay=None,reg_min_batch_epochs=None,reg_weight_decay=None,reg_opt_init=None,reg_init_params=None,loss=None,standardize=False):

    # get the data
    data_stream, num_batches = datasets.get_datastream(
        train_features,train_labels, batch_size
    )
    batches=data_stream()

    # set up results storage
    if do_accounting:
        etas_squared = jnp.zeros((num_epochs, train_features.shape[0]))
        epsilons = jnp.zeros(num_epochs)
        rdp_epsilons = jnp.zeros(num_epochs)
    if plot_grads:
        if do_fam:
            base_norms = jnp.zeros(num_epochs*num_batches)
            pert_norms = jnp.zeros(num_epochs*num_batches)
            base_norm_stds = jnp.zeros(num_epochs*num_batches)
            pert_norm_stds = jnp.zeros(num_epochs*num_batches)
        grad_norms = jnp.zeros(num_epochs*num_batches)
        grad_norm_stds = jnp.zeros(num_epochs*num_batches)
        train_mets0 = jnp.zeros(num_epochs)
        test_mets0 = jnp.zeros(num_epochs)
    num_iters = 0

    for epoch in range(num_epochs):
        if epoch > 0 and do_accounting:
            etas_squared = etas_squared.at[epoch].set(etas_squared[epoch-1])

        for batch_counter in range(num_batches):
            num_iters += 1
            inputs, targets, batch_idx = next(batches)

            if do_fam:
                # reset FIM model when statically standardizing
                if standardize:
                    reg_opt_state = reg_opt_init(reg_init_params)
                # do lambda and epoch decay
                reg_curr_epochs=max(reg_min_batch_epochs,reg_num_epochs-(reg_epoch_decay*(num_iters-1)))
                reg_curr_lam=max(reg_min_lam,reg_lam-(reg_lam_decay*(num_iters-1)))
                if reg_curr_epochs < 1:
                    rng, subrng = jnr.split(rng)
                    reg_curr_epochs=jnr.choice(subrng,2,p=[1-reg_curr_epochs,reg_curr_epochs])
                rng, subrng = jnr.split(rng)
                reg_opt_state,grads_mean,grads_std=reg_train(rng=subrng,verbosity=verbosity,reg_update=reg_update,reg_loss=reg_loss,reg_curr_lam=reg_curr_lam,reg_get_params=reg_get_params,reg_opt_state=reg_opt_state,get_params=get_params,opt_state=opt_state,train_features=reg_train_features,train_labels=reg_train_labels,test_features=inputs,test_labels=targets,batch_size=reg_batch_size,num_epochs=reg_curr_epochs,weight_decay=reg_weight_decay,prune_mask=prune_mask,prune_only_grads=prune_only_grads,loss=loss,standardize=standardize)
            else:
                grads_mean=None
                grads_std=None

            # update privacy loss:
            if sigma > 0 and do_accounting:
                rng, subrng = jnr.split(rng)
                # these etas_batch need to be divided by d
                etas_batch = fil_accountant(subrng, get_params(opt_state) if prune_only_grads else pruner(get_params(opt_state),prune_mask), reg_get_params(reg_opt_state) if do_fam else None,prune_mask,inputs,targets,grads_mean,grads_std) / sigma / multiplier
                etas_squared = etas_squared.at[epoch, batch_idx].add(
                    subsampling_factor * jnp.power(etas_batch, 2)/jnp.prod(jnp.asarray(train_features.shape[1:])), unique_indices=True
                )
                del etas_batch

            rng, subrng = jnr.split(rng)
            grads_flat_init,base_grads,pert,opt_state = update(i=num_iters, rng=subrng, opt_state=opt_state,reg_opt_state=reg_opt_state,prune_mask=prune_mask,inputs=inputs,targets=targets,reg_curr_lam=None, sigma=sigma, weight_decay=weight_decay,grads_mean=grads_mean,grads_std=grads_std)

            if plot_grads:
                grads_flat_init=jnp.concatenate([i.reshape(i.shape[0],-1) for i in grads_flat_init],axis=1)
                grad_norm=jnp.sqrt(jnp.sum(jnp.power(grads_flat_init,2),axis=1)+1e-7)
                grad_norm_std=jnp.std(grad_norm)
                grad_norms=grad_norms.at[num_iters-1].set(jnp.mean(grad_norm))
                grad_norm_stds=grad_norm_stds.at[num_iters-1].set(jnp.mean(grad_norm_std))
                plots.pca_plot(f'{num_iters} Final',grads_flat_init,jnp.argmax(targets, axis=-1))
                if do_fam:
                    base_grads=jnp.concatenate([i.reshape(i.shape[0],-1) for i in base_grads],axis=1)
                    pert=jnp.concatenate([i.reshape(i.shape[0],-1) for i in pert],axis=1)
                    base_grads=base_grads.at[:,jnp.nonzero(prune_mask)].get().squeeze()
                    base_norm=jnp.sqrt(jnp.sum(jnp.power(base_grads,2),axis=1)+1e-7)
                    base_norm_std=jnp.std(base_norm)
                    base_norms=base_norms.at[num_iters-1].set(jnp.mean(base_norm))
                    base_norm_stds=base_norm_stds.at[num_iters-1].set(jnp.mean(base_norm_std))
                    plots.pca_plot(f'{num_iters} Base',base_grads,jnp.argmax(targets, axis=-1))
                    pert_norm=jnp.sqrt(jnp.sum(jnp.power(pert,2),axis=1)+1e-7)
                    pert_norm_std=jnp.std(pert_norm)
                    pert_norms=pert_norms.at[num_iters-1].set(jnp.mean(pert_norm))
                    pert_norm_stds=pert_norm_stds.at[num_iters-1].set(jnp.mean(pert_norm_std))
                    plots.pca_plot(f'{num_iters} Perturbation',pert,jnp.argmax(targets, axis=-1))
                    plots.cos_sim_plot(num_iters,base_grads,pert,jnp.argmax(targets, axis=-1))
                    plots.pca_cos_sim_plot(num_iters,base_grads,pert,jnp.argmax(targets, axis=-1))
                    plots.average_cos_sim_plot(num_iters,base_grads,pert,jnp.argmax(targets, axis=-1))

        if prune_only_grads:
            params = get_params(opt_state)
        else:
            params = pruner(get_params(opt_state),prune_mask)

        train_predictions = utils.batch_predict(predict, params, train_features, batch_size)
        test_predictions = utils.batch_predict(predict, params, test_features, batch_size)
        train_mets=[metric_func(train_predictions, train_labels) for i,metric_func in enumerate(metric_funcs) if i==0 or epoch==num_epochs-1]
        test_mets=[metric_func(test_predictions, test_labels) for i,metric_func in enumerate(metric_funcs) if i==0 or epoch==num_epochs-1]
        if plot_grads:
            train_mets0=train_mets0.at[epoch].set(train_mets[0])
            test_mets0=test_mets0.at[epoch].set(test_mets[0])

        if sigma > 0 and do_accounting:
            mean_eta = jnp.mean(jnp.sqrt(etas_squared[epoch]))
            median_eta = jnp.median(jnp.sqrt(etas_squared[epoch]))
            max_eta = jnp.sqrt(etas_squared[epoch]).max()
            epsilon = dp_accountant(num_iters, train_features.shape[0], delta)
            epsilons = epsilons.at[epoch].set(epsilon)
            rdp_epsilon = dp_accountant(num_iters, train_features.shape[0], delta, alpha=2)
            rdp_epsilons = rdp_epsilons.at[epoch].set(rdp_epsilon)

        # print out progress:
        if verbosity > 0:
            logging.info(f"Epoch {epoch + 1}:")
            [logging.info(f" -> training metric {i+1} = {train_met:.4f}") for i,train_met in enumerate(train_mets)]
            [logging.info(f" -> test metric {i+1} = {test_met:.4f}") for i,test_met in enumerate(test_mets)]
            if sigma > 0 and do_accounting:
                logging.info(f" -> Mean FIL privacy loss = {mean_eta:.4f}")
                logging.info(f" -> Median FIL privacy loss = {median_eta:.4f}")
                logging.info(f" -> Max FIL privacy loss = {max_eta:.4f}")
                logging.info(f" -> DP privacy loss = ({epsilon:.4f}, {delta:.2e})")
                logging.info(f" -> 2-RDP privacy loss = {rdp_epsilon:.4f}")

        del params
    
    if plot_grads:
        if do_fam:
            plots.norm_plot(('Base','Final','Perturbation'),(base_norms,grad_norms,pert_norms),(base_norm_stds,grad_norm_stds,pert_norm_stds))
        else:
            plots.norm_plot(('Final',), (grad_norms,),(grad_norm_stds,))
        plots.acc_dfil_plot(test_mets0,etas_squared,'max')
        plots.acc_dfil_plot(test_mets0,etas_squared,'meanmed')
    return opt_state

@hydra.main(config_path="configs", config_name="e2")
def main(cfg):

    logging.info(f"Running using JAX {jax.__version__}...")
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Torch device: {torch_device}")
    logging.info(cfg)
    rng = jnr.PRNGKey(int(time.time()))

    if cfg.gen_data:

        # get image data
        splits=datasets.get_dataset_splits(cfg.dataset,cfg.binary,cfg.pca_dims,cfg.data_amount,cfg.public_data_amount,cfg.do_attack,cfg.attack.test_samples,cfg.attack.train_samples)
        if cfg.do_attack:
            test_images,test_labels,private_images,private_labels,public_images,public_labels,attack_train_images,attack_train_labels,attack_test_images,attack_test_labels=splits
        else:
            test_images,test_labels,private_images,private_labels,public_images,public_labels=splits
        
        image_input_shape=(-1,)+private_images.shape[1:]
        d=jnp.prod(jnp.asarray(private_images.shape[1:]))
        num_labels=private_labels.shape[-1]

        # get pretrain model
        rng, subrng = jnr.split(rng)
        init_params, predict = utils.get_model(subrng, cfg.model, image_input_shape, 1 if num_labels==2 else num_labels)

        # get pretrain model optimizer
        opt_init, opt_update, get_params=utils.get_opt(cfg.pretrain.optimizer,cfg.pretrain.step_size,cfg.pretrain.momentum_mass)
        opt_state = opt_init(init_params)

        # get pretrain model update method
        loss=trainer.get_loss_func(predict=predict,loss_type="entropy")
        regd_grad_func=trainer.get_grad_func(loss=loss, argnum=1,norm_clip=0, soft_clip=True,do_fam=False,reg_predict=None,size=None,cnn_reg=None)
        update = trainer.get_update_func(get_params=get_params, reg_get_params=None, grad_func=regd_grad_func, opt_update=opt_update,prune_only_grads=cfg.prune_only_grads, norm_clip=0,argnum=1)

        # initalize pruning mask
        topk_mask=get_topk_mask_func(cfg.prune_strat)
        rng, subrng = jnr.split(rng)
        prune_mask=topk_mask(subrng,get_params(opt_state),1.0)

        # train pretrain model
        rng, subrng = jnr.split(rng)
        opt_state=train(rng=subrng,verbosity=cfg.verbosity,metric_funcs=(utils.accuracy,),plot_grads=False,predict=predict,update=update,get_params=get_params,opt_state=opt_state,train_features=public_images,train_labels=public_labels,test_features=test_images,test_labels=test_labels,batch_size=cfg.pretrain.batch_size,num_epochs=cfg.pretrain.num_epochs,weight_decay=cfg.pretrain.weight_decay,prune_mask=prune_mask,prune_only_grads=cfg.prune_only_grads)

        # update prune mask
        rng, subrng = jnr.split(rng)
        prune_mask=topk_mask(subrng,get_params(opt_state),cfg.prune_dense)

        # use pruned, pretrained model weights
        if cfg.prune_only_grads:
            init_params = get_params(opt_state)
        else:
            init_params = pruner(get_params(opt_state),prune_mask)

        if cfg.do_fam:
            # get fam model
            if cfg.reg.model.startswith("cnn"):
                reg_input_shape = (image_input_shape, (-1,2*jnp.count_nonzero(prune_mask)+num_labels))
            else:
                reg_input_shape = (-1, 2*jnp.count_nonzero(prune_mask)+num_labels+d)
            rng, subrng = jnr.split(rng)
            reg_init_params, reg_predict = utils.get_model(subrng, cfg.reg.model, reg_input_shape, jnp.count_nonzero(prune_mask))

            # get optimizer for fam
            reg_opt_init, reg_opt_update, reg_get_params=utils.get_opt(cfg.reg.optimizer,cfg.reg.step_size,cfg.reg.momentum_mass)
        else:
            reg_predict=None
            reg_get_params=None
            reg_init_params=None
            reg_opt_update=None
            reg_opt_init=None

        # get optimizer for main model
        opt_init, opt_update, get_params=utils.get_opt(cfg.final.optimizer,cfg.final.step_size,cfg.final.momentum_mass)

        # get main model update method
        loss=trainer.get_loss_func(predict=predict,loss_type="entropy")
        regd_grad_func=trainer.get_grad_func(loss=loss, argnum=1,norm_clip=cfg.final.norm_clip, soft_clip=True,do_fam=cfg.do_fam,reg_predict=reg_predict,size=jnp.nonzero(prune_mask),cnn_reg=cfg.reg.model.startswith("cnn"),standardize=cfg.do_static_standardize,ret_grads=cfg.plot_grads)
        update = trainer.get_update_func(get_params=get_params, reg_get_params=reg_get_params, grad_func=regd_grad_func, opt_update=opt_update,prune_only_grads=cfg.prune_only_grads, norm_clip=cfg.final.norm_clip,argnum=1,ret_grads=cfg.plot_grads)

        if cfg.do_fam:
            # get fam model update method
            reg_loss=trainer.get_loss_func(predict=reg_predict,loss_type="reg",lam_pow=cfg.reg.lam_pow,entropy_loss=loss,regd_grad_func=regd_grad_func,size=jnp.nonzero(prune_mask),cnn_reg=cfg.reg.model.startswith("cnn"),standardize=cfg.do_static_standardize,fisher_iters=cfg.reg.fisher_iters,fisher_batch_size=cfg.reg.fisher_batch_size)
            reg_grad_func=trainer.get_grad_func(loss=reg_loss, argnum=2,norm_clip=0, soft_clip=True)
            reg_update=trainer.get_update_func(get_params=get_params,reg_get_params=reg_get_params,grad_func=reg_grad_func,opt_update=reg_opt_update,prune_only_grads=cfg.prune_only_grads,norm_clip=0,argnum=2)
        else:
            reg_loss=None
            reg_update=None

        # get privacy accountants
        gelu_approx = 1.115
        fil_accountant = accountant.get_grad_jacobian_trace_func(
            regd_grad_func,
            fisher_iters=cfg.fisher_iters,
            fisher_batch_size=cfg.fisher_batch_size
        )
        dp_accountant = accountant.get_dp_accounting_func(cfg.final.batch_size, cfg.final.sigma / gelu_approx)

        if cfg.do_attack:
            attack_train_params = jnp.zeros((cfg.attack.train_samples, jnp.count_nonzero(prune_mask)))
            attack_test_params = jnp.zeros((cfg.attack.test_samples, jnp.count_nonzero(prune_mask)))
            gen_steps=cfg.attack.train_samples+cfg.attack.test_samples
        else:
            gen_steps=1

        # train main models
        for gen_attack_data_step in range(gen_steps):
            logging.info(f"Attack Data Generation Step {gen_attack_data_step+1}:")
            if cfg.same_rng:
                rng=jnr.PRNGKey(0)
            if cfg.do_accounting and gen_attack_data_step==1:
                break
            
            # reinitialize models
            opt_state = opt_init(init_params)
            if cfg.do_fam:
                reg_opt_state = reg_opt_init(reg_init_params)
            else:
                reg_opt_state=None

            # make next attack dataset
            if cfg.do_attack:
                if gen_attack_data_step==0:
                    private_images=jnp.concatenate((private_images,jnp.expand_dims(attack_train_images[gen_attack_data_step],0)))
                    private_labels=jnp.concatenate((private_labels,jnp.expand_dims(attack_train_labels[gen_attack_data_step],0)))
                else:
                    if gen_attack_data_step<cfg.attack.train_samples:
                        private_images=private_images.at[-1].set(attack_train_images[gen_attack_data_step])
                        private_labels=private_labels.at[-1].set(attack_train_labels[gen_attack_data_step])
                    else:
                        private_images=private_images.at[-1].set(attack_test_images[gen_attack_data_step-cfg.attack.train_samples])
                        private_labels=private_labels.at[-1].set(attack_test_labels[gen_attack_data_step-cfg.attack.train_samples])

            # train the model
            rng, subrng = jnr.split(rng)
            subsampling_factor,multiplier=utils.calc_sub_fact(gelu_approx,cfg.final.sigma,cfg.final.norm_clip,cfg.final.delta,private_images.shape[0],cfg.final.batch_size)
            opt_state=train(rng=subrng,verbosity=cfg.verbosity,metric_funcs=(utils.accuracy,),plot_grads=cfg.plot_grads,predict=predict,update=update,get_params=get_params,opt_state=opt_state,train_features=private_images,train_labels=private_labels,test_features=test_images,test_labels=test_labels,batch_size=cfg.final.batch_size,num_epochs=cfg.final.num_epochs,weight_decay=cfg.final.weight_decay,prune_mask=prune_mask,prune_only_grads=cfg.prune_only_grads,sigma=cfg.final.sigma,delta=cfg.final.delta,do_accounting=cfg.do_accounting,subsampling_factor=subsampling_factor,multiplier=multiplier,fil_accountant=fil_accountant,dp_accountant=dp_accountant,do_fam=cfg.do_fam,reg_update=reg_update,reg_loss=reg_loss,reg_lam=cfg.reg.lam,reg_lam_decay=cfg.reg.lam_decay,reg_min_lam=cfg.reg.min_lam,reg_get_params=reg_get_params,reg_opt_state=reg_opt_state,reg_train_features=public_images,reg_train_labels=public_labels,reg_batch_size=cfg.reg.batch_size,reg_num_epochs=cfg.reg.num_epochs,reg_epoch_decay=cfg.reg.epoch_decay,reg_min_batch_epochs=cfg.reg.min_batch_epochs,reg_weight_decay=cfg.reg.weight_decay,reg_opt_init=reg_opt_init,reg_init_params=reg_init_params,loss=loss,standardize=cfg.do_static_standardize)

            # save the model parameters
            if cfg.do_attack:
                if cfg.prune_only_grads:
                    params = get_params(opt_state)
                else:
                    params = pruner(get_params(opt_state),prune_mask)            
                if gen_attack_data_step<cfg.attack.train_samples:
                    attack_train_params=attack_train_params.at[gen_attack_data_step].set(jnp.concatenate(tree_flatten(params)[0],axis=None).at[jnp.nonzero(prune_mask)].get())
                else:
                    attack_test_params=attack_test_params.at[gen_attack_data_step-cfg.attack.train_samples].set(jnp.concatenate(tree_flatten(params)[0],axis=None).at[jnp.nonzero(prune_mask)].get())

    # do the attack
    if cfg.do_attack and not cfg.do_accounting:

        if cfg.gen_data:
            if not cfg.no_labels:
                attack_train_params=jnp.concatenate((attack_train_params,attack_train_labels),axis=1)
                attack_test_params=jnp.concatenate((attack_test_params,attack_test_labels),axis=1)

            attack_train_params,attack_test_params=utils.standardize_data(attack_train_params,attack_test_params)

            with open('data.pkl', 'wb') as file:
                pickle.dump((attack_train_params,attack_test_params,attack_train_images,attack_test_images), file)
        else:
            with open(cfg.in_data_loc, 'rb') as file:
                attack_train_params,attack_test_params,attack_train_images,attack_test_images = pickle.load(file)

        full_im_shape=attack_train_images.shape[1:]

        attack_train_images=attack_train_images.reshape((attack_train_images.shape[0],-1))
        attack_test_images=attack_test_images.reshape((attack_test_images.shape[0],-1))
        
        # set up model:
        rng, subrng = jnr.split(rng)
        init_params, predict = utils.get_model(subrng, cfg.attack.model, (-1,attack_train_params.shape[-1]), jnp.prod(jnp.asarray(attack_train_images.shape[1:])))

        # get model attack optimizer
        opt_init, opt_update, get_params=utils.get_opt(cfg.attack.optimizer,cfg.attack.step_size,cfg.attack.momentum_mass)
        
        # initialize the attack model
        opt_state = opt_init(init_params)

        # get attack model methods
        loss=trainer.get_loss_func(predict=predict,loss_type="mse")
        regd_grad_func=trainer.get_grad_func(loss=loss, argnum=1,norm_clip=0, soft_clip=True)
        update = trainer.get_update_func(get_params=get_params, reg_get_params=None, grad_func=regd_grad_func, opt_update=opt_update,prune_only_grads=cfg.prune_only_grads, norm_clip=0,argnum=1)

        # reset the prune mask
        topk_mask=get_topk_mask_func(cfg.prune_strat)
        rng, subrng = jnr.split(rng)
        prune_mask=topk_mask(subrng,get_params(opt_state),1.0)

        # train the attack model
        rng, subrng = jnr.split(rng)
        opt_state=train(rng=subrng,verbosity=1,metric_funcs=(utils.re,utils.get_lpips_func(full_im_shape,torch_device)),plot_grads=False,predict=predict,update=update,get_params=get_params,opt_state=opt_state,train_features=attack_train_params,train_labels=attack_train_images,test_features=attack_test_params,test_labels=attack_test_images,batch_size=cfg.attack.batch_size,num_epochs=cfg.attack.num_epochs,weight_decay=cfg.attack.weight_decay,prune_mask=prune_mask,prune_only_grads=cfg.prune_only_grads)

        # visualize reconstructions
        if len(full_im_shape)!=1:
            im_shape=[i for i in full_im_shape if i!=1]
            cmap=None if len(im_shape)!=2 else "gray"
            if cfg.prune_only_grads:
                params = get_params(opt_state)
            else:
                params = pruner(get_params(opt_state),prune_mask)
            train_predictions = utils.batch_predict(predict, params, attack_train_params, cfg.attack.batch_size)
            test_predictions = utils.batch_predict(predict, params, attack_test_params, cfg.attack.batch_size)
            plots.recons_plot('train',train_predictions,attack_train_images,im_shape,cmap)
            plots.recons_plot('test',test_predictions,attack_test_images,im_shape,cmap)

    
    logging.info(cfg)


# run all the things:
if __name__ == "__main__":
    main()
