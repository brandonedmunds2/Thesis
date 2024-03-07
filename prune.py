# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import jax
import jax.numpy as jnp
from jax import jit
from flatten_util import ravel_pytree

# apply pruning mask
@jit
def pruner(value, prune_mask):
    flat,rev=ravel_pytree(value)
    return rev(flat*prune_mask)


def get_topk_mask_func(strategy):
    # generate pruning mask
    @jit
    def topk_mask(rng,value,density_fraction):
        value=abs(ravel_pytree(value)[0])
        shuffled_indices = jax.random.shuffle(rng, jnp.arange(0, jnp.size(value), dtype=jnp.int32))
        if(strategy=='magnitude'):
            indices = jnp.argsort(value[shuffled_indices])
        elif(strategy=='random' or strategy==None):
            indices=jnp.arange(0, jnp.size(value[shuffled_indices]), dtype=jnp.int32)
        k = jnp.round(density_fraction * jnp.size(value[shuffled_indices])).astype(jnp.int32)
        mask = jnp.greater_equal(jnp.arange(value[shuffled_indices].size), value[shuffled_indices].size - k)
        mask = jnp.zeros_like(mask).at[indices].set(mask)
        mask = mask.astype(jnp.int32)
        mask = jnp.zeros_like(mask).at[shuffled_indices].set(mask)
        return mask
    return topk_mask