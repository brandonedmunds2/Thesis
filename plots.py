import jax.numpy as jnp
from jax import vmap

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle
matplotlib.use('Agg')

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def norm_plot(names,norms,stds):
    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for name,n,s,c in zip(names,norms,stds,colors):
        plt.errorbar(jnp.arange(1,n.shape[0]+1),n,s,alpha=0.5,color=c)
        plt.plot(jnp.arange(1,n.shape[0]+1),n,label=name,color=c)
    plt.title(f'Gradient 2-norms')
    plt.xlabel('Iteration')
    plt.ylabel('2-norm')
    plt.legend()
    plt.savefig(f'norm2.png')
    plt.close()

def pca_plot(name,vals,targets,comp1=0,comp2=1):
    vals=PCA().fit_transform(vals)
    x = vals[:, comp1]
    y = vals[:, comp2]
    plt.figure()
    unique_targets = jnp.unique(targets)
    for target in unique_targets:
        indices = jnp.where(targets == target)[0]
        plt.scatter(x[indices], y[indices], label=f'{target}',alpha=0.5)
    plt.title(f'PCA Transformed Iteration {name} Gradients Scatter Plot')
    plt.xlabel(f'Principal Component {comp1+1}')
    plt.ylabel(f'Principal Component {comp2+1}')
    plt.legend()
    plt.savefig(f'pca_{name}.png')
    plt.close()

def recons_plot(name,preds,labels,im_shape,cmap):
    fig_dim=10
    f,ax=plt.subplots(2,fig_dim)
    for fd in range(fig_dim):
        ax[0][fd].imshow(labels[fd].reshape(im_shape), cmap=cmap)
        ax[1][fd].imshow(preds[fd].reshape(im_shape), cmap=cmap)
        ax[0][fd].axis('off')
        ax[1][fd].axis('off')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    f.savefig(f'recons_{name}.png')
    plt.close()

def cos_sim_plot(name,v1s,v2s,targets):
    sims= vmap(lambda x,y: jnp.dot(x,y)/(jnp.linalg.norm(x)*jnp.linalg.norm(y)+1e-7),in_axes=(0,0))(v1s,v2s)
    unique_targets=jnp.unique(targets)
    sims=[sims.at[jnp.where(targets==c)[0]].get() for c in unique_targets]
    plt.figure()
    for s,c in zip(sims,unique_targets):
        plt.hist(s,label=f'{c}',alpha=0.5)
    plt.title(f'Histogram of Iteration {name} Gradient Cosine Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'cossim_{name}.png')
    plt.close()

def pca_cos_sim_plot(name,v1s,v2s,targets):
    p=PCA().fit(v1s)
    sims= vmap(lambda x,y: jnp.dot(x,y)/(jnp.linalg.norm(x)*jnp.linalg.norm(y)+1e-7),in_axes=(0,None))(p.components_,v2s[0])
    plt.figure()
    plt.plot(jnp.arange(1,101),sims[:100],alpha=0.5,label='Cosine Similarity')
    plt.plot(jnp.arange(1,101),p.explained_variance_ratio_[:100],alpha=0.5,label='Explained Variance')
    plt.title(f'Iteration {name} Base PCA and Perturbation Gradient Cosine Similarities')
    plt.xlabel('Principal Component')
    plt.ylabel('Ratio')
    plt.legend()
    plt.savefig(f'pcacossim_{name}.png')
    plt.close()

def average_cos_sim_plot(name,v1s,v2s,targets):
    sims= vmap(lambda x,y: jnp.dot(x,y)/(jnp.linalg.norm(x)*jnp.linalg.norm(y)+1e-7),in_axes=(None,0))(jnp.mean(v1s,axis=0),v2s)
    plt.figure()
    plt.hist(sims)
    plt.title(f'Iteration {name} Base Average and Perturbation Gradient Cosine Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.savefig(f'avgcossim_{name}.png')
    plt.close()

def acc_dfil_plot(test_accs,etas_squared,name):
    iterations=jnp.arange(1,test_accs.shape[0]+1)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Test Accuracy', color='tab:blue')
    ax1.plot(iterations, test_accs, color='tab:blue', marker='o', label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('dFIL', color='tab:red')
    if name!='max':
        ax2.plot(iterations, jnp.mean(jnp.sqrt(etas_squared),axis=-1), color='tab:red', linestyle='--', marker='s', label=f'Mean dFIL')
        ax2.plot(iterations, jnp.median(jnp.sqrt(etas_squared),axis=-1), color='tab:red', linestyle='-.', marker='^', label=f'Median dFIL')
    else:
        ax2.plot(iterations, jnp.max(jnp.sqrt(etas_squared),axis=-1), color='tab:red', linestyle=':', marker='d', label=f'Max dFIL')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.title('Test Accuracy and dFIL vs. Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'accdfil_{name}.png')
    plt.close()