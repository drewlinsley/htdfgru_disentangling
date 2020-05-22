import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import sys
from tqdm import tqdm

def get_conv1(weight_key):
    conv1 = np.load(weight_key)
    conv1_rev = conv1.transpose(3,0,1,2)
    conv1_min = conv1_rev.min(1,keepdims=True).min(2,keepdims=True).min(-1,keepdims=True)
    conv1_max = conv1_rev.max(1,keepdims=True).max(2,keepdims=True).max(-1,keepdims=True)
    conv1_norm = (conv1_rev - conv1_min)/(conv1_max - conv1_min)
    return conv1_norm

def rectify_pr(weight_key):
    eps = 1e-7
    p_r = np.load(weight_key)

    p_r_rev = p_r.transpose(2,3,0,1)
    pr_pos = np.maximum(p_r_rev, 0)
    pr_neg = np.minimum(p_r_rev, 0)
    pr_neg = pr_neg*-1.
    pr_pos_min = pr_pos.min(2,keepdims=True).min(3,keepdims=True)
    pr_pos_max = pr_pos.max(2,keepdims=True).max(3,keepdims=True)
    pr_neg_min = pr_neg.min(2,keepdims=True).min(3,keepdims=True)
    pr_neg_max = pr_neg.max(2,keepdims=True).max(3,keepdims=True)
    pr_pos_norm = (pr_pos-pr_pos_min)/(eps+(pr_pos_max-pr_pos_min))
    pr_neg_norm = (pr_neg-pr_neg_min)/(eps+(pr_neg_max-pr_neg_min))
    pr_neg_norm = pr_neg_norm*-1.
    pr_norm = pr_pos_norm + pr_neg_norm
    print pr_norm.max(), pr_norm.min()
    return pr_norm

def plot_connectivity_matrix(i,j,norm_pr,conv1_norm,out_file,dpi=500):
    def plot_conv_kernels(gs1,conv1_norm):
        for jj in range(1,j):
            plt.subplot(gs1[i-1,jj])
            if conv1_norm.shape[-1] == 1:
                plt.imshow(conv1_norm[jj-1,:,:,:].squeeze(),cmap='gray')
            else:
                plt.imshow(conv1_norm[jj-1,:,:,:],cmap='gray')
            plt.axis('off')
        for ii in range(j-1):
            plt.subplot(gs1[ii,0])
            if conv1_norm.shape[-1] == 1:
                plt.imshow(conv1_norm[ii,:,:,:].squeeze(),cmap='gray')
            else:
                plt.imshow(conv1_norm[ii,:,:,:],cmap='gray')
            plt.axis('off')

    def plot_pr_kernels(gs1,norm_pr):
        for ii in tqdm(range(i),desc='I'):
            for jj in tqdm(range(j),desc='J'):
                if (ii+1)<jj or (ii==i-1 and jj==0):
                    continue
                if ii==i-1 and jj!=0:
                    continue
                if jj==0 and ii!=(j-1):
                    continue
                plt.subplot(gs1[ii,jj])
                plt.imshow(norm_pr[ii,jj-1,:,:],vmin=-1, vmax=1, cmap=plt.get_cmap('RdYlGn_r'))
                plt.axis('off')

    fig, ax = plt.subplots(dpi=dpi)
    gs1 = gridspec.GridSpec(i, j)
    gs1.update(wspace=0.0, hspace=0.1)
    plot_conv_kernels(gs1,conv1_norm)
    plot_pr_kernels(gs1,norm_pr)
    plt.subplots_adjust(right=0.8)
    cb_ax=fig.add_axes([0.85, 0.15, 0.015, 0.5])
    plt.colorbar(cax=cb_ax)
    #    plt.show()
    plt.savefig('%s_%s.eps'%(out_file,dpi),format='eps',dpi=dpi)
    plt.close()

if __name__=="__main__":
    conv_path, association_path, out_file_name = sys.argv[1], sys.argv[2], sys.argv[3]
    conv1_norm = get_conv1(conv_path)
    pr_norm = rectify_pr(association_path)
    nKernels = conv1_norm.shape[0]
    plot_connectivity_matrix(nKernels+1, nKernels+1, pr_norm, conv1_norm, out_file_name, dpi=1000)