from IPython import embed;embed()


import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage import transform
from skimage import filters
from skimage import segmentation


def sigmoid_fun(x):
    """Apply sigmoid to maps before mAP."""
    return 1 / (1 + np.exp(x))


im_names = [
    'EGFR_3361830_6.jpg',
    'EGFR_3361851_7.jpg',
    'KRAS_3361802_14.jpg',
    'KRAS_3361802_18.jpg',
    'KRAS_3361802_19.jpg'
]
for im_name in im_names:
    im = io.imread('/media/data_cifs/andreas/pathology/2018-04-26/mar2019/LMD/annotation_imgs/%s' % im_name)[:2000, :2000, :]
    feed_dict = {
        val_dict[dict_image_key]: im[None, ...],
        val_dict[dict_label_key]: it_labs}
    it_val_dict = sess.run(val_dict, feed_dict=feed_dict)

    if 'KRAS' in im_name:
        pos, neg = 0, 2
    elif 'EGFR' in im_name:
        pos, neg = 1, 3
    else:
        raise NotImplementedError

    f = plt.figure(figsize=(10, 10))
    trans_im = sigmoid_fun(transform.resize(filters.gaussian((it_val_dict['activity'].squeeze()[..., pos]), 3), [2000, 2000]))
    marked = segmentation.mark_boundaries(im, (trans_im > .5).astype(int).squeeze(), color=None, outline_color=[1, 0, 0], mode='outer')
    plt.imshow(marked[100:-100, 100:-100])
    # plt.imshow(np.concatenate((im.astype(np.float32) / 255., trans_im[..., None]), axis=-1))
    plt.savefig('pos_%s.png' % im_name.split('.')[0])
    # plt.show()
    plt.close(f)

    f = plt.figure(figsize=(10, 10))
    trans_im = 1 - sigmoid_fun(transform.resize(filters.gaussian((it_val_dict['activity'].squeeze()[..., neg]), 3), [2000, 2000]))
    marked = segmentation.mark_boundaries(im, (trans_im > .5).astype(int).squeeze(), color=None, outline_color=[0, 0, 1], mode='outer')
    plt.imshow(marked[100:-100, 100:-100])
    # plt.imshow(np.concatenate((im, trans_im[..., None]), axis=-1))
    plt.savefig('neg_%s.png' % im_name.split('.')[0])
    # plt.show()
    plt.close(f)
