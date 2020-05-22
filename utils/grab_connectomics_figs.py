from IPython import embed; embed()

# /media/data_cifs/cluttered_nist_experiments/checkpoints/gammanet_t8_per_pixel_snemi_combos_200_2019_03_22_22_57_22_303843/model_1775.ckpt-1775
# /media/data_cifs/cluttered_nist_experiments/checkpoints/gammanet_t8_per_pixel_snemi_combos_200_2019_03_22_23_09_46_988210/model_2525.ckpt-2525
# /media/data_cifs/cluttered_nist_experiments/checkpoints/gammanet_t8_per_pixel_snemi_combos_200_2019_03_22_22_59_37_216260/model_1750.ckpt-1750
# /media/data_cifs/cluttered_nist_experiments/checkpoints/gammanet_t8_per_pixel_snemi_combos_200_2019_03_22_23_00_27_802822/model_1650.ckpt-1650
# /media/data_cifs/cluttered_nist_experiments/checkpoints/gammanet_t8_per_pixel_snemi_combos_200_2019_03_22_11_11_05_488124/model_2475.ckpt-2475
from matplotlib import pyplot as plt
from skimage.filters import gaussian


def sigmoid_fun(x):
    """Apply sigmoid to maps before mAP."""
    return 1 / (1 + np.exp(x))

# tag = 'isbi'
# isbi_ims = np.load('/media/data_cifs/connectomics/datasets/isbi_2013_0.npz')['volume'];isbi_labs = np.load('/media/data_cifs/connectomics/datasets/isbi_2013_0.npz')['label'];isbi_ims = np.load('/media/data_cifs/connectomics/datasets/isbi_2013_0.npz')['volume']
# sel = -10

tag = 'berson'
isbi_ims = np.load('/media/data_cifs/connectomics/datasets/berson_0.npz')['volume']
isbi_labs = np.load('/media/data_cifs/connectomics/datasets/berson_0.npz')['label']
sel = -45

# tag = 'fib'
# isbi_ims = np.load('/media/data_cifs/connectomics/datasets/fib25_0.npz')['volume'][:512, :512, :512]
# isbi_labs = np.load('/media/data_cifs/connectomics/datasets/fib25_0.npz')['label'][:512, :512, :512]
# sel = -10



ims = isbi_ims[sel][None, ...]
labs = isbi_labs[sel][None, ...]

feed_dict = {test_dict['test_images']: ims, test_dict['test_labels']: labs}
output = sess.run(test_dict, feed_dict)
out_logits = sigmoid_fun(output['test_logits'][..., 0]).squeeze()
f = plt.figure()
plt.subplot(131)
plt.axis('off')
plt.imshow(ims.squeeze(), cmap='Greys_r')
plt.subplot(132)
plt.axis('off')
plt.imshow(gaussian(labs.squeeze(), sigma=2, preserve_range=True), cmap='Greys')
plt.subplot(133)
plt.axis('off')
plt.imshow(gaussian(out_logits, sigma=2, preserve_range=True), cmap='Greys')
plt.savefig(os.path.join('data_to_process_for_jk/generalization_membranes', '%s_%s.pdf' % (tag, exp_label)), dpi=300)
plt.show()
plt.close(f)


# Get first windowed version
windowed = []
for r in range(0, 1024, 256):
    for c in range(0, 1024, 256):
        windowed += [isbi_ims[-5, r:r + 256, c:c + 256][None, ...]]
ims = np.concatenate(windowed)
labs = np.zeros_like(ims)
feed_dict = {test_dict['test_images']: ims, test_dict['test_labels']: labs}
output = sess.run(test_dict, feed_dict)
out_logits = sigmoid_fun(output['test_logits'][..., 0])


# Get second windowed version
windowed = []
for r in range(128, 1024, 256)[:-1]:
    for c in range(0, 1024, 256)[:-1]:
        windowed += [isbi_ims[-5, r:r + 256, c:c + 256][None, ...]]
ims = np.concatenate(windowed)
ims = np.concatenate([ims, np.zeros((7, 256, 256))])
labs = np.zeros_like(ims)
feed_dict = {test_dict['test_images']: ims, test_dict['test_labels']: labs}
output = sess.run(test_dict, feed_dict)
sec_out_logits = sigmoid_fun(output['test_logits'][..., 0])



reassembled = np.zeros((1024, 1024))
count = 0
for r in range(0, 1024, 256):
    for c in range(0, 1024, 256):
        reassembled[r+5:r + 251, c+5:c + 251] = out_logits[count][5:-5, 5:-5]
        count += 1

reassembled_2 = np.zeros((1024, 1024))
count = 0
for r in range(128, 1024, 256)[:-1]:
    for c in range(0, 1024, 256)[:-1]:
        reassembled_2[r+5:r + 251, c+5:c + 251] = sec_out_logits[count][5:-5, 5:-5]
        count += 1
plt.imshow(np.maximum(reassembled, reassembled_2));plt.show()


reassembled = gaussian(reassembled, sigma=2, preserve_range=True)




reassembled = gaussian(reassembled, sigma=2, preserve_range=True)


