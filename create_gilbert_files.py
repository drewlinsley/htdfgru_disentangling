import os
from copy import deepcopy
from glob2 import glob


gpu = 5
data_dir = "/media/data_cifs_lrs/projects/prj_neural_circuits/"
target_dirs = glob(os.path.join(data_dir, "gilbert_datasets/*/*"))
template = "datasets/gilbert_length17_shear0_0.py"
with open(template, "r+") as f:
    template_script = f.readlines()

tf_lines = [
    "#!/usr/bin/env bash\n# Autogen script for creating gilbert tfrecords\n\n",
    # "python encode_dataset.py --dataset=gilbert_length17_shearp6",  # Rebuild the template
]
cmd = "CUDA_VISIBLE_DEVICES={} python run_job.py --no_db --experiment=gilbert --model=BSDS_vgg_gilbert --train=gilbert_length17_shearp6 --val=gilbert_length17_shearp6 --ckpt=/media/data_cifs_lrs/projects/prj_neural_circuits/gammanet/checkpoints/BSDS_vgg_gilbert_gilbert_2020_09_03_20_40_46_998983/model_2600.ckpt-2600 --test --out_dir=gilbert_length17_shearp6\n".format(gpu)  # noqa
cmds = [
    '#!/usr/bin/env bash\n# Autogen script for running models\n\n',
    cmd,
]
for target in target_dirs:
    # TFRecords
    target_file = deepcopy(template_script)
    # length = target.split("length_")[-1]
    length = target.split(os.path.sep)[-2].split("_")[-1]
    # shear = target.split("shear_")[1].split("_length")[0]
    shear = target.split(os.path.sep)[-1].split("_")[-1]
    target_file[25] = target_file[25].replace("5000", "500")
    #         self.contour_dir = '/media/data_cifs_lrs/projects/prj_neural_circuits/gilbert_datasets/contour_21/shear_val_0/imgs/1'
    target_file[15] = target_file[15].replace("17", length)
    # target_file[15] = target_file[15].replace("0_6", shear)
    target_file[16] = target_file[16].replace("/media/data_cifs_lrs/projects/prj_neural_circuits/gilbert_datasets/contour_17/shear_val_0", target)
    target_file[38] = target_file[38].replace("0.1", "1.0")
    out_file = os.path.join("datasets", "gilbert_length{}_shear{}.py".format(length, shear))  #noqa
    with open(out_file, "w") as f:
        f.writelines(target_file)
    tf_lines.append("python encode_dataset.py --dataset=gilbert_length{}_shear{}\n".format(length, shear))

    # CMDs
    cmds.append(cmd.replace("length17_shearp6", "length{}_shear{}".format(length, shear)))

# Create the tfrecords creation script
with open("create_gilbert_tfrecords.sh", "w") as f:
    f.writelines(tf_lines)

# Create the model eval script
with open("eval_gilbert_datasets.sh", "w") as f:
    f.writelines(cmds)

