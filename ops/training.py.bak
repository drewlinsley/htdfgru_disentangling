"""Model training with tfrecord queues or placeholders."""
import os
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
# from utils import logger
from utils import py_utils
from ops import data_to_tfrecords
from ops import tf_fun
from tqdm import tqdm
from utils import py_utils
try:
    from db import db
except Exception as e:
    print('Failed to import db in ops/training.py: %s' % e)
# from memory_profiler import profile


def sigmoid(x):
    """Per-element sigmoid in numpy."""
    return 1 / (1 + np.exp(-x))


def val_status(
        log,
        dt,
        step,
        train_loss,
        rate,
        timer,
        score_function,
        train_score,
        val_score,
        val_loss,
        best_val_loss,
        summary_dir):
    """Print training status."""
    format_str = (
        '%s: step %d, loss = %.2f (%.1f examples/sec; '
        '%.3f sec/batch) | Training %s = %s | '
        'Validation %s = %s | Validation loss = %s | '
        'Best validation loss = %s| logdir = %s')
    log.info(
        format_str % (
            dt,
            step,
            train_loss,
            rate,
            timer,
            score_function,
            train_score,
            score_function,
            val_score,
            val_loss,
            best_val_loss,
            summary_dir))


def train_status(
        log,
        dt,
        step,
        train_loss,
        rate,
        timer,
        io_timer,
        score_function,
        lr,
        train_score):
    """Print training status."""
    format_str = (
        '%s: step %d, loss = %.5f (%.3f examples/sec; '
        '%.3f sec/batch; %.3f IO time) lr = %.5f | Training %s = %s')
    log.info(
        format_str % (
            dt,
            step,
            train_loss,
            rate,
            timer,
            io_timer,
            lr,
            score_function,
            train_score))


def training_step(
        sess,
        train_dict,
        config,
        feed_dict=False):
    """Run a step of training."""
    start_time = time.time()
    if feed_dict:
        it_train_dict = sess.run(train_dict, feed_dict=feed_dict)
    else:
        it_train_dict = sess.run(train_dict)
    train_score = it_train_dict['train_score']
    # from matplotlib import pyplot as plt
    # plt.plot(it_train_dict['train_labels'].reshape(-1), label="GT")
    # plt.plot(it_train_dict['train_logits'].reshape(-1), label="optimized")
    # plt.legend()
    # plt.show()
    # Patch for accuracy... TODO: Fix the TF op
    if config.score_function == 'accuracy':
        preds = np.argsort(it_train_dict['train_logits'], -1)[:, -1]
        train_score = np.mean(
            preds == it_train_dict['train_labels'].astype(float))
    elif config.score_function == 'fixed_accuracy':
         train_labels = it_train_dict['train_labels'].astype(float)
         train_logits = np.round(sigmoid(it_train_dict['train_logits'])).astype(float)
         train_score = np.mean(train_labels == train_logits)

    train_loss = it_train_dict['train_loss']
    duration = time.time() - start_time
    timesteps = duration
    if train_loss == np.isnan:
        raise RuntimeError('NaN loss during training.')
    return train_score, train_loss, it_train_dict, timesteps


def validation_step(
        sess,
        val_dict,
        config,
        log,
        sequential=False,
        dict_image_key='val_images',
        dict_label_key='val_labels',
        eval_score_key='val_score',
        eval_loss_key='val_loss',
        map_im_key='val_images',
        map_log_key='val_logits',
        map_lab_key='val_labels',
        val_images=False,
        val_labels=False,
        val_batch_idx=None,
        val_batches=False):
    it_val_score = np.asarray([])
    it_val_loss = np.asarray([])
    start_time = time.time()
    save_val_dicts = (hasattr(config, 'get_map') and config.get_map) or \
        (hasattr(config, 'all_results') and config.all_results)
    # map_log_key="fgru"  # Hack to get activities for viz.
    if save_val_dicts:
        it_val_dicts = []
    for idx in range(config.validation_steps):
        # Validation accuracy as the average of n batches
        file_paths = ''  # Init empty full path image info for tfrecords
        if val_batch_idx is not None:
            if not sequential and len(val_batch_idx) > 1:
                it_val_batch_idx = val_batch_idx[
                    np.random.permutation(len(val_batch_idx))]
                val_step = np.random.randint(low=0, high=val_batch_idx.max())
            else:
                it_val_batch_idx = val_batch_idx
                val_step = idx
            if val_images.shape > 1:
                it_idx = it_val_batch_idx == val_step
                it_ims = val_images[it_idx]
                it_labs = val_labels[it_idx]
            # Correct for batch-size=1 cases
            if isinstance(it_ims[0], basestring):
                file_paths = np.copy(it_ims)
                it_ims = np.asarray(
                    [
                        data_to_tfrecords.load_image(im)
                        for im in it_ims])
            if isinstance(it_labs[0], basestring):
                it_labs = np.asarray(
                    [
                        data_to_tfrecords.load_image(im)
                        for im in it_labs])
            feed_dict = {
                val_dict[dict_image_key]: it_ims,
                val_dict[dict_label_key]: it_labs
            }
            it_val_dict = sess.run(val_dict, feed_dict=feed_dict)
        else:
            it_val_dict = sess.run(val_dict)

        if hasattr(config, 'plot_recurrence') and config.plot_recurrence:
            tf_fun.visualize_recurrence(
                idx=idx,
                image=it_ims,
                label=it_labs,
                logits=it_val_dict['test_logits'],
                h2s=it_val_dict['h2_list'],
                ff=it_val_dict['ff'],
                config=config,
                debug=False)

        # Patch for accuracy... TODO: Fix the TF op
        # if config.score_function == 'accuracy':
        #     preds = np.argsort(it_val_dict['val_logits'], -1)[:, -1]
        #     it_val_score = np.mean(
        #         preds == it_val_dict['val_labels'].astype(float))
        if config.score_function == 'fixed_accuracy':
            it_val_labels = it_val_dict[map_lab_key].astype(float)
            it_val_logits = np.round(sigmoid(it_val_dict[map_log_key])).astype(float)
            it_val_score = np.mean(it_val_labels == it_val_logits)
        it_val_score = np.append(
            it_val_score,
            it_val_dict[eval_score_key])
        it_val_loss = np.append(
            it_val_loss,
            it_val_dict[eval_loss_key])
        if save_val_dicts:
            trim_dict = {
                'images': it_val_dict[map_im_key],  # im_key='val_images'],
                'logits': it_val_dict[map_log_key],  # log_key='val_logits'],
                'labels': it_val_dict[map_lab_key],  # lab_key='val_labels']
                'image_paths': file_paths,
            }
            it_val_dicts += [trim_dict]
    val_score = it_val_score.mean()
    val_lo = it_val_loss.mean()
    duration = time.time() - start_time
    if save_val_dicts:
        it_val_dict = it_val_dicts
    return val_score, val_lo, it_val_dict, duration


def save_progress(
        config,
        weight_dict,
        it_val_dict,
        exp_label,
        step,
        directories,
        sess,
        saver,
        val_check,
        val_score,
        val_loss,
        val_perf,
        train_score,
        train_loss,
        timer,
        num_params,
        log,
        use_db,
        summary_op,
        summary_writer,
        save_activities,
        save_gradients,
        save_checkpoints):
    """Save progress and important data."""
    # Update best val
    if len(val_check):
        val_check_idx = val_check[0]
        val_perf[val_check_idx] = val_loss

    # Then trigger optional saves
    if config.save_weights and len(val_check):
        it_weights = {
            k: it_val_dict[k] for k in weight_dict.keys()}
        py_utils.save_npys(
            data=it_weights,
            model_name='%s_%s' % (
                exp_label,
                step),
            output_string=directories['weights'])

    if save_activities and len(val_check):
        py_utils.save_npys(
            data=it_val_dict,
            model_name='%s_%s' % (
                exp_label,
                step),
            output_string=directories['weights'])

    ckpt_path = os.path.join(
        directories['checkpoints'],
        'model_%s.ckpt' % step)
    if save_checkpoints and len(val_check):
        log.info('Saving checkpoint to: %s' % ckpt_path)
        saver.save(
            sess,
            ckpt_path,
            global_step=step)
        val_check = val_check[0]
        val_perf[val_check] = val_loss
    if save_gradients and len(val_check):
        # np.savez(
        #     os.path.join(
        #         config.results,
        #         '%s_train_gradients' % exp_label),
        #     **it_train_dict)
        np.savez(
            os.path.join(
                config.results,
                '%s_val_gradients' % exp_label),
            **it_val_dict)

    if use_db:
        db.update_performance(
            experiment_id=config._id,
            experiment=config.experiment,
            train_score=float(train_score),
            train_loss=float(train_loss),
            val_score=float(val_score),
            val_loss=float(val_loss),
            step=step,
            num_params=int(num_params),
            ckpt_path=ckpt_path,
            results_path=config.results,
            summary_path=directories['summaries'])

    # Summaries
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, step)
    return val_perf


def test_loop(
        config,
        sess,
        summary_op,
        summary_writer,
        saver,
        restore_saver,
        directories,
        test_dict,
        exp_label,
        num_params,
        log,
        map_out='test_maps',
        num_batches=None,
        placeholders=False,
        checkpoint=None,
        save_weights=False,
        save_checkpoints=False,
        save_activities=False,
        save_gradients=False):
    """Run the model test loop."""
    if checkpoint is not None:
        restore_saver.restore(sess, checkpoint)
        print 'Restored checkpoint %s' % checkpoint
    if placeholders:
        test_images = placeholders['test']['images']
        test_labels = placeholders['test']['labels']
        test_batches = len(test_images) / config.test_batch_size
        test_batch_idx = np.arange(
            test_batches).reshape(-1, 1).repeat(
                config.test_batch_size)
        test_images = test_images[:len(test_batch_idx)]
        test_labels = test_labels[:len(test_batch_idx)]
        assert len(test_labels), 'Test labels not found.'
        assert len(test_images), 'Test images not found.'

        # Check that labels are appropriate shape
        tf_label_shape = test_dict['test_labels'].get_shape().as_list()
        np_label_shape = test_labels.shape
        if len(tf_label_shape) == 2 and len(np_label_shape) == 1:
            test_labels = test_labels[..., None]
        elif len(tf_label_shape) == len(np_label_shape):
            pass
        elif len(tf_label_shape) == 4 and np.all(
                np.array(tf_label_shape)[1:3] == np_label_shape[1:3]):
            pass
        else:
            # raise RuntimeError(
            #     'Mismatch label shape np: %s vs. tf: %s' % (
            #         np_label_shape,
            #         tf_label_shape))
            print('Mismatch label shape np: %s vs. tf: %s' % (
                    np_label_shape,
                    tf_label_shape))


        # Loop through all the images
        if num_batches is not None:
            config.validation_steps = num_batches
        else:
            config.validation_steps = test_batches
        test_score, test_lo, it_test_dict, duration = validation_step(
            sequential=True,
            sess=sess,
            val_dict=test_dict,
            config=config,
            log=log,
            dict_image_key='test_images',
            dict_label_key='test_labels',
            eval_score_key='test_score',
            eval_loss_key='test_loss',
            map_im_key='test_proc_images',
            map_log_key='test_logits',
            # map_lab_key='test_proc_labels',
            map_lab_key='test_labels',
            val_images=test_images,
            val_labels=test_labels,
            val_batch_idx=test_batch_idx,
            val_batches=test_batches)
        if hasattr(
                config, 'get_map') and config.get_map and map_out is not None:
            maps, arands = tf_fun.calculate_map(
                it_test_dict,
                exp_label,
                config,
                map_dir=map_out)
            return {
                'scores': test_score,
                'losses': test_lo,
                'maps': maps,
                'arands': arands,
                'exp_label': exp_label,
                'test_dict': it_test_dict,
                'duration': duration}
        else:
            return {
                'scores': test_score,
                'losses': test_lo,
                'exp_label': exp_label,
                'test_dict': it_test_dict,
                'duration': duration}
    else:
        test_score, test_lo, it_test_dict, duration = validation_step(
            sess=sess,
            val_dict=test_dict,
            config=config,
            log=log,
            dict_image_key='test_images',
            dict_label_key='test_labels',
            eval_score_key='test_score',
            eval_loss_key='test_loss',
            # map_im_key='test_proc_images',
            map_im_key='test_images',
            map_log_key='test_logits',
            # map_lab_key='test_proc_labels')
            map_lab_key='test_labels')
        return {
            'scores': test_score,
            'losses': test_lo,
            'exp_label': exp_label,
            'test_dict': it_test_dict,
            'duration': duration}


# @profile
def training_loop(
        config,
        coord,
        sess,
        summary_op,
        summary_writer,
        saver,
        restore_saver,
        threads,
        directories,
        train_dict,
        val_dict,
        exp_label,
        num_params,
        use_db,
        log,
        placeholders=False,
        checkpoint=None,
        save_weights=False,
        save_checkpoints=False,
        save_activities=False,
        save_gradients=False):
    """Run the model training loop."""
    if checkpoint is not None:
        restore_saver.restore(sess, checkpoint)
        print 'Restored checkpoint %s' % checkpoint
    if not hasattr(config, 'early_stop'):
        config.early_stop = np.inf
    val_perf = np.asarray([np.inf])
    step = 0
    best_val_dict = None
    if save_weights:
        try:
            weight_dict = {v.name: v for v in tf.trainable_variables()}
            val_dict = dict(
                val_dict,
                **weight_dict)
        except Exception:
            raise RuntimeError('Failed to find weights to save.')
    else:
        weight_dict = None
    if hasattr(config, 'early_stop'):
        it_early_stop = config.early_stop
    else:
        it_early_stop = np.inf

    if hasattr(config, "adaptive_train"):
        adaptive_train = config.adaptive_train
    else:
        adaptive_train = False
    if placeholders:
        train_images = placeholders['train']['images']
        val_images = placeholders['val']['images']
        train_labels = placeholders['train']['labels']
        val_labels = placeholders['val']['labels']
        train_batches = len(train_images) / config.train_batch_size
        train_batch_idx = np.arange(
            train_batches).reshape(-1, 1).repeat(
                config.train_batch_size)
        train_images = train_images[:len(train_batch_idx)]
        train_labels = train_labels[:len(train_batch_idx)]
        val_batches = len(val_images) / config.val_batch_size
        val_batch_idx = np.arange(
            val_batches).reshape(-1, 1).repeat(
                config.val_batch_size)
        val_images = val_images[:len(val_batch_idx)]
        val_labels = val_labels[:len(val_batch_idx)]

        # Check that labels are appropriate shape
        tf_label_shape = train_dict['train_labels'].get_shape().as_list()
        np_label_shape = train_labels.shape
        if len(tf_label_shape) == 2 and len(np_label_shape) == 1:
            train_labels = train_labels[..., None]
            val_labels = val_labels[..., None]
        elif len(tf_label_shape) == len(np_label_shape):
            pass
        else:
            raise RuntimeError(
                'Mismatch label shape np: %s vs. tf: %s' % (
                    np_label_shape,
                    tf_label_shape))

        # Start training
        train_losses = []
        train_logits = []
        for epoch in tqdm(
                range(config.epochs),
                desc='Epoch',
                total=config.epochs):
            for train_batch in range(train_batches):
                io_start_time = time.time()
                data_idx = train_batch_idx == train_batch
                it_train_images = train_images[data_idx]
                it_train_labels = train_labels[data_idx]
                if isinstance(it_train_images[0], basestring):
                    it_train_images = np.asarray(
                        [
                            data_to_tfrecords.load_image(im)
                            for im in it_train_images])
                feed_dict = {
                    train_dict['train_images']: it_train_images,
                    train_dict['train_labels']: it_train_labels
                }
                (
                    train_score,
                    train_loss,
                    it_train_dict,
                    timer) = training_step(
                    sess=sess,
                    train_dict=train_dict,
                    config=config,
                    feed_dict=feed_dict)
                train_losses.append(train_loss)
                if step % config.validation_period == 0:
                    val_score, val_lo, it_val_dict, duration = validation_step(
                        sess=sess,
                        val_dict=val_dict,
                        config=config,
                        log=log,
                        val_images=val_images,
                        val_labels=val_labels,
                        val_batch_idx=val_batch_idx,
                        val_batches=val_batches)

                    # Save progress and important data
                    try:
                        val_check = np.where(val_lo < val_perf)[0]
                        if not len(val_check):
                            it_early_stop -= 1
                            print 'Deducted from early stop count.'
                        else:
                            it_early_stop = config.early_stop
                            best_val_dict = it_val_dict
                            print 'Reset early stop count.'
                        if it_early_stop <= 0:
                            print 'Early stop triggered. Ending early.'
                            print 'Best validation loss: %s' % np.min(val_perf)
                            return
                        save_progress(
                            config=config,
                            val_check=val_check,
                            weight_dict=weight_dict,
                            it_val_dict=it_val_dict,
                            exp_label=exp_label,
                            step=step,
                            directories=directories,
                            sess=sess,
                            saver=saver,
                            val_score=val_score,
                            val_loss=val_lo,
                            train_score=train_score,
                            train_loss=train_loss,
                            timer=duration,
                            num_params=num_params,
                            log=log,
                            summary_op=summary_op,
                            summary_writer=summary_writer,
                            save_activities=save_activities,
                            save_gradients=save_gradients,
                            save_checkpoints=save_checkpoints)
                    except Exception as e:
                        log.info('Failed to save checkpoint: %s' % e)

                    # Hack to get the visulations... clean this up later
                    if "BSDS500_test_orientation_viz" in config.experiment:  # .model == "BSDS_inh_perturb" or config.model == "BSDS_exc_perturb":
                        # from matplotlib import pyplot as plt;plt.plot(it_train_dict['train_logits'].squeeze(), "r", label="Perturb");plt.plot(it_train_dict['train_labels'].squeeze()[-6:], 'b', label="GT");plt.legend();plt.show()
                        # from matplotlib import pyplot as plt;plt.imshow((it_train_dict['impatch'].squeeze() + np.asarray([123.68, 116.78, 103.94])[None, None]).astype(np.uint8));plt.show()
                        # from matplotlib import pyplot as plt;dd = it_train_dict["grad0"];plt.imshow(np.abs(dd.squeeze()).mean(-1) / (np.abs(dd.squeeze()).std(-1) + 1e-4));plt.show()
                        # from matplotlib import pyplot as plt;dd = it_train_dict['mask'];plt.imshow(dd.squeeze().mean(-1));plt.show()
                        train_logits.append([it_train_dict["train_logits"].ravel()])
                        out_dir = "circuits_{}".format(config.out_dir)
                        py_utils.make_dir(out_dir)
                        out_target = os.path.join(out_dir, "{}_{}".format(config.model, config.train_dataset))
                        np.save("{}_optim".format(out_target), [sess.run(tf.trainable_variables())])  # , it_train_dict["conv"]])
                        np.save("{}_perf".format(out_target), train_losses)
                        np.save("{}_curves".format(out_target), train_logits)
                        np.save("{}_label".format(out_target), it_train_dict["train_labels"])
                    """
                    if config.model == "BSDS_inh_perturb":
                        np.save("inh_perturbs/optim", sess.run(tf.trainable_variables()[0]))
                        np.save("inh_perturbs/perf", train_losses)
                        np.save("inh_perturbs/curves", train_logits)
                        np.save("inh_perturbs/label", it_train_dict["train_labels"])

                    if config.model == "BSDS_exc_perturb":
                        np.save("exc_perturbs/optim", sess.run(tf.trainable_variables()[0]))
                        np.save("exc_perturbs/perf", train_losses)
                        np.save("exc_perturbs/curves", train_logits)
                        np.save("exc_perturbs/label", it_train_dict["train_labels"])
                    """

                    # Training status and validation accuracy
                    val_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        score_function=config.score_function,
                        train_score=train_score,
                        val_score=val_score,
                        val_loss=val_lo,
                        best_val_loss=np.min(val_perf),
                        summary_dir=directories['summaries'])
                else:
                    # Training status
                    io_duration = time.time() - io_start_time
                    train_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        io_timer=float(io_duration),
                        lr=it_train_dict['lr'],
                        score_function=config.score_function,
                        train_score=train_score)

                # End iteration
                val_perf = np.concatenate([val_perf, [val_lo]])
                step += 1
                
                # Adaptive ending
                if adaptive_train and train_loss <= adaptive_train:
                    break
            if adaptive_train and train_loss <= adaptive_train:
                break


    else:
        try:
            while not coord.should_stop():
                (
                    train_score,
                    train_loss,
                    it_train_dict,
                    duration) = training_step(
                    sess=sess,
                    config=config,
                    train_dict=train_dict)
                io_start_time = time.time()
                if step % config.validation_period == 0:
                    val_score, val_lo, it_val_dict, duration = validation_step(
                        sess=sess,
                        val_dict=val_dict,
                        config=config,
                        log=log)

                    # Save progress and important data
                    try:
                        val_check = np.where(val_lo < val_perf)[0]
                        if not len(val_check):
                            it_early_stop -= 1
                            print 'Deducted from early stop count.'
                        else:
                            it_early_stop = config.early_stop
                            best_val_dict = it_val_dict
                            print 'Reset early stop count.'
                        if it_early_stop <= 0:
                            print 'Early stop triggered. Ending early.'
                            print 'Best validation loss: %s' % np.min(val_perf)
                            break
                        val_perf = save_progress(
                            config=config,
                            val_check=val_check,
                            weight_dict=weight_dict,
                            it_val_dict=it_val_dict,
                            exp_label=exp_label,
                            step=step,
                            directories=directories,
                            sess=sess,
                            saver=saver,
                            val_score=val_score,
                            val_loss=val_lo,
                            val_perf=val_perf,
                            train_score=train_score,
                            train_loss=train_loss,
                            timer=duration,
                            num_params=num_params,
                            log=log,
                            use_db=use_db,
                            summary_op=summary_op,
                            summary_writer=summary_writer,
                            save_activities=save_activities,
                            save_gradients=save_gradients,
                            save_checkpoints=save_checkpoints)
                    except Exception as e:
                        log.info('Failed to save checkpoint: %s' % e)

                    # Training status and validation accuracy
                    val_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        score_function=config.score_function,
                        train_score=train_score,
                        val_score=val_score,
                        val_loss=val_lo,
                        best_val_loss=np.min(val_perf),
                        summary_dir=directories['summaries'])
                else:
                    # Training status
                    io_duration = time.time() - io_start_time
                    train_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        io_timer=float(io_duration),
                        lr=it_train_dict['lr'],
                        score_function=config.score_function,
                        train_score=train_score)

                # End iteration
                step += 1
        except tf.errors.OutOfRangeError:
            log.info(
                'Done training for %d epochs, %d steps.' % (
                    config.epochs, step))
            log.info('Saved to: %s' % directories['checkpoints'])
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
    print 'Best %s loss: %s' % (config.val_loss_function, val_perf[0])
    if hasattr(config, 'get_map') and config.get_map:
        tf_fun.calculate_map(best_val_dict, exp_label, config)
    return

