# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

import evalu
import lrs
import model
import bert
from data import Dataset
from tasks import get_task
from utils import parallel, cycle, util, queuer, saver, initializer


def tower_train_graph(train_features, optimizer, graph, params):
    # define multi-gpu training graph
    def _tower_train_graph(features):
        train_output = graph.train_fn(
            features, params,
            initializer=initializer.get_initializer(params))
        tower_gradients = optimizer.compute_gradients(
            train_output["loss"], colocate_gradients_with_ops=True)
        return {
            "loss": train_output["loss"],
            "gradient": tower_gradients
        }

    # feed model to multiple gpus
    tower_outputs = parallel.parallel_model(
        _tower_train_graph, train_features,
        params.gpus, use_cpu=(len(params.gpus) == 0))

    loss = tf.add_n(tower_outputs['loss']) / len(tower_outputs['loss'])

    gradients = parallel.average_gradients(tower_outputs['gradient'])

    return loss, gradients


def tower_infer_graph(eval_features, graph, params):
    # define multi-gpu inferring graph
    def _tower_infer_graph(features):
        pred_label, pred_distribution = graph.infer_fn(params, features)
        return pred_label

    # feed model to multiple gpus
    eval_outputs = parallel.parallel_model(
        _tower_infer_graph, eval_features,
        params.gpus, use_cpu=(len(params.gpus) == 0))

    return eval_outputs


def train(params):
    # status measure
    if params.recorder.estop or \
            params.recorder.epoch > params.epoches or \
            params.recorder.step > params.max_training_steps:
        tf.logging.info("Stop condition reached, you have finished training your model.")
        return 0.

    # loading dataset
    tf.logging.info("Begin Loading Training and Dev Dataset")
    start_time = time.time()

    bert_vocab, tokenizer = None, None

    if params.enable_bert:
        bert_vocab = params.bert.vocab
        tokenizer = bert.load_tokenizer(params)

    dataset = Dataset(get_task(params, True),
                      params.max_len,
                      params.max_w_len,
                      params.max_p_num,
                      params.word_vocab,
                      bert_vocab,
                      tokenizer,
                      enable_hierarchy=params.enable_hierarchy,
                      char_vocab=params.char_vocab if params.use_char else None,
                      enable_char=params.use_char,
                      batch_or_token=params.batch_or_token)

    tf.logging.info(
        "End Loading dataset, within {} seconds".format(time.time() - start_time))

    # Build Graph
    with tf.Graph().as_default():
        lr = tf.placeholder(tf.float32, [], "learn_rate")

        features = []
        for fidx in range(max(len(params.gpus), 1)):
            feature = {
                "t": tf.placeholder(tf.int32, [None, None], "t_{}".format(fidx)),
                "l": tf.placeholder(tf.int32, [None], "l_{}".format(fidx)),
            }
            if params.use_char:
                feature["c"] = tf.placeholder(tf.int32, [None, None, None], "c_{}".format(fidx))

            if params.enable_bert:
                feature["s"] = tf.placeholder(tf.int32, [None, None], "s_{}".format(fidx))
                feature["sb"] = tf.placeholder(tf.int32, [None, None], "sb_{}".format(fidx))

            features.append(feature)

        # session info
        sess = util.get_session(params.gpus)

        tf.logging.info("Begining Building Training Graph")
        start_time = time.time()

        # create global step
        global_step = tf.train.get_or_create_global_step()

        # set up optimizer
        optimizer = tf.train.AdamOptimizer(lr,
                                           beta1=params.beta1,
                                           beta2=params.beta2,
                                           epsilon=params.epsilon)

        # set up training graph
        loss, gradients = tower_train_graph(features, optimizer, model, params)

        # apply pseudo cyclic parallel operation
        vle, ops = cycle.create_train_op({"loss": loss},
                                         gradients, optimizer, global_step, params)

        tf.logging.info("End Building Training Graph, within {} seconds"
                        .format(time.time() - start_time))

        tf.logging.info("Begin Building Inferring Graph")
        start_time = time.time()

        # set up infer graph
        eval_pred = tower_infer_graph(features, model, params)

        tf.logging.info("End Building Inferring Graph, within {} seconds"
                        .format(time.time() - start_time))

        # initialize the model
        sess.run(tf.global_variables_initializer())

        # log parameters
        util.variable_printer()

        # create saver
        train_saver = saver.Saver(checkpoints=params.checkpoints,
                                  output_dir=params.output_dir)

        tf.logging.info("Training")
        cycle_counter = 1
        cum_loss, cum_gnorm = [], []

        # restore parameters
        tf.logging.info("Trying restore existing parameters")
        if params.enable_bert:
            bert.load_model(sess, params.bert_dir)
        train_saver.restore(sess)

        # setup learning rate
        adapt_lr = lrs.get_lr(params)

        start_time = time.time()
        start_epoch = params.recorder.epoch
        data_on_gpu = []
        for epoch in range(start_epoch, params.epoches + 1):

            params.recorder.epoch = epoch

            tf.logging.info("Training the model for epoch {}".format(epoch))
            size = params.batch_size if params.batch_or_token == 'batch' \
                else params.token_size
            train_batcher = dataset.batcher(size,
                                            buffer_size=params.buffer_size,
                                            shuffle=params.shuffle_batch,
                                            train="train")
            train_queue = queuer.EnQueuer(train_batcher,
                                          multiprocessing=params.data_multiprocessing,
                                          random_seed=params.random_seed)
            train_queue.start(workers=params.nthreads,
                              max_queue_size=params.max_queue_size)

            adapt_lr.before_epoch(eidx=epoch)
            for lidx, data in enumerate(train_queue.get()):

                if params.train_continue:
                    if lidx <= params.recorder.lidx:
                        segments = params.recorder.lidx // 5
                        if params.recorder.lidx < 5 or lidx % segments == 0:
                            tf.logging.info("Passing {}-th index according to record"
                                            .format(lidx))
                        continue
                params.recorder.lidx = lidx

                data_on_gpu.append(data)
                # use multiple gpus, and data samples is not enough
                # make sure the data is fully added
                # The actual batch size: batch_size * num_gpus * update_cycle
                if len(params.gpus) > 0 and len(data_on_gpu) < len(params.gpus):
                    continue

                if cycle_counter == 1:
                    sess.run(ops["zero_op"])

                    # calculate adaptive learning rate
                    adapt_lr.step(params.recorder.step)

                feed_dicts = {}
                for fidx, data in enumerate(data_on_gpu):
                    # define feed_dict
                    feed_dict = {
                        features[fidx]["t"]: data['token_ids'],
                        features[fidx]["l"]: data['l_id'],
                        lr: adapt_lr.get_lr(),
                    }
                    if params.use_char:
                        feed_dict[features[fidx]["c"]] = data['char_ids']
                    if params.enable_bert:
                        feed_dict[features[fidx]["s"]] = data['subword_ids']
                        feed_dict[features[fidx]["sb"]] = data['subword_back']

                    feed_dicts.update(feed_dict)

                # reset data points
                data_on_gpu = []

                if cycle_counter < params.update_cycle:
                    sess.run(ops["collect_op"], feed_dict=feed_dicts)
                if cycle_counter == params.update_cycle:
                    cycle_counter = 0

                    _, loss, gnorm, gstep = sess.run(
                        [ops["train_op"], vle["loss"],
                         vle["gradient_norm"], global_step],
                        feed_dict=feed_dicts
                    )

                    if np.isnan(loss) or np.isinf(loss):
                        tf.logging.error("Nan or Inf raised")
                        params.recorder.estop = True
                        break

                    cum_loss.append(loss)
                    cum_gnorm.append(gnorm)

                    if gstep % params.disp_freq == 0:
                        end_time = time.time()
                        tf.logging.info(
                            "{} Epoch {}, GStep {}~{}, LStep {}~{}, "
                            "Loss {:.3f}, GNorm {:.3f}, Lr {:.5f}, "
                            "Document {}, UD {:.3f} s"
                            .format(util.time_str(end_time), epoch,
                                    gstep - params.disp_freq + 1, gstep,
                                    lidx - params.disp_freq + 1, lidx,
                                    np.mean(cum_loss), np.mean(cum_gnorm),
                                    adapt_lr.get_lr(), data['token_ids'].shape,
                                    end_time - start_time)
                        )
                        start_time = time.time()
                        cum_loss, cum_gnorm = [], []

                    # trigger model saver
                    if gstep > 0 and gstep % params.save_freq == 0:
                        train_saver.save(sess, gstep)
                        params.recorder.save_to_json(
                            os.path.join(params.output_dir, "record.json"))

                    # trigger model evaluation
                    if gstep > 0 and gstep % params.eval_freq == 0:

                        if params.ema_decay > 0.:
                            sess.run(ops['ema_backup_op'])
                            sess.run(ops['ema_assign_op'])

                        tf.logging.info("Start Evaluating")
                        eval_start_time = time.time()
                        predictions, score = evalu.predict(
                            sess, features, eval_pred,
                            dataset, params, train="dev")
                        eval_end_time = time.time()
                        tf.logging.info("End Evaluating")

                        tf.logging.info(
                            "{} GStep {}, Score {}, Duration {:.3f} s"
                            .format(util.time_str(eval_end_time), gstep,
                                    score,
                                    eval_end_time - eval_start_time)
                        )

                        if params.ema_decay > 0.:
                            sess.run(ops['ema_restore_op'])

                        # save eval translation
                        evalu.dump_predictions(
                            predictions,
                            os.path.join(params.output_dir,
                                         "eval-{}.trans.txt".format(gstep))
                        )

                        # save parameters
                        train_saver.save(sess, gstep, score)

                        # check for early stopping
                        valid_scores = [v[1] for v in params.recorder.valid_script_scores]
                        if len(valid_scores) == 0 or score > np.max(valid_scores):
                            params.recorder.bad_counter = 0
                        else:
                            params.recorder.bad_counter += 1

                            if params.recorder.bad_counter > params.estop_patience:
                                params.recorder.estop = True
                                break

                        params.recorder.history_scores.append(
                            (gstep, float(score))
                        )
                        params.recorder.valid_script_scores.append(
                            (gstep, float(score))
                        )
                        params.recorder.save_to_json(
                            os.path.join(params.output_dir, "record.json"))

                        # handle the learning rate decay in a typical manner
                        adapt_lr.after_eval(float(score))

                    # trigger stopping
                    if gstep >= params.max_training_steps:
                        params.recorder.estop = True
                        break

                    # should be equal to global_step
                    params.recorder.step += 1.0

                cycle_counter += 1

            train_queue.stop()

            if params.recorder.estop:
                tf.logging.info("Early Stopped!")
                break

            # reset to 0
            params.recorder.lidx = -1

            adapt_lr.after_epoch(eidx=epoch)

    # Final Evaluation
    tf.logging.info("Start Evaluating")
    if params.ema_decay > 0.:
        sess.run(ops['ema_backup_op'])
        sess.run(ops['ema_assign_op'])

    gstep = int(params.recorder.step + 1)
    eval_start_time = time.time()
    predictions, score = evalu.predict(
        sess, features, eval_pred,
        dataset, params, train="dev")
    eval_end_time = time.time()
    tf.logging.info("End Evaluating")
    tf.logging.info(
        "{} GStep {}, Score {}, Duration {:.3f} s"
        .format(util.time_str(eval_end_time), gstep,
                score,
                eval_end_time - eval_start_time)
    )

    # save eval translation
    evalu.dump_predictions(
        predictions,
        os.path.join(params.output_dir,
                     "eval-{}.trans.txt".format(gstep)))

    if params.ema_decay > 0.:
        sess.run(ops['ema_restore_op'])

    tf.logging.info("Your training is finished :)")

    return train_saver.best_score


def evaluate(params):
    # loading dataset
    tf.logging.info("Begin Loading Test Dataset")
    start_time = time.time()

    bert_vocab, tokenizer = None, None

    if params.enable_bert:
        bert_vocab = params.bert.vocab
        tokenizer = bert.load_tokenizer(params)

    dataset = Dataset(get_task(params, False),
                      params.max_len,
                      params.max_w_len,
                      params.max_p_num,
                      params.word_vocab,
                      bert_vocab,
                      tokenizer,
                      enable_hierarchy=params.enable_hierarchy,
                      char_vocab=params.char_vocab if params.use_char else None,
                      enable_char=params.use_char,
                      batch_or_token=params.batch_or_token)
    tf.logging.info(
        "End Loading dataset, within {} seconds".format(time.time() - start_time))

    # Build Graph
    with tf.Graph().as_default():
        features = []
        for fidx in range(max(len(params.gpus), 1)):
            feature = {
                "t": tf.placeholder(tf.int32, [None, None], "t_{}".format(fidx)),
                "l": tf.placeholder(tf.int32, [None], "l_{}".format(fidx)),
            }
            if params.use_char:
                feature["c"] = tf.placeholder(tf.int32, [None, None, None], "c_{}".format(fidx))

            if params.enable_bert:
                feature["s"] = tf.placeholder(tf.int32, [None, None], "s_{}".format(fidx))
                feature["sb"] = tf.placeholder(tf.int32, [None, None], "sb_{}".format(fidx))

            features.append(feature)

        # session info
        sess = util.get_session(params.gpus)

        tf.logging.info("Begin Building Inferring Graph")
        start_time = time.time()

        # set up infer graph
        eval_pred = tower_infer_graph(features, model, params)

        tf.logging.info("End Building Inferring Graph, within {} seconds"
                        .format(time.time() - start_time))

        # set up ema
        if params.ema_decay > 0.:
            # recover from EMA
            ema = tf.train.ExponentialMovingAverage(decay=params.ema_decay)
            ema.apply(tf.trainable_variables())
            ema_assign_op = tf.group(*(tf.assign(var, ema.average(var).read_value())
                                       for var in tf.trainable_variables()))
        else:
            ema_assign_op = tf.no_op()

        # initialize the model
        sess.run(tf.global_variables_initializer())

        # log parameters
        util.variable_printer()

        # create saver
        eval_saver = saver.Saver(checkpoints=params.checkpoints,
                                 output_dir=params.output_dir)

        # restore parameters
        tf.logging.info("Trying restore existing parameters")
        eval_saver.restore(sess)
        sess.run(ema_assign_op)

        # Evaluation
        tf.logging.info("Start Evaluating")
        eval_start_time = time.time()
        predictions, score = evalu.predict(
            sess, features, eval_pred,
            dataset, params, train="test")
        eval_end_time = time.time()
        tf.logging.info("End Evaluating")
        tf.logging.info(
            "{} Score {}, Duration {:.3f} s"
            .format(util.time_str(eval_end_time),
                    score,
                    eval_end_time - eval_start_time)
        )

        # save eval translation
        evalu.dump_predictions(
            predictions,
            params.test_output
        )

        return score
