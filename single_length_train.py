import sys

from tensorflow.python.training.saver import latest_checkpoint

from config import *
from language_helpers import generate_argmax_samples_and_gt_samples, inf_train_gen, decode_indices_to_string
from objective import get_optimization_ops, define_objective
from summaries import define_summaries, \
    log_samples

sys.path.append(os.getcwd())

from model import *
import model_and_data_serialization


# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!

def run(iterations, seq_length, is_first, charmap, inv_charmap, prev_seq_length):
    if len(DATA_DIR) == 0:
        raise Exception('Please specify path to data directory in single_length_train.py!')

    lines, _, _ = model_and_data_serialization.load_dataset(seq_length=seq_length, b_charmap=False, b_inv_charmap=False,
    n_examples=FLAGS.MAX_N_EXAMPLES)

    '''
    print("Lines=", lines)
    print("Lines Len=", len(lines))
    print("Lines[1] Len=", lines[1], len(lines[1]))
    print("Real Inputs Discrete Args BSize=%d, Seqlen=%d" % (BATCH_SIZE, seq_length))   
    
    print("charmap=", charmap)
    print("Len charmap=", len(charmap))
    print("Inv charmap=", inv_charmap)
    print("Len inv_charmap=", len(inv_charmap))
    '''

    real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length])

    global_step = tf.Variable(0, trainable=False)
    disc_cost, gen_cost, fake_inputs, disc_fake, disc_real, disc_on_inference, inference_op = define_objective(charmap,
                                                                                                            real_inputs_discrete,
                                                                                                            seq_length)
    merged, train_writer = define_summaries(disc_cost, gen_cost, seq_length)
    disc_train_op, gen_train_op = get_optimization_ops(disc_cost, gen_cost, global_step)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as session:

        session.run(tf.initialize_all_variables())
        if not is_first:
            print("Loading previous checkpoint...")
            internal_checkpoint_dir = model_and_data_serialization.get_internal_checkpoint_dir(prev_seq_length)
            model_and_data_serialization.optimistic_restore(session,
                                                            latest_checkpoint(internal_checkpoint_dir, "checkpoint"))

            restore_config.set_restore_dir(
                load_from_curr_session=True)  # global param, always load from curr session after finishing the first seq

        gen = inf_train_gen(lines, charmap)

        for iteration in range(iterations):
            start_time = time.time()

            # Train critic - Discriminator
            for i in range(CRITIC_ITERS):
                _data = next(gen)
                _disc_cost, _, real_scores = session.run(
                    [disc_cost, disc_train_op, disc_real],
                    feed_dict={real_inputs_discrete: _data}
                )

            # Train G
            for i in range(GEN_ITERS):
                _data = next(gen)
                _ = session.run(gen_train_op, feed_dict={real_inputs_discrete: _data})

            print("iteration %s/%s"%(iteration, iterations))
            print("disc cost %f"%_disc_cost)

            # Summaries
            if iteration % 100 == 99:
                _data = next(gen)
                summary_str = session.run(
                    merged,
                    feed_dict={real_inputs_discrete: _data}
                )

                train_writer.add_summary(summary_str, global_step=iteration)
                fake_samples, samples_real_probabilites, fake_scores = generate_argmax_samples_and_gt_samples(session, inv_charmap,
                                                                                                              fake_inputs,
                                                                                                              disc_fake,
                                                                                                              gen,
                                                                                                              real_inputs_discrete,
                                                                                                              feed_gt=True)

                log_samples(fake_samples, fake_scores, iteration, seq_length, "train")
                log_samples(decode_indices_to_string(_data, inv_charmap), real_scores, iteration, seq_length,
                             "gt")
                test_samples, _, fake_scores = generate_argmax_samples_and_gt_samples(session, inv_charmap,
                                                                                      inference_op,
                                                                                      disc_on_inference,
                                                                                      gen,
                                                                                      real_inputs_discrete,
                                                                                      feed_gt=False)
                # disc_on_inference, inference_op
                log_samples(test_samples, fake_scores, iteration, seq_length, "test")


            if iteration % FLAGS.SAVE_CHECKPOINTS_EVERY == FLAGS.SAVE_CHECKPOINTS_EVERY-1:
                saver.save(session, model_and_data_serialization.get_internal_checkpoint_dir(seq_length) + "/ckp")

        saver.save(session, model_and_data_serialization.get_internal_checkpoint_dir(seq_length) + "/ckp")
        session.close()
