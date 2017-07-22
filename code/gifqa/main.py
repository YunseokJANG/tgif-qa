import time
import os
import tensorflow as tf
from data_util.tgif import DatasetTGIF
from util import log
import json
import hickle as hkl
import getpass

from models.count_models import *
from models.frameqa_models import *
from models.mc_models import *

# Training Params
tf.flags.DEFINE_string("task", "Count", "[Count, Action, FrameQA, Trans]")
tf.flags.DEFINE_string("name", "Tp", "[C3D, Resnet, Concat, Tp, Sp, SpTp]")
tf.flags.DEFINE_string("save_path", "./", "Save path")
tf.flags.DEFINE_integer("random_state", 42, "Random state initialization for reproductibility")
tf.flags.DEFINE_integer("max_sequence_length", 35, "Examples will be padded/truncated to this length")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs")
tf.flags.DEFINE_integer("log_every", 25, "Number of step size for training log")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this number of steps")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Evaluate model on dev set after this number of steps")
tf.flags.DEFINE_boolean("save_checkpoint_by_param", True, "checkpoint dir save as param")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate for training")

tf.flags.DEFINE_integer("hidden_dim", 512, "rnn hidden unit dimension")
tf.flags.DEFINE_integer("batch_size", 64, "Batch_size")
tf.flags.DEFINE_integer("num_layers", 2, "Number of stacked video RNN cells")
tf.flags.DEFINE_float("dropout_keep_prob_cell_input", 0.8, "RNN cell input connection dropout")
tf.flags.DEFINE_float("dropout_keep_prob_cell_output", 0.8, "RNN cell output connection dropout")
tf.flags.DEFINE_float("dropout_keep_prob_fully_connected", 0.8, "fully_connected output dropout")
tf.flags.DEFINE_float("dropout_keep_prob_output", 0.8, "Output layer dropout")
tf.flags.DEFINE_float("dropout_keep_prob_image_embed", 0.8, "image embedding dropout")

# Test configurations
tf.flags.DEFINE_string("checkpoint_path", "", "Path for checkpoint diretory you want to recover.")
tf.flags.DEFINE_boolean("test_phase", False, "use test set instead of validation set")

# Session Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow soft device placement(e.g. no GPU)")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("allow_growth", True, "Allow growth to session config")


FLAGS = tf.flags.FLAGS
model_name = FLAGS.task + FLAGS.name


# Model Params
class_name = model_name
if "Action" in class_name or "Trans" in class_name:
    class_name = "MC" + FLAGS.name
ModelEvaluator = globals()[class_name+"Evaluator"]
ModelTrainer = globals()[class_name+"Trainer"]
Model = globals()[class_name]
Model.add_flags(FLAGS)


# Restore FLAGS variables
if FLAGS.checkpoint_path:
    checkpoint = FLAGS.checkpoint_path
    params_path = os.path.join(os.path.dirname(checkpoint), '%s_%s_param.json' % (FLAGS.task.lower(), FLAGS.name.lower()))
    if os.path.exists(params_path):
        log.info("Restored FLAGS from {}".format(params_path))
        with open(params_path, 'r') as f:
            model_params = json.load(f)
        # Restore parameters for model build
        FLAGS.__flags = params_path

print("\nModel: %s" % model_name)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

print 'Start loading TGIF dataset'
train_dataset = DatasetTGIF(dataset_name='train',
                                         image_feature_net=FLAGS.image_feature_net,
                                         layer=FLAGS.layer,
                                         max_length=FLAGS.max_sequence_length,
                                         data_type=FLAGS.task)
train_dataset.load_word_vocabulary()

val_dataset = train_dataset.split_dataset(ratio=0.1)
val_dataset.share_word_vocabulary_from(train_dataset)

test_dataset = DatasetTGIF(dataset_name='test',
                            image_feature_net=FLAGS.image_feature_net,
                            layer=FLAGS.layer,
                            max_length=FLAGS.max_sequence_length,
                            data_type=FLAGS.task)

test_dataset.share_word_vocabulary_from(train_dataset)

# Parameters
SEQUENCE_LENGTH = FLAGS.max_sequence_length
VOCABULARY_SIZE = train_dataset.n_words
FEAT_DIM = train_dataset.get_video_feature_dimension()[1:]
train_iter = train_dataset.batch_iter(FLAGS.num_epochs, FLAGS.batch_size)

# Create a graph and session
graph = tf.Graph()
session_conf = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=FLAGS.allow_growth),
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement
)
sess = tf.Session(graph=graph, config=session_conf)

def main():
    tf.set_random_seed(FLAGS.random_state)
    model, model_params = init_model()

    # Directory for training and dev summaries
    timestamp = str(int(time.time()))
    #summary_root_dir = os.path.abspath(os.path.join(FLAGS.save_path, "runs"))
    summary_root_dir = os.path.join(FLAGS.save_path, "runs")
    if FLAGS.save_checkpoint_by_param:
        param_name = FLAGS.task + "/" + FLAGS.name + "_" + timestamp
        rundir = os.path.join(summary_root_dir, param_name)
    else:
        rundir = os.path.join(summary_root_dir, timestamp)
    train_dir = os.path.join(rundir, 'train')
    dev_dir = os.path.join(rundir, 'dev')

    # Build the Trainer/Evaluator
    trainer = ModelTrainer(model, optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
                           train_summary_dir=train_dir)
    evaluator = ModelEvaluator(model, summary_dir=dev_dir)
    # Saving/Checkpointing
    checkpoint_dir = os.path.join(rundir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        model.save_to_file(FLAGS.__flags, os.path.join(checkpoint_dir, '%s_%s_param.hkl' % (FLAGS.task.lower(), FLAGS.name.lower())))

    global checkpoint_file
    checkpoint_file = os.path.join(checkpoint_dir, "model.ckpt")
    saver = tf.train.Saver(max_to_keep=10000)

    # Initialization, optionally load from checkpoint
    sess.run(tf.initialize_all_variables())
    if FLAGS.checkpoint_path:
        checkpoint = FLAGS.checkpoint_path
        if checkpoint:
            log.info("Restoring checkpoint from {}".format(checkpoint))
            saver.restore(sess, checkpoint)
            log.info("Restored checkpoint from {}".format(checkpoint))

    # evaluation only
    global global_acc, acc
    global_acc = 0.0
    acc = None
    def run_evaluation(current_step):
        global global_acc, acc
        global checkpoint_file
        if FLAGS.test_phase:
            dev_iter = test_dataset.batch_iter(1, FLAGS.batch_size)
        else:
            dev_iter = val_dataset.batch_iter(1, FLAGS.batch_size)

        mean_loss, acc, _, result_json = evaluator.eval(
            dev_iter,
            global_step=trainer.global_step,
            generate_results=FLAGS.test_phase)

        log.info((" [{split_mode:5} step {step:4d}] " +
                  "Dev mean_loss: {mean_loss:.5f}, " +
                  "acc: {acc:.5f}"
                    ).format(split_mode='Dev',
                            step=current_step,
                            mean_loss=mean_loss,
                            acc=acc
                            )
                    )

        if not FLAGS.test_phase and global_acc <= acc:
            global_acc = acc
            checkpoint_file = os.path.join(os.path.dirname(checkpoint_file), str(acc)+"_"+str(mean_loss)+"_model.ckpt")
            # Checkpoint Model
            save_path = saver.save(sess, checkpoint_file, global_step=trainer.global_step)
            log.info("Saved {}".format(save_path))

        if FLAGS.test_phase:
            # dump result into JSON
            result_json_path = os.path.join(
                os.path.dirname(FLAGS.checkpoint_path),
                "%s_%s_results.json" % (FLAGS.task.lower(), FLAGS.name.lower()))
            with open(result_json_path, 'w') as f:
                json.dump(result_json, f, sort_keys=True, indent=4, separators=(',', ': '))
                log.infov("Dumped result into : %s", result_json_path)


    if FLAGS.test_phase:
        log.infov("Evaluation mode! use --test_phase")
        train_loss, train_acc, current_step, time_delta = next(trainer.train_loop(train_iter))
        current_step = sess.run(trainer.global_step)
        log.info("Sample training step %d: loss = %.5f, acc = %.5f", current_step, train_loss, train_acc)
        run_evaluation(current_step)
        return

    for train_loss, train_acc, current_step, time_delta in trainer.train_loop(train_iter):
        if current_step % FLAGS.log_every == 0:
            log.info((" [{split_mode:5} step {step:4d}] " +
                      "batch mean_loss: {mean_loss:.5f}, batch mean_acc: {mean_acc:.5f} " +
                      "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec)"
                      ).format(split_mode='train',
                               step=current_step,
                               mean_loss=train_loss, mean_acc=train_acc,
                               sec_per_batch=time_delta,
                               instance_per_sec=FLAGS.batch_size / time_delta
                               )
                     )

        # Evaluate dev/test set
        if current_step % FLAGS.evaluate_every == 0:
            run_evaluation(current_step)

        if current_step % FLAGS.checkpoint_every == 0:
            checkpoint_file = os.path.join(os.path.dirname(checkpoint_file), str(acc) + "_"+str(train_loss)+"_model.ckpt")
            save_path = saver.save(sess, checkpoint_file, global_step=trainer.global_step)
            log.info("Saved {}".format(save_path))

def init_model():
    task = FLAGS.task

    model_params = {"feat_dim": FEAT_DIM, "word_embed": train_dataset.word_matrix, "lstm_steps": SEQUENCE_LENGTH}
    if task == 'FrameQA':
        model_params["vocabulary_size"] = len(train_dataset.idx2word)
        model_params["answer_size"] = len(train_dataset.idx2ans)

    model_params.update(FLAGS.__flags)

    if FLAGS.checkpoint_path:
        checkpoint = FLAGS.checkpoint_path
        params_path = os.path.join(os.path.dirname(checkpoint), '%s_%s_param.hkl' % (FLAGS.task.lower(), FLAGS.name.lower()))
        log.info("Restored parameter set from {}".format(params_path))
        model_params = hkl.load(open(params_path))
        print "ewlifjwelifjwelfjewif"

    model = Model.from_dict(model_params)
    model.print_params()

    video = tf.placeholder(tf.float32, [FLAGS.batch_size] + list(train_dataset.get_video_feature_dimension()))
    video_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, SEQUENCE_LENGTH])
    answer = tf.placeholder(tf.int32, [FLAGS.batch_size, 1])
    train_flag = tf.placeholder(tf.bool)

    if task is 'Count' or task is 'FrameQA':
        question = tf.placeholder(tf.int32, [FLAGS.batch_size, SEQUENCE_LENGTH])
        question_mask = tf.placeholder(tf.int32, [FLAGS.batch_size, SEQUENCE_LENGTH])
    else:
        question = tf.placeholder(tf.int32, [FLAGS.batch_size, Model.MULTICHOICE_COUNT, SEQUENCE_LENGTH])
        question_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, Model.MULTICHOICE_COUNT, SEQUENCE_LENGTH])

    model.build_graph(video, video_mask, question, question_mask, answer, train_flag)
    return model, model_params


if __name__ == '__main__':
    with graph.as_default(), sess.as_default():
        main()
