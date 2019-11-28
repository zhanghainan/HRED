import os
import time
import argparse
from utils.vocab import *
from models.HRED import HREDModel
from models.model_base import ModelMode
from configs import *
from utils.iterator import *
import utils.misc_utils as utils
from utils.eval_utils import evaluate



def run_evaluate(sess, eval_model, infer_model,
                 vocab, config, global_step, data_dir, pred_dir,
                 data_iter, mode):
    assert mode in ["valid", "test"]
    loss, ppl = _run_internal_eval(sess, eval_model, data_iter)

    ctx_file = os.path.join(data_dir, "%s.context.txt" % mode)
    resp_file = os.path.join(data_dir, "%s.response.txt" % mode)

    infer_iter = get_infer_iter(context_file=ctx_file, vocab=vocab, config=config)
    pred_sents = _run_external_eval(sess, infer_model, infer_iter, vocab)

    pred_tgt_file = os.path.join(pred_dir, "%s_e%d_ppl_%.2f_loss_%.2f.pred.txt" %
                                 (mode, global_step, ppl, loss))

    utils.save_sentences(pred_sents, pred_tgt_file)

    bleu = evaluate(resp_file, pred_tgt_file)

    return loss, ppl, bleu


def _run_internal_eval(sess, eval_model, eval_iter):
    eval_loss, eval_predict_count, eval_samples = 0.0, 0, 0
    for batch_data in eval_iter.next_batch():
        step_loss, step_word_count, step_predict_count, batch_size, _ = eval_model.eval(sess, batch_data)

        eval_samples += batch_size
        eval_loss += step_loss * batch_size
        eval_predict_count += step_predict_count

    return eval_loss / eval_samples, utils.safe_exp(eval_loss / eval_predict_count)


def _run_external_eval(sess, infer_model, infer_iter, tgt_vocab, num_response_per_input=1):
    predict_sents = []
    for infer_batch_data in infer_iter.next_batch():
        batch_ids, batch_size = infer_model.infer(sess, infer_batch_data)

        for sent_id in range(batch_size):
            for beam_id in range(num_response_per_input):
                predict_id = batch_ids[sent_id, :, beam_id].tolist()
                predict_sent = tgt_vocab.convert2words(predict_id)
                predict_sents.append(predict_sent)
    return predict_sents


def load_vocab_setup_config(args):
    # load vocab from precessed vocab file
    vocab_file = os.path.join(args.data_dir, "vocab.dialog.txt")
    vocab = load_vocabulary(vocab_file)

    # get config
    config = eval(args.config)
    # setup vocab related config
    config.vocab_size = vocab.size
    config.sos_idx = vocab.sos_idx
    config.eos_idx = vocab.eos_idx

    return vocab, config


def run_prediction(path_in,path_out):
    '''
    file1 = open(path_in)
    cc = open("./data/ubuntu-10k/test.context.txt","w")
    for line in file1:
        split = line.split("<s>")
        print(line)
        sen=""
        for ss in split:
            sen += " ".join(w for w in ss[:-1])
            sen+=" </d> "
        cc.write("%s\n"%(sen))
    cc.close()
    '''
    start_time = time.time()
    args = parse_args()
    vocab, model_config = load_vocab_setup_config(args)
    print('... load vocab and setup model config over, cost:\t%.2f s' % (time.time() - start_time))
    print('... vocab size:\t%d' % vocab.size)

    start_time = time.time()
    train_iter, valid_iter, test_iter = get_train_iter(args.data_dir, vocab=vocab, config=model_config)
    print('-' * 100)
    print('... load train and valid data iterator over, cost:\t%.2f s' % (time.time() - start_time))
    print('... train iterator samples:\t%d' % train_iter.num_samples)
    print('... valid iterator samples:\t%d' % valid_iter.num_samples)
    print('... test iterator samples:\t%d' % test_iter.num_samples)

    # prepare output dir
    output_dir = args.output_dir
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    ckpt_dir2 = os.path.join(output_dir, "checkpoints2")
    ckpt_dir3 = os.path.join(output_dir, "checkpoints3")
    log_dir = os.path.join(output_dir, "train_log")
    pred_dir = os.path.join(output_dir, "pred")

    utils.mkdir_not_exists(output_dir)
    utils.mkdir_not_exists(ckpt_dir)
    utils.mkdir_not_exists(ckpt_dir2)
    utils.mkdir_not_exists(ckpt_dir3)
    utils.mkdir_not_exists(log_dir)
    utils.mkdir_not_exists(pred_dir)
    ckpt_path = os.path.join(ckpt_dir, model_config.model)
    ckpt_path2 = os.path.join(ckpt_dir2, model_config.model)
    ckpt_path3 = os.path.join(ckpt_dir3, model_config.model)
    print('=' * 100)
    print('... building model')
    start_time = time.time()
    if model_config.model == 'HRED':
        model = HREDModel
    else:
        raise NotImplementedError("No such model")

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config_proto) as sess:
        initializer = tf.random_uniform_initializer(-1.0 * model_config.init_w, model_config.init_w)
        scope = model_config.model
        with tf.variable_scope(scope, reuse=None, initializer=initializer):
            train_model = model(config=model_config, mode=ModelMode.train, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            eval_model = model(config=model_config, mode=ModelMode.eval, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            infer_model = model(config=model_config, mode=ModelMode.infer, scope=scope)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        ckpt2 = tf.train.get_checkpoint_state(ckpt_dir2)
        ckpt3 = tf.train.get_checkpoint_state(ckpt_dir3)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            train_model.saver.restore(sess, ckpt.model_checkpoint_path)
            print("Reading model parameters from %s" % ckpt2.model_checkpoint_path)
            eval_model.saver.restore(sess, ckpt2.model_checkpoint_path)
            print("Reading model parameters from %s" % ckpt3.model_checkpoint_path)
            infer_model.saver.restore(sess, ckpt3.model_checkpoint_path)
        else:
            print('... create %s model over, time cost: %.2fs' % (model_config.model, time.time() - start_time))
            print('=' * 100)
            sess.run(tf.global_variables_initializer())

        start_time = time.time()
        ckpt_loss, ckpt_ppl, ckpt_predict_count, ckpt_samples = 0.0, 0.0, 0, 0
        test_loss, test_ppl, test_bleu = run_evaluate(sess, eval_model, infer_model,
                                                              vocab, model_config,
                                                              0,
                                                              args.data_dir,
                                                              pred_dir,
                                                              test_iter, "test")

    file2 = open("./data/ubuntu_10k_output/pred/test_e0_ppl_%.2f_loss_%.2f.pred.txt"%(test_ppl,test_loss))
    output = open(path_out,"w")
    for line in file2:
        output.write("%s\n"%("".join(w for w in line.split())))
    output.close()

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Dialog Generation')

    parser.add_argument("--config", type=str,
                        default="HREDTestConfig", help="model config")
    parser.add_argument("--data_dir", type=str,
                        default="./data/ubuntu-10k/", help="training input dir")
    parser.add_argument("--output_dir", type=str,
                        default="./data/ubuntu_10k_output", help="training output dir")

    return parser.parse_args()


if __name__ == '__main__':
    run_prediction("./test/test.txt","./test/result.txt")


