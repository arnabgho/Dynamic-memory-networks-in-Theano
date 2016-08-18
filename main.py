import sys
import numpy as np
import sklearn.metrics as metrics
import argparse
import time
import json
import re
import utils
import nn_utils
import h5py
from progress.bar import Bar
print "==> parsing input arguments"
parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default="plus_scan_vqa_bounding_shared_att_dmn_batch", help='network type: dmn_basic, dmn_smooth, vqa_dmn_batch,vqa_image_dmn_batch or dmn_batch')
parser.add_argument('--word_vector_size', type=int, default=300, help='embeding size (50, 100, 200, 300 only)')
parser.add_argument('--dim', type=int, default=320, help='number of hidden units in input module GRU')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--load_state', type=str, default="", help='state file path')
parser.add_argument('--answer_module', type=str, default="recurrent", help='answer module type: feedforward or recurrent')
parser.add_argument('--mode', type=str, default="train", help='mode: train or test. Test mode required load_state')
parser.add_argument('--input_mask_mode', type=str, default="sentence", help='input_mask_mode: word or sentence')
parser.add_argument('--memory_hops', type=int, default=5, help='memory GRU steps')
parser.add_argument('--batch_size', type=int, default=50, help='no commment')
parser.add_argument('--babi_id', type=str, default="1", help='babi task ID')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--normalize_attention', type=bool, default=True, help='flag for enabling softmax on attention vector')
parser.add_argument('--log_every', type=int, default=1, help='print information every x iteration')
parser.add_argument('--save_every', type=int, default=1, help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="", help='optional prefix of network name')
parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
parser.add_argument('--babi_test_id', type=str, default="", help='babi_id of test set (leave empty to use --babi_id)')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate (between 0 and 1)')
parser.add_argument('--batch_norm', type=bool, default=True, help='batch normalization')
parser.add_argument('--h5file',type=str,default="data/vqa_fact_data_prepro.h5",help="The h5 file containing ")
parser.add_argument('--json_dict_file',type=str,default="data/vqa_fact_data_prepro.json",help="The json file containing dicts")
parser.add_argument('--num_answers',type=int,default=1000,help="The number of answers")
parser.add_argument('--learning_rate_decay',type=float,default=0.5,help="The learning rate decay")
parser.add_argument('--img_h5file_train',type=str,default="data/vqa_box_feat_train.h5",help="The h5 file containing img features")
parser.add_argument('--img_h5file_test',type=str,default="data/vqa_box_feat_test.h5",help="The h5 file containing img features")
parser.add_argument('--img_seq_len',type=int,default=20,help="The number of image memories present")
parser.add_argument('--img_vector_size',type=int,default=4096,help="The img memory size")

parser.set_defaults(shuffle=True)
args = parser.parse_args()

print args

assert args.word_vector_size in [50, 100, 200, 300]

network_name = args.prefix + '%s.mh%d.n%d.bs%d%s%s%s.babi%s' % (
    args.network,
    args.memory_hops,
    args.dim,
    args.batch_size,
    ".na" if args.normalize_attention else "",
    ".bn" if args.batch_norm else "",
    (".d" + str(args.dropout)) if args.dropout>0 else "",
    args.babi_id)


babi_train_raw, babi_test_raw = utils.get_babi_raw(args.babi_id, args.babi_test_id)

word2vec = utils.load_glove(args.word_vector_size)

args_dict = dict(args._get_kwargs())
args_dict['babi_train_raw'] = babi_train_raw
args_dict['babi_test_raw'] = babi_test_raw
args_dict['word2vec'] = word2vec

accuracies=[]
# init class
if args.network == 'dmn_batch':
    import dmn_batch
    dmn = dmn_batch.DMN_batch(**args_dict)

elif args.network == 'dmn_basic':
    import dmn_basic
    if (args.batch_size != 1):
        print "==> no minibatch training, argument batch_size is useless"
        args.batch_size = 1
    dmn = dmn_basic.DMN_basic(**args_dict)

elif args.network == 'dmn_smooth':
    import dmn_smooth
    if (args.batch_size != 1):
        print "==> no minibatch training, argument batch_size is useless"
        args.batch_size = 1
    dmn = dmn_smooth.DMN_smooth(**args_dict)

elif args.network == 'dmn_qa':
    import dmn_qa_draft
    if (args.batch_size != 1):
        print "==> no minibatch training, argument batch_size is useless"
        args.batch_size = 1
    dmn = dmn_qa_draft.DMN_qa(**args_dict)

elif args.network == 'vqa_dmn_batch':
    import vqa_dmn_batch
    dmn=vqa_dmn_batch.VQA_DMN_batch(**args_dict)

elif args.network == 'vqa_image_dmn_batch':
    import vqa_image_dmn_batch
    dmn=vqa_image_dmn_batch.VQA_IMAGE_DMN_batch(**args_dict)

elif args.network == 'vqa_image_only_dmn_batch':
    import vqa_image_only_dmn_batch
    dmn=vqa_image_only_dmn_batch.VQA_IMAGE_DMN_batch(**args_dict)


elif args.network == 'vqa_bounding_dmn_batch':
    import vqa_bounding_dmn_batch
    dmn=vqa_bounding_dmn_batch.VQA_BOUNDING_DMN_batch(**args_dict)

elif args.network == 'vqa_bounding_shared_att_dmn_batch':
    import vqa_bounding_shared_att_dmn_batch
    dmn=vqa_bounding_shared_att_dmn_batch.VQA_BOUNDING_SHARED_ATT_DMN_batch(**args_dict)

elif args.network == 'plus_vqa_bounding_shared_att_dmn_batch':
    import plus_vqa_bounding_shared_att_dmn_batch
    dmn=plus_vqa_bounding_shared_att_dmn_batch.PLUS_VQA_BOUNDING_SHARED_ATT_DMN_batch(**args_dict)

elif args.network == 'plus_scan_vqa_bounding_shared_att_dmn_batch':
    import plus_scan_vqa_bounding_shared_att_dmn_batch
    dmn=plus_scan_vqa_bounding_shared_att_dmn_batch.PLUS_VQA_BOUNDING_SHARED_ATT_DMN_batch(**args_dict)


else:
    raise Exception("No such network known: " + args.network)

epoch=0
if args.load_state != "":
    dmn.load_state(args.load_state)
    epoch=1+int(re.search("epoch(\d)+",args.load_state).group(1))


def do_epoch(mode, epoch, skipped=0):
    # mode is 'train' or 'test'
    y_true = []
    y_pred = []
    avg_loss = 0.0
    prev_time = time.time()

    batches_per_epoch = dmn.get_batches_per_epoch(mode)

    if mode=="test":
        batches_per_epoch=min(1000,batches_per_epoch)
    bar=Bar('processing',max=batches_per_epoch)
    for i in range(0, batches_per_epoch):
        step_data = dmn.step(i, mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]
        current_skip = (step_data["skipped"] if "skipped" in step_data else 0)
        log = step_data["log"]

        skipped += current_skip

        if current_skip == 0:
            avg_loss += current_loss

            for x in answers:
                y_true.append(x)

            for x in prediction.argmax(axis=1):
                y_pred.append(x)

            # TODO: save the state sometimes
            if (i % args.log_every == 0):
                cur_time = time.time()
                #print ("  %sing: %d.%d / %d \t loss: %.3f \t avg_loss: %.3f \t skipped: %d \t %s \t time: %.2fs" %
                #    (mode, epoch, i * args.batch_size, batches_per_epoch * args.batch_size,
                #     current_loss, avg_loss / (i + 1), skipped, log, cur_time - prev_time))
                prev_time = cur_time

        if np.isnan(current_loss):
            print "==> current loss IS NaN. This should never happen :) "
            exit()
        bar.next()
    bar.finish()

    avg_loss /= batches_per_epoch
    print "\n  %s loss = %.5f" % (mode, avg_loss)
    print "confusion matrix:"
    print metrics.confusion_matrix(y_true, y_pred)

    accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    print "accuracy: %.2f percent" % (accuracy * 100.0 / batches_per_epoch / args.batch_size)

    if len(accuracies)>0 and accuracies[-1]>accuracy:
        dmn.lr=dmn.lr*args.learning_rate_decay
    accuracies.append(accuracy)
    return avg_loss, skipped


if args.mode == 'train':
    print "==> training"
    skipped = 0
    while epoch < (args.epochs):
        start_time = time.time()

        if args.shuffle:
            dmn.shuffle_train_set()

        _, skipped = do_epoch('train', epoch, skipped)

        epoch_loss, skipped = do_epoch('test', epoch, skipped)

        state_name = 'states/%s.epoch%d.test%.5f.state' % (network_name, epoch, epoch_loss)

        if (epoch % args.save_every == 0):
            print "==> saving ... %s" % state_name
            dmn.save_params(state_name, epoch)

        print "epoch %d took %.3fs" % (epoch, float(time.time()) - start_time)
        epoch=epoch+1

elif args.mode == 'test':
    file = open('last_tested_model.json', 'w+')
    data = dict(args._get_kwargs())
    data["id"] = network_name
    data["name"] = network_name
    data["description"] = ""
    data["vocab"] = dmn.vocab.keys()
    json.dump(data, file, indent=2)
    do_epoch('test', 0)

else:
    raise Exception("unknown mode")
