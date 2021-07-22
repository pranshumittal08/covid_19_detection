import os
import platform
import sys
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import pathlib
sys.path.append(os.path.join(os.getcwd(), 'automl'))
sys.path.append(os.path.join(os.getcwd(), 'automl', 'efficientnetv2'))
sys.path.insert(0, os.path.join(os.getcwd(), 'automl', 'efficientdet'))

import covid_19_dataloader
import hparams_config
import utils
from keras import tfmot
from keras import train_lib
from keras import util_keras

def setup_model(model, config):
    """Build and compile model."""
    model.build((None, *config.image_size, 3))
    model.compile(
          steps_per_execution=config.steps_per_execution,
          optimizer=train_lib.get_optimizer(config.as_dict()),
          loss={
              train_lib.BoxLoss.__name__:
                  train_lib.BoxLoss(
                      config.delta, reduction=tf.keras.losses.Reduction.NONE),
              train_lib.BoxIouLoss.__name__:
                  train_lib.BoxIouLoss(
                      config.iou_loss_type,
                      config.min_level,
                      config.max_level,
                      config.num_scales,
                      config.aspect_ratios,
                      config.anchor_scale,
                      config.image_size,
                      reduction=tf.keras.losses.Reduction.NONE),
              train_lib.FocalLoss.__name__:
                  train_lib.FocalLoss(
                      config.alpha,
                      config.gamma,
                      label_smoothing=config.label_smoothing,
                      reduction=tf.keras.losses.Reduction.NONE),
              tf.keras.losses.CategoricalCrossentropy.__name__:
                  tf.keras.losses.CategoricalCrossentropy(
                      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
          })
    return model

def init_experimental(config):
    """Serialize train config to model directory."""
    tf.io.gfile.makedirs(config.model_dir)
    config_file = os.path.join(config.model_dir, 'config.yaml')
    if not tf.io.gfile.exists(config_file):
        tf.io.gfile.GFile(config_file, 'w').write(str(config))

model_dir = os.path.join('C:\\Users\\prans\\Python files\\Kaggle Competitions\\Covid_19_object_detection', 'object_detection', 'models')
val_file_pattern = 'D:\\Datasets\\siim_covid19_detection\\1080px\\tfrecords\\fold_0\\valid\\*'
train_file_pattern = 'D:\\Datasets\\siim_covid19_detection\\1080px\\tfrecords\\fold_0\\train\\*'
val_json_file = r'D:\Datasets\siim_covid19_detection\1080px\object_detection_files\files_fold_0\valid\object_detection_info.json'
hparams = ''
model_name = 'efficientdetv2-s'
num_epochs = 1
batch_size = 16
num_of_examples_per_epoch = 6000
mode = 'traineval'
debug = False
use_fake_data = False
eval_samples = 2000
steps_per_execution = 1
lr_warmup_init = 0.0005
weight_decay = 0.001
num_scales = 2
# backbone_config = {'pretrained_path': None,
#                    'weights': 'imagenet'}

def main():
    config = hparams_config.get_detection_config(model_name)
    config.override(hparams)
    config.num_epochs = num_epochs
    config.image_size = utils.parse_image_size(config.image_size)
    steps_per_epoch = num_of_examples_per_epoch//batch_size
    params = dict(
        model_name = model_name,
        steps_per_execution = steps_per_execution,
        model_dir = model_dir,
        steps_per_epoch = 1,
        batch_size = batch_size,
        tf_random_seed = 42,
        debug = debug,
        val_json_file = val_json_file,
        lr_warmup_init = lr_warmup_init,
        weigth_decay = weight_decay,
        # backbone_config = backbone_config,
        num_scales = num_scales)
    config.override(params, True)

    def get_dataset(is_training, config):
        file_pattern = (
            train_file_pattern
            if is_training else val_file_pattern)
        if not file_pattern:
            raise ValueError('No matching files.')

        return covid_19_dataloader.InputReader(
            file_pattern,
            is_training=is_training,
            use_fake_data=use_fake_data,
            max_instances_per_image=config.max_instances_per_image,
            debug=debug)(config.as_dict())

    model = train_lib.EfficientDetNetTrain(config=config)
    model = setup_model(model, config)
    print('!!!!!!!model loaded successfully!!!!!!')
    if debug:
        tf.config.run_functions_eagerly(True)
    
    if 'train' in mode:
        val_dataset = get_dataset(False, config) if 'eval' in mode else None
        print('!!!!!!Data loaded!!!!!!')
        model.fit(get_dataset(True, config),
                 epochs = config.num_epochs,
                 steps_per_epoch = steps_per_epoch,
                
                 validation_data = val_dataset,
                 validation_steps = (eval_samples//batch_size))
    else:
        for ckpt in tf.train.checkpoints_iterator(model_dir, min_interval_secs = 180):
            logging.info('Starting to evaluate.')
            try:
                current_epoch = int(os.path.basename(ckpt).split('-')[1])
            except IndexError:
                current_epoch = 0
            
            val_dataset = get_dataset(False, config)
            logging.info('start loading model.')
            model.load_weights(tf.train.latest_checkpoint(model_dir))
            logging.info('finish loading model.')

#  callbacks = train_lib.get_callbacks(config.as_dict(), 
#                                                       val_dataset),
if __name__ == '__main__':
  main()