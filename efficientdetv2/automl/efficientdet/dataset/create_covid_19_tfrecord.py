import collections
import hashlib
import io
import json
import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import PIL.Image
import tensorflow as tf
import pandas as pd
from dataset import label_map_util
from dataset import tfrecord_util


flags.DEFINE_string('image_dir', '', 'Directory containing images.')
flags.DEFINE_string(
    'image_info_file', '', 'File containing image information. '
    'Tf Examples in the output files correspond to the image '
    'info entries in this file. If this file is not provided '
    'object_annotations_file is used if present. Otherwise, '
    'caption_annotations_file is used to get image info.')
flags.DEFINE_string(
    'object_annotations_file', '', 'File containing object '
    'annotations - boxes and instance masks.')

flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
flags.DEFINE_integer('num_threads', 10, 'Number of threads to run.')
FLAGS = flags.FLAGS


def create_tf_example(image, 
                      image_dir, 
                      bbox_annotations = None,
                      category_index = None):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys: [ u'file_name',  u'height',
      u'width']
    image_dir: directory containing the image files.
    bbox_annotations:
      list of dicts with keys: [u'image_id', u'bbox', u'category_id', 'id'] Notice that bounding box
        coordinates are given as [x, y, width,
        height] tuples using absolute coordinates where x, y represent the
        top-left (0-indexed) corner.  This function converts to the format
        expected by the Tensorflow Object Detection API (which is
        [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
        size).
    category_index: a dict containing category information keyed by the
      'id' field of each category.  See the label_map_util.create_category_index
      function.


  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()
  feature_dict = {
      'image/height':
          tfrecord_util.int64_feature(image_height),
      'image/width':
          tfrecord_util.int64_feature(image_width),
      'image/filename':
          tfrecord_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          tfrecord_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          tfrecord_util.bytes_feature(encoded_jpg),
      'image/format':
          tfrecord_util.bytes_feature('png'.encode('utf8')),
  }

  num_annotations_skipped = 0
  if bbox_annotations:
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    # is_crowd = []
    area = []
    category_names = []
    category_ids = []
    for object_annotations in bbox_annotations:
      (x, y, width, height) = tuple(object_annotations['bbox'])
      if width <= 0 or height <= 0:
        num_annotations_skipped += 1
        continue
      if x + width > image_width or y + height > image_height:
        num_annotations_skipped += 1
        continue
      xmin.append(float(x) / image_width)
      xmax.append(float(x + width) / image_width)
      ymin.append(float(y) / image_height)
      ymax.append(float(y + height) / image_height)
      # is_crowd.append(object_annotations['is_crowd'])
      category_id = int(object_annotations['category_id'])
      category_ids.append(category_id)
      category_names.append(category_index[category_id]['name'].encode('utf8'))
      area.append(object_annotations['area'])
      
    feature_dict.update({
        'image/object/bbox/xmin':
            tfrecord_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            tfrecord_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            tfrecord_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            tfrecord_util.float_list_feature(ymax),
        'image/object/class/text':
            tfrecord_util.bytes_list_feature(category_names),
        'image/object/class/label':
            tfrecord_util.int64_list_feature(category_ids),
        'image/object/area':
            tfrecord_util.float_list_feature(area),
    })

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped


def _pool_create_tf_example(args):
  return create_tf_example(*args)


def _load_object_annotations(object_annotations_file):
  """Loads object annotation JSON file."""
  with tf.io.gfile.GFile(object_annotations_file, 'r') as fid:
    obj_annotations = json.load(fid)

  images = obj_annotations['images']
  category_index = label_map_util.create_category_index(
      obj_annotations['categories'])

  img_to_obj_annotation = collections.defaultdict(list)
  logging.info('Building bounding box index.')
  # Assigning all the bboxes to their corresponding images and storing in dictionary format
  for annotation in obj_annotations['annotations']:
    image_id = annotation['image_id']
    img_to_obj_annotation[image_id].append(annotation)

  # Computing the total number of images with no annotations
  missing_annotation_count = 0
  for image in images:
    image_id = image['id']
    if image_id not in img_to_obj_annotation:
      missing_annotation_count += 1

  logging.info('%d images are missing bboxes.', missing_annotation_count)

  return img_to_obj_annotation, category_index

def _load_images_info(image_info_file):
  with tf.io.gfile.GFile(image_info_file, 'r') as fid:
    info_dict = json.load(fid)
  return info_dict['images']



def _create_tf_record_from_covid_19_annotations(image_info_file, image_dir, output_path, num_shards,
object_annotations_file=None):
  """Loads covid 19 chest x ray annotation json files and converts to tf.Record format.

  Args:
    image_info_file: JSON file containing image info. The number of tf.Examples
      in the output tf Record files is exactly equal to the number of image info
      entries in this file. This can be any of train/val/test annotation json
      files Eg. 'image_info_test-dev2017.json',
      'instance_annotations_train2017.json',
      'caption_annotations_train2017.json', etc.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    num_shards: Number of output files to create.
    object_annotations_file: JSON file containing bounding box annotations.
  """

  if not tf.io.gfile.isdir(output_path):
    tf.io.gfile.mkdir(output_path)

  logging.info('writing to output path: %s', output_path)
  writers = [
      tf.io.TFRecordWriter(os.path.join(output_path, output_path.split(os.sep)[-1]+ f'-{i}-of-{num_shards}.tfrecord')) for i in range(num_shards)
  ]
  images = _load_images_info(image_info_file)

  img_to_obj_annotation = None
  category_index = None
  if object_annotations_file:
    img_to_obj_annotation, category_index = (
        _load_object_annotations(object_annotations_file))


  def _get_object_annotation(image_id):
    if img_to_obj_annotation:
      return img_to_obj_annotation[image_id]
    else:
      return None

  print(category_index)

  pool = multiprocessing.Pool(10)
  total_num_annotations_skipped = 0
  for idx, (_, tf_example, num_annotations_skipped) in enumerate(
      pool.imap(
          _pool_create_tf_example,
          [(image, image_dir, _get_object_annotation(image['id']),
            category_index)
           for image in images])):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(images))

    total_num_annotations_skipped += num_annotations_skipped
    writers[idx % num_shards].write(tf_example.SerializeToString())

  pool.close()
  pool.join()

  for writer in writers:
    writer.close()

  logging.info('Finished writing, skipped %d annotations.',
               total_num_annotations_skipped)

# if __name__ == '__main__':
  
#   data_path = r'D:\Datasets\siim_covid19_detection'
  
#   data_path_1080px = os.path.join(data_path, '1080px')
  
#   main_df = pd.read_csv(os.path.join(data_path, 'main.csv'))

#   for fold in range(5):
#     image_info_file = os.path.join(data_path_1080px, 'object_detection_files', f'files_fold_{fold}', 'train', 'images_info.json')
    
#     image_dir = os.path.join(data_path_1080px, 'train', 'image')

#     output_path =os.path.join(data_path_1080px, 'tfrecords', f'fold_{fold}', 'train')

#     os.makedirs(output_path, exist_ok= True)
#     num_shards = 32
#     object_annotations_file= os.path.join(data_path_1080px,'object_detection_files', f'files_fold_{fold}', 'train', 'object_detection_info.json')
    # _create_tf_record_from_covid_19_annotations(image_info_file, image_dir, output_path, num_shards,
                                                                        # object_annotations_file)  