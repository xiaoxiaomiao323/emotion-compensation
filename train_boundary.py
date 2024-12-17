# python3.7
"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os.path
import argparse
import numpy as np

from utils.logger import setup_logger
from utils.manipulator import train_boundary
from kaldiio import ReadHelper
import pickle
import joblib

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Train semantic boundary with given latent codes and '
                  'attribute scores.')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-c', '--latent_codes_path', type=str, required=True,
                      help='Path to the input latent codes. (required)')
  parser.add_argument('-s', '--scores_path', type=str, required=True,
                      help='Path to the input attribute scores. (required)')
  parser.add_argument('-norm', '--norm_type', type=str, required='std',
                      help='norm type.std inter (required)')
  parser.add_argument('-n', '--chosen_num_or_ratio', type=float, default=0.9,
                      help='How many samples to choose for training. '
                           '(default: 0.2)')
  parser.add_argument('-nt', '--number_train', type=float, default=453151,
                      help='How many training samples in total. '
                      '(gender:453151)')
  parser.add_argument('-r', '--split_ratio', type=float, default=0.7,
                      help='Ratio with which to split training and validation '
                           'sets. (default: 0.7)')
  parser.add_argument('-V', '--invalid_value', type=float, default=None,
                      help='Sample whose attribute score is equal to this '
                           'field will be ignored. (default: None)')

  return parser.parse_args()

def norm_std(data):
    data_mean = data.mean(axis=1)
    data_std = data.std(axis=1)
    data_norm = (data - data_mean)/data_std
    return data_norm

def norm(data):
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    data_norm = data / norm * np.sqrt(192)
    return data_norm

def main():
  """Main function."""
  args = parse_args()
  if os.path.exists(args.output_dir):
      os.system("rm -rf %s" % args.output_dir)
  logger = setup_logger(args.output_dir, logger_name='generate_data')

  logger.info('Loading latent codes.')
  if not os.path.isfile(args.latent_codes_path):
    raise ValueError(f'Latent codes `{args.latent_codes_path}` does not exist!')
  
  logger.info('Loading attribute scores.')
  if not os.path.isfile(args.scores_path):
    raise ValueError(f'Attribute scores `{args.scores_path}` does not exist!')
  with open(args.scores_path,'rb') as tf:
      scores_temp = pickle.load(tf)

  scores = np.zeros(int(args.number_train))
  latent_codes = np.zeros((int(args.number_train),192))
  i = 0
  with ReadHelper('scp:'+args.latent_codes_path) as reader:
    for key, mat in reader:
      if args.norm_type == 'std':
        mat_norm = norm_std(mat)
      elif args.norm_type == 'inter':
        mat_norm = norm(mat)
      else:
        raise ValueError(f'norm type does not exist!')
      latent_codes[i,:] = mat_norm
      #print(key)
      # spk_id = key.split('-')[0]
      # scores[i] = scores_temp[spk_id]
      scores[i] = scores_temp[key]
      i = i + 1

  if i != args.number_train:
      raise ValueError("check the number of training samples")

  scores = np.expand_dims(scores, axis=1)
  # boundary = train_boundary(latent_codes=latent_codes,
  latent_space_dim = latent_codes.shape[1]
  classifier = train_boundary(latent_codes=latent_codes,
                            scores=scores,
                            chosen_num_or_ratio=args.chosen_num_or_ratio,
                            split_ratio=args.split_ratio,
                            invalid_value=args.invalid_value,
                            logger=logger)
  a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
  boundary = a / np.linalg.norm(a)
  np.save(os.path.join(args.output_dir, 'boundary.npy'), boundary)
  joblib.dump(classifier, os.path.join(args.output_dir, "svm.m"))


if __name__ == '__main__':
  main()
