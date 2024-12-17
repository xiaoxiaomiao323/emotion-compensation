import os.path
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
from utils.logger import setup_logger
from utils.manipulator import linear_interpolate
from kaldiio import ReadHelper
import joblib
from sklearn import svm

def read_raw_mat(filename,col,format='f4',end='l'):
    """read_raw_mat(filename,col,format='float',end='l')
       Read the binary data from filename
       Return data, which is a (N, col) array
       
       filename: the name of the file, take care about '\\'
       col:      the number of column of the data
       format:   please use the Python protocal to write format
                 default: 'f4', float32
                 see for more format:
       end:      little endian 'l' or big endian 'b'?
                 default: 'l'
       
       dependency: numpy
       Note: to read the raw binary data in python, the question
             is how to interprete the binary data. We can use
             struct.unpack('f',read_data) to interprete the data
             as float, however, it is slow.
    """
    f = open(filename,'rb')
    if end=='l':
        format = '<'+format
    elif end=='b':
        format = '>'+format
    else:
        format = '='+format
    datatype = np.dtype((format,(col,)))
    data = np.fromfile(f,dtype=datatype)
    f.close()
    if data.ndim == 2 and data.shape[1] == 1:
        return data[:,0]
    else:
        return data

def write_raw_mat(data, filename, format='f4', end='l'):
    """write_raw_mat(data,filename,format='',end='l')
       Write the binary data from filename.
       Return True

       data:     np.array
       filename: the name of the file, take care about '\\'
       format:   please use the Python protocal to write format
                 default: 'f4', float32
       end:      little endian 'l' or big endian 'b'?
                 default: '', only when format is specified, end
                 is effective

       dependency: numpy
       Note: we can also write two for loop to write the data using
             f.write(data[a][b]), but it is too slow
    """
    if not isinstance(data, np.ndarray):
        print("Error write_raw_mat: input shoul be np.array")
        return False
    f = open(filename, 'wb')
    if len(format) > 0:
        if end == 'l':
            format = '<' + format
        elif end == 'b':
            format = '>' + format
        else:
            format = '=' + format
        datatype = np.dtype((format, 1))
        temp_data = data.astype(datatype)
    else:
        temp_data = data
    temp_data.tofile(f, '')
    f.close()
    return True




def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Edit image synthesis with given semantic boundary.')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-b', '--boundary_path', type=str, required=True,
                      help='Path to the semantic boundary. (required)')
  parser.add_argument('-i', '--input_latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                        'path instead of randomly sampling. (optional)')
  parser.add_argument('-u2e', '--utt2emo_file', type=str, required=True)
  parser.add_argument('-norm', '--norm_type', type=str, required='std',
                      help='norm type.std inter (required)')
  parser.add_argument('-n', '--number_sample', type=int, default=1,
                      help='Number of images for edicting.')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('--start_distance', type=float, default=30.0,
                      help='Start point for manipulation in latent space. '
                           '(default: -3.0)')
  parser.add_argument('--end_distance', type=float, default=40.0,
                      help='End point for manipulation in latent space. '
                           '(default: 3.0)')
  parser.add_argument('--steps', type=int, default=2,
                      help='Number of steps for image editing. (default: 2)')

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
  logger = setup_logger(args.output_dir, logger_name='generate_data')

  logger.info(f'Preparing boundary.')
  if not os.path.isdir(args.boundary_path):
    raise ValueError(f'Boundary `{args.boundary_path}` does not exist!')
  # boundary = np.load(args.boundary_path)
  # np.save(os.path.join(args.output_dir, 'boundary.npy'), boundary)

  logger.info(f'Preparing latent codes.')
  if os.path.isfile(args.input_latent_codes_path):
    logger.info(f'  Load latent codes from `{args.input_latent_codes_path}`.')
    latent_codes = np.zeros((args.number_sample, 192))
    i = 0
    names = []
    utt2emo = {}
    with open(args.utt2emo_file, 'r') as f:
      for line in f:
        utt, emo = line.strip().split()
        if emo == 'ang':
          emo = 'angry'
        elif emo == 'hap':
          emo = 'happy'
        elif emo == 'neu':
          emo = 'neutral'
        elif emo == 'sad':
          emo = 'sad'
        else:
          raise ValueError(f'emo `{emo}` does not exist!')
        utt2emo[utt] = emo
    with ReadHelper('scp:' + args.input_latent_codes_path) as reader:
        for key, mat in reader:
            if args.norm_type == 'std':
                mat_norm = norm_std(mat)
            elif args.norm_type == 'inter':
                mat_norm = norm(mat)
            else:
                raise ValueError(f'norm type does not exist!')
            latent_codes[i, :] = mat_norm
            names.append(key)
            i = i + 1
  else:
    raise ValueError(f'latent `{args.input_latent_codes_path}` does not exist!')
  np.save(os.path.join(args.output_dir, 'latent_codes.npy'), latent_codes)
  total_num = latent_codes.shape[0]

  angry_svm = joblib.load(os.path.join(args.boundary_path, 'ESD_MSP_emo_inter_angry/svm.m'))
  happy_svm = joblib.load(os.path.join(args.boundary_path, 'ESD_MSP_emo_inter_happy/svm.m'))
  neutral_svm = joblib.load(os.path.join(args.boundary_path, 'ESD_MSP_emo_inter_neutral/svm.m'))
  sad_svm = joblib.load(os.path.join(args.boundary_path, 'ESD_MSP_emo_inter_sad/svm.m'))

  svms = {'angry':angry_svm, 'happy':happy_svm, 'neutral':neutral_svm, 'sad':sad_svm}

  logger.info(f'Editing {total_num} samples.')
  for sample_id in tqdm(range(total_num), leave=False):
    cur_name = names[sample_id]
    emo = utt2emo[cur_name]
    boundary = np.load(os.path.join(args.boundary_path, 'ESD_MSP_emo_inter_{}/boundary.npy'.format(emo)))
    if emo == 'sad':
      start = -1*args.start_distance
      end = -1*args.end_distance
    else:
      start = args.start_distance
      end = args.end_distance
    interpolations = linear_interpolate(latent_codes[sample_id:sample_id + 1],
                                        boundary,
                                        start_distance=start,
                                        end_distance=end,
                                        steps=args.steps)
    for i in range(interpolations.shape[0]):
        write_raw_mat(interpolations[i,:], join(args.output_dir, names[sample_id] + '_' + str(i) +'.xvector'))
    logger.debug(f'  Finished sample {sample_id:3d}.')
  logger.info(f'Successfully edited {total_num} samples.')


if __name__ == '__main__':
  main()
