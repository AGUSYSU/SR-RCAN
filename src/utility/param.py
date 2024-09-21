import argparse
import os

parser = argparse.ArgumentParser(description='RCAN')
parser.add_argument("--debug", action='store_true', help="Enablesdebug mode")
parser.add_argument("--template",
                    default='.',
                    help="You can set various templates in option.py")

# Hardware
parser.add_argument("--n_threads",
                    type=int,
                    default=3,
                    help="number of threads for data loading")

parser.add_argument("--cpu", action="store_true", help="use cpu only")

parser.add_argument("--gpu_id", type=int, default=0, help="index of GPUs")

parser.add_argument("--n_GPUs", type=int, default=1, help="number of GPUs")

parser.add_argument("--seed", type=int, default=114514, help="randon seed")

# Data
parser.add_argument("--dir_data",
                    type=str,
                    default='.',
                    help="dataset directory")

parser.add_argument("--data_train",
                    type=str,
                    default='MYDATA',
                    help="train dataset name")

parser.add_argument("--data_test",
                    type=str,
                    default='MYDATA',
                    help="test dataset name")

parser.add_argument("--offset_val",
                    type=int,
                    default=800,
                    help="validation index offest")

parser.add_argument("--scale",
                    type=int,
                    default=4,
                    help="super resolution scale")

parser.add_argument("--patch_size",
                    type=int,
                    default=192,
                    help="output patch size")

parser.add_argument("--rgb_range",
                    type=int,
                    default=255,
                    help="maximum value of RGB")

parser.add_argument("--n_colors",
                    type=int,
                    default=3,
                    help="number of color channels to use")

parser.add_argument('--noise',
                    type=str,
                    default='.',
                    help='Gaussian noise std.')

parser.add_argument('--chop',
                    action='store_true',
                    help='enable memory-efficient forward')

# Model

parser.add_argument('--model', default='RCAN', help='model name')
parser.add_argument('--act',
                    type=str,
                    default='relu',
                    help='activation function')

parser.add_argument('--pre_train',
                    type=str,
                    default='.',
                    help='pre_trian model path')

parser.add_argument('--n_resblocks',
                    type=int,
                    default=20,
                    help='number of residual blocks')

parser.add_argument('--n_feats',
                    type=int,
                    default=64,
                    help='number of feature maps')

parser.add_argument('--res_scale',
                    type=float,
                    default=1,
                    help='residual scaling')

parser.add_argument('--shift_mean',
                    default=True,
                    help='subtract pixel mean from the input')

parser.add_argument('--precision',
                    type=str,
                    default='single',
                    choices=("single", "half"),
                    help='FP precision for test (single | half)')

# Training
parser.add_argument('--reset', action='store_true', help='reset the training')

parser.add_argument('--epochs',
                    type=int,
                    default=1000,
                    help='number of epochs')

parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='batch size to train')

parser.add_argument('--repeat',
                    type=int,
                    default=1,
                    help='dataset repeat times')

parser.add_argument('--split_batch',
                    type=int,
                    default=1,
                    help='split the batch into smaller chunks')

parser.add_argument('--self_ensemble',
                    action='store_true',
                    help='use self-ensemble method for test')

parser.add_argument('--test-only', action='store_true', help='test the model')

parser.add_argument('--gan_k',
                    type=int,
                    default=1,
                    help='k value for adversarial loss')

# Optimization
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument('--lr_decay',
                    type=int,
                    default=200,
                    help='lr decay per N epochs')

parser.add_argument('--decay_type',
                    type=str,
                    default="step",
                    help='lr decay type')

parser.add_argument('--gamma',
                    type=float,
                    default=0.5,
                    help='learning rate decay factor for step decay')

parser.add_argument('--optimizer',
                    default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer (SGD | ADAM | RMSprop)')

parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')

parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')

parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')

parser.add_argument('--epsilon',
                    type=float,
                    default=1e-8,
                    help='ADAM epsilon for numerical stability')

parser.add_argument('--weight_decay',
                    type=float,
                    default=0,
                    help='weight_decay')

# loss
parser.add_argument('--loss',
                    type=str,
                    default="1*L1",
                    help='loss function configuration')

parser.add_argument('--skip_threshold',
                    type=float,
                    default='1e6',
                    help='skipping batch that has large error')

# Log
parser.add_argument('--save',
                    type=str,
                    default='RCAN',
                    help='file name to save')

parser.add_argument('--load', type=str, default='.', help='file name to load')

parser.add_argument('--resume',
                    type=int,
                    default=0,
                    help='resume from specific checkpoint')

parser.add_argument('--print_model', action='store_true', help='print model')

parser.add_argument('--save_models', action="store_true", help='save model')

parser.add_argument(
    '--print_every',
    type=int,
    default=100,
    help='how many batches to wait before logging training status')

parser.add_argument('--save_results', action="store_true", help='save results')

# residual
parser.add_argument('--n_resgroups',
                    type=int,
                    default=10,
                    help='number of residual groups')
parser.add_argument('--reduction',
                    type=int,
                    default=16,
                    help='number of feature maps reduction')

# test
parser.add_argument('--test_path',
                    type=str,
                    default='DIV2K/DIV2K_valid_LR_bicubic/X4',
                    help='dataset directory for testing')

parser.add_argument('--testset',
                    type=str,
                    default='DIV2k',
                    help='dataset name for testing')

# eval
parser.add_argument('--img_path',
                    type=str,
                    default='.',
                    help='image path')

parser.add_argument('--is_sharp',
                    type=bool,
                    default=True,
                    help='image path')
####
args = parser.parse_args()
# template.set_template(args)

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

if args.dir_data == '.':
    param_path = os.path.abspath(__file__)
    args.dir_data = os.path.dirname(os.path.dirname(os.path.dirname(param_path)))