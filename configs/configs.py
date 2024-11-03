import argparse
import torchvision
import os

class BaseOptions():
    """This class defines options used during both training and test time"""

    def __init__(self):
        parser = argparse.ArgumentParser()
        """Define the common options that are used in both training and test."""
        # Device parameters
        parser.add_argument('--gpu_id', type=int, default=0)
        parser.add_argument('--type', type=str, default='Visual and Near-infrared Fusion')
        # Data parameters
        parser.add_argument('--train_root',type=str, default='../data/vision/MSRS/train')
        parser.add_argument('--valid_root',type=str, default='../data/vision/MSRS/test')
        parser.add_argument('--test_root',type=str, default='../data/vision/MSRS/test')

        """ Pansharpening Network """
        # train parameters
        parser.add_argument('--n_epochs', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--save_per_epoch', type=int, default=1)
        parser.add_argument('--test_per_epoch', type=int, default=499)
        parser.add_argument('--isshuffle', type=bool, default=True)
        parser.add_argument('--iscontinue', type=bool, default=False)
        parser.add_argument('--continue_load_name', type=str, default='pretrain')
        
        # Optimizer parameters
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.999)
        parser.add_argument('--initial_loss', type=float, default=999)
        # Scheduler parameters
        parser.add_argument('--lr_scheduler_step', type=int, default=150)
        parser.add_argument('--lr_scheduler_decay', type=float, default=0.5)

        # Save directory parameters
        parser.add_argument('--net_params_dir', type=str, default='../fusion/checkpoint/net')
        parser.add_argument('--opt_params_dir', type=str, default='../fusion/checkpoint/opt')
        parser.add_argument('--result_dir', type=str, default='../fusion/result')
        parser.add_argument('--writer_dir', type=str, default='../fusion/blog')
    
        # test parameters
        parser.add_argument('--test_epoch_params', type=int, default=499)
        self.opt = parser.parse_args()
        self.transform = {'train': torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),torchvision.transforms.ToTensor()])
                          ,'test': torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),torchvision.transforms.ToTensor()])
                          }


    def initialize(self):
        if not os.path.exists(self.opt.net_params_dir):
            os.makedirs(self.opt.net_params_dir)
        if not os.path.exists(self.opt.opt_params_dir):
            os.makedirs(self.opt.opt_params_dir)
        if not os.path.exists(self.opt.writer_dir):
            os.makedirs(self.opt.writer_dir)
            
        if not os.path.exists(self.opt.result_dir):
            os.makedirs(self.opt.result_dir)
            

    def print_options(self):
        """Print and save options"""
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)


if __name__ == "__main__":
    Baseopt = BaseOptions()
    Baseopt.print_options()
    Baseopt.initialize()