# from vis_mgp import get_plot_file
import torch
from cycle_gan_new import CycleGANModel
from train_options import TrainOptions, get_skeleton_aware_options
from dataloader_new import *
import wandb
import util
from tqdm import tqdm
import time
import os

import matplotlib.pyplot as plt
import wandb as WWandb
import dataclasses
#tensorboard
from torch.utils.tensorboard import SummaryWriter
from data_util import remove_t_dim

opt = TrainOptions().parse()   # get training options
torch.cuda.set_device(opt.gpu_ids[0])
torch.set_num_threads(opt.num_threads)
# prior_shape,dino_pred, model_mgp = get_plot_file(opt.gpu_ids[0])
prior_shape,dino_pred, model_mgp= None,None,None
from cycle_gan_new import *
util.seed_everything(24)

datasetA = AinoZoo(
    dataset_path = Path("/home/nguyen/projects/pose2motion_submission/data/aino_zoo"),
    class_names=["adult_dog_new"]
)
datasetB = AinoZoo(
    dataset_path = Path("/home/nguyen/projects/pose2motion_submission/data/aino_zoo"),
    class_names=["adult_dog_new"]
)

dataset_testA = AinoZoo(
    dataset_path = Path("/home/nguyen/projects/pose2motion_submission/data/aino_zoo"),
    class_names=["adult_dog_new"]
)
dataset_testB = AinoZoo(
    dataset_path = Path("/home/nguyen/projects/pose2motion_submission/data/aino_zoo"),
    class_names=["adult_dog_new"]
)

opt.num_joints_A = datasetA.num_joints
opt.num_joints_B = datasetB.num_joints

if opt.skeleton_aware:
    args = get_skeleton_aware_options()
    args.topology_a = datasetA.joint_parents_idx
    args.topology_b = datasetB.joint_parents_idx
    args.fix_virtual_bones = opt.fix_virtual_bones # False
    args.rotation_data = opt.rotation_data # rotation_6d
    args.rotation = opt.rotation # rotation_6d
    args.num_layers = opt.skeleton_num_layers
    args.kernel_size = opt.skeleton_kernel_size
    args.window_size = opt.window_size # 64
    args.with_root = opt.with_root # True
    args.connect_end_site = opt.connect_end_site # False
    args.velocity_virtual_node = opt.velocity_virtual_node # True
    args.acGAN = opt.acGAN # False
    args.v_connect_all_joints = opt.v_connect_all_joints # False
    args.last_sigmoid = opt.last_sigmoid # False
    args.scale_factor = opt.scale_factor # 1.0
    args.skeleton_dist = opt.skeleton_dist # 0.0
    args.end_sites_a = None
    args.end_sites_b = None
    opt.skeleton_aware_args = args

    

#setup tensorboard
# writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir,opt.name))
writer = None
opt.lr_G_A = opt.lr
opt.lr_G_B = opt.lr
if opt.load_pretrained:
    opt.lr_G_A = 0
model: CycleGANModel  = CycleGANModel(opt,datasetA,datasetB).cuda()
if opt.load_pretrained:
    print("load pretrain model")
    model_dict = torch.load(os.path.join(opt.pretrained_path,"latest_netG.pth"))
    model.netG_A.encoder.load_state_dict(model_dict['encoder'],strict=True)
    model.netG_B.decoder.load_state_dict(model_dict['decoder'],strict=True)

dataloadersA = torch.utils.data.DataLoader(datasetA, batch_size=opt.batchSize, shuffle=True if opt.overfit==0 else False, num_workers=1)
dataloadersB = torch.utils.data.DataLoader(datasetB, batch_size=opt.batchSize, shuffle=True if opt.overfit==0 else False, num_workers=1)

def cycle_dataloader(dataloaderA,dataloaderB):
    while True:
        for dataA, idxA in dataloaderA:
            for dataB, idxB in dataloaderB:
                yield {
                    "A": dataA,
                    "B": dataB,
                    "infoA": dataclasses.asdict(datasetA.infos[0]),
                    "infoB": dataclasses.asdict(datasetB.infos[0]),
                    "A_paths": idxA,
                    "B_paths": idxB
                }

dataloaders = cycle_dataloader(dataloadersA,dataloadersB)

wandb = wandb.init(
    project=opt.wandb_project_name,
    name="animal"+opt.checkpoints_dir, 
    config=opt,
    settings=wandb.Settings(code_dir=".")
)

wandb.log_code(".")
wandb.watch(model)
total_iters = 0
torch.cuda.set_device(opt.gpu_ids[0])
opt.path = os.path.join(opt.checkpoints_dir,opt.name)
dict_var = vars(opt)
np.save(os.path.join(opt.checkpoints_dir,opt.name,"opt.npy"),dict_var)
dict_args = vars(args)
np.save(os.path.join(opt.checkpoints_dir,opt.name,"args.npy"),dict_var)
# copy code "model_modules.py" to checkpoints folder
import shutil
shutil.copyfile("model_modules.py", os.path.join(opt.checkpoints_dir,opt.name,"model_modules.py"))
# dataloader 
shutil.copyfile("dataloader.py", os.path.join(opt.checkpoints_dir,opt.name,"dataloader.py"))

# model.save_networks('latest')
for epoch in range(opt.n_epochs):
    # model.update_learning_rate()  
    epoch_start_time = time.time()  # timer for entire epoch
    for i, data in enumerate(tqdm(dataloaders)):
        if opt.load_pretrained:
            model.netG_A.encoder.eval()
            model.netG_B.decoder.eval()
      
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        if opt.gen_freq+opt.disc_freq>0:
            if total_iters%(opt.gen_freq+opt.disc_freq) < opt.disc_freq:
                model.optimize_parameters(wandb,opt_gen=False,opt_disc=True,writer=writer)     # calculate loss functions, get gradients, update network weights
            elif total_iters%(opt.gen_freq+opt.disc_freq) >= opt.disc_freq:
                model.optimize_parameters(wandb,opt_gen=True,opt_disc=False,writer=writer)     # calculate loss functions, get gradients, update network weights
        else:
            model.optimize_parameters(wandb,opt_gen=True,opt_disc=True,writer=writer)

        total_iters+=1
        
        # if total_iters % opt.save_bvh == 0 or total_iters==1:
        #     model.save_networks('latest')
    
        #     with torch.no_grad():
        #         model.eval()
        #         if opt.window_size>0:
        #             num_windows = dataset_testB.poses.shape[0]//opt.window_size
        #             skeleton_B = dataset_testB.poses[:num_windows*opt.window_size].clone()
        #             skeleton_B = skeleton_B.reshape([1,num_windows*opt.window_size,dataset_testB.poses.shape[1],dataset_testB.poses.shape[2]]).permute(0,2,3,1)
        #         # else:
        #         #     skeleton_B = dataset_testB.pose.clone()
        #         skeleton_A = model.netG_B(skeleton_B.cuda().clone(),input_type="matrot").detach()
        #         skeleton_B = dataset_testB.poses.clone()
        #         if opt.dataset.startswith("biped"):
        #             skeleton_A_gt = dataset_testA.poses.clone()
        #         # import pdb;pdb.set_trace()
        #         if opt.window_size>0:
        #             skeleton_A = skeleton_A.permute(0,3,1,2).reshape([-1,dataset_testA.poses.shape[1],dataset_testA.poses.shape[2]])
        #         # if opt.normalize_all:
        #         #     skeleton_A = dataset_testA.denormalize(skeleton_A,"A")
        #         #     skeleton_B = dataset_testB.denormalize(skeleton_B.cuda(),"B")
        #         #     if opt.dataset.startswith("biped"):
        #         #         skeleton_A_gt = dataset_test.denormalize(skeleton_A_gt.cuda(),"A")

        #         # if opt.remove_virtual_node:
        #         #     skeleton_A = to_full_joint(skeleton_A,dataset_test.infoA['remain_index'],dataset_test.infoA['pose_virtual_index'][0],dataset_test.infoA['pose_virtual_val'])
        #         #     skeleton_B = to_full_joint(skeleton_B,dataset_test.infoB['remain_index'],dataset_test.infoB['pose_virtual_index'][0],dataset_test.infoB['pose_virtual_val'])
                
        #         #     if opt.dataset.startswith("biped"):
        #         #         skeleton_A_gt = to_full_joint(skeleton_A_gt,dataset_test.infoA['remain_index'],dataset_test.infoA['pose_virtual_index'][0],dataset_test.infoA['pose_virtual_val'])
        #         if opt.velocity_virtual_node:
        #             root_A = skeleton_A[:,-1,:3].clone()
        #             root_B = skeleton_B[:,-1,:3].clone()
        #             skeleton_A[:,-1,:] = torch.eye(3).cuda().reshape(-1)[:6]
        #             skeleton_B[:,-1,:] = torch.eye(3).cuda().reshape(-1)[:6]
                   
        #             skeleton_A_rotation = to_matrix(skeleton_A)
        #             skeleton_B_rotation = to_matrix(skeleton_B)
                   
        #             if opt.dataset.startswith("biped"):
        #                 root_A_gt = skeleton_A_gt[:,-1,:3].clone()
        #                 skeleton_A_gt[:,-1,:] = torch.eye(3).cuda().reshape(-1)[:6]
        #                 skeleton_A_gt_rotation = to_matrix(skeleton_A_gt)

        #             if opt.use_global_y:
        #                 root_A[:,[0,2]] = root_A[:,[0,2]].cumsum(dim=0)
        #                 root_B[:,[0,2]] = root_B[:,[0,2]].cumsum(dim=0)   
        #                 root_A = root_A.detach().cpu().numpy()
        #                 root_B = root_B.detach().cpu().numpy()

        #                 if opt.dataset.startswith("biped"):
        #                     root_A_gt[:,[0,2]] = root_A_gt[:,[0,2]].cumsum(dim=0)
        #                     root_A_gt = root_A_gt.detach().cpu().numpy()
        #             else:
        #                 root_A = root_A.cumsum(dim=0).detach().cpu().numpy()
        #                 root_B = root_B.cumsum(dim=0).detach().cpu().numpy()
        #                 if opt.dataset.startswith("biped"):
        #                     root_A_gt = root_A_gt.cumsum(dim=0).detach().cpu().numpy()
        #         else:
        #             skeleton_A_rotation = to_matrix(skeleton_A)
        #             skeleton_B_rotation = to_matrix(skeleton_B)
        #             root_A = None
        #             root_B = None

        #             if opt.dataset.startswith("biped"):
        #                 skeleton_A_gt_rotation = to_matrix(skeleton_A_gt)
        #                 root_A_gt = None

        #         #create path if not exist:
        #         path = os.path.join(opt.checkpoints_dir,opt.name)
        #         bvh_path = "bvh"

        #         if not os.path.exists(os.path.join(path,bvh_path)):
        #             os.makedirs(os.path.join(path,bvh_path))
                
        #         if opt.dataset.startswith("biped"):
        #             util.save_bvh(skeleton_A_gt_rotation[:].detach().cpu().numpy(),dataset_test.infoA,"{}/{}/B2A_gt_A.bvh".format(path,bvh_path),root_A_gt,normalize=True)
        #         util.save_bvh(skeleton_A_rotation[:].detach().cpu().numpy(),dataset_test.infoA,"{}/{}/B2A_2_{}.bvh".format(path,bvh_path,total_iters),root_A,normalize=True)
        #         util.save_bvh(skeleton_B_rotation[:].detach().cpu().numpy(),dataset_test.infoB,"{}/{}/B2A_gt_B.bvh".format(path,bvh_path),root_B,normalize=True)
        #         model.train()

    if epoch % opt.save_epoch_freq == 0 or epoch==0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks(str(epoch))

    model.save_networks('latest')
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
