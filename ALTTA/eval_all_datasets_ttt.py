import random
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
from torch.utils.data import DataLoader

from model.eval_network import STCN
from model.losses import EntropyLoss

from ttt.config.load_config import load_config
from ttt.model.model_ttt import STCN_TTT
from ttt.utils.meter import AverageMeterDict
from ttt.utils.helper import *
from ttt.dataset.vos_dataset_ttt import VOSDataset
from dataset.SurgicalVideo import SurgicalVideo
from dataset.SurgicalVideo import SurgicalTsVideo
from dataset.SurgicalVideo import SurgicalTestDataset

os.environ['CUDA_VISIBLE_DEVICES'] = "5"

def test_time_train_and_evaluate_one_video(args, video_data, pretrained_model,ema_model=None):
    """
    For a given video, runs the test time training process which updates the weights of the pre-trained model.
    Then evaluates the updated model on the given video.
    """
    video_name = video_data['info']['name'][0]

    # Fix the seed for this video
    seed = args.seed
    output_seed_dir = os.path.join(args.output_dir, str(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize the training and inference models
    test_time_training_model = STCN_TTT().cuda().eval()
    test_time_inference_model = STCN().cuda().eval()

    weights = pretrained_model.state_dict()

    print('\nvideo:', video_name, 'shape', video_data['rgb'][0].shape, 'objects:', len(video_data['info']['labels'][0]))
    if args.ttt_number_iterations_per_jump_step > 0:
        test_time_training_model.copy_weights_from(weights)
        test_time_training_model.freeze_parse(args.ttt_frozen_layers)

        result_dir = os.path.join(output_seed_dir, 'final', video_name)
        if not os.path.exists(result_dir) or not len(os.listdir(result_dir)) or args.overwrite:

            # Run test time training
            logs, ttt_models,ema_model = test_time_train_one_video(
                test_time_training_model, video_name, video_data, output_seed_dir, args,ema_model=ema_model)

            #test_time_training_model.copy_weights_from(ttt_models,False)

            # Run test time inference
            test_time_evaluate_one_video(ttt_models, output_seed_dir, test_time_inference_model, video_data)

            # Save in logs
            if logs is not None:
                log_dir = os.path.join(output_seed_dir, 'logs')
                os.makedirs(log_dir, exist_ok=True)
                dump_logs(os.path.join(log_dir, video_name + '.txt'), logs)

    return ttt_models

def test_time_train_one_video(model, video_name, vid_reader, result_dir, args,ema_model=None):
    """ For a given video, runs the test time training process which updates the weights of the pre-trained model. """
    ce_criterion = nn.CrossEntropyLoss()
    ent_criterion = EntropyLoss(dim=1)

    val_model = STCN().cuda().eval()
    model.copy_weights_to(val_model)
    if ema_model==None:
        ema_model=model
    video_inference(vid_reader, val_model, os.path.join(result_dir, 'temp'), args)

    # Parameters for VOSDataset
    frames_with_gt = sorted(list(vid_reader['info']['gt_obj'].keys())) if args.dataset_name == "youtube" else [0]
    max_obj = 6
    all_objects = len(frames_with_gt) == 1
    frame_dir, all_frames_dir = get_frame_dirs(args)

    iteration, ttt_models, logs = 0, dict(), []
    for e in range(args.ttt_number_jump_steps):
    #for e in range(2):
    #for e in range(5):
        # print("#########################################")
        print(f"current test training epoch is {e},{video_name} begins test time training")
        # print("#########################################")
        for max_jump, num_frames in zip(args.ttt_max_jump_step, args.ttt_sequence_length):

            # Evaluate the current model and save the results in the temp folder
            if args.ttt_loss == "tt-mcc":
                model.copy_weights_to(val_model)
                video_inference(vid_reader, val_model, os.path.join(result_dir, 'temp'), args)

            #dataset=SurgicalTsVideo("/data4/jjj/video_data/ts_setv2",output_size=(272,480),clip_n=7,max_obj_n=2)
            

            dataset = SurgicalTsVideo(frame_dir,
                                 #os.path.join(result_dir, 'temp'),
                                os.path.join("/data5/jjj/VOS/ttt-matching_vos-Surgical/GTS/1234/temp"),
                                 video_name,
                                 max_jump,
                                 num_frames,
                                 total_sequences=args.ttt_number_iterations_per_jump_step * args.ttt_batch_size,
                                 resolution=args.ttt_resolution,
                                 # scale=args.ttt_scale,
                                 # ratio=args.ttt_ratio,
                                 augmentations=args.ttt_augmentations,
                                 check_last=args.ttt_loss == "tt-mcc",
                                 all_objects=all_objects,
                                 max_obj=max_obj,
                                 frames_with_gt=frames_with_gt,
                                 im_root_all_frames=all_frames_dir,
                                 ema_model=ema_model,
                                 device=torch.device("cuda"),
                                 total_needed=args.ttt_number_iterations_per_jump_step * args.ttt_batch_size,
                                 cur_epoch=e-1)


            ### old version
            # train_loader = DataLoader(dataset, args.ttt_batch_size, num_workers=16, pin_memory=True,
            #                           worker_init_fn=worker_init_fn)

            train_loader = DataLoader(dataset, args.ttt_batch_size, num_workers=0, pin_memory=True,
                                      worker_init_fn=worker_init_fn)

            optimizer = torch.optim.Adam(filter(
                lambda p: p.requires_grad, model.parameters()), lr=args.ttt_lr, weight_decay=1e-7)
            scaler = torch.cuda.amp.GradScaler()

            meters = AverageMeterDict()
            dataset.set_epoch(e)
            dataset.curriculumnbuilder.id=-1
            for data in train_loader:
                optimizer.zero_grad()

                for k, v in data.items():
                    if type(v) != list and type(v) != dict and type(v) != int:
                        data[k] = v.cuda(non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.amp):

                    if args.ttt_loss == "tt-ae":
                        logits_f, masks_f = model.do_single_pass(data)
                    
                    elif args.multif:
                        logits_f, logits_b, masks_f, masks_b,uncertainties_f = model.do_cycle_pass(
                            data, backwards=args.ttt_loss == "tt-mcc", encode_first=False,ema_model=ema_model,multi_fpass=True)
                    else:
                        logits_f, logits_b, masks_f, masks_b = model.do_cycle_pass(
                            data, backwards=args.ttt_loss == "tt-mcc", encode_first=False,ema_model=ema_model,multi_fpass=True)

                    # Loss
                    if args.ttt_loss == "tt-mcc":  # Mask Cycle Consistency
                        loss = ce_criterion(logits_b[-1], data['cls_gt'][:, 0])
                    elif args.ttt_loss == "tt-ae":  # Auto Encoder
                        loss = ce_criterion(logits_f[0], data['cls_gt'][:, 0])
                    elif args.ttt_loss == "tt-ent":  # Entropy
                        loss = ent_criterion(torch.cat(logits_f, 0))
                    ##consistency loss
                    pixel_uncertainty, _ = torch.max(uncertainties_f[0], dim=1, keepdim=True)
                    # beta 是敏感度超参数
                    beta = 10.0
                    # 越不确定(uncertainty大) -> 权重越接近 0
                    reliability_weight = torch.exp(-beta * pixel_uncertainty)
                    
                    #print(f"{reliability_weight.mean().item():.10f}")

                    cons_loss=ce_criterion(logits_b[1],logits_f[0])
                    #cons_loss=(cons_loss*reliability_weight).mean()

                meters.update('loss', loss)

                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    (loss+0.01*cons_loss).backward() ### 默认是0.01cons
                    #loss.backward()
                    #optimizer.param_groups[0]['lr']=(1.0-reliability_weight.mean().item())*(12e-5) #adjust learning rate
                    
                    #optimizer.param_groups[0]['lr']=(reliability_weight.mean().item())*(8e-7) #1e-6
                    optimizer.param_groups[0]['lr']=(reliability_weight.mean().item())*(1e-6) 
                    #print("lr :{}".format(optimizer.param_groups[0]['lr']))
                    optimizer.step()
                iteration += 1
                _update_ema(ema_model,model,iteration)

    ttt_models['final'] = model.state_dict()
    model.copy_weights_to(val_model)

    return logs, ttt_models,ema_model.state_dict()

def _update_ema(ema_model,student_model,iter,alpha=0.999):
    #alpha_teacher = min(1 - 1 / (iter + 1), alpha)
    alpha_teacher=0.99
    for ema_param, param in zip(ema_model.parameters(),
                                student_model.parameters()):
        if not param.data.shape:  # scalar tensor
            ema_param.data = \
                alpha_teacher * ema_param.data + \
                (1 - alpha_teacher) * param.data
        else:
            ema_param.data[:] = \
                alpha_teacher * ema_param[:].data[:] + \
                (1 - alpha_teacher) * param[:].data[:]



def frozen_ema(ema_model):
    for param in ema_model.parameters():
        param.requires_grad = False

def test_time_evaluate_one_video(ttt_models, output_seed_dir, test_time_inference_model, data):
    for k, v in ttt_models.items():
        result_dir = os.path.join(output_seed_dir, k)
        os.makedirs(result_dir, exist_ok=True)
        test_time_inference_model.load_state_dict(v)
        video_inference(data, test_time_inference_model, result_dir, args)

def get_parameters():
    args = load_config()

    if len(args.ttt_max_jump_step) > 1 and len(args.ttt_sequence_length) == 1:
        args.ttt_sequence_length = args.ttt_sequence_length * len(args.ttt_max_jump_step)
    elif len(args.ttt_sequence_length) > 1 and len(args.ttt_max_jump_step) == 1:
        args.ttt_max_jump_step = len(args.ttt_sequence_length) * args.ttt_max_jump_step
    elif len(args.ttt_max_jump_step) != len(args.ttt_sequence_length):
        raise Exception('ttt_max_jump_step and ttt_sequence_length should be of equal size or 1.')
    args.palette = get_palette(args)  # load palette
    os.makedirs(args.output_dir, exist_ok=True)  # create the output dir

    print('\nInput Arguments')
    print('---------------')
    for k, v in sorted(dict(vars(args)).items()):
        print('%s: %s' % (k, str(v)))
    print()
    return args



def load_model(prop_saved):
    stcn_model = STCN().cuda().eval()
    # Performs input mapping such that stage 0 model can be loaded
    #prop_saved = torch.load(args.model_filename)
    prop_saved=prop_saved['final']
    for k in list(prop_saved.keys()):
        if k == 'value_encoder.conv1.weight':
            if prop_saved[k].shape[1] == 4:
                pads = torch.zeros((64, 1, 7, 7), device=prop_saved[k].device)
                prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
    stcn_model.load_state_dict(prop_saved)
    return stcn_model

def print_model_grad(model):
    for param in model.parameters():
        print(param.requires_grad)

if __name__ == '__main__':
    """
    Arguments loading
    """
    args = get_parameters()

    # Setup Dataset
    test_dataset = get_test_dataset(args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load our checkpoint
    pretrained_model = get_stcn_model(args)

    #### get teacher model
    ema_model=STCN_TTT().cuda().eval()
    ema_model.copy_weights_from(pretrained_model)
    ema_model=frozen_ema(ema_model)
    
    original_ema_model = ema_model

    #print_model_grad(pretrained_model)

    for _, video_data in enumerate(test_loader):
        print("hello world!")
        ttt_pro_weights=test_time_train_and_evaluate_one_video(args, video_data, pretrained_model,ema_model)
        
        #pretrained_model=load_model(ttt_pro_weights)
        # print(ema_model is original_ema_model)
