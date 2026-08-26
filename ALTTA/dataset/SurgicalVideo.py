

import os
import random
import numpy as np
from glob import glob

import torch
from torch.utils import data
import torchvision.transforms as TF

from dataset import transforms as mytrans
from dataset.reseed import reseed

from os import path
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataset.range_transform import im_normalization, im_mean
from PIL import Image
import copy
from util.tensor_util import pad_divide_by
from dataset.util import all_to_onehot
# import myutils
from curriculumn import *





class SurgicalVideo(data.Dataset):
    def __init__(self, root, output_size, imset='SurgicalVideo/ImageSets/train.txt',max_jump=5, clip_n=3, max_obj_n=2,is_bl=False):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n #每个frame最多有的target数量

        dataset_path = os.path.join("/data5/jjj/VOS/AFB-URR-SURGICAL", imset)
        self.dataset_list = list()
        with open(os.path.join(dataset_path), 'r') as lines:
            for line in lines:
                dataset_name = line.strip()
                if len(dataset_name) > 0:
                    self.dataset_list.append(dataset_name)

        self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.02)
        self.random_affine = mytrans.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10)
        self.random_resize_crop = mytrans.RandomResizedCrop(output_size, (0.8, 1), (0.95, 1.05))
        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=True)
        

        ##new
        self.max_jump=max_jump
        self.is_bl=is_bl


    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):

        try:
            info={}
            
            video_name = self.dataset_list[idx]
            info['name']=video_name
            img_dir = os.path.join(self.root, video_name)
            mask_dir = os.path.join(self.root, video_name)

            img_list = sorted(glob(os.path.join(img_dir, '*.npz.npy')))
            mask_list = sorted(glob(os.path.join(mask_dir, '*.npz.npy')))

            # idx_list = list(range(len(img_list)))
            # random.shuffle(idx_list)
            # idx_list = idx_list[:self.clip_n]
            

            # frames = torch.zeros((self.clip_n, 3, self.output_size[0], self.output_size[1]), dtype=torch.float)
            # masks = torch.zeros((self.clip_n, self.max_obj_n, self.output_size[0], self.output_size[1]), dtype=torch.float)
            
            trials=0
            #info['frames'] = []
            while trials<5:
                info['frames']=[]
                this_max_jump = min(len(img_list), self.max_jump)
                start_idx = np.random.randint(len(img_list)-this_max_jump+1)
                f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1
                f1_idx = min(f1_idx, len(img_list)-this_max_jump, len(img_list)-1)
                f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
                f2_idx = min(f2_idx, len(img_list)-this_max_jump//2, len(img_list)-1)
                frames_idx = [start_idx, f1_idx, f2_idx]
                if np.random.rand() < 0.5:
                    # Reverse time
                    frames_idx = frames_idx[::-1]
                sequence_seed = np.random.randint(2147483647)
                images = []
                masks = []
                target_object = None
                for idx,f_idx in enumerate(frames_idx):
                    data_path=img_list[f_idx]
                    info['frames'].append(data_path) ##有问题
                    reseed(sequence_seed)

                    ##preprocess
                    data=np.load(data_path,allow_pickle=True)
                    data=data.item()
                    this_im=data['image']
                    img_roi=this_im

                    

                    reseed(sequence_seed)
                    this_gt=data['label']
                    mask_roi=np.array(this_gt,np.uint8)

                    # if idx==0:
                    #     mask_roi,obj_list=self.to_onehot(mask_roi)
                    # else:
                    #     mask_roi,_=self.to_onehot(mask_roi,obj_list)
                    
                    # if torch.any(mask_roi[0]==0).item():
                    #     break
                    image=self.to_tensor(img_roi)
                    images.append(image)
                    masks.append(mask_roi)
                
                images = torch.stack(images, 0)
                print(f"images1111{images.shape}")
                labels = np.unique(masks[0])
                # Remove background
                labels = labels[labels!=0]
                if self.is_bl:
                    # Find large enough labels
                    good_lables = []
                    for l in labels:
                        pixel_sum = (masks[0]==l).sum()
                        if pixel_sum > 10*10:
                            # OK if the object is always this small
                            # Not OK if it is actually much bigger
                            if pixel_sum > 30*30:
                                good_lables.append(l)
                            elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
                                good_lables.append(l)
                    labels = np.array(good_lables, dtype=np.uint8)
                
                if len(labels) == 0:
                    target_object = -1  # all black if no objects
                    has_second_object = False
                    trials += 1
                else:
                    target_object = np.random.choice(labels)
                    has_second_object = (len(labels) > 1)
                    if has_second_object:
                        labels = labels[labels!=target_object]
                        second_object = np.random.choice(labels)
                    break

            masks = np.stack(masks, 0)
            print(masks.shape)
            print(f"images2222{images.shape}")
            if (masks.shape!=torch.Size([3,272,480])):
                print(f"nask shape is {masks.shape}")
            tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:]
            if has_second_object:
                sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
                selector = torch.FloatTensor([1, 1])
            else:
                sec_masks = np.zeros_like(tar_masks)
                selector = torch.FloatTensor([1, 0])

            
            #print(F"tar_masks {tar_masks.shape}")
            #print(F"sec_masks {sec_masks.shape}")
            cls_gt=np.zeros((3,272,480),dtype=np.int8)
            cls_gt[tar_masks[:,0] > 0.5] = 1
            cls_gt[sec_masks[:,0] > 0.5] = 2
            #print(f"cls_gt {cls_gt.shape}")

                    # pairwise_seed = np.random.randint(2147483647)
                    # reseed(pairwise_seed)

            data = {
                'rgb': images,
                'gt': tar_masks,
                'cls_gt': cls_gt,
                'sec_gt': sec_masks,
                'selector': selector,
                'info': info,
            }

            assert data['rgb'].shape!=torch.Size([4,3,3,272,480])
            return data
        except RuntimeError as e:
              if "each element in list of batch should be of equal size" in str(e):
                print(f"An error occurred: {e}")
                print("The program will stop here for debugging purposes.")



        # for i, frame_idx in enumerate(idx_list):
        #     data_path=img_list[frame_idx]
        #     data=np.load(data_path,allow_pickle=True)
        #     data=data.item()
        #     img=data['image']
        #     mask=data['label']
            
        #     roi_cnt = 0
        #     while roi_cnt < 10:
        #         #img_roi, mask_roi = self.random_resize_crop(img, mask)

        #         ##mask_roi = np.array(mask_roi, np.uint8)
        #         img_roi=img
        #         mask_roi = np.array(mask, np.uint8)
        #         if i == 0:
        #             mask_roi, obj_list = self.to_onehot(mask_roi)
        #             obj_n = len(obj_list) + 1
        #         else:
        #             mask_roi, _ = self.to_onehot(mask_roi, obj_list)

        #         if torch.any(mask_roi[0] == 0).item():
        #             break

        #         roi_cnt += 1

        #     frames[i] = self.to_tensor(img_roi)
        #     masks[i] = mask_roi

        # info = {
        #     'name': video_name,
        #     'idx_list': idx_list
        # }
        # data={
        #     'rgb':frames,
        #     'gt':masks[:,:,obj_n],


        # }

        # return frames, masks[:, :obj_n], obj_n, info   
    


class SurgicalTsVideo(data.Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self,
                    im_root,
                    gt_root,
                    video_name,
                    max_jump,
                    num_frames,
                    total_sequences=1,
                    video_percentage=1.,
                    mem_every=None,
                    resolution=480,
                    scale=(1., 1.),
                    ratio=(1., 1.),
                    augmentations='none',
                    coverage=0.,
                    all_objects=False,
                    check_last=True,
                    max_obj=6,
                    frames_with_gt=[0],
                    im_root_all_frames=None,  # needed for yt
                    ema_model=None,
                    device=None,
                    total_needed=None,
                    cur_epoch=0
                    ):

        self.im_root = path.join(im_root, video_name)
        self.gt_root = path.join(gt_root, video_name)
        self.video = video_name

        if im_root_all_frames is None:
            self.im_root_all_frames = self.im_root
            self.frames = sorted(os.listdir(self.im_root))
            #mask = np.load(path.join(self.gt_root, self.frames[0][:-4] + '.npy')).item()['label']
            #mask = Image.fromarray(mask).convert('P')
            mask = Image.open(path.join(self.gt_root, self.frames[0][:-8] + '.png')).convert('P')
            self.all_labels = np.unique(np.array(mask))
            self.relative_position_first_frame = 0
        else:
            self.im_root_all_frames = path.join(im_root_all_frames, video_name)
            subsampled_frames = sorted(os.listdir(self.im_root))
            self.all_frames = sorted(os.listdir(self.im_root_all_frames))
            first_subsampled_frame_id = int(subsampled_frames[0].split(".")[0])
            self.frames = [(i_frame, frame) for i_frame, frame in enumerate(self.all_frames) if
                            int(frame.split(".")[0]) >= first_subsampled_frame_id]
            self.relative_position_first_frame = int(self.frames[0][0])
            self.frames = [frame for i_frame, frame in self.frames]

        self.frames_with_gt = frames_with_gt  # this is needed for the yt dataset
        self.total_sequences = total_sequences
        self.len_frames = int(len(self.frames) * video_percentage)
        self.check_last = check_last
        self.max_obj = max_obj
        self.max_jump = max_jump
        self.num_frames = num_frames
        self.mem_every = mem_every
        self.all_objects = all_objects
        self.coverage = coverage

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ]) if 'colour' in augmentations else lambda x: x

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ]) if 'geometric' in augmentations else lambda x: x

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ]) if 'geometric' in augmentations else lambda x: x

        # These transforms are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ]) if 'colour' in augmentations else lambda x: x

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip() if 'geometric' in augmentations else transforms.Lambda(lambda x: x),
            transforms.RandomResizedCrop((resolution, resolution), scale=scale, ratio=ratio,
                                            interpolation=Image.BICUBIC)
            if ratio != (1., 1.) or scale != (1., 1.) else
            transforms.Resize(resolution, interpolation=Image.BICUBIC)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip() if 'geometric' in augmentations else transforms.Lambda(lambda x: x),
            transforms.RandomResizedCrop((resolution, resolution), scale=scale, ratio=ratio,
                                            interpolation=Image.NEAREST) if ratio != (1., 1.) or scale != (1., 1.)
            else transforms.Resize(resolution, interpolation=Image.NEAREST)
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
        
        self.curriculumnbuilder=CurriculumnBuilder(im_root,video_name,ema_model,device,total_needed,cur_epoch)
        self.pairs=self.curriculumnbuilder.build_curriculum_pairs()
        self.cur_epoch=-1


    def select_first_frame_id(self):
        """ Selects the id of the first frame of the triplet. In DAVIS, DAVIS-C and Mose, it is always the first
        frame but for the YouTube dataset, it might be a later frame. Additionally, in the youtube dataset, each
        object may be annotated for the first time in a different frame (represented in self.frames_with_gt).
        """
        frame_with_gt = np.random.choice(self.frames_with_gt)
        first_frame_id = (frame_with_gt - self.relative_position_first_frame)
        filename = path.join(self.gt_root, self.frames[first_frame_id][:-8] + '.png')
        mask = Image.open(filename).convert('P')
        all_labels = np.unique(np.array(mask))
        return first_frame_id, all_labels

    def get_sequence(self, first_frame_id):
        if self.num_frames == 1:
            return [first_frame_id]

        # Don't want to bias towards beginning/end
        if isinstance(self.max_jump, tuple):
            this_max_jump = np.random.randint(*self.max_jump)
        else:
            this_max_jump = self.max_jump
        this_max_jump = min(self.len_frames - first_frame_id - 1, this_max_jump)

        frames_idx = [first_frame_id]
        for i in range(self.num_frames - 2):
            if self.mem_every is not None:
                r = np.arange(self.mem_every, self.len_frames, self.mem_every)
                r = r[np.logical_and(r > frames_idx[-1], r <= frames_idx[-1] + this_max_jump)]
                f_idx = np.random.choice(r) if len(r) else frames_idx[-1] + self.mem_every
            else:
                f_idx = frames_idx[-1] + np.random.randint(this_max_jump) + 1
            f_idx = min(f_idx, self.len_frames - this_max_jump, self.len_frames - 1)
            frames_idx.append(f_idx)

        f_idx = frames_idx[-1] + np.random.randint(this_max_jump) + 1
        f_idx = min(f_idx, self.len_frames - 1)
        frames_idx.append(f_idx)

        return frames_idx

    def set_epoch(self,cur_epoch):
        self.cur_epoch=cur_epoch

    def __getitem__(self, idx):
        info = {'name': self.video}

        trials, limit = 0, 100
        self.curriculumnbuilder.update(self.cur_epoch)
        self.pairs=self.curriculumnbuilder.build_curriculum_pairs()
        
        while trials < limit:
            while True:
                first_frame_id, all_labels = self.select_first_frame_id()
                #frames_idx = self.get_sequence(first_frame_id)
                
                frames_idx=list(self.pairs[self.curriculumnbuilder.id+1])
                frames_idx.insert(0,0)
                #print("cur_id is {},the pair is {}".format(self.curriculumnbuilder.id+1,frames_idx))
                self.curriculumnbuilder.id=self.curriculumnbuilder.id+1
                
                if len(frames_idx) == 1:  # for the tt-AE baseline, no triplet is used
                    break
                if frames_idx[-1] != frames_idx[-2]:
                    break

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None
            for f_idx in frames_idx:
                data_name= self.frames[f_idx]
                data_path= path.join(self.im_root_all_frames,data_name)
                data=np.load(data_path,allow_pickle=True).item()
                #jpg_name = self.frames[f_idx][:-8] + '.jpg'
                #png_name = self.frames[f_idx][:-8] + '.png'

                reseed(sequence_seed)
                #filename_img = path.join(self.im_root_all_frames, jpg_name)
                this_im = Image.fromarray(data['image'])
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)

                reseed(sequence_seed)
                #mask_name = path.join(self.gt_root, png_name)
                #this_gt = Image.open(mask_name).convert('P')
                this_gt= Image.fromarray(data['label']).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            labels = np.unique(masks[0]) #前景和背景的个数
            # Remove background
            labels = labels[labels != 0] #删掉背景 只剩前景

            if len(labels) == 0: ##没有前景
                target_object = -1  # all black if no objects
                has_second_object = False
            else:
                if self.all_objects:
                    target_objects = np.random.choice(labels, np.minimum(len(labels), self.max_obj), replace=False)
                    if not self.check_last or len(set(target_objects) - set(np.unique(masks[-1]))) == 0: ##检查所有物体是否出现在了最后一帧上
                        break

                    if len(all_labels) == len(np.unique(masks[0])) == len(np.unique(masks[-1])):
                        break
                    if trials > limit // 2 and len(all_labels) == len(np.unique(masks[0])):
                        break
                else:
                    target_object = np.random.choice(labels)
                    has_second_object = (len(labels) > 1)
                    if has_second_object:
                        second_object = np.random.choice(labels[labels != target_object])
                    ratio = (masks[-1] == target_object).mean() / (masks[0] == target_object).mean()
                    if self.coverage <= 0. or ratio > self.coverage:
                        break
            trials += 1

        if self.check_last and (
                self.all_objects and len(np.unique(masks[0])) != len(np.unique(masks[-1]))) or trials >= limit: ##第一帧和最后一帧obj数量是否相同
            images[-1] = copy.deepcopy(images[0])
            masks[-1] = copy.deepcopy(masks[0])
            frames_idx[-1] = first_frame_id
        info['frames_idx'] = frames_idx

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        images = pad_divide_by(images, 16)[0]
        masks = pad_divide_by(torch.from_numpy(masks), 16)[0].numpy()

        if self.all_objects: ##true
            labels = np.unique(masks[0])
            labels = labels[labels != 0]

            cls_gt = np.zeros(masks.shape, dtype=np.int64)
            for i, l in enumerate(labels):
                cls_gt[masks == l] = i + 1 ##前景提取

            obj_masks = torch.from_numpy(all_to_onehot(cls_gt, labels)).float()##只有前景
            obj_masks = obj_masks.unsqueeze(2)  # O x T x 1 x H x W

            object_count = obj_masks.shape[0] ##前景的数量
            if object_count > 1:
                other_mask = torch.sum(obj_masks, dim=0, keepdim=True) - obj_masks
                selector = torch.FloatTensor([1 for _ in range(object_count)])
                if len(target_objects) < len(labels):
                    obj_masks = obj_masks[(target_objects - 1).tolist()]
                    other_mask = other_mask[(target_objects - 1).tolist()]
                    selector = selector[(target_objects - 1).tolist()]
                    cls_gt = np.zeros(masks.shape, dtype=np.int64)
                    for i, l in enumerate(target_objects):
                        cls_gt[masks == l] = i + 1
            else:
                other_mask = torch.cat([torch.zeros_like(obj_masks), obj_masks], 0) ##全0 mask拼接上了前景
                obj_masks = torch.cat([obj_masks, torch.zeros_like(obj_masks)], 0) ##前景拼接上了全0
                selector = torch.FloatTensor([1, 0])
        else:
            tar_masks = (masks == target_object).astype(np.float32)[:, None, :, :]
            if has_second_object:
                sec_masks = (masks == second_object).astype(np.float32)[:, None, :, :]
                selector = torch.FloatTensor([1, 1])
            else:
                sec_masks = np.zeros_like(tar_masks)
                selector = torch.FloatTensor([1, 0])

            obj_masks = np.stack([tar_masks, sec_masks])
            other_mask = np.stack([sec_masks, tar_masks])
            cls_gt = np.zeros(masks.shape, dtype=np.int64)
            cls_gt[tar_masks[:, 0] > 0.5] = 1
            cls_gt[sec_masks[:, 0] > 0.5] = 2

        labels = np.unique(masks[0])
        labels = labels[labels != 0]
        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': obj_masks,
            'cls_gt': cls_gt,
            'sec_gt': other_mask,
            'selector': selector,
            'info': info,
        }
        return data

    def __len__(self):
        return self.total_sequences




class SurgicalTestDataset(data.Dataset):
    def __init__(self, root, imset='SurgicalVideo/ImageSets/test.txt', resolution=-1, single_object=True,
                 target_name=None, dataset_name="SurgicalVideo", jpeg_path="", corrupt_dir=None, video_set=None):
        self.root = root
        if resolution == 480:
            res_tag = '480p'
        else:
            res_tag = 'Full-Resolution'
        self.resolution = resolution
        if dataset_name =="SurgicalVideo":
            self.mask_dir= path.join(root)
            self.image_dir=path.join(root)
            #_imset_dir=path.join("/data5/jjj/VOS/AFB-URR-SURGICAL",imset)
            ###############################################################################################################################################################################################################
            _imset_dir="/data5/jjj/VOS/tta-matching-vos-cross/list/test1.txt" ##/data5/jjj/VOS/tta-matching-vos-cross/list/test1.txt
            #_imset_dir="/data5/jjj/VOS/CROSS-EMA-CURRICULUMV2-Dropout/list/test5.txt"
        _imset_f = _imset_dir
        self.videos=[]
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(path.join(_imset_f),"r") as lines:
            for line in lines:
                _video = line.rstrip('\n').split("/")[-1]
                if target_name is not None and target_name != _video:
                    continue
                self.videos.append(_video)
                self.num_frames[_video] = len(os.listdir(path.join(self.image_dir, _video)))
                data=np.load(path.join(self.mask_dir,_video,"frame_0.npz.npy"),allow_pickle=True)
                data=data.item()
                _mask=data['label']
                # _mask = np.array(
                #     Image.open(path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                # _mask480 = np.array(
                #     Image.open(path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask) 
        #self.videos=list(sorted(self.videos))
        ###############################################################################################################################################################################################################
        self.videos=list(self.videos)
        print(f"self.videos {self.videos}")
        self.single_object = single_object
        if resolution == -1:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=InterpolationMode.NEAREST),
            ])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['frames'] = []
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        images = []
        masks = []
        for f in range(self.num_frames[video]):
           #img_file = path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            img_file = path.join(self.image_dir,video,'frame_{}.npz.npy'.format(f))
            data=np.load(img_file,allow_pickle=True).item()
            #images.append(self.im_transform(Image.open(img_file).convert('RGB')))
            images.append(self.im_transform(data['image']))
            info['frames'].append('frame_{}.npz.npy'.format(f))
            
            # mask_file = path.join(self.mask_dir, video, '{:05d}.png'.format(f))
            mask_file = path.join(self.image_dir,video,'frame_{}.npz.npy'.format(f))
            if path.exists(mask_file):
                # masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
                masks.append(data['label'])
            else:
                # Test-set maybe?
                masks.append(np.zeros_like(masks[0]))
        
        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)
        gt = masks.copy()
        
        if self.single_object:
            labels = [1]
            masks = (masks > 0.5).astype(np.uint8)
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()
        else:
            labels = np.unique(masks[0])
            labels = labels[labels!=0]
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        if self.resolution != -1:
            masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'cls_gt': gt,
            'info': info,
        }

        return data



# class SurgicalTsVideo(data.Dataset):
#     def __init__(self, root, output_size, imset='SurgicalVideo/ImageSets/test.txt', clip_n=3, max_obj_n=2):
#         self.root = root
#         self.clip_n = clip_n
#         self.output_size = output_size
#         self.max_obj_n = max_obj_n #每个frame最多有的target数量

#         dataset_path = os.path.join("/data5/jjj/VOS/AFB-URR-SURGICAL", imset)
#         self.dataset_list = list()
#         with open(os.path.join(dataset_path), 'r') as lines:
#             for line in lines:
#                 dataset_name = line.strip()
#                 if len(dataset_name) > 0:
#                     self.dataset_list.append(dataset_name)

#         self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
#         self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.02)
#         self.random_affine = mytrans.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10)
#         self.random_resize_crop = mytrans.RandomResizedCrop(output_size, (0.8, 1), (0.95, 1.05))
#         self.to_tensor = TF.ToTensor()
#         self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=False)

#     def __len__(self):
#         return len(self.dataset_list)

#     def __getitem__(self, idx):

#         video_name = self.dataset_list[idx]
#         img_dir = os.path.join(self.root, video_name)
#         mask_dir = os.path.join(self.root, video_name)

#         img_list = sorted(glob(os.path.join(img_dir, '*.npz.npy')))
#         mask_list = sorted(glob(os.path.join(mask_dir, '*.npz.npy')))

#         idx_list = list(range(len(img_list)))
#         #random.shuffle(idx_list)
#         #idx_list = idx_list[:self.clip_n]

#         frames = torch.zeros((len(idx_list), 3, self.output_size[0], self.output_size[1]), dtype=torch.float)
#         masks = torch.zeros((len(idx_list), self.max_obj_n, self.output_size[0], self.output_size[1]), dtype=torch.float)
#         gts=torch.zeros((len(idx_list),self.output_size[0],self.output_size[1]))
#         for i, frame_idx in enumerate(idx_list):
#             data_path=img_list[frame_idx]
#             data=np.load(data_path,allow_pickle=True)
#             data=data.item()
#             img=data['image']
#             mask=data['label']

#             # img = myutils.load_image_in_PIL(img_list[frame_idx], 'RGB')
#             # mask = myutils.load_image_in_PIL(mask_list[frame_idx], 'P')

#             # if i > 0:
#             #     img = self.color_jitter(img)
#             #     img, mask = self.random_affine(img, mask)
            
#             roi_cnt = 0
#             while roi_cnt < 10:
#                 #img_roi, mask_roi = self.random_resize_crop(img, mask)

#                 ##mask_roi = np.array(mask_roi, np.uint8)
#                 img_roi=img
#                 mask_roi = np.array(mask, np.uint8)
#                 if i == 0:
#                     mask_roi, obj_list = self.to_onehot(mask_roi)
#                     obj_n = len(obj_list) + 1
#                 else:
#                     mask_roi, _ = self.to_onehot(mask_roi, obj_list)

#                 if torch.any(mask_roi[0] == 0).item():
#                     break

#                 roi_cnt += 1
#             #print(img_roi.shape)
#             frames[i] = self.to_tensor(img_roi)
#             masks[i] = mask_roi
#             gts[i]=torch.from_numpy(mask).to(torch.float)

#         info = {
#             'name': video_name,
#             'idx_list': idx_list,
#             'num_frames': len(idx_list)
#         }

#         return frames, masks[:, :obj_n], obj_n, info,gts   