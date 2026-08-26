"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import torch
import torch.nn as nn

from model.network import STCN


import matplotlib.pyplot as plt
import os
import numpy as np

def save_debug_vis(clean_mask, mean_mask, entropy_map, frame_idx, video_name="debug", save_dir="./vis_debug"):
    """
    保存 Clean Mask, Mean Mask, 差异图 和 熵图 的对比可视化
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 转换为 Numpy (假设 shape 是 [1, 1, H, W] 或 [1, H, W])
    # 取第一个 object (channel 1) 进行可视化，通常 channel 0 是背景
    clean = clean_mask[0, 0].detach().cpu().numpy() 
    mean = mean_mask[0, 0].detach().cpu().numpy()
    entropy = entropy_map[0, 0].detach().cpu().numpy()
    
    # 计算差异 (绝对值差)
    diff = np.abs(clean - mean)
    
    plt.figure(figsize=(20, 5))
    
    # 1. Clean Pass (Target)
    plt.subplot(1, 4, 1)
    plt.title(f"Clean Pass (Sharp)\nFrame {frame_idx}")
    plt.imshow(clean, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    # 2. Mean Prediction (Blurred?)
    plt.subplot(1, 4, 2)
    plt.title(f"Mean Prediction (MC Dropout)\nFrame {frame_idx}")
    plt.imshow(mean, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    # 3. Difference (Clean - Mean)
    plt.subplot(1, 4, 3)
    plt.title(f"Difference (Jitter Area)\nHigh = Mismatch")
    plt.imshow(diff, cmap='jet') # 热力图显示差异
    plt.colorbar()
    plt.axis('off')
    
    # 4. Entropy (Uncertainty)
    plt.subplot(1, 4, 4)
    plt.title(f"Entropy (Uncertainty Weight)\nHigh = Unreliable")
    plt.imshow(entropy, cmap='jet') # 热力图显示不确定性
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{video_name}_f{frame_idx:03d}.png"))
    plt.close()

def open_ema_dropout(ema_model):
    ema_model.eval()
    for m in ema_model.decoder.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train() 
    return ema_model

def close_ema_dropout(ema_model):
    ema_model.eval()
    return ema_model

class STCN_TTT(STCN):
    def __init__(self,):
        super().__init__(False)
        #super().__init__(True)

    def propagate(self, frames, mask_first, mask_second, selector, k16, kf16_thin, kf16, kf8, kf4,
                  encode_first=False, mem_frame=None,multi_fpass=False,frame_idx=None,video_name=None):
        ref_v = torch.stack([
            self.encode_value(frames[:, 0], kf16[:, 0], mask_first[:, j, 0], mask_second[:, j, 0])
            for j in range(mask_first.shape[1])
        ], 1)
        ref_k = k16[:, :, 0].unsqueeze(2) ###取了第一帧的key

        logits, masks = [], []
        uncertainties=[]
        if encode_first:
            first_logits, first_mask = self.decode(ref_v, kf16_thin[:, 0], kf8[:, 0], kf4[:, 0], selector)
            logits.append(first_logits)
            masks.append(first_mask)
            uncertainties.append(torch.zeros_like(first_mask))

        if multi_fpass:
            # Segment frame 1 with frame 0
            l = frames.shape[1] - 1
            mem_loop = range(1, l) if mem_frame is None else [mem_frame]
            for i in mem_loop:
                
                # ---------------------------------------------------------
                # 1. Clean Pass (用于生成伪标签 Target 和 存入 Memory)
                # ---------------------------------------------------------
                self.eval() # 关闭 Dropout
                with torch.no_grad():
                    clean_logits, clean_mask = self.segment(k16[:, :, i], kf16_thin[:, i], kf8[:, i], kf4[:, i],
                                                            ref_k, ref_v, selector)

                # ---------------------------------------------------------
                # 2. Noisy Pass (用于计算不确定性权重 & 可视化对比)
                # ---------------------------------------------------------
                # 强制开启 Dropout
                ema_model_open = open_ema_dropout(self) 
                
                noisy_probs = []
                noisy_logits= []
                for q in range(5): # 采样 5 次
                    with torch.no_grad():
                        noisy_logit, noisy_mask = ema_model_open.segment(k16[:, :, i], kf16_thin[:, i], kf8[:, i], kf4[:, i],
                                                               ref_k, ref_v, selector)
                        noisy_probs.append(noisy_mask)
                        if False:
                            from PIL import Image
                            noisy_pred=(noisy_mask>0.5).float()
                            pred_sq = noisy_pred.squeeze(0).cpu().detach()

                            pred_sq = noisy_pred.squeeze(0).cpu().detach()

                            # 2. 创建一个空的 [H, W] 画布
                            # zeros 代表背景 (label 0)
                            final_mask = torch.zeros(pred_sq.shape[1], pred_sq.shape[2], dtype=torch.uint8)

                            # 3. 填充像素值
                            # 通道 0 的位置设为 1
                            final_mask[pred_sq[0] == 1] = 1 
                            # 通道 1 的位置设为 2 (如果两个物体有重叠，这里后赋值的会覆盖前面的)
                            final_mask[pred_sq[1] == 1] = 2 

                            # 4. 转为 PIL Image
                            mask_img = Image.fromarray(final_mask.numpy())

                            # === 修改重点：黑白/灰度调色板 ===
                            # 我们不再使用红绿蓝，而是使用不同亮度的“黑白灰”
                            # 格式: [R, G, B,  R, G, B, ...]
                            palette = [
                                0,   0,   0,     # index 0: 黑色 (背景)
                                255, 255, 255,   # index 1: 灰色 (物体1)
                                0, 0, 0    # index 2: 白色 (物体2)
                            ]

                            # 补齐剩下的颜色防止报错 (256个颜色 * 3)
                            palette.extend([0, 0, 0] * (256 - 3)) 

                            # 应用调色板
                            mask_img.putpalette(palette)

                            mask_img.save("./figs/video5_frame_{}_repeat_{}.png".format(frame_idx,q))
                            print("已保存为./figs/video5_frame_{}_repeat_{}.png".format(frame_idx,q))
                            
                        noisy_logits.append(noisy_logit)

                
                # 计算 Mean Prediction (仅用于对比，不用于传播)
                mean_logit=torch.mean(torch.stack(noisy_logits,dim=0),dim=0)
                prob_stack = torch.stack(noisy_probs, dim=0)
                mean_prob = torch.mean(prob_stack, dim=0)
                mean_mask=(mean_prob>0.6).float()
                if False:
                    from PIL import Image
                    pred_sq = mean_mask.squeeze(0).cpu().detach()

                    # 2. 创建一个空的 [H, W] 画布
                    # zeros 代表背景 (label 0)
                    final_mask = torch.zeros(pred_sq.shape[1], pred_sq.shape[2], dtype=torch.uint8)

                    # 3. 填充像素值
                    # 通道 0 的位置设为 1
                    final_mask[pred_sq[0] == 1] = 1 
                    # 通道 1 的位置设为 2 (如果两个物体有重叠，这里后赋值的会覆盖前面的)
                    final_mask[pred_sq[1] == 1] = 2 

                    # 4. 转为 PIL Image
                    mask_img = Image.fromarray(final_mask.numpy())

                    # === 修改重点：黑白/灰度调色板 ===
                    # 我们不再使用红绿蓝，而是使用不同亮度的“黑白灰”
                    # 格式: [R, G, B,  R, G, B, ...]
                    palette = [
                        0,   0,   0,     # index 0: 黑色 (背景)
                        255, 255, 255,   # index 1: 灰色 (物体1)
                        0, 0, 0    # index 2: 白色 (物体2)
                    ]

                    # 补齐剩下的颜色防止报错 (256个颜色 * 3)
                    palette.extend([0, 0, 0] * (256 - 3)) 

                    # 应用调色板
                    mask_img.putpalette(palette)

                    mask_img.save("./figs/video5_frame_{}_mean.png".format(frame_idx))
                    print("已保存为./figs/video5_frame_{}_mean.png".format(frame_idx))


                # 计算熵 (Uncertainty)
                #entropy = -torch.sum(mean_mask * torch.log(mean_mask + 1e-6), dim=1, keepdim=True)
                entropy = -torch.sum(mean_mask * torch.log(mean_prob + 1e-6), dim=1, keepdim=True)
                # ---------------------------------------------------------
                # 3. 可视化对比 (Debug 核心)
                # ---------------------------------------------------------
                # 每隔几帧保存一次，防止IO太慢，或者针对特定视频保存
                # 你可以把 True 改成 i % 10 == 0
                if False: 
                    print(f"Saving debug visualization for frame {frame_idx[1]}...")
                    save_debug_vis(clean_mask, mean_mask, entropy, frame_idx=frame_idx[1],video_name=video_name)

                # ---------------------------------------------------------
                # 4. Memory Update (关键：使用 Clean Mask!)
                # ---------------------------------------------------------
                #prev_mask=mean_mask
                prev_mask = clean_mask # 确保传入 memory 的是清晰的
                prev_other = torch.sum(prev_mask, dim=1, keepdim=True) - prev_mask
                
                prev_v = torch.stack([
                    self.encode_value(frames[:, i].clone(), kf16[:, i].clone(), prev_mask[:, j, None], prev_other[:, j, None])
                    for j in range(prev_mask.shape[1])
                ], 1)

                ref_v = torch.cat([ref_v, prev_v], 3)
                ref_k = torch.cat([ref_k, k16[:, :, i].unsqueeze(2)], 2)

                #logits.append(clean_logits) # 返回 Clean Logits
                logits.append(mean_logit)
                masks.append(clean_mask)    # 返回 Clean Mask
                # 如果需要返回 uncertainty 给 loss 使用:
                uncertainties.append(entropy)

                # ... (最后一帧的处理逻辑同理，可以使用 Clean Pass) ...
                self.eval()
                last_logits, last_mask = self.segment(k16[:, :, l], kf16_thin[:, l], kf8[:, l], kf4[:, l],
                                                    ref_k, ref_v, selector)
                logits.append(last_logits)
                masks.append(last_mask)

                uncertainties.append(torch.zeros_like(last_mask)) # 占位

                return logits,masks,uncertainties



                # self.eval()
                # with torch.no_grad():
                #     clean_logit,clean_mask=self.segment(k16[:, :, i], kf16_thin[:, i], kf8[:, i], kf4[:, i],
                #                                         ref_k, ref_v, selector)
                # for m in self.decoder.modules():
                #     if m.__class__.__name__.startswith('Dropout'):
                #         m.train()
                
                # noisy_probs=[]
                # for _ in range(5):
                #     with torch.no_grad():
                #         _,noisy_mask=self.segment(k16[:, :, i], kf16_thin[:, i], kf8[:, i], kf4[:, i],
                #                                         ref_k, ref_v, selector)
                #         noisy_probs.append(noisy_mask)
                # prob_stack=torch.stack(noisy_probs,dim=0)
                # mean_prob=torch.mean(prob_stack,dim=0)
                # entropy=-torch.sum(mean_prob*torch.log(mean_prob+1e-6),dim=1,keepdim=True)
                # # 3. Memory Update: 使用 CLEAN mask (关键!)
                # prev_other = torch.sum(clean_mask, dim=1, keepdim=True) - clean_mask
                
                # prev_v = torch.stack([
                #     self.encode_value(frames[:, i].clone(), kf16[:, i].clone(), clean_mask[:, j, None], prev_other[:, j, None])
                #     for j in range(clean_mask.shape[1])
                # ], 1)

                # ref_v = torch.cat([ref_v, prev_v], 3)
                # ref_k = torch.cat([ref_k, k16[:, :, i].unsqueeze(2)], 2)

                # # 4. 收集结果
                # logits.append(clean_logit)    # 返回 clean logits
                # masks.append(clean_mask)      # 返回 clean mask (target)
                # uncertainties.append(entropy) # 返回不确定性 (weight)



                # prev_others=[]
                # prev_masks=[]
                # prev_logits=[]
                # for _ in range(5):
                #     prev_logit, prev_mask = self.segment(k16[:, :, i], kf16_thin[:, i], kf8[:, i], kf4[:, i],
                #                                         ref_k, ref_v, selector) #frame1
                #     prev_other = torch.sum(prev_mask, dim=1, keepdim=True) - prev_mask
                #     prev_masks.append(prev_mask)
                #     prev_others.append(prev_other)        
                #     prev_logits.append(prev_logit)     
                
                # prev_mask=torch.stack(prev_masks,dim=0)
                # prev_other=torch.stack(prev_others,dim=0)
                # prev_mask=torch.mean(prev_mask,dim=0)
                # prev_other=torch.mean(prev_other,dim=0)
                # prev_logits=torch.stack(prev_logits,dim=0)
                # prev_logits=torch.mean(prev_logits,dim=0)
                
                # prev_v = torch.stack([
                #     self.encode_value(frames[:, i].clone(), kf16[:, i].clone(), prev_mask[:, j, None], prev_other[:, j, None])
                #     for j in range(prev_mask.shape[1])
                # ], 1) ##frame 1 pack

                # ref_v = torch.cat([ref_v, prev_v], 3) ##frame0 and frame 1 group
                # ref_k = torch.cat([ref_k, k16[:, :, i].unsqueeze(2)], 2)

                # logits.append(prev_logits)
                # masks.append(prev_mask)

            # Segment frame 2 with frame 0 and 1
            last_logits=[]
            last_masks=[]
            for _ in range(5): 
                last_logit, last_mask = self.segment(k16[:, :, l], kf16_thin[:, l], kf8[:, l], kf4[:, l],
                                                    ref_k, ref_v, selector)
                last_logits.append(last_logit)
                last_masks.append(last_mask)
            last_logits=torch.stack(last_logits,dim=0)
            last_logits=torch.mean(last_logits,dim=0)
            last_masks=torch.stack(last_masks,dim=0)
            last_masks=torch.mean(last_mask,dim=0)

                

            logits.append(last_logits)
            masks.append(last_masks)
            return logits, masks



        # Segment frame 1 with frame 0
        l = frames.shape[1] - 1
        mem_loop = range(1, l) if mem_frame is None else [mem_frame]
        for i in mem_loop:
            prev_logits, prev_mask = self.segment(k16[:, :, i], kf16_thin[:, i], kf8[:, i], kf4[:, i],
                                                  ref_k, ref_v, selector) #frame1

            if False:
                from PIL import Image
                prev_pred=(prev_mask>0.5).float()
                pred_sq = prev_pred.squeeze(0).cpu().detach()
                # 2. 创建一个空的 [H, W] 画布
                # zeros 代表背景 (label 0)
                final_mask = torch.zeros(pred_sq.shape[1], pred_sq.shape[2], dtype=torch.uint8)

                # 3. 填充像素值
                # 通道 0 的位置设为 1
                final_mask[pred_sq[0] == 1] = 1 
                # 通道 1 的位置设为 2 (如果两个物体有重叠，这里后赋值的会覆盖前面的)
                final_mask[pred_sq[1] == 1] = 2 

                # 4. 转为 PIL Image
                mask_img = Image.fromarray(final_mask.numpy())

                # === 修改重点：黑白/灰度调色板 ===
                # 我们不再使用红绿蓝，而是使用不同亮度的“黑白灰”
                # 格式: [R, G, B,  R, G, B, ...]
                palette = [
                    0,   0,   0,     # index 0: 黑色 (背景)
                    255, 255, 255,   # index 1: 灰色 (物体1)
                    0, 0, 0    # index 2: 白色 (物体2)
                ]

                # 补齐剩下的颜色防止报错 (256个颜色 * 3)
                palette.extend([0, 0, 0] * (256 - 3)) 

                # 应用调色板
                mask_img.putpalette(palette)

                mask_img.save("./figs/video5_frame_{}_back1.png".format(frame_idx))
                print("已保存为./figs/video5_frame_{}_back1.png".format(frame_idx))


            prev_other = torch.sum(prev_mask, dim=1, keepdim=True) - prev_mask
            prev_v = torch.stack([
                self.encode_value(frames[:, i].clone(), kf16[:, i].clone(), prev_mask[:, j, None], prev_other[:, j, None])
                for j in range(prev_mask.shape[1])
            ], 1) ##frame 1 pack

            ref_v = torch.cat([ref_v, prev_v], 3) ##frame0 and frame 1 group
            ref_k = torch.cat([ref_k, k16[:, :, i].unsqueeze(2)], 2)

            logits.append(prev_logits)
            masks.append(prev_mask)

        # Segment frame 2 with frame 0 and 1
        last_logits, last_mask = self.segment(k16[:, :, l], kf16_thin[:, l], kf8[:, l], kf4[:, l],
                                              ref_k, ref_v, selector)
        
        if False:
            from PIL import Image
            last_pred=(last_mask>0.5).float()
            pred_sq = last_pred.squeeze(0).cpu().detach()
            # 2. 创建一个空的 [H, W] 画布
            # zeros 代表背景 (label 0)
            final_mask = torch.zeros(pred_sq.shape[1], pred_sq.shape[2], dtype=torch.uint8)

            # 3. 填充像素值
            # 通道 0 的位置设为 1
            final_mask[pred_sq[0] == 1] = 1 
            # 通道 1 的位置设为 2 (如果两个物体有重叠，这里后赋值的会覆盖前面的)
            final_mask[pred_sq[1] == 1] = 2 

            # 4. 转为 PIL Image
            mask_img = Image.fromarray(final_mask.numpy())

            # === 修改重点：黑白/灰度调色板 ===
            # 我们不再使用红绿蓝，而是使用不同亮度的“黑白灰”
            # 格式: [R, G, B,  R, G, B, ...]
            palette = [
                0,   0,   0,     # index 0: 黑色 (背景)
                255, 255, 255,   # index 1: 灰色 (物体1)
                0, 0, 0    # index 2: 白色 (物体2)
            ]

            # 补齐剩下的颜色防止报错 (256个颜色 * 3)
            palette.extend([0, 0, 0] * (256 - 3)) 

            # 应用调色板
            mask_img.putpalette(palette)

            mask_img.save("./figs/video5_frame_{}_back2.png".format(frame_idx))
            print("已保存为./figs/video5_frame_{}_back2.png".format(frame_idx))
        
        logits.append(last_logits)
        masks.append(last_mask)

        return logits, masks

    def do_cycle_pass(self, data, backwards=True, encode_first=True,ema_model=None,multi_fpass=False):
        Fs = data['rgb']
        Ms = data['gt']
        sec_Ms = data['sec_gt']
        selector = data['selector']
        # === [新增] 获取真实帧索引 ===
        # data['info']['frames_idx'] 在 DataLoader batch_size=1 时通常是一个 list of tensors
        # 例如: [tensor([0]), tensor([5]), tensor([10])]
        # 我们需要把它转成 list of int: [0, 5, 10]
        try:
            raw_idx = data['info']['frames_idx']
            video_name=data['info']['name']
            # 处理 collate_fn 带来的格式差异
            if isinstance(raw_idx, list):
                frames_idx_list = [x.item() for x in raw_idx]
            elif isinstance(raw_idx, torch.Tensor):
                frames_idx_list = raw_idx[0].tolist()
            else:
                frames_idx_list = None
        except KeyError:
            frames_idx_list = None

        if ema_model==None:

            # key features never change, compute once
            k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(Fs)

            # forward pass
            logits_f, masks_f = self.propagate(Fs, Ms, sec_Ms, selector, k16, kf16_thin, kf16, kf8, kf4, encode_first)

            # backward pass
            logits_b, masks_b = None, None
            if backwards:
                Ms_b = masks_f[-1][:, :, None, None]
                sec_Ms_b = torch.sum(Ms_b, dim=1, keepdim=True) - Ms_b
                logits_b, masks_b = self.propagate(Fs.flip(dims=(1,)), Ms_b, sec_Ms_b, selector,
                                                k16.flip(dims=(2,)), kf16_thin.flip(dims=(1,)),
                                                kf16.flip(dims=(1,)), kf8.flip(dims=(1,)), kf4.flip(dims=(1,)),
                                                encode_first)

        elif multi_fpass:
            #k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(Fs)
            with torch.no_grad():
                k16, kf16_thin, kf16, kf8, kf4 = ema_model.encode_key(Fs)
                # multiple forward passes
                ema_model=open_ema_dropout(ema_model) ## open dropout

                logits_f, masks_f,uncertainties_f = ema_model.propagate(Fs, Ms, sec_Ms, selector, k16, kf16_thin, kf16, kf8, kf4, encode_first,multi_fpass=True,frame_idx=frames_idx_list,video_name=video_name)
                #masks_f[-1]=masks_f[-1].unsqueeze(0)
                # for m in masks_f:
                #     m=m.unsqueeze(0)

            # backward pass
            logits_b, masks_b = None, None
            if backwards:
                Ms_b = masks_f[-1][:, :, None, None]
                sec_Ms_b = torch.sum(Ms_b, dim=1, keepdim=True) - Ms_b
                logits_b, masks_b = self.propagate(Fs.flip(dims=(1,)), Ms_b, sec_Ms_b, selector,
                                                k16.flip(dims=(2,)), kf16_thin.flip(dims=(1,)),
                                                kf16.flip(dims=(1,)), kf8.flip(dims=(1,)), kf4.flip(dims=(1,)),
                                                encode_first,frame_idx=frames_idx_list,video_name=video_name)
            return logits_f,logits_b, masks_f, masks_b,uncertainties_f


        else:
            # key features never change, compute once
            #k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(Fs)
            with torch.no_grad():
                k16, kf16_thin, kf16, kf8, kf4 = ema_model.encode_key(Fs)
                # forward pass
                ema_model=close_ema_dropout(ema_model)
                logits_f, masks_f = ema_model.propagate(Fs, Ms, sec_Ms, selector, k16, kf16_thin, kf16, kf8, kf4, encode_first)

            # backward pass
            logits_b, masks_b = None, None
            if backwards:
                Ms_b = masks_f[-1][:, :, None, None]
                sec_Ms_b = torch.sum(Ms_b, dim=1, keepdim=True) - Ms_b
                logits_b, masks_b = self.propagate(Fs.flip(dims=(1,)), Ms_b, sec_Ms_b, selector,
                                                k16.flip(dims=(2,)), kf16_thin.flip(dims=(1,)),
                                                kf16.flip(dims=(1,)), kf8.flip(dims=(1,)), kf4.flip(dims=(1,)),
                                                encode_first)

        return logits_f, logits_b, masks_f, masks_b

    def do_single_pass(self, data):
        Fs = data['rgb']
        Ms = data['gt']
        sec_Ms = data['sec_gt']
        selector = data['selector']

        # key features never change, compute once
        k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(Fs[:, :1])

        ref_v = torch.stack([
            self.encode_value(Fs[:, 0], kf16[:, 0], Ms[:, j, 0], sec_Ms[:, j, 0])
            for j in range(Ms.shape[1])
        ], 1)
        logits, masks = self.decode(ref_v, kf16_thin[:, 0], kf8[:, 0], kf4[:, 0], selector)

        return [logits], [masks]

    def forward(self, data):
        return self.do_cycle_pass(data)

    def copy_weights_from(self, model,flag=True):
        if flag:
            if isinstance(model, nn.Module):
                self.load_state_dict(model.state_dict())
            else:
                self.load_state_dict(model)
        else:
            self.load_state_dict(model['final'])

    def copy_weights_to(self, model):
        model.load_state_dict(self.state_dict())

    def freeze_encoders(self):
        self.freeze_key_encoder()
        self.freeze_value_encoder()

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_all_keys(self):
        self.freeze_key_encoder()
        self.freeze_key_proj()
        self.freeze_key_comp()

    def freeze_key_encoder(self):
        for param in self.key_encoder.parameters():
            param.requires_grad = False

    def freeze_value_encoder(self):
        for param in self.value_encoder.parameters():
            param.requires_grad = False

    def freeze_key_proj(self):
        for param in self.key_proj.parameters():
            param.requires_grad = False

    def freeze_key_comp(self):
        for param in self.key_comp.parameters():
            param.requires_grad = False

    def freeze_network(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_batch_norms(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def unfreeze_batch_norms(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(True)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(True)
                module.eval()

    def freeze_parse(self, freeze_str):
        if freeze_str is not None:
            for fm in freeze_str.lower().split(','):
                if 'enc' == fm:
                    self.freeze_encoders()
                if 'dec' == fm:
                    self.freeze_decoder()
                if 'all_keys' == fm:
                    self.freeze_all_keys()
                if 'key_enc' == fm:
                    self.freeze_key_encoder()
                if 'val_enc' == fm:
                    self.freeze_value_encoder()
                if 'key_proj' == fm:
                    self.freeze_key_proj()
                if 'key_comp' == fm:
                    self.freeze_key_comp()
                if 'net' == fm:
                    self.freeze_network()
                if 'bn' == fm:
                    self.freeze_batch_norms()
                if 'ubn' == fm:
                    self.unfreeze_batch_norms()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name)
