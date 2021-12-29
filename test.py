"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

from numpy.lib.type_check import real

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

from metrics.psnr import PSNR
from metrics.ssim import SSIM

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

opt = TestOptions().parse()
opt.status = 'test'

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
# web_dir = os.path.join(opt.results_dir, opt.name,
#                        '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir,
#                     'Experiment = %s, Phase = %s, Epoch = %s' %
#                     (opt.name, opt.phase, opt.which_epoch))

# test
ssim_scores = []
psnr_scores = []
visualize = False
eval = True

if visualize:
    web_dir = os.path.join(opt.results_dir, opt.name,
                    '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir,
                'Experiment = %s, Phase = %s, Epoch = %s' %
                (opt.name, opt.phase, opt.which_epoch))

for i, data_i in enumerate(tqdm(dataloader)):
    # if i * opt.batchSize >= opt.how_many:
    #     break
    with torch.no_grad():
        
        
        fakes, attentions = model(data_i, mode='inference')

        finals = data_i['image'] * data_i['valid'] + fakes * (1- data_i['valid'])


        valids_raw = valid = F.interpolate(data_i['valid'], (64,64), mode='nearest')
        valids_raw = valids_raw.view(data_i['valid'].shape[0], 64*64)

        img_path = data_i['path']
        for b in range(fakes.shape[0]):
            #print('process image... %s' % img_path[b])

            if visualize and i < 10:
                indices = torch.nonzero(1-valids_raw[b]).squeeze(1)
                selected_attention = attentions[b][indices]

                #Randomly select index 

                index = np.random.randint(low=selected_attention.shape[0], dtype=int)
                real_idx = indices[index].item()
                att_dict = (selected_attention, index, real_idx)

                visuals = OrderedDict([
                                ('input_label', data_i['label'][b]),
                                ('real', data_i['image'][b]* data_i['valid'][b]),
                                ('fake', fakes[b]),
                                ('final_image', finals[b]),
                                ('attention', att_dict)
                                ])

                visualizer.save_images(webpage, visuals, img_path[b:b + 1])

            if eval:
                visuals =  OrderedDict([('final', finals[b]),
                                        ('real', data_i['image'][b])
                                        ])
                visuals = visualizer.convert_visuals_to_numpy(visuals) 
                ssim = SSIM(visuals['final'], visuals['real'])
                psnr = PSNR(visuals['final'], visuals['real'])

                ssim_scores.append(ssim)
                psnr_scores.append(psnr)
        
        if i > 10 and visualize:
            break

if eval:
    print("SSIM: ", np.mean(ssim_scores))
    print("PSNR: ", np.mean(psnr_scores))


if visualize:
    webpage.save()


