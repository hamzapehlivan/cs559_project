"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

from metrics.psnr import PSNR
from metrics.ssim import SSIM

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
for i, data_i in enumerate(tqdm(dataloader)):
    # if i * opt.batchSize >= opt.how_many:
    #     break

    generated = model(data_i, mode='inference')
    img_path = data_i['path']
    final = data_i['image'] * data_i['valid'] + generated * (1- data_i['valid'])

    for b in range(generated.shape[0]):
        #print('process image... %s' % img_path[b])
        # visuals = OrderedDict([('input_label', data_i['label'][b]),
        #                        ('synthesized_image', generated[b])])

        visuals =  OrderedDict([('final', final[b]),
                                ('real', data_i['image'][b])])
        #visualizer.save_images(webpage, visuals, img_path[b:b + 1])
        visuals = visualizer.convert_visuals_to_numpy(visuals) 
        ssim = SSIM(visuals['final'], visuals['real'])
        psnr = PSNR(visuals['final'], visuals['real'])

        ssim_scores.append(ssim)
        psnr_scores.append(psnr)
    
print("SSIM: ", np.mean(ssim_scores))
print("PSNR: ", np.mean(psnr_scores))


        
#webpage.save()


