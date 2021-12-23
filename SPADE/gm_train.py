"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

#####
# 2021-12-23
# 1. opt(base): 학습이 끝날 때 마다 원본 iter.text 다시 불러오기(사실상 필요 없음, opt.txt는 test에 필요하기 때문에 일단 보류).
# 2. 
########

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.gm_visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import torch
import time

###
# 시간 체크
start_time=time.time()


# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# # create tool for visualization
visualizer = Visualizer(opt)

base_epoch=50
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        
        # Training

        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)
        trainer.run_discriminator_one_step(data_i)
       

        if not opt.skip_losses: # Loss 추이 출력 - 학습 20~25% 느려짐
            if epoch % opt.print_freq == 0:
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, losses) 
        else: # loss 추이 출력 안 할때는 그냥 학습이 잘 되고있는지만 출력.
            if epoch % 10 ==0:
                mid_time=time.time()
                print('End of epoch %d / %d \t Time Taken(Total): %d sec' %
                      (epoch, opt.niter, mid_time-start_time))
        
        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

#     trainer.update_learning_rate(epoch)
        

    if epoch == opt.niter:
        print('Training was successfully finished.')
        print('saving the model at %d_net_G(D).pth' % (epoch))
        trainer.save(epoch)

end_time=time.time()

print('Time for %d iterations : %d sec' %(opt.niter-50, end_time-start_time))