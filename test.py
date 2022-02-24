import os
import time
import yaml
import argparse
import torch
import os.path as op
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import utils  # my tool box
import dataset
from model import TSAN
# from new_model import SwinTSAN
from skimage.measure import compare_psnr, compare_ssim

def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='tsan_config.yml', 
        help='Path to option YAML file.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log_test.log"
        )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
        )
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )
    opts_dict['test']['checkpoint_save_path'] = (
        f"{opts_dict['train']['checkpoint_save_path_pre']}"
        f"{opts_dict['test']['restore_iter']}"
        '.pt'
        )

    return opts_dict


def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    unit = opts_dict['test']['criterion']['unit']

    # ==========
    # open logger
    # ==========

    log_fp = open(opts_dict['train']['log_path'], 'w')
    msg = (
        f"{'<' * 10} Test {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"\n{'<' * 10} Options {'>' * 10}\n"
        f"{utils.dict2str(opts_dict['test'])}"
        )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    # ========== 
    # Ensure reproducibility or Speed up
    # ==========

    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up

    # ==========
    # create test data prefetchers
    # ==========
    
    # create datasets
    test_ds_type = opts_dict['dataset']['test']['type']
    radius = opts_dict['network']['radius']
    assert test_ds_type in dataset.__all__, \
        "Not implemented!"
    test_ds_cls = getattr(dataset, test_ds_type)
    test_ds = test_ds_cls(
        opts_dict=opts_dict['dataset']['test'], 
        radius=radius
        )

    test_num = len(test_ds)
    test_vid_num = test_ds.get_vid_num()

    # create datasamplers
    test_sampler = None  # no need to sample test data

    # create dataloaders
    test_loader = utils.create_dataloader(
        dataset=test_ds, 
        opts_dict=opts_dict, 
        sampler=test_sampler, 
        phase='val'

        )
    assert test_loader is not None

    # create dataloader prefetchers
    test_prefetcher = utils.CPUPrefetcher(test_loader)

    # ==========
    # create & load model
    # ==========

    model = TSAN(opts_dict=opts_dict['network'])

    checkpoint_save_path =  opts_dict['train']['checkpoint_save_path']+'/ckp_'+str(opts_dict['test']['restore_iter'])+'.pt'
        
    msg = f'loading model {checkpoint_save_path}...'
    print(msg)
    log_fp.write(msg + '\n')

    checkpoint = torch.load(checkpoint_save_path)

    new_state_dict = OrderedDict()
    
    for k, v in checkpoint['state_dict'].items():
        name = k  
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    msg = f'> model {checkpoint_save_path} loaded.'
    print(msg)
    log_fp.write(msg + '\n')

    model = model.cuda()
    model.eval()

    # ==========
    # define criterion
    # ==========

    # define criterion
    assert opts_dict['test']['criterion'].pop('type') == \
        'PSNR', "Not implemented."
    criterion = utils.PSNR()

    # ==========
    # validation
    # ==========
                
    # create counters
    per_aver_dict = dict()
    ori_aver_dict = dict()
    ssim_ori_aver_dict = dict()
    ssim_rec_aver_dict = dict()
    name_vid_dict = dict()
    for index_vid in range(test_vid_num):
        per_aver_dict[index_vid] = utils.Counter()
        ori_aver_dict[index_vid] = utils.Counter()
        ssim_ori_aver_dict[index_vid] = utils.Counter()
        ssim_rec_aver_dict[index_vid] = utils.Counter()
        name_vid_dict[index_vid] = ""

    pbar = tqdm(
        total=test_num, 
        ncols=opts_dict['test']['pbar_len']
        )

    # fetch the first batch
    test_prefetcher.reset()
    val_data = test_prefetcher.next()

    with torch.no_grad():
        while val_data is not None:
            # get data
            gt_data = val_data['gt'].cuda()  # (B [RGB] H W)
            lq_data = val_data['lq'].cuda()  # (B T [RGB] H W)
            index_vid = val_data['index_vid'].item()
            name_vid = val_data['name_vid'][0]  # bs must be 1!
            
            b, _, c, height, weight  = lq_data.shape
            assert b == 1, "Not supported!"
            input_data = lq_data

            y_out = input_data.new_zeros(1, 1, height , weight)
            y_out,_ = model(input_data)
            enhanced_data = y_out
            # eval

            psnr_ori = compare_psnr(np.float32(lq_data[0, radius, ...].cpu()), np.float32(gt_data[0].cpu()), data_range=1.0)
            psnr_rec = compare_psnr(np.float32(enhanced_data[0].cpu()), np.float32(gt_data[0].cpu()), data_range=1.0)

            ssim_ori = compare_ssim(np.reshape(np.float32(lq_data[0, radius, ...].cpu()),[height, weight]), np.reshape(np.float32(gt_data[0].cpu()),[height, weight]),data_range=1.0)
            ssim_rec = compare_ssim(np.reshape(np.float32(enhanced_data[0].cpu()),[height, weight]), np.reshape(np.float32(gt_data[0].cpu()),[height, weight]),data_range=1.0)

            pbar.set_description(
                "{:s}: [{:.3f}]->[{:.3f}] [{:.3f}]"
                .format(name_vid, psnr_ori,  psnr_rec,  psnr_rec-psnr_ori)
                )
            pbar.update()

            # log
            per_aver_dict[index_vid].accum(volume=psnr_rec)
            ori_aver_dict[index_vid].accum(volume=psnr_ori)
            ssim_ori_aver_dict[index_vid].accum(volume=ssim_ori)
            ssim_rec_aver_dict[index_vid].accum(volume=ssim_rec)
            if name_vid_dict[index_vid] == "":
                name_vid_dict[index_vid] = name_vid
            else:
                assert name_vid_dict[index_vid] == name_vid, "Something wrong."

            # fetch next batch
            val_data = test_prefetcher.next()
        
    # end of val
    pbar.close()

    # log
    msg = '\n' + '<' * 10 + ' Results ' + '>' * 10
    print(msg)
    log_fp.write(msg + '\n')
    for index_vid in range(test_vid_num):
        per = per_aver_dict[index_vid].get_ave()
        ori = ori_aver_dict[index_vid].get_ave()
        ssim_ori = ssim_ori_aver_dict[index_vid].get_ave()
        ssim_rec = ssim_rec_aver_dict[index_vid].get_ave()
        name_vid = name_vid_dict[index_vid]
        msg = "{:s}: [{:.3f}] -> [{:.3f}] [{:.3f}]".format(
            name_vid, ori, per, per-ori
            )
        print(msg)
        log_fp.write(msg + '\n')
    ave_per = np.mean([
        per_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    ave_ori = np.mean([
        ori_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    ssim_ave_ori = np.mean([
        ssim_ori_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    ssim_ave_rec = np.mean([
        ssim_rec_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    msg = (
        f"{'> ori: [{:.3f}] {:s}'.format(ave_ori, unit)}\n"
        f"{'> ave: [{:.3f}] {:s}'.format(ave_per, unit)}\n"
        f"{'> delta: [{:.3f}] {:s}'.format(ave_per - ave_ori, unit)}"
        f"{'> ssim_ori: [{:.3f}]'.format(ssim_ave_ori)}\n"
        f"{'> ssim_ave: [{:.3f}]'.format(ssim_ave_rec)}\n"
        f"{'> ssim_delta: [{:.3f}]'.format(ssim_ave_rec - ssim_ave_ori)}"
        )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    # ==========
    # final log & close logger
    # ==========
    
    msg = (
        f"\n{'<' * 10} Goodbye {'>' * 10}\n"
        )
    print(msg)
    log_fp.write(msg + '\n')
    
    log_fp.close()


if __name__ == '__main__':
    main()
    