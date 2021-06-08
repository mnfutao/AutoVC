import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from data.dataloader import VCDataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from data import SpeakerDataset
from modules.models import AutoVC
from tqdm import tqdm
import pickle

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def main(
    config_path: Path,
    data_dir: Path,
    save_dir: Path,
    warm_up_path: Path,
    opt_warm_up_path: Path,
    embeding_path: Path,
    n_steps: int,
    save_steps: int,
    log_steps: int,
    batch_size: int,
    seg_len: int,
):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir.mkdir(exist_ok=True)
    config = yaml.load(config_path.open(mode="r"), Loader=yaml.FullLoader)
    writer = SummaryWriter(save_dir)
    
    assert embeding_path.exists()
    
    if warm_up_path.exists():
        print('【warm up model from saved model !!!】 ')
        model = torch.jit.load(warm_up_path).to(device)
    else:
        model = AutoVC(config)

    model = torch.jit.script(model).to(device)
    #train_set = SpeakerDataset(['mels', 'embed'], data_dir, seg_len=seg_len)
    train_set = SpeakerDataset(['mels', 'embed'], data_dir, embeding_path, seg_len=seg_len)

    data_loader = VCDataLoader(train_set, batch_size=batch_size, mode='train')

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    if opt_warm_up_path.exists():
        print('【warm up optimizer from saved optimizer !!!】 ')
        optimizer.load_state_dict(torch.load(opt_warm_up_path))

    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    lambda_cnt = 1.0

   
    
    global_step = 0
    global_epoch = 0
    step_epoch_path = save_dir / f'step_epoch.pkl'
    if step_epoch_path.exists():
        print('【warm up global step && epoch from saved file !!!】 ')
        
        with open(step_epoch_path, 'rb') as fr:
            global_value = pickle.load(fr)
            global_step = global_value['global_step']
            global_epoch = global_value['global_epoch']

    print()
    print(f'【current_global_step:{global_step}】')
    print(f'【current_global_epoch:{global_epoch}】')
    print()

    for step in range(global_step, n_steps):
        pbar = tqdm(data_loader, unit="mels", unit_scale=data_loader.batch_size, disable=False)
        for batch in pbar:  

            mels, embs = batch['mels'], batch['embed']

            mels = mels.to(device)
            embs = embs.to(device)
            rec_org, rec_pst, codes = model(mels, embs)

            fb_codes = torch.cat(model.content_encoder(rec_pst, embs), dim=-1)

            # reconstruction loss
            org_loss = MSELoss(rec_org, mels)
            pst_loss = MSELoss(rec_pst, mels)
            # content consistency
            cnt_loss = L1Loss(fb_codes, codes)

            loss = org_loss + pst_loss + lambda_cnt * cnt_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (global_step + 1) % save_steps == 0:
                print('model_save!')
                model.save(str(save_dir / f"model-{global_step + 1}.pt"))
                model.save(str(save_dir / f"model.pt"))
                torch.save(optimizer.state_dict(), str(save_dir / f"optimizer.pt"))
                global_value = {}
                global_value['global_step'] = global_step + 1
                global_value['global_epoch'] = global_epoch
                with open(step_epoch_path, 'wb') as fw:
                    pickle.dump(global_value, fw)
                

            if (global_step + 1) % log_steps == 0:
                writer.add_scalar("loss/org_rec", org_loss.item(), global_step + 1)
                writer.add_scalar("loss/pst_rec", pst_loss.item(), global_step + 1)
                writer.add_scalar("loss/content", cnt_loss.item(), global_step + 1)
            pbar.set_postfix(
                {
                    "org_rec": org_loss.item(),
                    "pst_rec": pst_loss.item(),
                    "cnt": cnt_loss.item(),
                }
            )
            global_step += 1
        global_epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default = 'config.yaml', type=Path)
    parser.add_argument("--data_dir", default='./data/other_speech/', type=Path)
    parser.add_argument("--save_dir", default='./logdir', type=Path)
    parser.add_argument("--warm_up_path", default='./logdir/model.pt', type=Path)
    parser.add_argument("--opt_warm_up_path", default='./logdir/optimizer.pt', type=Path)
    parser.add_argument("--embeding_path", default='./data/new_char_emb.npy', type=Path)
    parser.add_argument("--n_steps", type=int, default=int(2e7))
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--log_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seg_len", type=int, default=128)
    main(**vars(parser.parse_args()))
