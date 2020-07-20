import os
import json
import pickle
import datetime
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter

from models import LogisticPARAFAC2, PULoss, SmoothnessConstraint
from utils import EarlyStopping, AverageMeter, PaddedDenseTensor


def validate(model, dataloader):
    with torch.no_grad():
        targets = []
        predictions = []
        for pids, Xdense, masks, _ in dataloader:
            pids, Xdense, masks = pids.cuda(), Xdense.cuda(), masks.cuda()
            output = model(pids)
            output = output[masks==1]
            target = Xdense[masks==1]
            targets.append(target)
            predictions.append(output)
        targets = torch.cat(targets, dim=0)
        predictions = torch.cat(predictions, dim=0)
        ap = average_precision_score(targets.cpu().numpy(), 
                                     predictions.cpu().numpy())
    return ap


def train_logistic_parafac2(indata, 
                            num_visits, 
                            num_feats, 
                            log_path, 
                            pos_prior, 
                            reg_weight, 
                            smooth_weight,
                            rank, 
                            weight_decay, 
                            alpha, 
                            gamma, 
                            lr, 
                            seed, 
                            batch_size, 
                            smooth_shape,
                            iters, 
                            patience, 
                            num_workers=5):

    if seed is not None:
        torch.manual_seed(seed)

    model = LogisticPARAFAC2(num_visits, 
                             num_feats, 
                             rank, 
                             alpha=alpha, 
                             gamma=gamma).cuda()
    smoothness = SmoothnessConstraint(beta=smooth_shape)

    tf_loss_func = PULoss(prior=pos_prior)

    optimizer_pt_reps = Adam([model.U, model.S], 
                             lr=lr, weight_decay=weight_decay)
    optimizer_phenotypes = Adam([model.V], lr=lr, weight_decay=weight_decay)

    lr_scheduler_pt_reps = ReduceLROnPlateau(optimizer_pt_reps, 
                                             mode='max', 
                                             cooldown=10, 
                                             min_lr=1e-6)
    lr_scheduler_phenotypes = ReduceLROnPlateau(optimizer_phenotypes, 
                                                mode='max', 
                                                cooldown=10, 
                                                min_lr=1e-6)

    writer = SummaryWriter(log_path)

    collators = [PaddedDenseTensor(indata, num_feats, subset=subset) 
                 for subset in ('train', 'validation', 'test')]
    loaders = [DataLoader(TensorDataset(torch.arange(len(num_visits))), 
                          shuffle=True, 
                          num_workers=num_workers, 
                          batch_size=batch_size, 
                          collate_fn=collator)
               for collator in collators]
    train_loader, valid_loader, test_loader = loaders

    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(iters):
        epoch_tf_loss = AverageMeter()
        epoch_uni_reg = AverageMeter()

        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}')
        lr = optimizer_pt_reps.param_groups[0]['lr']
        for pids, Xdense, masks, deltas in train_loader:
            num_visits_batch = masks.squeeze(-1).sum(dim=1).cuda()

            num_visits_batch, pt_idx = num_visits_batch.sort(descending=True)
            pids = pids[pt_idx].cuda()
            Xdense = Xdense[pt_idx].cuda()
            masks = masks[pt_idx].cuda()

            deltas = deltas[pt_idx].cuda() / 7  # transform days to weeks

            # update U & S
            model.S.requires_grad = True
            model.U.requires_grad = True
            model.V.requires_grad = False
            
            optimizer_pt_reps.zero_grad()
            
            output = model(pids)
            loss, out = tf_loss_func(output, Xdense, masks=masks)
            uni_reg = model.uniqueness_regularization(pids)
            out = out + reg_weight * uni_reg

            smoothness_reg = smoothness(model.U[pids], 
                                        num_visits_batch, 
                                        deltas=deltas)
            out = out + smooth_weight * smoothness_reg
            out.backward()
            optimizer_pt_reps.step()
            model.projection()
            epoch_uni_reg.update(uni_reg.item(), n=pids.shape[0])

            # update V
            model.S.requires_grad = False
            model.U.requires_grad = False
            model.V.requires_grad = True
            optimizer_phenotypes.zero_grad()
            output = model(pids)
            loss, out = tf_loss_func(output, Xdense, masks=masks)
            out.backward()
            optimizer_phenotypes.step()
            model.projection()

            epoch_tf_loss.update(loss.item(), n=masks.sum())

            pbar.update()
        model.update_phi()
        
        ap_valid = validate(model, valid_loader)
        lr_scheduler_pt_reps.step(ap_valid)
        lr_scheduler_phenotypes.step(ap_valid)
        pbar.set_description(f'Epoch {epoch+1}: loss={epoch_tf_loss.avg:.5e}, '
                             f'uni_reg={epoch_uni_reg.avg:.5e}'
                             f', lr={lr:.2e}'
                             f', completion@valid: PR-AUC={ap_valid:.3f}')
        pbar.close()

        writer.add_scalar('training/loss', epoch_tf_loss.avg, epoch+1)
        writer.add_scalar('training/uniqueness_regularization', 
                          epoch_uni_reg.avg, epoch+1)
        writer.add_scalar('validation/completion-AP', ap_valid, epoch+1)
        
        # early stopping
        if early_stopping(ap_valid, model):
            print('Early Stopped.')
            break
    model = early_stopping.best_model
    print('Model with best validation performance is restored.')

    ap_valid = validate(model, valid_loader)
    print(f'Best PR-AUC for completion@validation set: {ap_valid:.3f}')
    ap_test = validate(model, test_loader)
    print(f'PR-AUC for completion@test set: {ap_test:.3f}\n\n')

    return model.cpu(), ap_valid, ap_test


def project_unseen_tensor(model,
                          test_data,
                          num_visits,
                          num_feats,
                          pos_prior,
                          reg_weight,
                          smooth_weight,
                          lr,
                          seed,
                          batch_size,
                          smooth_shape,
                          iters,
                          num_workers=5):

    if seed is not None:
        torch.manual_seed(seed)

    # set up the projector
    projector = LogisticPARAFAC2(num_visits,
                                 num_feats,
                                 model.rank,
                                 alpha=model.alpha,
                                 gamma=model.gamma,
                                 is_projector=True)
    projector.V.data = model.V.data.clone()
    projector.V.requires_grad = False
    projector.Phi.data = model.Phi.data.clone()
    projector.cuda()

    smoothness = SmoothnessConstraint(beta=smooth_shape)
    tf_loss_func = PULoss(prior=pos_prior)
    optimizer = Adam([projector.U, projector.S], lr=lr)

    collator_train = PaddedDenseTensor(test_data, num_feats, subset='train')
    data_loader = DataLoader(TensorDataset(torch.arange(len(num_visits))), 
                             shuffle=True, 
                             num_workers=num_workers, 
                             batch_size=batch_size, 
                             collate_fn=collator_train)

    pbar = tqdm(total=iters, 
                desc=f'Projecting unseen test tensor onto factor matrices')
    for epoch in range(iters):
        epoch_tf_loss = AverageMeter()
        epoch_uni_reg = AverageMeter()

        for pids, Xdense, masks, deltas in data_loader:
            num_visits_batch = masks.squeeze(-1).sum(dim=1).cuda()

            num_visits_batch, pt_idx = num_visits_batch.sort(descending=True)
            pids = pids[pt_idx].cuda()
            Xdense = Xdense[pt_idx].cuda()
            masks = masks[pt_idx].cuda()
            deltas = deltas[pt_idx].cuda() / 7

            optimizer.zero_grad()
            
            output = projector(pids)
            loss, out = tf_loss_func(output, Xdense, masks=masks)
            uni_reg = projector.uniqueness_regularization(pids)
            out = out + reg_weight * uni_reg

            smoothness_reg = smoothness(projector.U[pids], 
                                        num_visits_batch, deltas)
            out = out + smooth_weight * smoothness_reg

            out.backward()
            optimizer.step()
            projector.projection()
            epoch_tf_loss.update(loss.item(), n=masks.sum())
            epoch_uni_reg.update(uni_reg.item(), n=pids.shape[0])
        pbar.update()
    pbar.set_description(f'Projection done.')
    pbar.close()
    return projector.cpu()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str,
                        help='Name of the experiment.')
    parser.add_argument('--data_path', '-d', type=str, 
                        default='./demo_data.pkl',
                        help='The path of input data.')
    parser.add_argument('--pi', '-p', type=float, default=0.005,
                        help='Class prior for the positive observations.')
    parser.add_argument('--uniqueness', '-u', type=float, default=1e-3,
                        help='Weighting for the uniqueness regularization.')
    parser.add_argument('--rank', '-r', type=int, default=30,
                        help='Target rank of the PARAFAC2 factorization.')
    parser.add_argument('--seed', type=int, 
                        help='Random seed')
    parser.add_argument('--alpha', type=float, default=2,
                        help='Maximam infinity norm allowed for the factor '\
                             'matrices.')
    parser.add_argument('--gamma', type=float, default=1,
                        help='Shape parameter for the sigmoid function.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate of the optimizers.')
    parser.add_argument('--wd', type=float, default=0,
                        help='Weight decay for the optimizers.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--proj_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20,
                        help='Epochs to wait before early stopping, use 0 to '\
                             'switch off early stopping.')
    parser.add_argument('--smooth', type=float, default=1,
                        help='Weighting for the smoothness regularization.')
    parser.add_argument('--smooth_shape', type=float, default=1,
                        help='shape parameter for the time-aware TV smoothing')
    parser.add_argument('--out', type=str, default='./results/',
                        help='Directory to save the results, a subfolder will '\
                             'be created.')
    parser.add_argument('--log', type=str, default='./results/tb_logs',
                        help='Path to store the tensorboard logging file.')

    args = parser.parse_args()

    ##########################
    ## Load data #############
    ##########################
    with open(args.data_path, 'rb') as f:
        indata = pickle.load(f)
        labels = [pt['label'] for pt in indata]
        if any([x is None for x in labels]):  
            # no label for prediction task: random train/test split
            train_idx, test_idx = train_test_split(range(len(indata)), 
                                                   train_size=0.8)
        else:  
            # train/test split by labels
            train_idx, test_idx, *_ = train_test_split(range(len(indata)), 
                                                       labels, 
                                                       train_size=0.8)
        num_feats = max([pt['train'][:, 1].max() + 1 for pt in indata])
        
        data_train = [indata[x] for x in train_idx]
        num_visits_train = [len(pt['times']) for pt in data_train]

        data_test = [indata[x] for x in test_idx]
        num_visits_test = [len(pt['times']) for pt in data_test]

    ##################################
    ## Set up experiment #############
    ##################################
    exp_id = f'LogPar_rank{args.rank}'
    if args.name is not None:
        exp_id = args.name + '_' + exp_id
    if args.seed is not None:
        exp_id += f'_seed{args.seed}'
    exp_id += f'_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    results_out_dir = Path(args.out) / exp_id

    results_out_dir.mkdir(parents=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    with open(results_out_dir/'config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    ###############################
    ## Run experiment #############
    ###############################
    print(f'rank={args.rank}, seed={args.seed}')

    model, *ap_scores = train_logistic_parafac2(
        data_train, num_visits_train, num_feats, 
        patience=args.patience or args.epochs, 
        log_path=os.path.join(args.log, exp_id),
        pos_prior=args.pi,
        smooth_weight=args.smooth,
        seed=args.seed,
        rank=args.rank,
        reg_weight=args.uniqueness,
        weight_decay=args.wd,
        alpha=args.alpha,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        smooth_shape=args.smooth_shape,
        iters=args.epochs)

    torch.save(model.state_dict(), results_out_dir / 'model.pt')

    projector = project_unseen_tensor(model, 
                                      data_test, 
                                      num_visits_test, 
                                      num_feats,
                                      pos_prior=args.pi,
                                      reg_weight=args.uniqueness,
                                      smooth_weight=args.smooth,
                                      lr=args.lr,
                                      batch_size=args.batch_size,
                                      seed=args.seed,
                                      smooth_shape=args.smooth_shape,
                                      iters=args.proj_epochs)
    torch.save(projector.state_dict(), results_out_dir / 'projector.pt')

    print(f'\nExperiment done. Completion performance summary: '
          f'@validation_entries = {ap_scores[0]:.3f}, '
          f'@test_entries = {ap_scores[1]:.3f}')