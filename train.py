import argparse
import timm
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.layers.patch_embed import Format
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, ContrastiveLearningViewGenerator, get_params_groups,DistillLoss
from models.token_transformer import TokViT
from util.loss import cdr

from models.linear import CosineLinear
class MyModel(nn.Module):
    def __init__(self, backbone:TokViT, head:DINOHead,base_num):
        super(MyModel, self).__init__()
        self.backbone = backbone
        self.head = head
        self.head.mlp = nn.Identity()
        in_dim = 768

        self.aux_fc = CosineLinear(in_dim, base_num)
        self.aux_fc.sigma = 1.0

    def forward(self, x):
        ori_feat,tok_feat = self.backbone(x)
        logit_tok = self.aux_fc(tok_feat)
        proj_ori,logit_ori = self.head.forward(ori_feat)
        
        
        return logit_tok, proj_ori,logit_ori



def train(student:MyModel, train_loader, test_loader, unlabelled_train_loader, args):
    if args.difflr:
        params_groups = [
            {"params": [param for param in student.backbone.blocks.parameters() if param.requires_grad],"lr":args.lr*0.1},
            {"params": student.head.parameters(),"lr":args.lr},
            {"params": student.aux_fc.parameters(),"lr":args.lr},
            {"params": student.backbone.Prompt_Token,"lr":args.lr},
            
        ]
        args.logger.info("Different learning rates for backbone and head")
    else:
        params_groups = get_params_groups(student)
 
    # params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )


    
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        LCE = AverageMeter()
        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
             
            images = torch.cat(images, dim=0).cuda(non_blocking=True)
            mask_lab_all = torch.cat([mask_lab for _ in range(2)], dim=0)
            class_lab_all = torch.cat([class_labels for _ in range(2)], dim=0)
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                ori_feat,tok_feat = student.backbone(images)
                tok_feat_norm = F.normalize(tok_feat, dim=-1)
                fc_out = student.aux_fc(tok_feat_norm)
                student_proj,student_out = student.head(ori_feat)

                teacher_out = student_out.detach()
                all_pred_base_mask = student_out.argmax(1) < args.num_labeled_classes
                all_pred_base_mask = all_pred_base_mask | mask_lab_all
                pred_base_mask_0,pred_base_mask_1 = (all_pred_base_mask).chunk(2)
                pred_base_mask_single = pred_base_mask_0 & pred_base_mask_1

                teach_temp = cluster_criterion.teacher_temp_schedule[epoch]
                tok_logits_0,tok_logits_1 = fc_out.chunk(2)
                tok_prob = F.softmax(fc_out.div(0.1),dim=1)
                tok_prob_0, tok_prob_1 = tok_prob.chunk(2)
                
        

                
                stu_prob = F.softmax(student_out / 0.1, dim=1)#cluster_criterion.teacher_temp_schedule[epoch]
                stu_prob_ft = F.softmax(student_out / args.fc_temp, dim=1)

                stu_prob_0,stu_prob_1 = stu_prob_ft.chunk(2)

                
                #teach_temp = 0.05
                #temp = 0.1
                tok_feat_norm_0,tok_feat_norm_1 = tok_feat_norm.chunk(2)
                semi_loss = torch.tensor(0.0).cuda()
                semi_loss = F.kl_div(F.log_softmax(tok_logits_0.div(0.1),dim=1)[pred_base_mask_single], F.softmax(tok_logits_1.div(0.1),dim=1)[pred_base_mask_single],reduction='batchmean') + \
                            F.kl_div(F.log_softmax(tok_logits_1.div(0.1),dim=1)[pred_base_mask_single], F.softmax(tok_logits_0.div(0.1),dim=1)[pred_base_mask_single],reduction='batchmean')
                semi_loss *= 0.5
                
                #semi_loss = F.kl_div(F.log_softmax(tok_logits_1.div(0.1),dim=1)[pred_base_mask_single], F.softmax(tok_logits_1.div(0.1)[pred_base_mask_single],dim=1),reduction='batchmean') 
                
                lab_tok_feat = torch.cat([tok_feat_norm_0[mask_lab].unsqueeze(1),tok_feat_norm_1[mask_lab].unsqueeze(1)], dim=1)
                lab = class_labels[mask_lab]
                tok_supcon_loss = SupConLoss(temperature=args.suptemp,base_temperature=args.suptemp,contrast_mode='one')(lab_tok_feat,lab)

                # clustering, sup
                
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                fc_logits  = torch.cat([f[mask_lab] for f in (fc_out / args.fc_temp).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                fc_loss  = nn.CrossEntropyLoss()(fc_logits, sup_labels)
                
               
                # clustering, unsup

                stu_base_prob   = F.softmax(student_out[:,:args.num_labeled_classes] / 0.1, dim=1)
                tok_prob_shrink = F.softmax(fc_out / args.shrink_temp, dim=1)

            
                cluster_loss = cluster_criterion.forward(student_out, teacher_out, epoch) 

               
                avg_probs = stu_prob.mean(dim=0)
                                
                me_max_loss =  torch.sum(avg_probs * torch.log(avg_probs+1e-6)) + math.log(float(len(avg_probs))) 
                cluster_loss += args.memax_weight * me_max_loss
                
                ori_feat_norm = F.normalize(ori_feat, dim=-1)
               
         
                y1,y2 = stu_prob_0.argmax(1) , stu_prob_1.argmax(1)
                y_tok_1,y_tok_2 = tok_prob_0.argmax(1) , tok_prob_1.argmax(1)
                y1[mask_lab] = class_labels[mask_lab]
                y2[mask_lab] = class_labels[mask_lab]
                y_tok_1[mask_lab] = class_labels[mask_lab]
                y_tok_2[mask_lab] = class_labels[mask_lab]
                cdr_loss      = cdr(stu_prob_0,stu_prob_1,y1,y2) #+ 1 - 1 * torch.diagonal(prob_mat).mean()
                cdr_aux_loss  = cdr(tok_prob_0[pred_base_mask_single],tok_prob_1[pred_base_mask_single],y_tok_1[pred_base_mask_single],y_tok_2[pred_base_mask_single]) #+ 1 - 1 * torch.diagonal(prob_tok_mat).mean()    

                
                lab_feat = torch.cat([f[mask_lab].unsqueeze(1) for f in ori_feat_norm.chunk(2)], dim=1)
    
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss(temperature=args.suptemp,base_temperature=args.suptemp)(lab_feat, labels=sup_con_labels) 
                cluster_loss += args.dw * cdr_loss
                semi_loss    += args.dw * cdr_aux_loss
                #cross_loss = torch.tensor(0.0).cuda()
                tok_prob_max_value = tok_prob_shrink.max(dim=1).values.data
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj,temperature=args.temp) # Is temp=1 here? Standard infonce
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
            
      
                    
                cross_loss = torch.sum(-tok_prob_shrink.data * torch.log(stu_base_prob+1e-6),dim=1)
                cross_loss = cross_loss * tok_prob_max_value
                cross_loss = cross_loss[all_pred_base_mask].mean() 


                pstr = ''
                
                pstr += f'fc: {fc_loss.item():.4f} '
                pstr += f'semi_loss: {semi_loss.item():.4f} '
                pstr += f'cls: {cls_loss.item():.4f} '
                pstr += f'cluster: {cluster_loss.item():.4f} '
                pstr += f'cross: {cross_loss.item():.4f}'
  
                
                loss = 0

                loss += args.cross_weight * cross_loss
                
                loss += (1 - args.fc_weight) * semi_loss     + args.fc_weight  * (fc_loss  + tok_supcon_loss)  
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * (cls_loss)
                #loss += (1 - args.sup_weight) * contra
                loss +=  args.sup_weight * sup_con_loss #+ (1 - args.sup_weight) * contrastive_loss

    
                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            LCE.update(cls_loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}] loss {:.5f} {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f},CE:{:.4f} '.format(epoch, loss_record.avg,LCE.avg))

        #args.logger.info('Testing on unlabelled examples in the training data...')
        if (epoch + 1) % 5 ==0 :
            all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args) # This is actually a clustering algorithm
        # args.logger.info('Testing on disjoint test set...')
        # all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)

       
            args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'epoch': epoch + 1,
        }

        
        if (epoch + 1) % 50 == 0:
            torch.save(save_dict, args.model_path)
            args.logger.info("model saved to {}.".format(args.model_path))

def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    preds_base=[]
    fc_pred_base = []
    mask = np.array([])
    pmax = np.array([])
    #all_feat = []
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            #feat = model.backbone(images)
            fc_logits, _, logits = model(images)
            probs = F.softmax(logits.div(0.1), dim=1)
            prob_max = probs.max(dim=1).values
            pmax = np.append(pmax, prob_max.cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            preds_base.append(logits[:,:len(args.train_classes)].argmax(1).cpu().numpy())
            fc_pred_base.append(fc_logits[:,:len(args.train_classes)].argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))
            #all_feat.append(fc_logits.cpu().numpy())
    preds = np.concatenate(preds)
    preds_base_mask = preds < len(args.train_classes)
    preds_base = np.concatenate(preds_base)
    targets = np.concatenate(targets)
    fc_pred_base = np.concatenate(fc_pred_base)
    
    base_num = len(args.train_classes)
    base_mask = targets < len(args.train_classes) 
    b2n_num = (preds[base_mask]>base_num-1).sum()
    b2b_num = (preds[base_mask] <base_num).sum()
    n2n_num = (preds[~base_mask]>base_num-1).sum()
    n2b_num = (preds[~base_mask]<base_num).sum()
    
    oracle_base_acc = (preds[base_mask] == targets[base_mask]).sum() / len(targets[base_mask])
    oracle2 = (preds_base[base_mask] == targets[base_mask]).sum() / len(targets[base_mask])
    oracle2_fc = (fc_pred_base[base_mask] == targets[base_mask]).sum() / len(targets[base_mask])
    try:
        args.logger.info(f'confidence_mean: {pmax.mean():.4f} varience:{pmax.var():.4f} base conf:{pmax[preds_base_mask].mean():.4f} novel conf {pmax[~preds_base_mask].mean():.4f}' ) 
        args.logger.info(f'Base to New: {b2n_num}, Base to Base: {b2b_num}, New to New: {n2n_num}, New to Base: {n2b_num}')
    
        args.logger.info(f'Oracle Base Accuracy: {oracle_base_acc:.2%} oracle2 {oracle2:.2%} oracle_fc {oracle2_fc:.2%}')
    except:
        print((f'confidence_mean: {pmax.mean():.4f} varience:{pmax.var():.4f} base conf:{pmax[preds_base_mask].mean():.4f} novel conf {pmax[~preds_base_mask].mean():.4f}' ) )
        print((f'Base to New: {b2n_num}, Base to Base: {b2b_num}, New to New: {n2n_num}, New to Base: {n2b_num}'))
        print((f'Oracle Base Accuracy: {oracle_base_acc:.2%} oracle2 {oracle2:.2%} oracle_fc {oracle2_fc:.2%}'))
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('-d','--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--no-con', action='store_true', default=False)
    parser.add_argument('--onlymain', action='store_true', default=False)
    parser.add_argument('--semi_type', type=str, default='kl')

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--diag_type', type=str, default='zero')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--cov_weight', type=float, default=0.000)#0.001
    parser.add_argument('--adv_weight', type=float, default=0.0)
    parser.add_argument('--cross_weight', type=float, default=0.2)
    parser.add_argument('--fc_weight', type=float, default=0.5)
    parser.add_argument('--fc_temp', type=float, default=0.1)
    parser.add_argument('--thr', type=float, default=0.9)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--cross_type', default='kl', type=str)
    parser.add_argument('--model_type', default='dino', type=str)
    
    parser.add_argument('-m','--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--shrink_temp', default=0.1, type=float, help='Temperature for shrinking the logits.')
    parser.add_argument('--suptemp', default=0.1, type=float, help='Temperature forsupcon.')
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--difflr', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--cnum', default=0, type=int)
    parser.add_argument('-n','--exp_name', default=None, type=str)
    parser.add_argument('--dname', default=None, type=str)
    parser.add_argument('-p','--prob_type', default='avg', type=str)
    parser.add_argument('--dw', type=float, default=1.0)
    parser.add_argument('--mid_class', type=int, default=0)
 

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_class_splits(args)

    if args.mid_class > 0:
        args.unlabeled_classes = args.train_classes[args.mid_class:] + args.unlabeled_classes
        args.train_classes = args.train_classes[:args.mid_class]

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes) 

    init_experiment(args, runner_name=['RLCD'],exp_id=args.dname)
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    #backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    if args.model_type == 'dino':
        dino = timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=0)
        patch_size = 16
      
        backbone = TokViT(img_size=224, patch_size=patch_size)
    elif args.model_type == 'dinov2':
        dino = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
        patch_size = 14
        backbone = TokViT(img_size=518, patch_size=patch_size,num_classes=0,init_values=1e-5)
        backbone.dynamic_img_size = True
        backbone.patch_embed.strict_img_size = False
        backbone.patch_embed.flatten = False
        backbone.patch_embed.output_fmt = Format.NHWC

    args.image_size = 224
    backbone.load_state_dict(dino.state_dict(), strict=False)

    
    
    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    

    
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes # This is actually the total number of classes
    if args.cnum > 0:
        args.mlp_out_dim = args.cnum
    args.logger.info(f'Using {args.mlp_out_dim} output dimensions for the projection head')
    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    
    backbone.Prompt_Token.requires_grad_(True)

 
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)

    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)#

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader           = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model     = MyModel(backbone, projector,args.num_labeled_classes).to(device)

    # ----------------------
    # TRAIN

    train(model, train_loader, None, test_loader_unlabelled, args)
    