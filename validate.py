import numpy as np
import time
import torch


def validate(val_loader, model, criterion, T, flops):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    times=[]
    total_flops=np.zeros(len(flops))

    model.eval()
    
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            batch_flops=[]
            target = target
            input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            
            data_time.update(time.time() - end)
            
            output, batch_flops = model(input_var, T, flops)
            total_flops+=batch_flops
            loss = 0.0
            
            loss += criterion(output, target_var)

            losses.update(loss.item(), input.size(0))

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

#             if i % 10 == 0:
#                 print('Epoch: [{0}/{1}]\t'
#                       'Time {batch_time.avg:.3f}\t'
#                       'Data {data_time.avg:.3f}\t'
#                       'Loss {loss.val:.4f}\t'
#                       'Acc@1 {top1.val:.4f}\t'
#                       'Acc@5 {top5.val:.4f}'.format(
#                         i + 1, len(val_loader),
#                         batch_time=batch_time, data_time=data_time,
#                         loss=losses, top1=top1, top5=top5))
            times.append(data_time.sum+batch_time.sum)
#    print('prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    
    return top1, top5, times, total_flops

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print(pred,target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res