import os
import torch
from yolov6.core.evaler import Evaler

class EvalerWrapper(object):
    def __init__(self, eval_cfg):
        task = eval_cfg['task']
        save_dir = eval_cfg['save_dir']
        half = eval_cfg['half']
        data = eval_cfg['data']
        batch_size = eval_cfg['batch_size']
        img_size = eval_cfg['img_size']
        device = eval_cfg['device']
        dataloader = None

        Evaler.check_task(task)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # reload thres/device/half/data according task
        conf_thres, iou_thres = Evaler.reload_thres(conf_thres=0.001, iou_thres=0.65, task=task)
        device = Evaler.reload_device(device, None, task)
        data = Evaler.reload_dataset(data) if isinstance(data, str) else data

        # init
        val = Evaler(data, batch_size, img_size, conf_thres, \
                     iou_thres, device, half, save_dir)
        val.stride = eval_cfg['stride']
        dataloader = val.init_data(dataloader, task)

        self.eval_cfg = eval_cfg
        self.half = half
        self.device = device
        self.task = task
        self.val = val
        self.val_loader = dataloader

    def eval(self, model):
        model.eval()
        model.to(self.device)
        if self.half is True:
            model.half()

        with torch.no_grad():
            pred_result = self.val.predict_model(model, self.val_loader, self.task)
            eval_result = self.val.eval_model(pred_result, model, self.val_loader, self.task)

        return eval_result
