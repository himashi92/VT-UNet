import os
import pathlib
import pprint

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)
from numpy import logical_and as l_and, logical_not as l_not
from torch import distributed as dist


def save_args(args):
    """Save parsed arguments to config file.
    """
    config = vars(args).copy()
    del config['save_folder']
    del config['seg_folder']
    config_file = args.save_folder / (args.exp_name + ".yaml")
    with open(config_file, "w") as file:
        yaml.dump(config, file)


def save_args_1(args):
    """Save parsed arguments to config file.
    """
    config = vars(args).copy()
    del config['save_folder_1']
    del config['seg_folder_1']
    config_file = args.save_folder_1 / (args.exp_name + ".yaml")
    with open(config_file, "w") as file:
        yaml.dump(config, file)


def master_do(func, *args, **kwargs):
    """Help calling function only on the rank0 process id ddp"""
    try:
        rank = dist.get_rank()
        if rank == 0:
            return func(*args, **kwargs)
    except AssertionError:
        # not in DDP setting, just do as usual
        func(*args, **kwargs)


def save_checkpoint(state: dict, save_folder: pathlib.Path):
    """Save Training state."""
    best_filename = f'{str(save_folder)}/model_best.pth.tar'
    torch.save(state, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def reload_ckpt(args, model, optimizer, device=torch.device("cuda:0")):
    if os.path.isfile(args):
        print("=> loading checkpoint '{}'".format(args))
        checkpoint = torch.load(args, map_location=device)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args))


def reload_ckpt_bis(ckpt, model, device=torch.device("cuda:0")):
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        try:
            checkpoint = torch.load(ckpt, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            return start_epoch
        except RuntimeError:
            # TO account for checkpoint from Alex nets
            print("Loading model Alex style")
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_metrics(preds, targets, patient, tta=False):
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []

    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )

        if np.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            dice = 1 if np.sum(preds[i]) == 0 else 0

        else:
            tp = np.sum(l_and(preds[i], targets[i]))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            dice = 2 * tp / (2 * tp + fp + fn)

        metrics[DICE] = dice
        pp.pprint(metrics)
        metrics_list.append(metrics)

    return metrics_list


def save_metrics(epoch, metrics, writer, current_epoch, teacher=False, save_folder=None):
    metrics = list(zip(*metrics))
    # print(metrics)
    # TODO check if doing it directly to numpy work
    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    # print(metrics)
    labels = ("ET", "TC", "WT")
    metrics = {key: value for key, value in zip(labels, metrics)}
    # print(metrics)
    fig, ax = plt.subplots()
    ax.set_title("Dice metrics")
    ax.boxplot(metrics.values(), labels=metrics.keys())
    ax.set_ylim(0, 1)
    writer.add_figure(f"val/plot", fig, global_step=epoch)
    print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    with open(f"{save_folder}/val{'_teacher' if teacher else ''}.txt", mode="a") as f:
        print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
              [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()], file=f)
    for key, value in metrics.items():
        tag = f"val{'_teacher' if teacher else ''}{''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)


dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
)

VAL_AMP = True


# define inference method
def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


def generate_segmentations_monai(data_loader, model, writer_1, args):
    metrics_list = []
    model.eval()
    for idx, val_data in enumerate(data_loader):
        print(f"Validating case {idx}")
        patient_id = val_data["patient_id"][0]
        ref_path = val_data["seg_path"][0]
        crops_idx = val_data["crop_indexes"]

        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)

        val_inputs, val_labels = (
            val_data["image"].cuda(),
            val_data["label"].cuda(),
        )

        with torch.no_grad():
            val_outputs_1 = inference(val_inputs, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs_1)]

        segs = torch.zeros((1, 3, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs[0]
        segs = segs[0].numpy() > 0.5

        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap = sitk.GetImageFromArray(labelmap)

        refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
        refmap_et = ref_seg == 4
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])

        patient_metric_list = calculate_metrics(segs, refmap, patient_id)
        metrics_list.append(patient_metric_list)
        labelmap.CopyInformation(ref_seg_img)

        print(f"Writing {args.seg_folder_1}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{args.seg_folder_1}/{patient_id}.nii.gz")

    val_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(val_metrics)
    overlap = df.boxplot(METRICS[1:], by="label", return_type="axes")
    overlap_figure = overlap[0].get_figure()
    writer_1.add_figure("benchmark/overlap_measures", overlap_figure)
    haussdorf_figure = df.boxplot(METRICS[0], by="label").get_figure()
    writer_1.add_figure("benchmark/distance_measure", haussdorf_figure)
    grouped_df = df.groupby("label")[METRICS]
    summary = grouped_df.mean().to_dict()
    for metric, label_values in summary.items():
        for label, score in label_values.items():
            writer_1.add_scalar(f"benchmark_{metric}/{label}", score)
    df.to_csv((args.save_folder_1 / 'results.csv'), index=False)


HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
SSIM = "ssim"
METRICS = [HAUSSDORF, DICE, SENS, SPEC, SSIM]
