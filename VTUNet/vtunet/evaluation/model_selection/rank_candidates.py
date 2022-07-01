#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from vtunet.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "vtunetPlans"

    overwrite_plans = {
        'vtunetTrainerV2_2': ["vtunetPlans", "vtunetPlansisoPatchesInVoxels"], # r
        'vtunetTrainerV2': ["vtunetPlansnonCT", "vtunetPlansCT2", "vtunetPlansallConv3x3",
                            "vtunetPlansfixedisoPatchesInVoxels", "vtunetPlanstargetSpacingForAnisoAxis",
                            "vtunetPlanspoolBasedOnSpacing", "vtunetPlansfixedisoPatchesInmm", "vtunetPlansv2.1"],
        'vtunetTrainerV2_warmup': ["vtunetPlans", "vtunetPlansv2.1", "vtunetPlansv2.1_big", "vtunetPlansv2.1_verybig"],
        'vtunetTrainerV2_cycleAtEnd': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_cycleAtEnd2': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_reduceMomentumDuringTraining': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_graduallyTransitionFromCEToDice': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_independentScalePerAxis': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_Mish': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_Ranger_lr3en4': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_fp32': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_GN': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_momentum098': ["vtunetPlans", "vtunetPlansv2.1"],
        'vtunetTrainerV2_momentum09': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_DP': ["vtunetPlansv2.1_verybig"],
        'vtunetTrainerV2_DDP': ["vtunetPlansv2.1_verybig"],
        'vtunetTrainerV2_FRN': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_resample33': ["vtunetPlansv2.3"],
        'vtunetTrainerV2_O2': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_ResencUNet': ["vtunetPlans_FabiansResUNet_v2.1"],
        'vtunetTrainerV2_DA2': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_allConv3x3': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_ForceBD': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_ForceSD': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_LReLU_slope_2en1': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_lReLU_convReLUIN': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_ReLU': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_ReLU_biasInSegOutput': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_ReLU_convReLUIN': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_lReLU_biasInSegOutput': ["vtunetPlansv2.1"],
        #'vtunetTrainerV2_Loss_MCC': ["vtunetPlansv2.1"],
        #'vtunetTrainerV2_Loss_MCCnoBG': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_Loss_DicewithBG': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_Loss_Dice_LR1en3': ["vtunetPlansv2.1"],
        'vtunetTrainerV2_Loss_Dice': ["vtunetPlans", "vtunetPlansv2.1"],
        'vtunetTrainerV2_Loss_DicewithBG_LR1en3': ["vtunetPlansv2.1"],
        # 'vtunetTrainerV2_fp32': ["vtunetPlansv2.1"],
        # 'vtunetTrainerV2_fp32': ["vtunetPlansv2.1"],
        # 'vtunetTrainerV2_fp32': ["vtunetPlansv2.1"],
        # 'vtunetTrainerV2_fp32': ["vtunetPlansv2.1"],
        # 'vtunetTrainerV2_fp32': ["vtunetPlansv2.1"],

    }

    trainers = ['vtunetTrainer'] + ['vtunetTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'vtunetTrainerNewCandidate24_2',
        'vtunetTrainerNewCandidate24_3',
        'vtunetTrainerNewCandidate26_2',
        'vtunetTrainerNewCandidate27_2',
        'vtunetTrainerNewCandidate23_always3DDA',
        'vtunetTrainerNewCandidate23_corrInit',
        'vtunetTrainerNewCandidate23_noOversampling',
        'vtunetTrainerNewCandidate23_softDS',
        'vtunetTrainerNewCandidate23_softDS2',
        'vtunetTrainerNewCandidate23_softDS3',
        'vtunetTrainerNewCandidate23_softDS4',
        'vtunetTrainerNewCandidate23_2_fp16',
        'vtunetTrainerNewCandidate23_2',
        'vtunetTrainerVer2',
        'vtunetTrainerV2_2',
        'vtunetTrainerV2_3',
        'vtunetTrainerV2_3_CE_GDL',
        'vtunetTrainerV2_3_dcTopk10',
        'vtunetTrainerV2_3_dcTopk20',
        'vtunetTrainerV2_3_fp16',
        'vtunetTrainerV2_3_softDS4',
        'vtunetTrainerV2_3_softDS4_clean',
        'vtunetTrainerV2_3_softDS4_clean_improvedDA',
        'vtunetTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'vtunetTrainerV2_3_softDS4_radam',
        'vtunetTrainerV2_3_softDS4_radam_lowerLR',

        'vtunetTrainerV2_2_schedule',
        'vtunetTrainerV2_2_schedule2',
        'vtunetTrainerV2_2_clean',
        'vtunetTrainerV2_2_clean_improvedDA_newElDef',

        'vtunetTrainerV2_2_fixes', # running
        'vtunetTrainerV2_BN', # running
        'vtunetTrainerV2_noDeepSupervision', # running
        'vtunetTrainerV2_softDeepSupervision', # running
        'vtunetTrainerV2_noDataAugmentation', # running
        'vtunetTrainerV2_Loss_CE', # running
        'vtunetTrainerV2_Loss_CEGDL',
        'vtunetTrainerV2_Loss_Dice',
        'vtunetTrainerV2_Loss_DiceTopK10',
        'vtunetTrainerV2_Loss_TopK10',
        'vtunetTrainerV2_Adam', # running
        'vtunetTrainerV2_Adam_vtunetTrainerlr', # running
        'vtunetTrainerV2_SGD_ReduceOnPlateau', # running
        'vtunetTrainerV2_SGD_lr1en1', # running
        'vtunetTrainerV2_SGD_lr1en3', # running
        'vtunetTrainerV2_fixedNonlin', # running
        'vtunetTrainerV2_GeLU', # running
        'vtunetTrainerV2_3ConvPerStage',
        'vtunetTrainerV2_NoNormalization',
        'vtunetTrainerV2_Adam_ReduceOnPlateau',
        'vtunetTrainerV2_fp16',
        'vtunetTrainerV2', # see overwrite_plans
        'vtunetTrainerV2_noMirroring',
        'vtunetTrainerV2_momentum09',
        'vtunetTrainerV2_momentum095',
        'vtunetTrainerV2_momentum098',
        'vtunetTrainerV2_warmup',
        'vtunetTrainerV2_Loss_Dice_LR1en3',
        'vtunetTrainerV2_NoNormalization_lr1en3',
        'vtunetTrainerV2_Loss_Dice_squared',
        'vtunetTrainerV2_newElDef',
        'vtunetTrainerV2_fp32',
        'vtunetTrainerV2_cycleAtEnd',
        'vtunetTrainerV2_reduceMomentumDuringTraining',
        'vtunetTrainerV2_graduallyTransitionFromCEToDice',
        'vtunetTrainerV2_insaneDA',
        'vtunetTrainerV2_independentScalePerAxis',
        'vtunetTrainerV2_Mish',
        'vtunetTrainerV2_Ranger_lr3en4',
        'vtunetTrainerV2_cycleAtEnd2',
        'vtunetTrainerV2_GN',
        'vtunetTrainerV2_DP',
        'vtunetTrainerV2_FRN',
        'vtunetTrainerV2_resample33',
        'vtunetTrainerV2_O2',
        'vtunetTrainerV2_ResencUNet',
        'vtunetTrainerV2_DA2',
        'vtunetTrainerV2_allConv3x3',
        'vtunetTrainerV2_ForceBD',
        'vtunetTrainerV2_ForceSD',
        'vtunetTrainerV2_ReLU',
        'vtunetTrainerV2_LReLU_slope_2en1',
        'vtunetTrainerV2_lReLU_convReLUIN',
        'vtunetTrainerV2_ReLU_biasInSegOutput',
        'vtunetTrainerV2_ReLU_convReLUIN',
        'vtunetTrainerV2_lReLU_biasInSegOutput',
        'vtunetTrainerV2_Loss_DicewithBG_LR1en3',
        #'vtunetTrainerV2_Loss_MCCnoBG',
        'vtunetTrainerV2_Loss_DicewithBG',
        # 'vtunetTrainerV2_Loss_Dice_LR1en3',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
        # 'vtunetTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
