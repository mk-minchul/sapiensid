import os
import torch

from .body_ltcc_evaluator import BodyLTCCEvaluator
from .body_market1501_evaluator import BodyMarket1501Evaluator
from .body_prcc_evaluator import BodyPRCCEvaluator
from .body_msmt17_evaluator import BodyMSMT17Evaluator
from .body_wb_test_evaluator import BodyWebBodyTestEvaluator
from .body_ccvid_evaluator import BodyCCVIDEvaluator
from .body_deepchange_evaluator import BodyDeepChangeEvaluator
from .body_celebreid_evaluator import BodyCelebReIDEvaluator
from .body_ccda_evaluator import BodyCCDAEvaluator
from .body_occluded_reid_evaluator import BodyOccludedReidEvaluator
from .body_partial_reid_evaluator import BodyPartialReidEvaluator
from .body_occluded_duke_evaluator import BodyOccludedDukeEvaluator
from .body_ccvid_vid_evaluator import BodyCCVIDVIDEvaluator


def get_evaluator_by_name(eval_type, name, eval_data_path, transform, fabric, batch_size, num_workers, save_artifacts_path=''):

    assert os.path.isdir(eval_data_path), (f'Evaluation Dataset does not exist. Check that cvlface/.env file is set correctly '
                                           f'and the dataset is downloaded. {eval_data_path}')

    if eval_type == 'body_ltcc':
        return BodyLTCCEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers, save_artifacts_path)
    elif eval_type == 'body_market_1501':
        return BodyMarket1501Evaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'body_prcc':
        return BodyPRCCEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers, save_artifacts_path)
    elif eval_type == 'body_msmt17':
        return BodyMSMT17Evaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'body_wb_test':
        return BodyWebBodyTestEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'body_ccvid':
        return BodyCCVIDEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'body_ccvid_vid':
        return BodyCCVIDVIDEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'body_deepchange':
        return BodyDeepChangeEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'body_celebreid':
        return BodyCelebReIDEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'body_ccda':
        return BodyCCDAEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'body_occluded_reid':
        return BodyOccludedReidEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'body_partial_reid':
        return BodyPartialReidEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    elif eval_type == 'body_occluded_duke':
        return BodyOccludedDukeEvaluator(name, eval_data_path, transform, fabric, batch_size, num_workers)
    else:
        raise ValueError('Unknown evaluation type: %s' % eval_type)


def summary(save_result, epoch, step, n_images_seen):
    key_metrics = [
                   'ltcc_test/diff_clothes_top1',
                   'prcc_test/diff_clothes_top1',
                   'market_1501_test/overall_top1',
                    'webbody_test_25/overall_top1',
                   'webbody_test_full/overall_top1',
                   'ccvid_test/diff_clothes_top1',
                   'deepchange_test/diff_clothes_top1',
                   'celebreid_test/overall_top1',
                   'ccda_test/overall_top1',
                   ]
    key_metrics_in_save_result = [k for k in key_metrics if k in save_result.index]
    briar_metrics = [k for k in save_result.index if 'briar' in k]
    key_metrics_in_save_result += briar_metrics
    if key_metrics_in_save_result:
        summary = save_result.loc[key_metrics_in_save_result]
        summary.index = ['summary/'+k.replace('/', '_') for k in summary.index]
        summary.index = [k.replace('Norm:False_Det:True_tpr_at_fpr_0.0001', 'TPR@FPR0.01') for k in summary.index]
        summary.index = [k.replace('_gt_aligned', '') for k in summary.index]
        mean = summary['val'].mean()

        summary_dict = summary['val'].to_dict()
        summary_dict['summary/mean'] = mean
        summary_dict['epoch'] = epoch
        summary_dict['step'] = step
        summary_dict['n_images_seen'] = n_images_seen
        summary_dict['trainer/global_step'] = step
        summary_dict['trainer/epoch'] = epoch

    else:
        mean = save_result['val'].mean()
        summary_dict = save_result['val'].to_dict()
        summary_dict['summary/mean'] = mean
        summary_dict['epoch'] = epoch
        summary_dict['step'] = step
        summary_dict['n_images_seen'] = n_images_seen
        summary_dict['trainer/global_step'] = step
        summary_dict['trainer/epoch'] = epoch
    return mean, summary_dict


class IsBestTracker():

    def __init__(self, fabric):
        self._is_best = True
        self.prev_best_metric = -1
        self.fabric = fabric


    def set_is_best(self, metric):
        metric_tensor = torch.tensor(metric, device=self.fabric.device)
        self.fabric.barrier()
        self.fabric.broadcast(metric_tensor, 0)
        metric = metric_tensor.item()

        if metric > self.prev_best_metric:
            self.prev_best_metric = metric
            self._is_best = True
        else:
            self._is_best = False


    def is_best(self):
        return self._is_best