"""
assign tracklet confidence score 
predict, update, punish tracklet score under category-specific way.

TODO: to support Confidence-based method to init and kill tracklets
Code URL: CBMOT(https://github.com/cogsys-tuebingen/CBMOT)
"""
import pdb
from data.script.NUSC_CONSTANT import *
from motion_module.nusc_object import FrameObject

class ScoreObject:
    def __init__(self) -> None:
        self.raw_score = self.final_score = None
        self.predict_score = self.update_score = None
        
    def __repr__(self) -> str:
        repr_str = 'Raw score: {}, Predict score: {}, Update score: {}, Final score: {}.'
        return repr_str.format(self.raw_score, self.predict_score, self.update_score, self.final_score)

class ScoreManagement:
    def __init__(self, timestamp: int, cfg: dict, cls_label: int, det_infos: dict) -> None:
        self.initstamp, self.cfg, self.frame_objects, self.trk_avg_score = timestamp, cfg['life_cycle'], {}, None
        self.dr, self.predict_mode = self.cfg['basic']['decay_rate'][cls_label], self.cfg['score']['predict_mode']
        self.score_dc, self.update_mode = self.cfg['score']['score_decay'][cls_label], self.cfg['score']['update_mode']
        assert self.predict_mode in SCORE_PREDICT and self.update_mode in SCORE_UPDATE
        self.initialize(det_infos)
    
    def initialize(self, det_infos: dict) -> None:
        """init tracklet confidence score, no inplace ops needed

        Args:
            det_infos (dict): dict, detection infos under different data format
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
        score_obj = ScoreObject()
        score_obj.raw_score = score_obj.final_score = det_infos['nusc_box'].score
        score_obj.predict_score = score_obj.update_score = det_infos['nusc_box'].score
        self.frame_objects[self.initstamp] = score_obj

        # calu tracklet average score
        self.trk_avg_score = self.calu_trk_avg_score()
    
    def predict(self, timestamp: int, pred_obj: FrameObject = None) -> None:
        """decay tracklet confidence score, change score in the predict infos inplace.

        Args:
            timestamp (int): current frame id
            pred_obj (FrameObject): nusc box/infos predicted by the filter
        """
        score_obj = ScoreObject()
        score_obj.raw_score = prev_score = self.frame_objects[timestamp - 1].final_score

        if self.predict_mode == 'Normal':
            # Consider cov convergence time
            score_obj.predict_score = prev_score * self.dr
        elif self.predict_mode == 'Minus':
            score_obj.predict_score = max(prev_score - self.score_dc, 0)
        else:
            raise Exception("unsupport score predict function")
        self.frame_objects[timestamp] = score_obj
        
        # assign tracklet score inplace
        pred_obj.predict_box.score = pred_obj.predict_infos[-5] = max(score_obj.predict_score, 0)
        
        
    def update(self, timestamp: int, update_obj: FrameObject, raw_det: dict = None) -> None:
        """Update trajectory confidence scores inplace directly using matched det

        Args:
            timestamp (int): current frame id
            update_obj (FrameObject): nusc box/infos updated by the filter
            raw_det (dict, optional): same as data format in the init function. Defaults to None.
        """
        score_obj = self.frame_objects[timestamp]
        if raw_det is None:
            score_obj.final_score = score_obj.predict_score
            # calu tracklet average score
            self.trk_avg_score = self.calu_trk_avg_score()
            return

        # update tracklet score by heuristic function
        if self.update_mode == 'Normal':
            # Consider cov convergence time
            update_score = raw_det['nusc_box'].score
        elif self.update_mode == 'Multi':
            update_score = 1 - (1 - raw_det['nusc_box'].score) * (1 - score_obj.predict_score)
        elif self.update_mode == 'Parallel':
            update_score = (1 - (1 - raw_det['nusc_box'].score) * (1 - score_obj.predict_score) /
                            (2 - raw_det['nusc_box'].score - score_obj.predict_score))
        else:
            raise Exception("unsupport score update function")

        # assign score objects and output scores
        score_obj.update_score = score_obj.final_score = update_score
        update_obj.update_box.score = update_obj.update_infos[-5] = max(update_score, 0)

        # calu tracklet average score
        self.trk_avg_score = self.calu_trk_avg_score()

    def calu_trk_avg_score(self) -> float:
        return sum([score_obj.final_score for _, score_obj in self.frame_objects.items()]) / len(self.frame_objects)

    def __getitem__(self, item) -> ScoreObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)
        
        
    
    