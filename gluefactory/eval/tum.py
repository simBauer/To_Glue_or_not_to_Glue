# Code for the evaluation of TUM dataset
# - Code based on eval/megadepth1500.py from gluefactory (https://github.com/cvg/glue-factory)

import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import torch

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH, EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import eval_matches_projection, eval_poses, eval_absolute_pose_robust

logger = logging.getLogger(__name__)


# Function to convert pixel to intermediate coordinate system
def pixel_to_st_coords(pix_coords, shape):
    W = shape.flatten()[0]
    H = shape.flatten()[1]
    st_coords = torch.zeros_like(pix_coords)
    st_coords[:, 0] = pix_coords[:, 0] / W
    st_coords[:, 1] = 1.0 - pix_coords[:, 1] / H

    return st_coords


# Function to compute world coordinates from the pixels
def get_world_coordinates_from_pixels(
        st_edges,
        xyz_edges, 
        pix_coords, 
        shape):
    
    st_edges = st_edges.view(-1, 2)
    xyz_edges = xyz_edges.view(-1, 3)
    
    n_edges = st_edges.size(0) // 2
    idx2 = 2

    new_basis_i = st_edges[[1, idx2], :] - st_edges[0, :]
    new_basis_w = xyz_edges[[1, idx2], :] - xyz_edges[0, :]

    dot_prod = torch.dot(new_basis_i[0, :], new_basis_i[1, :]) / torch.prod(torch.norm(new_basis_i, dim=1))
    while (torch.abs(dot_prod) > 0.5) and (idx2 < n_edges - 1):
        idx2 += 1
        new_basis_i = st_edges[[1, idx2], :] - st_edges[0, :]
        new_basis_w = xyz_edges[[1, idx2], :] - xyz_edges[0, :]
        dot_prod = torch.dot(new_basis_i[0, :], new_basis_i[1, :]) / torch.prod(torch.norm(new_basis_i, dim=1))

    A = new_basis_i.t()
    assert torch.abs(torch.linalg.cond(A) - 1) < 3
    A_inv = torch.linalg.inv(A)

    pix_coords = pix_coords.view(-1, 2)
    st_coords = pixel_to_st_coords(pix_coords=pix_coords, shape=shape)
    st_coords = st_coords - st_edges[0, :]
    b_coords = torch.matmul(A_inv, st_coords.t()).t()
    xyz_coords = xyz_edges[0, :] + torch.outer(b_coords[:, 0], new_basis_w[0, :]) + torch.outer(b_coords[:, 1], new_basis_w[1, :])
    return xyz_coords


class TumFacadePipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "image_pairs",
            "pairs": "tum_facade/pairs.txt",
            "root": "tum_facade/",
            "extra_data": "absolute_pose",
            "preprocessing": {},
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "opencv",
            "ransac_th": 10.0,  # -1 runs a bunch of thresholds and selects the best
        },
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]
    optional_export_keys = []

    def _init(self, conf):
        if not (DATA_PATH / "tum_facade").exists():
            logger.info("No dataset found.")
            
    @classmethod
    def get_dataloader(self, data_conf=None, pairs=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf if data_conf else self.default_conf["data"]
        data_conf["pairs"] = pairs if pairs else data_conf["pairs"]
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        conf = self.conf.eval
        results = defaultdict(list)
        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [4.0, 8.0, 12.0, 20.0, 30.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)
            
            # Convert kpts in facade image (kpts0) to world coords
            # Scale x and y coordinates with scales
            scales = data["facade"]["scales"]
            shape = data["facade"]["ori_size"]
            pred["keypoints0"] /= scales
            kpts0_world = get_world_coordinates_from_pixels(
                st_edges=data["facade"]["st_edges"],
                xyz_edges=data["facade"]["xyz_edges"],
                pix_coords=pred["keypoints0"],
                shape=shape)
            pred["keypoints0_world"] = kpts0_world
            
            results_i = eval_matches_projection(data, pred)
            for th in test_thresholds:
                pose_results_i = eval_absolute_pose_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        best_pose_results_t, best_th = eval_poses(
            pose_results, auc_ths=[.5, 1., 5.], key="abs_pose_error_t", unit="m"
        )
        best_pose_results_r, best_th = eval_poses(
            pose_results, auc_ths=[1., 3., 10.], key="abs_pose_error_R", unit="°"
        )
        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results_t,
            **best_pose_results_r
        }

        figures = {
            "pose_recall_R": plot_cumulative(
                {self.conf.eval.estimator: results["abs_pose_error_R"]},
                [0, 30],
                unit="°",
                title="Pose (rotation)",
            ),
            "pose_recall_t": plot_cumulative(
                {self.conf.eval.estimator: results["abs_pose_error_t"]},
                [0, 10],
                unit="m",
                title="Pose (translation)",
            )
        }

        return summaries, figures, results


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(TumFacadePipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = TumFacadePipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
