# final_test.py
# 최종 테스트 전용 스크립트 (Train 데이터 불필요 버전)
#
# 사용법:
# 1. 기본 테스트 (추론 결과 저장 없음):
#    python test.py
#
# 2. 추론 결과를 인덱스와 함께 저장:
#    python test.py --save-predictions
#
# 3. 특정 분류기 체크포인트 사용:
#    python test.py --classifier-checkpoint /path/to/checkpoint.pth --save-predictions
#
# 저장되는 추론 결과 형식 (predictions_*.json):
# [
#   {
#     "index": 0,
#     "prediction": 5,
#     "confidence": 0.95,
#     "target": 5,
#     "all_probs": [0.01, 0.02, ..., 0.95, ...],
#     "file_path": "/path/to/image.jpg"
#   },
#   ...
# ]

import argparse
import json
import logging
import os
import re
import sys
from glob import glob
from functools import partial

import numpy as np
import torch
import torch.nn as nn

# dinov2 import 경로 가정: 현재 파일이 eval 스크립트와 같은 repo 트리 안에 있다고 가정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import ModelWithIntermediateLayers, evaluate
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.data import SamplerType, make_dataset, make_data_loader
from dinov2.data.transforms import make_classification_eval_transform
from fvcore.common.checkpoint import Checkpointer
import dinov2.distributed as distributed  # 남겨두지만 이 파일에서는 사용하지 않음

logger = logging.getLogger("dinov2.final_test")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


############################
# 유틸 / 모델 정의
############################

def _pad_and_collate(batch):
    import numpy as np
    maxlen = max(len(targets) for image, targets in batch)
    padded_batch = [
        (image, np.pad(targets, (0, maxlen - len(targets)), constant_values=-1)) for image, targets in batch
    ]
    return torch.utils.data.default_collate(padded_batch)

def _pad_and_collate_with_indices(batch):
    import numpy as np
    # batch는 (image, targets, index) 형태로 가정
    maxlen = max(len(targets) for image, targets, index in batch)
    padded_batch = [
        (image, np.pad(targets, (0, maxlen - len(targets)), constant_values=-1), index) 
        for image, targets, index in batch
    ]
    return torch.utils.data.default_collate(padded_batch)

def _collate_with_indices(batch):
    # batch는 (image, targets, index) 형태로 가정
    return torch.utils.data.default_collate(batch)

class IndexPreservingDataset(torch.utils.data.Dataset):
    """원본 데이터셋의 인덱스를 보존하는 래퍼"""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, (list, tuple)):
            return (*sample, idx)
        else:
            return (sample, idx)

def get_file_paths_from_dataset(dataset_str):
    """데이터셋에서 파일 경로 목록을 추출"""
    try:
        # 데이터셋을 다시 생성하여 파일 경로 추출
        temp_dataset = make_dataset(
            dataset_str=dataset_str,
            transform=make_classification_eval_transform(),
        )
        
        file_paths = []
        for i in range(len(temp_dataset)):
            try:
                # 데이터셋에서 파일 경로 정보 추출 시도
                if hasattr(temp_dataset, 'get_image_relpath'):
                    # MyDataset 형태 - 상대 경로 사용
                    file_paths.append(temp_dataset.get_image_relpath(i))
                elif hasattr(temp_dataset, 'get_image_full_path'):
                    # MyDataset 형태 - 전체 경로 사용
                    file_paths.append(temp_dataset.get_image_full_path(i))
                elif hasattr(temp_dataset, 'samples'):
                    # ImageFolder 형태
                    file_paths.append(temp_dataset.samples[i][0])
                elif hasattr(temp_dataset, 'imgs'):
                    # ImageFolder 형태 (구버전)
                    file_paths.append(temp_dataset.imgs[i][0])
                elif hasattr(temp_dataset, 'data'):
                    # 다른 형태의 데이터셋
                    file_paths.append(f"sample_{i}")
                else:
                    file_paths.append(f"sample_{i}")
            except:
                file_paths.append(f"sample_{i}")
        
        return file_paths
    except Exception as e:
        logger.warning(f"파일 경로 추출 실패: {e}")
        return None

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()

class LinearClassifier(nn.Module):
    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)

class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

def make_eval_data_loader(dataset_str, batch_size, num_workers, metric_type):
    test_dataset = make_dataset(
        dataset_str=dataset_str,
        transform=make_classification_eval_transform(),
    )
    # 인덱스 보존을 위한 래퍼 추가
    test_dataset = IndexPreservingDataset(test_dataset)
    
    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=_pad_and_collate_with_indices if metric_type == MetricType.IMAGENET_REAL_ACCURACY else _collate_with_indices,
    )
    return test_data_loader

def evaluate_linear_classifiers(
    feature_model,
    linear_classifiers,
    data_loader,
    metric_type,
    training_num_classes,
    class_mapping=None,
    best_classifier_on_val=None,
    save_predictions=False,
    output_dir=None,
    dataset_str=None,
):
    num_classes = len(class_mapping) if class_mapping is not None else training_num_classes
    metric = build_metric(metric_type, num_classes=num_classes)
    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
    
    

    # 파일 경로 정보 추출
    file_paths = None
    if save_predictions and dataset_str:
        file_paths = get_file_paths_from_dataset(dataset_str)

    class LinearPostprocessor(nn.Module):
        def __init__(self, linear_classifier, class_mapping=None):
            super().__init__()
            self.linear_classifier = linear_classifier
            self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))
        def forward(self, samples, targets):
            preds = self.linear_classifier(samples)
            return {
                "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
                "target": targets,
            }

    postprocessors = {k: LinearPostprocessor(v, class_mapping) for k, v in linear_classifiers.classifiers_dict.items()}
    for p in postprocessors.values():
        p.to(device)
    metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}
    for m in metrics.values():
        m.to(device)

    # 추론 결과 저장을 위한 리스트
    all_predictions = []
    all_targets = []
    all_indices = []

    def custom_evaluate(model, data_loader, postprocessors, metrics, device):
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                # 인덱스가 포함된 배치 처리
                if len(batch) == 3:  # (images, targets, indices)
                    images, targets, indices = batch
                else:  # (images, targets) - 기존 호환성
                    images, targets = batch
                    indices = torch.arange(len(images), device=device)
                
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                indices = indices.to(device, non_blocking=True)
                
                # feature extraction
                samples = model(images)
                
                # 각 분류기에 대해 예측
                for name, postprocessor in postprocessors.items():
                    outputs = postprocessor(samples, targets)
                    metrics[name].update(outputs["preds"], outputs["target"])
                    
                    # 추론 결과 저장
                    if save_predictions and name == best_classifier_on_val:
                        preds = outputs["preds"].cpu().numpy()
                        targets_np = outputs["target"].cpu().numpy()
                        indices_np = indices.cpu().numpy()
                        
                        for i in range(len(preds)):
                            prediction_data = {
                                'index': int(indices_np[i]),
                                'prediction': int(np.argmax(preds[i])),
                                'confidence': float(np.max(preds[i])),
                                'target': int(targets_np[i]),
                                'all_probs': preds[i].tolist()
                            }
                            
                            # 파일 경로 정보 추가
                            if file_paths and int(indices_np[i]) < len(file_paths):
                                prediction_data['file_path'] = str(file_paths[int(indices_np[i])])
                            
                            all_predictions.append(prediction_data)

    # 평가 실행
    custom_evaluate(feature_model, data_loader, postprocessors, metrics, device)

    # 메트릭 계산
    results_dict_temp = {}
    for name, metric in metrics.items():
        results_dict_temp[name] = metric.compute()

    # best classifier 선택
    max_acc = -1.0
    best_name = None
    for cls_name, m in results_dict_temp.items():
        acc = m["top-1"].item()
        if (best_classifier_on_val is None and acc > max_acc) or (cls_name == best_classifier_on_val):
            max_acc = acc
            best_name = cls_name

    # 추론 결과 저장
    if save_predictions and output_dir and all_predictions:
        predictions_file = os.path.join(output_dir, f"predictions_{best_name}.json")
        # 인덱스 순으로 정렬
        all_predictions.sort(key=lambda x: x['index'])
        with open(predictions_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        logger.info(f"Predictions saved to {predictions_file}")
        #########################################################
    return {"best_classifier": {"name": best_name, "accuracy": max_acc},
            "all": {k: float(v["top-1"].item()) for k, v in results_dict_temp.items()}}


############################
# 체크포인트 기반 복원 (Train 데이터 불필요)
############################

_NAME_RE = re.compile(r"classifier_(\d+)_blocks_avgpool_(True|False)_lr_([0-9_]+)")

def _parse_classifier_name(name: str):
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"Unexpected classifier name: {name}")
    n_last_blocks = int(m.group(1))
    use_avgpool = (m.group(2) == "True")
    return n_last_blocks, use_avgpool

def build_linear_classifiers_from_ckpt(ckpt_path, device="cuda"):
    """체크포인트 내부의 분류기 파라미터 shape로부터 out_dim/num_classes를 복원하고, 이름으로 하이퍼파라미터를 복원"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)  # fvcore Checkpointer 포맷 호환

    meta = {}  # name -> (out_dim, num_classes)
    for k, v in state.items():
        # expected: "classifiers_dict.<name>.linear.weight"
        if k.endswith(".linear.weight") and k.startswith("classifiers_dict."):
            name = k.split(".")[1]  # classifiers_dict.<name>.linear.weight
            num_classes, out_dim = v.shape
            meta[name] = (out_dim, num_classes)

    if not meta:
        raise RuntimeError("No linear classifier weights found in checkpoint.")

    classifiers = {}
    for name, (out_dim, num_classes) in meta.items():
        n_last_blocks, use_avgpool = _parse_classifier_name(name)
        clf = LinearClassifier(
            out_dim=out_dim,
            use_n_blocks=n_last_blocks,
            use_avgpool=use_avgpool,
            num_classes=num_classes,
        ).to(device)
        classifiers[name] = clf

    all_cls = AllClassifiers(nn.ModuleDict(classifiers)).to(device)

    # 가중치 로드
    checkpointer = Checkpointer(all_cls)
    # resume=False 의미로 명시적 load 사용
    _ = checkpointer.load(ckpt_path)

    # 모든 분류기의 클래스 수는 동일하다고 가정
    any_name = next(iter(meta))
    _, training_num_classes = meta[any_name]
    return all_cls, training_num_classes


############################
# 보조 함수
############################

def read_best_from_metrics(metrics_path):
    """results_eval_linear.json에서 마지막 best_classifier 이름을 추출"""
    best = None
    if not os.path.isfile(metrics_path):
        return best
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "best_classifier" in obj:
                    best = obj["best_classifier"]["name"]
            except Exception:
                # "iter: N" 같은 라인 무시
                pass
    return best

def pick_checkpoint(output_dir, prefer_running=True):
    """러닝 체크포인트 우선, 없으면 가장 최근의 model_*.pth 선택"""
    run_ckpt = os.path.join(output_dir, "running_checkpoint_linear_eval.pth")
    if prefer_running and os.path.isfile(run_ckpt):
        return run_ckpt
    cands = sorted(glob(os.path.join(output_dir, "model_*.pth")))
    if cands:
        # 숫자가 큰(최근) iteration이 뒤에 오도록 정렬되어 있으므로 마지막 선택
        return cands[-1]
    raise FileNotFoundError("No checkpoint found in output_dir.")


############################
# 메인
############################

def main():
    # 학습용 setup args 포함
    p = argparse.ArgumentParser("Final Test Only", parents=[get_setup_args_parser(add_help=False)], add_help=True)
    p.add_argument("--best-classifier-name", type=str, default=None)
    p.add_argument("--no-resume", action="store_true", help="분산 초기화 생략 플래그 전달용(일반 단일 GPU 테스트 시 무시)")
    p.add_argument(
        "--classifier-checkpoint",
        dest="classifier_checkpoint",
        type=str,
        default=None,
        help="분류기 체크포인트(.pth) 경로"
    )
    p.add_argument(
        "--save-predictions",
        action="store_true",
        help="추론 결과를 인덱스와 함께 JSON 파일로 저장"
    )
    # 기본값들 (learning_rates 등은 더 이상 사용하지 않지만 get_args_parser와 호환 위해 둠)
    p.set_defaults(
        train_dataset_str="unused",  # 사용 안 함
        val_dataset_str="unused",    # 사용 안 함
        test_dataset_strs=["MyDataset:split=TEST:root=/root/vlm_classification/out_classifier/jpg_data_2:extra=/root/vlm_classification/out_classifier/jpg_data_2_extra"],
        output_dir="/root/vlm_classification/out_classifier/out/dinov2_vitg14_reg4_pretrain",
        epochs=10,
        batch_size=128,
        num_workers=8,
        epoch_length=64,
        save_checkpoint_frequency=20,
        eval_period_iterations=64,
        learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
        save_predictions=True,
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "results_eval_linear.json")

    # 모델 준비
    model, autocast_dtype = setup_and_build_model(args)
    # feature extractor with intermediate layers
    n_last_blocks_list = [1, 4]
    n_last_blocks = max(n_last_blocks_list)
    autocast_ctx = partial(torch.amp.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx).cuda()

    # 체크포인트에서 분류기 구조 및 파라미터 복원 (Train 데이터 불필요)
    ckpt_path = args.classifier_checkpoint or pick_checkpoint(args.output_dir, prefer_running=True)
    logger.info(f"Loading checkpoint: {ckpt_path}")
    linear_classifiers, training_num_classes = build_linear_classifiers_from_ckpt(ckpt_path, device="cuda")

    # best classifier 이름 결정
    best_name = args.best_classifier_name or read_best_from_metrics(metrics_path)
    if best_name is None:
        logger.warning("best_classifier 이름을 metrics에서 찾지 못했습니다. 테스트 중 다시 산출합니다.")
    else:
        logger.info(f"Best classifier from metrics: {best_name}")

    # test metric 타입/매핑 정리
    if args.test_metric_types is None:
        test_metric_types = [MetricType.MEAN_ACCURACY] * len(args.test_dataset_strs)
    else:
        assert len(args.test_metric_types) == len(args.test_dataset_strs)
        test_metric_types = args.test_metric_types

    if args.test_class_mapping_fpaths is None:
        test_class_mappings = [None] * len(args.test_dataset_strs)
    else:
        assert len(args.test_class_mapping_fpaths) == len(args.test_dataset_strs)
        test_class_mappings = []
        for pth in args.test_class_mapping_fpaths:
            if pth is not None and pth != "None":
                test_class_mappings.append(np.load(pth))
            else:
                test_class_mappings.append(None)

    # 각 테스트셋 실행
    model.eval()
    linear_classifiers.eval()
    torch.cuda.synchronize()

    all_results = {}
    for ds_str, mtype, cmap in zip(args.test_dataset_strs, test_metric_types, test_class_mappings):
        logger.info(f"[TEST] dataset={ds_str} metric={mtype}")
        test_loader = make_eval_data_loader(ds_str, args.batch_size, args.num_workers, mtype)
        res = evaluate_linear_classifiers(
            feature_model=feature_model,
            linear_classifiers=linear_classifiers,
            data_loader=test_loader,
            metric_type=mtype,
            training_num_classes=training_num_classes,
            class_mapping=cmap,
            best_classifier_on_val=best_name,
            save_predictions=args.save_predictions,
            output_dir=args.output_dir,
            dataset_str=ds_str,
        )
        # 퍼센트 스케일로 보고 싶으면 *100
        all_results[ds_str] = {
            "best_classifier_used": res["best_classifier"]["name"],
            "top1": 100.0 * float(res["best_classifier"]["accuracy"]),
            "all_top1": {k: 100.0 * v for k, v in res["all"].items()},
        }
        logger.info(f"[TEST][{ds_str}] top-1={all_results[ds_str]['top1']:.3f} (best={all_results[ds_str]['best_classifier_used']})")

    # 요약 출력
    print("\n====== FINAL TEST RESULTS ======")
    for ds, r in all_results.items():
        print(f"{ds}: top-1={r['top1']:.3f} | best={r['best_classifier_used']}")
    print("================================")


if __name__ == "__main__":
    # 단일 GPU 테스트 기준
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # 분산 사용 안하면 dinov2.distributed 초기화 없이도 동작
    main()
