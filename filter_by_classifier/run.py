import os, json, re
import torch
import torch.nn as nn
from functools import partial
from PIL import Image

# === dinov2 쪽 유틸 (학습 코드와 동일) ===
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import ModelWithIntermediateLayers
from dinov2.data.transforms import make_classification_eval_transform

# ------- 학습 때 썼던 입력 구성 로직 복사 -------
def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    # x_tokens_list: List[Tuple[patch_tokens, cls_token]] from ModelWithIntermediateLayers
    intermediate_output = x_tokens_list[-use_n_blocks:]
    # class_token: shape [B, 1, C] or [B, C]? (학습 코드 기준: concat dim=-1)
    # 학습 코드에서 class_token은 이미 [B, C] 형태로 쓴다고 가정
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens mean
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, linear_input):
        return self.linear(linear_input)

# ------- 베스트 헤드 이름 파싱 -------
# 예: "classifier_4_blocks_avgpool_True_lr_0_00080"
PATTERN = r"classifier_(\d+)_blocks_avgpool_(True|False)_lr_([0-9_]+)"

def parse_best_name(name: str):
    m = re.match(PATTERN, name)
    if not m:
        raise ValueError(f"Unexpected best classifier name: {name}")
    n_blocks = int(m.group(1))
    use_avgpool = (m.group(2) == "True")
    return n_blocks, use_avgpool

# ------- 체크포인트에서 특정 헤드 가중치만 뽑기 -------
def extract_head_state_dict(ckpt_path: str, best_name: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    # fvcore.Checkpointer 포맷: {"model": state_dict, "optimizer":..., "scheduler":..., "iteration":...}
    state = obj.get("model", obj)
    prefix = f"classifiers_dict.{best_name}."
    head_sd = {}
    for k, v in state.items():
        if k.startswith(prefix):
            head_sd[k[len(prefix):]] = v  # "linear.weight", "linear.bias"
    if not head_sd:
        # 혹시 DDP 래핑으로 "module.classifiers_dict"로 저장된 경우
        prefix = f"module.classifiers_dict.{best_name}."
        for k, v in state.items():
            if k.startswith(prefix):
                head_sd[k[len(prefix):]] = v
    if not head_sd:
        keys_preview = "\n".join(list(state.keys())[:20])
        raise KeyError(f"Could not find head '{best_name}' in checkpoint.\nKeys preview:\n{keys_preview}")
    return head_sd

# ------- Predictor -------
class DinoV2LinearProbePredictor:
    def __init__(
        self,
        cfg,                         # argparse.Namespace (학습 때 setup_and_build_model에 넘기던 args와 동일한 것)
        output_dir: str,             # 학습 결과 폴더
        ckpt_name: str = "model_final.pth",
        device: str = "cuda",
        topk: int = 5,
        val_class_mapping_path: str | None = None,  # (선택) 클래스 매핑 사용 시
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.topk = topk

        # 1) 백본 로드(+freeze)
        model, autocast_dtype = setup_and_build_model(cfg)
        for p in model.parameters():
            p.requires_grad = False
        model.eval().to(self.device)

        # 2) results_eval_linear.json에서 베스트 헤드 이름 읽기
        results_path = os.path.join(output_dir, "results_eval_linear.json")
        with open(results_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        # 파일 말미의 {"best_classifier": {"name": "...", "accuracy": ...}} 라인 사용
        best_name = None
        for ln in reversed(lines):
            if ln.startswith("{") and '"best_classifier"' in ln:
                best_name = json.loads(ln)["best_classifier"]["name"]
                break
        if best_name is None:
            raise RuntimeError(f"best_classifier not found in {results_path}")

        n_blocks, use_avgpool = parse_best_name(best_name)

        # 3) feature extractor: 학습 때 max(n_last_blocks_list)=4였으므로 안전하게 그 이상으로 설정
        n_last_blocks = max(4, n_blocks)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
        self.feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx).to(self.device)
        self.feature_model.eval()

        # 4) 체크포인트에서 베스트 헤드 가중치만 추출해 shape 파악
        ckpt_path = os.path.join(output_dir, ckpt_name)
        head_sd = extract_head_state_dict(ckpt_path, best_name)

        # 5) in_dim/num_classes 결정
        #    in_dim은 실제 피처를 한 번 흘려서 계산 (샘플 한 장 필요)
        self.eval_transform = make_classification_eval_transform()
        # 더미 한 장을 만들 수 없으니, 실제 추론 때 첫 이미지를 통해 in_dim을 lazy init
        self._lazy_initialized = False
        self._lazy = {
            "best_name": best_name,
            "n_blocks": n_blocks,
            "use_avgpool": use_avgpool,
            "head_sd": head_sd,
            "val_class_mapping": None,
        }

        if val_class_mapping_path is not None and os.path.exists(val_class_mapping_path):
            import numpy as np
            self._lazy["val_class_mapping"] = torch.LongTensor(np.load(val_class_mapping_path)).to(self.device)

    @torch.no_grad()
    def _ensure_head(self, images_tensor):
        if self._lazy_initialized:
            return
        # images_tensor: [B,3,H,W]
        x_tokens_list = self.feature_model(images_tensor.to(self.device))
        linear_input = create_linear_input(
            x_tokens_list,
            use_n_blocks=self._lazy["n_blocks"],
            use_avgpool=self._lazy["use_avgpool"],
        )
        in_dim = linear_input.shape[1]
        # num_classes: 가중치 모양에서 직접 추정
        num_classes = self._lazy["head_sd"]["linear.weight"].shape[0]
        self.head = LinearClassifier(in_dim, num_classes).to(self.device)
        self.head.load_state_dict(self._lazy["head_sd"])
        self.head.eval()
        self._lazy_initialized = True

    @torch.no_grad()
    def predict_pil(self, pil_img: Image.Image):
        tensor = self.eval_transform(pil_img).unsqueeze(0)  # [1,3,H,W]
        return self.predict_tensor(tensor)

    @torch.no_grad()
    def predict_tensor(self, images_tensor: torch.Tensor):
        self._ensure_head(images_tensor)
        x_tokens_list = self.feature_model(images_tensor.to(self.device))
        linear_input = create_linear_input(
            x_tokens_list,
            use_n_blocks=self._lazy["n_blocks"],
            use_avgpool=self._lazy["use_avgpool"],
        )
        logits = self.head(linear_input)  # [B, C]
        if self._lazy["val_class_mapping"] is not None:
            logits = logits[:, self._lazy["val_class_mapping"]]

        probs = torch.softmax(logits, dim=-1)
        vals, inds = torch.topk(probs, k=min(self.topk, probs.shape[-1]), dim=-1)
        return vals.cpu(), inds.cpu()

# 1) 학습 때 썼던 args 그대로(모델/체크포인트 경로 등) 준비
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
args = get_setup_args_parser().parse_args([])  # 예시: 코드 내부에서 기본값 사용
args.model_arch = "vitb14"           # 학습 때와 동일하게
args.pretrained_weights = "/path/to/dinov2_weights.pth"  # 필요시
args.output_dir = "/path/to/output_dir"  # 학습 결과 폴더

# 2) Predictor 생성
pred = DinoV2LinearProbePredictor(
    cfg=args,
    output_dir=args.output_dir,
    ckpt_name="model_final.pth",      # 또는 원하는 체크포인트 파일명
    device="cuda",
    topk=1,
    val_class_mapping_path=None,      # 썼다면 경로 지정
)

# 3) 추론
from PIL import Image
img = Image.open("/path/to/image.jpg").convert("RGB")
top_vals, top_inds = pred.predict_pil(img)
print(top_vals, top_inds)  # 확률/클래스 인덱스