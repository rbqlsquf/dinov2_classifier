from enum import Enum
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from gzip import GzipFile
from io import BytesIO
from mmap import ACCESS_READ, mmap
import os
from typing import Any, Callable, List, Optional, Set, Tuple, Union
import warnings
import csv
import logging
import numpy as np

from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")


_Labels = int

_DEFAULT_MMAP_CACHE_SIZE = 16  # Warning: This can exhaust file descriptors

_Labels = int

_DEFAULT_MMAP_CACHE_SIZE = 16  # Warning: This can exhaust file descriptors

# ExtendedVisionDataset, logger, _Split, _Target 이 프로젝트 어딘가에 이미 있다고 가정
# from your_pkg import ExtendedVisionDataset, logger, _Split, _Target

class MyDataset(ExtendedVisionDataset):
    Target = Union[int]  # 라벨이 int라고 가정
    class Split(Enum):
        TRAIN = "train"
        VAL   = "val"
        TEST  = "test"

        # split별 샘플 수를 미리 알고 있다면 상수로, 모르면 dump 시 계산해서 검증만 건너뛰도록 바꿔도 됨
        @property
        def length(self) -> int:
            # 필요하면 lazy하게 npy 생성 후 len(entries)로 맞춰도 OK
            if self == MyDataset.Split.TRAIN:
                # 임시 값; 실제로는 dump_extra에서 계산된 entries 길이로 assert 제거 가능
                return 0  
            if self == MyDataset.Split.VAL:
                return 0
            if self == MyDataset.Split.TEST:
                return 0

        def get_dirname(self) -> str:
            return self.value

        # actual_index, class_id → 이미지 상대경로 규칙
        def get_image_relpath(self, actual_index: int, class_id: Optional[str]) -> str:
            # if self == MyDataset.Split.TEST:
            #     # test 이미지는 번호 기반 단일 폴더라고 가정
            #     return os.path.join(self.get_dirname(), "images", f"{actual_index:06d}.jpg")
            # else:
            # train/val은 class_id/파일명 규칙이라고 가정
            return os.path.join(self.get_dirname(), class_id, f"{actual_index:06d}.jpg")

        # 이미지 상대경로 → (class_id, actual_index)
        def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
            parts = image_relpath.split(os.sep)
            # 예: train/classA/000123.jpg  또는 test/images/000123.jpg
            split_dir = parts[0]
            # if split_dir == "test":
            #     fname = parts[-1]
            #     actual_index = int(os.path.splitext(fname)[0])
            #     class_id = ""  # test에선 없음
            # else:
            class_id = parts[1]
            fname = parts[-1]
            actual_index = int(os.path.splitext(fname)[0])
            return class_id, actual_index

    def __init__(
        self,
        *,
        split: "MyDataset.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None

    @property
    def split(self) -> "MyDataset.Split":
        return self._split

    # --- extra I/O 유틸 ---
    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        return np.load(self._get_extra_full_path(extra_path), mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(self._get_extra_full_path(extra_path), extra_array)

    # --- extra 파일 이름 규칙 ---
    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    # --- lazy getters ---
    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        # if self._split == MyDataset.Split.TEST:
        #     raise AssertionError("Class IDs are not available in TEST split")
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        # if self._split == MyDataset.Split.TEST:
        #     raise AssertionError("Class names are not available in TEST split")
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        return self._class_names

    # --- public API (ImageNet과 동일) ---
    def find_class_id(self, class_index: int) -> str:
        return str(self._get_class_ids()[class_index])

    def find_class_name(self, class_index: int) -> str:
        return str(self._get_class_names()[class_index])

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        actual_index = entries[index]["actual_index"]
        class_id = self.get_class_id(index)

        image_relpath = self.split.get_image_relpath(int(actual_index), class_id if class_id else None)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Optional[int]:
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        return str(class_id) if class_id else None

    def get_class_name(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_name = entries[index]["class_name"]
        return str(class_name) if class_name else None

    def get_image_relpath(self, index: int) -> str:
        """인덱스에 해당하는 이미지의 상대 경로를 반환합니다."""
        entries = self._get_entries()
        return str(entries[index]["image_relpath"])

    def get_image_full_path(self, index: int) -> str:
        """인덱스에 해당하는 이미지의 전체 경로를 반환합니다."""
        relpath = self.get_image_relpath(index)
        return os.path.join(self.root, relpath)

    def get_all_image_relpaths(self) -> List[str]:
        """모든 이미지의 상대 경로 리스트를 반환합니다."""
        entries = self._get_entries()
        return [str(entry["image_relpath"]) for entry in entries]

    def get_all_image_full_paths(self) -> List[str]:
        """모든 이미지의 전체 경로 리스트를 반환합니다."""
        relpaths = self.get_all_image_relpaths()
        return [os.path.join(self.root, relpath) for relpath in relpaths]

    def __len__(self) -> int:
        entries = self._get_entries()
        # 길이를 미리 모르면 이 assert는 제거해도 됨
        # assert len(entries) == self.split.length
        return len(entries)

    # --- 메타 로딩/덤프 ---
    def _load_labels(self, labels_path: str) -> List[Tuple[str, str]]:
        labels_full_path = os.path.join(self.root, labels_path)
        labels: List[Tuple[str, str]] = []
        try:
            with open(labels_full_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    class_id, class_name = row
                    labels.append((class_id, class_name))
        except OSError as e:
            raise RuntimeError(f'can not read labels file "{labels_full_path}"') from e
        return labels

    def _dump_entries(self) -> None:
        split = self.split
        # if split == MyDataset.Split.TEST:
        #     # test: 라벨 없음, images/000001.jpg 형태라고 가정
        #     images_dir = os.path.join(self.root, split.get_dirname(), "images")
        #     image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        #     sample_count = len(image_files)
        #     max_class_id_length, max_class_name_length = 0, 0
        #     dtype = np.dtype(
        #         [("actual_index", "<u4"), ("class_index", "<u4"),
        #          ("class_id", f"U{max_class_id_length}"), ("class_name", f"U{max_class_name_length}")]
        #     )
        #     entries_array = np.empty(sample_count, dtype=dtype)
        #     for i, fname in enumerate(image_files):
        #         actual_index = int(os.path.splitext(fname)[0])
        #         # TEST split에서는 class_index를 0으로 설정 (기본값)
        #         entries_array[i] = (actual_index, np.uint32(0), "", "")
        # else:
        # train/val: ImageFolder와 labels.txt를 함께 사용
        from torchvision.datasets import ImageFolder
        labels = self._load_labels("labels.txt")
        dataset_root = os.path.join(self.root, split.get_dirname())
        ds = ImageFolder(dataset_root)
        sample_count = len(ds)

        class_id_map = {idx: labels[idx][0] for idx in range(len(labels))}
        class_name_map = {cid: labels[idx][1] for idx, (cid, _) in enumerate(labels)}

        max_class_id_length = max(len(cid) for cid, _ in labels) if labels else 0
        max_class_name_length = max(len(cn) for _, cn in labels) if labels else 0
        relpaths = [os.path.relpath(p, self.root) for (p, _) in ds.samples]
        max_rel_len = max((len(r) for r in relpaths), default=0)
        dtype = np.dtype(
            [("actual_index", "<u4"), ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"), ("class_name", f"U{max_class_name_length}"),("image_relpath", f"U{max_rel_len}")]
        )
        entries_array = np.empty(sample_count, dtype=dtype)
        
        for i, (image_full_path, class_index) in enumerate(ds.samples):
            image_relpath = os.path.relpath(image_full_path, self.root)
            # class_id는 labels에서, actual_index는 파일명 숫자라고 가정
            class_id = class_id_map[class_index]
            class_name = class_name_map[class_id]
            fname = os.path.basename(image_full_path)
            actual_index = int(os.path.splitext(fname)[0])
            entries_array[i] = (actual_index, class_index, class_id, class_name, image_relpath)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _dump_class_ids_and_names(self) -> None:
        # if self.split == MyDataset.Split.TEST:
        #     return
        entries_array = self._load_extra(self._entries_path)

        max_class_index = -1
        class_id_set, class_name_map = set(), {}
        for e in entries_array:
            class_index = int(e["class_index"])
            class_id = str(e["class_id"])
            class_name = str(e["class_name"])
            max_class_index = max(max_class_index, class_index)
            class_id_set.add((class_index, class_id))
            class_name_map[class_id] = class_name

        class_count = max_class_index + 1
        max_class_id_len = max(len(cid) for _, cid in class_id_set) if class_id_set else 0
        max_class_name_len = max(len(nm) for nm in class_name_map.values()) if class_name_map else 0

        class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_len}")
        class_names_array = np.empty(class_count, dtype=f"U{max_class_name_len}")
        for class_index, class_id in class_id_set:
            class_ids_array[class_index] = class_id
            class_names_array[class_index] = class_name_map[class_id]

        logger.info(f'saving class IDs to "{self._class_ids_path}"')
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f'saving class names to "{self._class_names_path}"')
        self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        self._dump_entries()
        self._dump_class_ids_and_names()
