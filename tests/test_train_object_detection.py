# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""End-to-end tests: object detection finetuning -> artifacts -> resume -> load."""

import pytest
import torch

from autovision import load_task_model as load_model, train_object_detection
from tests.conftest import create_image


def make_coco_dataset(tmp_path, num_images: int = 8):
    images_dir = tmp_path / "images"
    rows, annotations, categories = [], [], [
        {"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}
    ]
    for index in range(num_images):
        path = create_image(images_dir / f"img_{index:03d}.jpg", size=(64, 64), seed=index)
        rows.append({"id": index + 1, "file_name": path.name, "width": 64, "height": 64})
        annotations.append({
            "id": index + 1, "image_id": index + 1,
            "category_id": 1 + (index % 2),
            "bbox": [8.0, 8.0, 32.0, 32.0], "area": 1024.0, "iscrowd": 0,
        })
    payload = {"images": rows, "annotations": annotations, "categories": categories}
    train_json = tmp_path / "train.json"
    train_json.write_text(__import__("json").dumps(payload))
    val_json = tmp_path / "val.json"
    val_payload = {"images": rows[:2], "annotations": annotations[:2], "categories": categories}
    val_json.write_text(__import__("json").dumps(val_payload))
    return train_json, val_json, images_dir


def data_config(train_json, val_json, images_dir):
    return {
        "format": "coco",
        "train": {"annotations": str(train_json), "images": str(images_dir)},
        "val": {"annotations": str(val_json), "images": str(images_dir)},
    }


def model_args():
    return {
        "backbone": "torchvision/resnet18",
        "image_size": (64, 64),
        "hidden_dim": 32,
        "num_queries": 10,
        "dec_layers": 2,
        "num_heads": 4,
        "num_top_queries": 10,
    }


def transform_args():
    # Small, fast recipe: no scale jitter (fixed 64px), short windows.
    return {
        "scale_jitter": None,
        "photometric_distort_prob": 0.5,
        "random_zoom_out_prob": None,
        "random_iou_crop_prob": None,
    }


@pytest.fixture(scope="module")
def coco(tmp_path_factory):
    return make_coco_dataset(tmp_path_factory.mktemp("coco"))


class TestTrainObjectDetection:
    def test_full_run_produces_all_artifacts(self, tmp_path, coco):
        train_json, val_json, images_dir = coco
        result = train_object_detection(
            out=tmp_path / "det1",
            data=data_config(train_json, val_json, images_dir),
            steps=4,
            batch_size=4,
            num_workers=0,
            model_args=model_args(),
            transform_args=transform_args(),
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.steps_run == 4
        assert result.checkpoint_path and result.checkpoint_path.exists()
        assert result.exported_model_path and result.exported_model_path.exists()
        assert (tmp_path / "det1" / "checkpoints" / "last.ckpt").exists()
        assert (tmp_path / "det1" / "metrics.jsonl").exists()
        assert result.classes == {0: "cat", 1: "dog"}
        # Validation ran twice; best checkpoint written.
        assert result.best_checkpoint_path and result.best_checkpoint_path.exists()
        assert "val_metric/map" in result.best_metrics

    def test_exported_model_loads_and_predicts(self, tmp_path, coco):
        train_json, val_json, images_dir = coco
        result = train_object_detection(
            out=tmp_path / "det2",
            data=data_config(train_json, val_json, images_dir),
            steps=2,
            batch_size=4,
            num_workers=0,
            model_args=model_args(),
            transform_args=transform_args(),
        )
        model = load_model(result.exported_model_path, device="cpu")
        assert not model.training
        image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        prediction = model.predict(image, threshold=0.0)
        assert set(prediction) == {"labels", "bboxes", "scores"}

    def test_resume_interrupted(self, tmp_path, coco):
        train_json, val_json, images_dir = coco
        common = dict(
            data=data_config(train_json, val_json, images_dir),
            batch_size=4, num_workers=0,
            model_args=model_args(), transform_args=transform_args(),
        )
        train_object_detection(out=tmp_path / "det3", steps=2, **common)
        result = train_object_detection(
            out=tmp_path / "det3", steps=4, resume_interrupted=True, **common
        )
        assert result.steps_run == 2  # continued, not restarted
        from autovision._checkpoint import Checkpoint

        metadata = Checkpoint.from_path(
            tmp_path / "det3" / "checkpoints" / "last.ckpt"
        ).autovision
        assert metadata.epoch == 3  # final step index

    def test_weights_only_checkpoint(self, tmp_path, coco):
        train_json, val_json, images_dir = coco
        common = dict(
            data=data_config(train_json, val_json, images_dir),
            batch_size=4, num_workers=0,
            model_args=model_args(), transform_args=transform_args(),
        )
        first = train_object_detection(out=tmp_path / "det4a", steps=2, **common)
        train_object_detection(
            out=tmp_path / "det4b", steps=2, checkpoint=first.checkpoint_path, **common
        )
        assert (tmp_path / "det4b" / "checkpoints" / "last.ckpt").exists()

    def test_default_transform_args_end_to_end(self, tmp_path, coco):
        """Regression (P0): default transform args must train without
        crashing (RandomIoUCrop probability wrapping + collate contract)."""
        train_json, val_json, images_dir = coco
        result = train_object_detection(
            out=tmp_path / "det6",
            data=data_config(train_json, val_json, images_dir),
            steps=3,
            batch_size=4,
            num_workers=0,
            model_args=model_args(),
            # NO transform_args override: exercises the full default recipe
            trainer_args={"val_every_num_steps": 2},
        )
        assert result.steps_run == 3
        assert result.checkpoint_path.exists()

    def test_unknown_format_raises(self, tmp_path, coco):
        train_json, val_json, images_dir = coco
        with pytest.raises(ValueError, match="Unknown detection data format"):
            train_object_detection(
                out=tmp_path / "det5",
                data={"format": "pascal_voc"},
                steps=2, batch_size=4, num_workers=0,
                model_args=model_args(), transform_args=transform_args(),
            )
