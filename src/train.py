"""U-Net(ResNet34) 다중 클래스 Segmentation 학습.

사용 예:
    python src/train.py --config configs/carpet.yaml
    python src/train.py --config configs/carpet.yaml --epochs 2 --no-pretrained   # 빠른 동작 확인
    python src/train.py --config configs/carpet.yaml --batch-size 2 --accum 4     # GPU 메모리 4GB (유효 배치 8)

작은 배치 대응
  - 인코더 BatchNorm은 ImageNet 통계로 고정하고, 디코더는 정규화 없이 사용 (common.build_model)
  - gradient accumulation 으로 유효 배치 크기를 유지
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import build_model, freeze_encoder_bn, load_config, seed_everything
from dataset import TrainPatches, load_or_make_split
from evaluate import evaluate
from losses import ComboLoss, frequency_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--no-pretrained", action="store_true", help="ImageNet 가중치 없이 학습 (오프라인 테스트용)")
    ap.add_argument("--accum", type=int, help="gradient accumulation 단계 수 (유효 배치 = batch-size x accum)")
    ap.add_argument("--val-every", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.accum:
        cfg["train"]["accum_steps"] = args.accum
    seed_everything(cfg["data"]["seed"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = load_or_make_split(cfg)
    for k, v in split.items():
        by_cls = {}
        for s in v:
            by_cls[s.label] = by_cls.get(s.label, 0) + 1
        print(f"[split] {k}: {len(v)} images {dict(sorted(by_cls.items()))}")

    tcfg = cfg["train"]
    ds = TrainPatches(split["train"], cfg)
    dl = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                    num_workers=tcfg["num_workers"], pin_memory=device.type == "cuda",
                    persistent_workers=tcfg["num_workers"] > 0)

    c = cfg["model"]["num_classes"]
    cw = None
    if tcfg["loss"]["class_weights"] == "auto":
        cw = frequency_weights([m for _, m in ds.cache], c)
    elif isinstance(tcfg["loss"]["class_weights"], list):
        cw = torch.tensor(tcfg["loss"]["class_weights"], dtype=torch.float32)
    print(f"[loss] class weights: {None if cw is None else [round(x, 3) for x in cw.tolist()]}")

    model = build_model(cfg["model"], pretrained=not args.no_pretrained).to(device)
    criterion = ComboLoss(c, cw, tcfg["loss"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    accum = max(1, int(tcfg.get("accum_steps", 1)))
    steps_per_epoch = math.ceil(len(dl) / accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=tcfg["epochs"] * steps_per_epoch,
                                                pct_start=0.1)
    print(f"[train] batch {tcfg['batch_size']} x accum {accum} = effective batch {tcfg['batch_size'] * accum}, "
          f"{steps_per_epoch} optimizer steps/epoch")
    use_amp = tcfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, history = -1.0, []
    for ep in range(1, tcfg["epochs"] + 1):
        model.train()
        freeze_encoder_bn(model)
        t0, run, parts_sum = time.time(), 0.0, {"ce": 0.0, "dice": 0.0, "focal": 0.0}
        opt.zero_grad(set_to_none=True)
        steps_done = 0
        for i, (x, y) in enumerate(dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss, parts = criterion(logits.float(), y)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(dl):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if steps_done < steps_per_epoch:
                    sched.step()
                    steps_done += 1
            run += loss.item()
            for k in parts_sum:
                parts_sum[k] += parts[k]
        n = len(dl)
        log = {"epoch": ep, "loss": round(run / n, 4), **{k: round(v / n, 4) for k, v in parts_sum.items()},
               "sec": round(time.time() - t0, 1)}

        if ep % args.val_every == 0 or ep == tcfg["epochs"]:
            res = evaluate(model, split["val"], cfg, device)
            score = res["pixel"]["mIoU_defect"]
            iou = res["pixel"]["iou"]
            log.update(val_mIoU_defect=score, val_iou={k: v for k, v in iou.items()},
                       val_macro_f1=res["image"]["macro_f1"], val_ng_recall=res["image"]["ng_recall"])
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep, "val": res},
                           out_dir / "best.pth")
                log["saved"] = True
        torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep}, out_dir / "last.pth")
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))

    (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"done. best val mIoU(defect)={best:.4f} -> {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
