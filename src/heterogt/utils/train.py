import torch
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_recall_fscore_support
import numpy as np
from contextlib import nullcontext
import pandas as pd

PHENO_ORDER = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and related",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Pneumonia",
    "Septicemia (except in labor)",
]

def train_with_early_stopping(model, train_dataloader, val_dataloader, test_dataloader,
                              optimizer, loss_fn, device, early_stop_patience, task_type, epochs, dec_loss_lambda = 0, 
                              val_long_seq_idx=None, test_long_seq_idx=None, eval_metric="prauc", return_model=False):

    # ---- Device & AMP switch ----
    device_type = device.type  # 'cuda' | 'cpu' | 'mps'
    use_amp = (device_type == "cuda")   # Enable AMP/GradScaler only on CUDA to avoid CPU/MPS warnings
    scaler = GradScaler(enabled=use_amp)

    best_score = 0.0
    best_val_metric = None
    best_test_metric = None
    best_test_long_seq_metric = None
    best_model_state = deepcopy(model.state_dict())
    epochs_no_improve = 0

    # Choose the proper autocast context (use nullcontext on CPU/MPS, or set enabled=False manually)
    amp_ctx = (autocast() if use_amp else nullcontext())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch:03d}"
        )

        for step, batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)

            # Move to target device
            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            labels = batch[-1].float()

            try:
                with amp_ctx:
                    preds, dec_loss = model(*batch[:-1])
                    task_loss = loss_fn(preds.view(-1), labels.view(-1))
                    loss = task_loss + dec_loss_lambda * dec_loss

                if use_amp:
                    # AMP path
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)  # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # FP32 path (CPU/MPS)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                running_loss += loss.item()
                num_steps = step + 1
                progress_bar.set_postfix({"loss": f"{running_loss / num_steps:.4f}"})

            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    print(f"[OOM Warning] Skipping batch {step} due to OOM.")
                    if device_type == "cuda":
                        torch.cuda.empty_cache()
                    elif device_type == "mps":
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                    continue
                else:
                    raise

        epoch_loss = running_loss / max(1, (step + 1))
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        # Validation + Early stopping
        (best_score, best_val_metric, best_test_metric, best_test_long_seq_metric, best_model_state,
         epochs_no_improve, early_stop_triggered) = evaluate_and_early_stop(
            model, val_dataloader, test_dataloader, device, task_type,
            val_long_seq_idx, test_long_seq_idx, eval_metric,
            best_score, best_val_metric, best_test_metric, best_test_long_seq_metric, 
            best_model_state, epochs_no_improve, early_stop_patience
        )
        if early_stop_triggered:
            break

    print("\nBest validation performance:")
    print(best_val_metric)
    print("Corresponding test performance:")
    print(best_test_metric)
    if best_test_long_seq_metric is not None:
        print("Corresponding test-long performance:")
        print(best_test_long_seq_metric)

    model.load_state_dict(best_model_state)
    if return_model:
        return (best_test_metric, best_test_long_seq_metric, model)
    else:
        return best_test_metric, best_test_long_seq_metric


def evaluate_and_early_stop(model, val_dataloader, test_dataloader, device, task_type,
                                  val_long_seq_idx, test_long_seq_idx, eval_metric,
                                  best_score, best_val_metric, best_test_metric, best_test_long_seq_metric, 
                                  best_model_state, epochs_no_improve, early_stop_patience):
    """
    Run evaluation on the validation and test sets, and perform early-stopping checks.
    Returns:
        - best_score
        - best_val_metric
        - best_test_metric
        - best_test_long_seq_metric
        - best_model_state
        - epochs_no_improve
        - early_stop_triggered (bool)
    """
    # --- Evaluation ---
    if val_long_seq_idx is not None:
        val_metric, val_long_seq_metric = evaluate(model, val_dataloader, device, task_type, val_long_seq_idx)
    else:
        val_metric = evaluate(model, val_dataloader, device, task_type)
        val_long_seq_metric = None

    if test_long_seq_idx is not None:
        test_metric, test_long_seq_metric = evaluate(model, test_dataloader, device, task_type, test_long_seq_idx)
    else:
        test_metric = evaluate(model, test_dataloader, device, task_type)
        test_long_seq_metric = None
        
    if task_type != "binary":
        per_class_val_df = val_metric["per_class"]
        val_metric = val_metric["global"]
        per_class_test_df = test_metric["per_class"]
        test_metric = test_metric["global"]
        
        if val_long_seq_metric is not None:
            per_class_val_long_seq_df = val_long_seq_metric["per_class"]
            val_long_seq_metric = val_long_seq_metric["global"]
            
        if test_long_seq_metric is not None:
            per_class_test_long_seq_df = test_long_seq_metric["per_class"]
            test_long_seq_metric = test_long_seq_metric["global"]
            

    print(f"Validation: {val_metric}")
    print(f"Test:      {test_metric}\n")
    if val_long_seq_metric is not None:
        print(f"Validation-long: {val_long_seq_metric}")
    if test_long_seq_metric is not None:
        print(f"Test-long: {test_long_seq_metric}\n")

    # --- Early Stopping ---
    current_score = val_metric[eval_metric]
    early_stop_triggered = False

    if current_score > best_score:
        best_score = current_score
        best_val_metric = val_metric if task_type == "binary" else {"global": val_metric, "per_class": per_class_val_df}
        best_test_metric = test_metric if task_type == "binary" else {"global": test_metric, "per_class": per_class_test_df}
        best_test_long_seq_metric = test_long_seq_metric if task_type == "binary" else {"global": test_long_seq_metric, "per_class": per_class_test_long_seq_df}
        best_model_state = deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print(f"\nEarly stopping triggered (no improvement for {early_stop_patience} epochs).")
            early_stop_triggered = True

    return best_score, best_val_metric, best_test_metric, best_test_long_seq_metric, best_model_state, epochs_no_improve, early_stop_triggered

def run_multilabel_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
    predictions_are_logits: bool = True,
):

    assert predictions.ndim == 2 and labels.ndim == 2, "predictions/labels must be [B, C]"
    assert predictions.shape == labels.shape, "shape mismatch [B, C]"
    B, C = predictions.shape

    # 1) Continuous scores (for AUC / PR-AUC)
    with torch.no_grad():
        if predictions_are_logits:
            # Upgrade to fp32 when CPU half has no sigmoid implementation
            if predictions.device.type == "cpu" and predictions.dtype == torch.float16:
                scores_t = torch.sigmoid(predictions.float())
            else:
                scores_t = torch.sigmoid(predictions)
        else:
            # Already [0,1] probabilities
            scores_t = predictions

        # 2) Thresholding (for P/R/F1)
        # Always apply threshold in probability space: when logits are given, scores_t is sigmoid(logits)
        # If you prefer strict logits>0 criterion, fix threshold at 0.5 (equivalent)
        y_pred_t = (scores_t >= threshold).to(torch.int32)

    # Convert to numpy
    scores = scores_t.cpu().numpy()                  # continuous scores
    y_pred = y_pred_t.cpu().numpy().astype(np.int32) # binary predictions
    y_true = labels.cpu().numpy().astype(np.int32)   # ground-truth labels

    # 3) Macro-averaged Precision/Recall/F1 (thresholded 0/1)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # 4) Per-class Precision/Recall/F1
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # 5) Per-class AUC / PR-AUC (continuous scores; single-class case -> NaN)
    aucs, praucs = [], []
    for c in range(C):
        yt, ys = y_true[:, c], scores[:, c]
        if yt.max() == yt.min():
            aucs.append(np.nan)
            praucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(yt, ys))
            prec_curve, rec_curve, _ = precision_recall_curve(yt, ys)
            praucs.append(auc(rec_curve, prec_curve))

    # 6) Percentage formatting
    def pct_scalar(x):
        if x is None:
            return None
        try:
            return None if np.isnan(x) else round(float(x) * 100.0, 4)
        except TypeError:
            return round(float(x) * 100.0, 4)

    def pct_array(arr):
        out = []
        for v in arr:
            if isinstance(v, float) and np.isnan(v):
                out.append(None)
            else:
                out.append(round(float(v) * 100.0, 4))
        return out

    global_metrics = {
        "precision": pct_scalar(p_macro),
        "recall":    pct_scalar(r_macro),
        "f1":        pct_scalar(f1_macro),
        "auc":       pct_scalar(np.nanmean(aucs)) if np.any(~np.isnan(aucs)) else None,
        "prauc":     pct_scalar(np.nanmean(praucs)) if np.any(~np.isnan(praucs)) else None,
    }

    assert len(PHENO_ORDER) == C, "len(PHENO_ORDER) must equal C"

    per_class_df = pd.DataFrame({
        "precision": pct_array(p),
        "recall":    pct_array(r),
        "f1":        pct_array(f1),
        "auc":       pct_array(aucs),
        "prauc":     pct_array(praucs),
    }, index=PHENO_ORDER)

    return global_metrics, per_class_df


def run_binary_metrics(predictions, labels):
    predictions = predictions.view(-1)
    labels = labels.view(-1).float()
    scores = predictions.numpy()
    binary_preds = (predictions > 0).float().numpy()  # logit > 0 ≈ prob > 0.5

    tp = (binary_preds * labels.numpy()).sum()
    precision = tp / (binary_preds.sum() + 1e-8)
    recall = tp / (labels.sum().item() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    rocauc = roc_auc_score(labels.numpy(), scores)
    prec_curve, rec_curve, _ = precision_recall_curve(labels.numpy(), scores)
    prauc = auc(rec_curve, prec_curve)
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": rocauc,
        "prauc": prauc,
    }
    for m, v in metrics.items():
        metrics[m] = round(v * 100, 4)
    return metrics

@torch.no_grad()
def evaluate(model, dataloader, device, task_type, long_seq_idx=None):
    model.eval()

    # Enable autocast only on CUDA to avoid CPU/MPS warnings
    device_type = device.type  # 'cuda' | 'cpu' | 'mps'
    use_amp = (device_type == "cuda")
    amp_ctx = autocast() if use_amp else nullcontext()

    all_preds, all_labels = [], []

    for _, batch in enumerate(tqdm(dataloader, desc="Running inference")):
        # Move to device
        batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        labels = batch[-1]

        with amp_ctx:
            output = model(*batch[:-1])

        # Compatible with tensor / tuple / list
        preds = output[0] if isinstance(output, (tuple, list)) else output

        all_preds.append(preds)
        all_labels.append(labels)

    predictions = torch.cat(all_preds, dim=0).detach().cpu()
    labels = torch.cat(all_labels, dim=0).detach().cpu()

    # If long_seq_idx is provided, ensure it can be used for tensor indexing
    def _select_long_seq(t):
        if long_seq_idx is None:
            return None
        if isinstance(long_seq_idx, torch.Tensor):
            idx = long_seq_idx
        else:
            idx = torch.as_tensor(long_seq_idx, dtype=torch.long)
        return t[idx]

    if task_type == "binary":
        results = run_binary_metrics(predictions, labels)
        if long_seq_idx is not None:
            long_seq_results = run_binary_metrics(
                _select_long_seq(predictions), _select_long_seq(labels)
            )
            return results, long_seq_results
        else:
            return results
    else:
        results, per_class_df = run_multilabel_metrics(predictions, labels)
        if long_seq_idx is not None:
            long_seq_results, long_seq_per_class_df = run_multilabel_metrics(
                _select_long_seq(predictions), _select_long_seq(labels)
            )
            return {"global": results, "per_class": per_class_df}, {"global": long_seq_results, "per_class": long_seq_per_class_df}
        else:
            return {"global": results, "per_class": per_class_df}