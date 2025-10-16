import numbers


def compute_metrics_from_cm(tn, fp, fn, tp):
    fpr = fp / (fp + tn) if (fp + tn) > 0 else "nan"
    fnr = fn / (tp + fn) if (tp + fn) > 0 else "nan"

    tnr_spec = tn / (tn + fp) if (tn + fp) > 0 else "nan"
    tpr_recall_sens = tp / (tp + fn) if (tp + fn) > 0 else "nan"

    ppv_prec = tp / (tp + fp) if (tp + fp) > 0 else "nan"
    npv = tn / (tn + fn) if (tn + fn) > 0 else "nan"

    acc = (tp + tn) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else "nan"

    if ppv_prec is not "nan" and tpr_recall_sens is not "nan":
        f1_score = 2 * ppv_prec * tpr_recall_sens / (ppv_prec + tpr_recall_sens)
    else:
        f1_score = "nan"

    return {
        "tn": tn.item() if tn is not "nan" else "nan",
        "fp": fp.item() if fp is not "nan" else "nan",
        "fn": fn.item() if fn is not "nan" else "nan",
        "tp": tp.item() if tp is not "nan" else "nan",
        "tp_rate_recall_sens": tpr_recall_sens.item() if tpr_recall_sens is not "nan" else "nan",
        "tn_rate_spec": tnr_spec.item() if tnr_spec is not "nan" else "nan",
        "accuracy": acc.item() if acc is not "nan" else "nan",
        "fp_rate": fpr.item() if fpr is not "nan" else "nan",
        "fn_rate": fnr.item() if fnr is not "nan" else "nan",
        "ppv_precision": ppv_prec.item() if ppv_prec is not "nan" else "nan",
        "npv": npv.item() if npv is not "nan" else "nan",
        "f1_score": f1_score.item() if f1_score is not "nan" else "nan",
    }


def get_keys():
    return ["tn", "fp", "fn", "tp", "tp_rate_recall_sens", "tn_rate_spec", "accuracy", "fp_rate", "fn_rate",
            "ppv_precision", "npv", "f1_score", ]
