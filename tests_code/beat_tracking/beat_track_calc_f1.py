import json
import glob
import os
import numpy as np
import mir_eval

GT_DIR = ".hidden/beat_this_annotations/gtzan/annotations/beats" # Beat this! released also a set of gt annotations for lots of datasets!
PRED_DIR = ".hidden/beats"

def load_gt_beats(txt_path):
    """
    Load ground-truth beats from Harmonix-style txt.
    Columns:
        0: time (seconds)
        1: beat index in bar
        2: bar index
    """
    data = np.loadtxt(txt_path)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return data[:, 0]  # beat times only


def load_predictions(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    beat_this_beats = np.array(data["file2beats_beats"])
    our_beats = np.array(data["our_beats"])

    return beat_this_beats, our_beats


results = []

json_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.json")))

for json_path in json_files:
    stem = os.path.splitext(os.path.basename(json_path))[0]
    gt_path = os.path.join(GT_DIR, f"gtzan_{stem.replace(".","_")}.beats")

    if not os.path.exists(gt_path):
        print(f"[WARN] Missing GT for {stem}, skipping.")
        continue

    f_measure_threshold = 0.07
    try:
        gt_beats = load_gt_beats(gt_path)
    except Exception as e:
        raise Exception(f"[ERROR] Failed to load GT for {stem}: {e}")

    beat_this_beats, our_beats = load_predictions(json_path)

    matching = mir_eval.util.match_events(gt_beats, beat_this_beats, f_measure_threshold)

    if beat_this_beats.size == 0 or gt_beats.size == 0:
        p_bt = 0.0
        r_bt = 0.0
        f1_bt = 0.0
    else:
        p_bt = float(len(matching)) / len(beat_this_beats)
        r_bt = float(len(matching)) / len(gt_beats)
        f1_bt = mir_eval.util.f_measure(p_bt, r_bt)

    scores = mir_eval.beat.evaluate(gt_beats, beat_this_beats)
    amlt_bt = scores['Any Metric Level Total']
    amlc_bt = scores['Any Metric Level Continuous']

    matching = mir_eval.util.match_events(gt_beats, our_beats, f_measure_threshold)

    if our_beats.size == 0 or gt_beats.size == 0:
        p_ours = 0.0
        r_ours = 0.0
        f1_ours = 0.0
    else:
        p_ours = float(len(matching)) / len(our_beats)
        r_ours = float(len(matching)) / len(gt_beats)
        f1_ours = mir_eval.util.f_measure(p_ours, r_ours)

    scores = mir_eval.beat.evaluate(gt_beats, our_beats)
    amlt_ours = scores['Any Metric Level Total']
    amlc_ours = scores['Any Metric Level Continuous']


    results.append({
        "track": stem,
        "precision_beat_this": p_bt,
        "recall_beat_this": r_bt,
        "f1_beat_this": f1_bt,
        "amlt_beat_this": amlt_bt,
        "amlc_beat_this": amlc_bt,
        "precision_ours": p_ours,
        "recall_ours": r_ours,
        "f1_ours": f1_ours,
        "amlt_ours": amlt_ours,
        "amlc_ours": amlc_ours,
    })

    print(
        f"{stem:30s} | "
        f"beat_this F1: {f1_bt:.3f} | "
        f"our F1: {f1_ours:.3f}"
    )


# ---- Summary statistics ----
if results:
    mean_precision_beat_this = np.mean([r["precision_beat_this"] for r in results])
    mean_recall_beat_this = np.mean([r["recall_beat_this"] for r in results])
    mean_beat_this = np.mean([r["f1_beat_this"] for r in results])
    mean_amlt_beat_this = np.mean([r["amlt_beat_this"] for r in results])
    mean_amlc_beat_this = np.mean([r["amlc_beat_this"] for r in results])

    mean_precision_ours = np.mean([r["precision_ours"] for r in results])
    mean_recall_ours = np.mean([r["recall_ours"] for r in results])
    mean_ours = np.mean([r["f1_ours"] for r in results])
    mean_amlt_ours = np.mean([r["amlt_ours"] for r in results])
    mean_amlc_ours = np.mean([r["amlc_ours"] for r in results])

    print("\n=== Overall Results ===")
    print(f"Mean Precision (beat_this): {mean_precision_beat_this:.3f}")
    print(f"Mean Recall (beat_this):    {mean_recall_beat_this:.3f}")
    print(f"Mean F1 (beat_this):        {mean_beat_this:.3f}")
    print(f"Mean AMLt (beat_this):      {mean_amlt_beat_this:.3f}")
    print(f"Mean AMLc (beat_this):      {mean_amlc_beat_this:.3f}")
    print(f"Mean Precision (ours):      {mean_precision_ours:.3f}")
    print(f"Mean Recall (ours):         {mean_recall_ours:.3f}")
    print(f"Mean F1 (ours):             {mean_ours:.3f}")
    print(f"Mean AMLt (ours):           {mean_amlt_ours:.3f}")
    print(f"Mean AMLc (ours):           {mean_amlc_ours:.3f}")
