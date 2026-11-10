"""
Offline Context-Aware Clinical Chat Engine.
Analyzes user queries using rule- and context-based natural language understanding
against live benchmark metrics and dataset statistics stored in Django sessions.
Zero external API calls or network dependencies.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


MODEL_ALIASES = {
    'Logistic Regression': ['logistic regression', 'logistic', 'logreg', 'lr'],
    'SVM': ['svm', 'support vector machine', 'support vector', 'rbf'],
    'Random Forest': ['random forest', 'rf', 'random forests', 'forest'],
    'KNN': ['knn', 'k-nearest neighbors', 'k nearest neighbors', 'k-nn', 'k nearest'],
    'Naive Bayes': ['naive bayes', 'gaussian nb', 'gaussiannb', 'nb', 'bayes'],
    'Decision Tree': ['decision tree', 'dt', 'tree', 'decision trees'],
}


def extract_mentioned_models(prompt: str) -> List[str]:
    """Finds all unique model names mentioned in the prompt."""
    prompt_lower = prompt.lower()
    found = []
    for canonical_name, aliases in MODEL_ALIASES.items():
        for alias in aliases:
            # Use regex word boundary to avoid partial matches
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, prompt_lower):
                if canonical_name not in found:
                    found.append(canonical_name)
                break
    return found


def format_model_card(model: Dict[str, Any]) -> str:
    """Formats a concise markdown summary for a model."""
    return (
        f"**{model['name']}**\n"
        f"- **Accuracy**: `{model['accuracy']:.2f}%`\n"
        f"- **Precision**: `{model['precision']:.2f}%`\n"
        f"- **Recall (Sensitivity)**: `{model['recall']:.2f}%`\n"
        f"- **F1-Score**: `{model['f1']:.2f}%`\n"
        f"- **Confusion Breakdown**: TN: `{model['tn']}` | FP: `{model['fp']}` | **FN (Missed Cancers): `{model['fn']}`** | TP: `{model['tp']}`"
    )


def respond_to_query(prompt: str, session_data: Optional[Dict[str, Any]]) -> str:
    """
    Evaluates the user's prompt against live session benchmark data.
    """
    if not prompt or not prompt.strip():
        return "Please ask a question about your benchmarked algorithms, dataset, or clinical safety metrics."

    query = prompt.strip().lower()

    # Check if session has benchmark data
    if not session_data or 'results' not in session_data:
        return (
            "⚠️ **No Active Benchmark Data Found**\n\n"
            "Please upload a clinical CSV dataset or click **'Download Sample Format .CSV'** to populate and run "
            "the 6-model benchmark pipeline. Once evaluated, you can ask me detailed questions about accuracy, "
            "clinical safety, confusion matrices, or model comparisons!"
        )

    results: List[Dict[str, Any]] = session_data.get('results', [])
    dataset_stats: Dict[str, Any] = session_data.get('dataset_stats', {})
    best_model: Dict[str, Any] = session_data.get('best_model', {})
    worst_model: Dict[str, Any] = session_data.get('worst_model', {})
    safest_model: Dict[str, Any] = session_data.get('safest_model', {})

    model_lookup = {r['name']: r for r in results}
    mentioned = extract_mentioned_models(query)

    # ------------------------------------------------------------------
    # 1. Head-to-Head Comparison (Two models mentioned)
    # ------------------------------------------------------------------
    if len(mentioned) >= 2 or ('compare' in query and len(mentioned) == 2) or ('vs' in query and len(mentioned) == 2):
        m1_name, m2_name = mentioned[0], mentioned[1]
        m1 = model_lookup.get(m1_name)
        m2 = model_lookup.get(m2_name)

        if m1 and m2:
            acc_diff = round(abs(m1['accuracy'] - m2['accuracy']), 2)
            if m1['accuracy'] > m2['accuracy']:
                acc_winner = f"**{m1_name}** leads by `{acc_diff}%` accuracy."
            elif m2['accuracy'] > m1['accuracy']:
                acc_winner = f"**{m2_name}** leads by `{acc_diff}%` accuracy."
            else:
                acc_winner = "Both models have identical accuracy."

            if m1['fn'] < m2['fn']:
                safety_winner = f"🛡️ **{m1_name}** is clinically safer with only `{m1['fn']}` False Negatives vs `{m2['fn']}` for {m2_name}."
            elif m2['fn'] < m1['fn']:
                safety_winner = f"🛡️ **{m2_name}** is clinically safer with only `{m2['fn']}` False Negatives vs `{m1['fn']}` for {m1_name}."
            else:
                safety_winner = f"Both models share identical False Negatives (`{m1['fn']}`)."

            return (
                f"### ⚖️ Head-to-Head Comparison: {m1_name} vs {m2_name}\n\n"
                f"| Metric | {m1_name} | {m2_name} | Difference |\n"
                f"| :--- | :--- | :--- | :--- |\n"
                f"| **Accuracy** | `{m1['accuracy']:.2f}%` | `{m2['accuracy']:.2f}%` | `{acc_diff}%` |\n"
                f"| **Precision** | `{m1['precision']:.2f}%` | `{m2['precision']:.2f}%` | `{abs(m1['precision'] - m2['precision']):.2f}%` |\n"
                f"| **Recall (Sensitivity)** | `{m1['recall']:.2f}%` | `{m2['recall']:.2f}%` | `{abs(m1['recall'] - m2['recall']):.2f}%` |\n"
                f"| **F1-Score** | `{m1['f1']:.2f}%` | `{m2['f1']:.2f}%` | `{abs(m1['f1'] - m2['f1']):.2f}%` |\n"
                f"| **False Negatives (FN)** | `{m1['fn']}` | `{m2['fn']}` | `{abs(m1['fn'] - m2['fn'])}` |\n"
                f"| **False Positives (FP)** | `{m1['fp']}` | `{m2['fp']}` | `{abs(m1['fp'] - m2['fp'])}` |\n\n"
                f"**Clinical Verdict**:\n"
                f"- **Accuracy Performance**: {acc_winner}\n"
                f"- **Diagnostic Safety**: {safety_winner}\n"
            )

    # ------------------------------------------------------------------
    # 2. Single Model Deep Dive
    # ------------------------------------------------------------------
    if len(mentioned) == 1 and not any(k in query for k in ['best', 'worst', 'safest', 'ranking', 'compare']):
        model_name = mentioned[0]
        m = model_lookup.get(model_name)
        if m:
            is_best = (m['name'] == best_model.get('name'))
            is_safest = (m['name'] == safest_model.get('name'))
            badges = []
            if is_best:
                badges.append("🏆 Highest Accuracy")
            if is_safest:
                badges.append("🛡️ Safest Clinical Model")
            badge_str = f" ({', '.join(badges)})" if badges else ""

            return (
                f"### 🔍 Deep Dive: {model_name}{badge_str}\n\n"
                f"- **Overall Accuracy**: `{m['accuracy']:.2f}%`\n"
                f"- **Precision**: `{m['precision']:.2f}%` (percentage of predicted malignancies that were actual tumors)\n"
                f"- **Recall (Sensitivity)**: `{m['recall']:.2f}%` (percentage of actual malignancies correctly flagged)\n"
                f"- **F1-Score**: `{m['f1']:.2f}%` (harmonic balance of precision and recall)\n\n"
                f"**Confusion Matrix Counts (Test Set: {dataset_stats.get('test_samples', 0)} samples)**:\n"
                f"- **True Negatives (TN)**: `{m['tn']}` benign cases correctly identified\n"
                f"- **False Positives (FP)**: `{m['fp']}` benign cases mistakenly classified as malignant\n"
                f"- **False Negatives (FN)**: `{m['fn']}` malignant cases dangerously missed as benign\n"
                f"- **True Positives (TP)**: `{m['tp']}` malignant tumors correctly detected\n\n"
                f"**Clinical Assessment**: "
                f"{'Exceptional diagnostic safety with minimal missed malignant cases.' if m['fn'] <= 2 else 'Moderate False Negative risk; clinical second-opinion recommended for negative predictions.'}"
            )

    # ------------------------------------------------------------------
    # 3. Best Model & Safest Clinical Model
    # ------------------------------------------------------------------
    if any(k in query for k in ['best', 'top', 'winner', 'highest', 'champion', 'lead']):
        response = (
            f"### 🏆 Top Performing Algorithm: **{best_model.get('name')}**\n\n"
            f"- **Accuracy**: `{best_model.get('accuracy', 0):.2f}%`\n"
            f"- **Precision**: `{best_model.get('precision', 0):.2f}%`\n"
            f"- **Recall**: `{best_model.get('recall', 0):.2f}%`\n"
            f"- **F1-Score**: `{best_model.get('f1', 0):.2f}%`\n"
            f"- **False Negatives (FN)**: `{best_model.get('fn', 0)}` missed cases\n\n"
        )
        if safest_model.get('name') != best_model.get('name'):
            response += (
                f"⚠️ **Note on Clinical Safety**:\n"
                f"While **{best_model.get('name')}** achieved the highest overall accuracy, "
                f"**{safest_model.get('name')}** is the **safest clinical choice** with only "
                f"`{safest_model.get('fn')}` False Negatives (higher Recall `{safest_model.get('recall'):.2f}%`). "
                f"In oncology diagnostics, minimizing False Negatives takes precedence over raw accuracy."
            )
        else:
            response += (
                f"🛡️ **{best_model.get('name')}** is ALSO the **safest clinical choice**, "
                f"holding the lowest False Negative count (`{best_model.get('fn')}` missed cases)."
            )
        return response

    if any(k in query for k in ['safe', 'safest', 'safety', 'clinical safety', 'fewest false negative', 'minimize false negative']):
        return (
            f"### 🛡️ Safest Clinical Algorithm: **{safest_model.get('name')}**\n\n"
            f"In cancer diagnostics, a **False Negative (FN)** occurs when a malignant tumor is misdiagnosed as benign, "
            f"which can delay critical treatment and endanger patient lives.\n\n"
            f"- **Safest Model**: **{safest_model.get('name')}**\n"
            f"- **False Negatives**: `{safest_model.get('fn')}` (Lowest in benchmark)\n"
            f"- **Recall (Clinical Sensitivity)**: `{safest_model.get('recall'):.2f}%`\n"
            f"- **Overall Accuracy**: `{safest_model.get('accuracy'):.2f}%`\n"
            f"- **True Positives Caught**: `{safest_model.get('tp')}`\n\n"
            f"**Recommendation**: When deploying for clinical decision support, prioritize **{safest_model.get('name')}** "
            f"to protect patients from undetected malignancies."
        )

    # ------------------------------------------------------------------
    # 4. Worst / Lowest Performing Model
    # ------------------------------------------------------------------
    if any(k in query for k in ['worst', 'lowest', 'least accurate', 'poor', 'bottom', 'weakest']):
        return (
            f"### ⚠️ Lowest Performing Algorithm: **{worst_model.get('name')}**\n\n"
            f"- **Accuracy**: `{worst_model.get('accuracy', 0):.2f}%`\n"
            f"- **Precision**: `{worst_model.get('precision', 0):.2f}%`\n"
            f"- **Recall**: `{worst_model.get('recall', 0):.2f}%`\n"
            f"- **F1-Score**: `{worst_model.get('f1', 0):.2f}%`\n"
            f"- **False Negatives (Missed Cancers)**: `{worst_model.get('fn', 0)}`\n"
            f"- **False Positives (False Alarms)**: `{worst_model.get('fp', 0)}`\n\n"
            f"This model underperformed relative to the benchmark suite and is not recommended as the primary classifier."
        )

    # ------------------------------------------------------------------
    # 5. Accuracy Ranking across all 6 models
    # ------------------------------------------------------------------
    if any(k in query for k in ['rank', 'ranking', 'leaderboard', 'order', 'all models', 'overall', 'scores', 'table']):
        sorted_models = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        rows = []
        for i, m in enumerate(sorted_models, 1):
            tag = " 🏆 Best" if i == 1 else (" ⚠️ Lowest" if i == len(sorted_models) else "")
            rows.append(
                f"{i}. **{m['name']}**{tag} — Accuracy: `{m['accuracy']:.2f}%` | Recall: `{m['recall']:.2f}%` | FN: `{m['fn']}`"
            )
        return "### 📊 Benchmark Accuracy Leaderboard\n\n" + "\n".join(rows)

    # ------------------------------------------------------------------
    # 6. False Negatives, False Positives, and Confusion Matrix Explanations
    # ------------------------------------------------------------------
    if any(k in query for k in ['false negative', 'false positive', 'fn', 'fp', 'confusion matrix', 'confusion', 'tn', 'tp', 'matrix']):
        return (
            "### 🧬 Confusion Matrix & Clinical Significance\n\n"
            "In diagnostic classification, error types have asymmetric clinical consequences:\n\n"
            "1. **False Negative (FN) — Dangerous / Critical Risk**:\n"
            "   - Patient **has malignant cancer**, but the algorithm predicts **Benign**.\n"
            "   - *Consequence*: The patient is sent home without urgent treatment, allowing disease progression.\n"
            "   - *Optimization Target*: We maximize **Recall / Sensitivity** to push FN to zero.\n\n"
            "2. **False Positive (FP) — Psychological / Economic Burden**:\n"
            "   - Patient has a **benign condition**, but the algorithm predicts **Malignant**.\n"
            "   - *Consequence*: Patient undergoes follow-up biopsies and emotional stress, but tumor is not missed.\n\n"
            "3. **True Negative (TN)**: Healthy/benign cases correctly classified as benign.\n"
            "4. **True Positive (TP)**: Malignant tumors correctly diagnosed as malignant.\n\n"
            "💡 *Tip: Check the Confusion Matrices 2x3 grid to inspect individual cell counts for each classifier.*"
        )

    # ------------------------------------------------------------------
    # 7. Dataset Statistics & Class Distribution
    # ------------------------------------------------------------------
    if any(k in query for k in ['dataset', 'data', 'stat', 'rows', 'samples', 'distribution', 'class', 'benign', 'malignant', 'split']):
        total = dataset_stats.get('total_rows', 0)
        benign = dataset_stats.get('benign_count', 0)
        malignant = dataset_stats.get('malignant_count', 0)
        train_s = dataset_stats.get('train_samples', 0)
        test_s = dataset_stats.get('test_samples', 0)
        b_pct = (benign / total * 100) if total else 0
        m_pct = (malignant / total * 100) if total else 0

        return (
            f"### 📋 Dataset Overview & Class Distribution\n\n"
            f"- **Total Valid Clinical Records**: `{total}` rows\n"
            f"- **Benign Cases (0)**: `{benign}` ({b_pct:.1f}%)\n"
            f"- **Malignant Cases (1)**: `{malignant}` ({m_pct:.1f}%)\n"
            f"- **Train Set (80% Stratified)**: `{train_s}` samples\n"
            f"- **Test Set (20% Stratified)**: `{test_s}` samples\n"
            f"- **Clinical Features Evaluated**: `10` continuous numeric variables (`radius_mean`, `texture_mean`, etc.)"
        )

    # ------------------------------------------------------------------
    # 8. Fallback / General Capabilities Overview
    # ------------------------------------------------------------------
    return (
        f"👋 **I am your Offline Clinical ML Benchmark Assistant.**\n\n"
        f"You can ask me anything about the live evaluation results, including:\n"
        f"- **Best Model**: *'What is the best performing model?'*\n"
        f"- **Clinical Safety**: *'Which model has the fewest False Negatives?'*\n"
        f"- **Head-to-head Comparison**: *'Compare Logistic Regression and SVM'* or *'Random Forest vs KNN'*\n"
        f"- **Single Model Breakdown**: *'Tell me about Naive Bayes'* or *'Decision Tree breakdown'*\n"
        f"- **Diagnostic Concepts**: *'Explain False Negatives and the confusion matrix'*\n"
        f"- **Data Distribution**: *'What are the dataset statistics?'*\n"
        f"- **Leaderboard**: *'Rank all models by accuracy'*"
    )
