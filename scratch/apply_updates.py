import os
import re

# 1. Update src/private_llm_analytics/planner/query.py
query_path = 'src/private_llm_analytics/planner/query.py'
with open(query_path, 'r', encoding='utf-8') as f:
    q_content = f.read().replace('\r\n', '\n')

old_q = """    seed: Optional[int] = None"""
new_q = """    seed: Optional[int] = None
    k: Optional[int] = None"""
if old_q in q_content and "k: Optional[int] = None" not in q_content:
    q_content = q_content.replace(old_q, new_q)
    with open(query_path, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(q_content.replace('\n', '\r\n'))
    print("Updated query.py")

# 2. Update src/private_llm_analytics/service/analytics.py
svc_path = 'src/private_llm_analytics/service/analytics.py'
with open(svc_path, 'r', encoding='utf-8') as f:
    s_content = f.read().replace('\r\n', '\n')

old_compute_sig = """    def compute(
        self,
        metric: str,
        recipient: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        mechanism: Optional[str] = None,
        clip_norm: Optional[float] = None,
        include_exact: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:"""

new_compute_sig = """    def compute(
        self,
        metric: str,
        recipient: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        mechanism: Optional[str] = None,
        clip_norm: Optional[float] = None,
        include_exact: bool = False,
        k: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:"""

old_query_init = """        query = AnalyticsQuery(
            metric=metric,
            recipient=recipient,
            window_start=start,
            window_end=end,
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism,
        )"""

new_query_init = """        query = AnalyticsQuery(
            metric=metric,
            recipient=recipient,
            window_start=start,
            window_end=end,
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism,
            k=k,
        )"""

if old_compute_sig in s_content:
    s_content = s_content.replace(old_compute_sig, new_compute_sig)
if old_query_init in s_content:
    s_content = s_content.replace(old_query_init, new_query_init)
with open(svc_path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(s_content.replace('\n', '\r\n'))
print("Updated analytics.py")

# 3. Update src/private_llm_analytics/planner/planner.py
planner_path = 'src/private_llm_analytics/planner/planner.py'
with open(planner_path, 'r', encoding='utf-8') as f:
    p_content = f.read().replace('\r\n', '\n')

old_canon = """            "providers": sorted(list(self.config.scenario.providers)),
            "public_population_size": int(resolved_pop_size) if resolved_pop_size is not None else -1,
            "statistic_id": str(stat_id),
            "tasks": sorted(list(self.config.scenario.tasks)),
            "taxonomy_fingerprint": tax_fp,
            "threshold": float(threshold_val) if threshold_val is not None else None,
            "top_k": int(top_k_val) if top_k_val is not None else None,"""

new_canon = """            "providers": (
                sorted(list(self.config.recipients.recipients[query.recipient].providers))
                if getattr(self.config, "recipients", None)
                and query.recipient in getattr(self.config.recipients, "recipients", {})
                and "*" not in getattr(self.config.recipients.recipients[query.recipient], "providers", ["*"])
                and metric_name == "top_k_tasks"
                else sorted(list(self.config.scenario.providers))
            ),
            "public_population_size": int(resolved_pop_size) if resolved_pop_size is not None else -1,
            "statistic_id": str(stat_id),
            "tasks": sorted(list(self.config.scenario.tasks)),
            "taxonomy_fingerprint": tax_fp,
            "threshold": float(threshold_val) if threshold_val is not None else None,
            "top_k": None if metric_name == "top_k_tasks" else (int(top_k_val) if top_k_val is not None else None),"""

if old_canon in p_content:
    p_content = p_content.replace(old_canon, new_canon)
    with open(planner_path, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(p_content.replace('\n', '\r\n'))
    print("Updated planner.py")

# 4. Update src/private_llm_analytics/release/exporter.py
exp_path = 'src/private_llm_analytics/release/exporter.py'
with open(exp_path, 'r', encoding='utf-8') as f:
    e_content = f.read().replace('\r\n', '\n')

old_top_k_row = """        for item in top_k_items:
            noisy_score = round(item.get("noisy_count", 0.0), 2)
            est_prev = round(item.get("estimated_prevalence", 0.0), 4) if item.get("estimated_prevalence") is not None else None
            rows.append({
                "rank": item.get("rank"),
                "task": item.get("task"),
                "cell_display": str(item.get("task")),
                "metric": "top_k_tasks",
                "metric_name": metric_friendly,
                "dp_value": noisy_score,
                "dp_estimate": f"Rank #{item.get('rank')} (score: {noisy_score})",
                "statistic_unit": stat_unit,
                "status": "NORMAL",
                "dp_noisy_score": noisy_score,
                "estimated_prevalence": est_prev,"""

new_top_k_row = """        for item in top_k_items:
            noisy_score = round(item.get("noisy_count", 0.0), 2)
            est_prev = round(item.get("estimated_prevalence", 0.0), 4) if item.get("estimated_prevalence") is not None else None
            rows.append({
                "rank": item.get("rank"),
                "task": item.get("task"),
                "noisy_count": noisy_score,
                "estimated_prevalence": est_prev,
                "cell_display": str(item.get("task")),
                "metric": "top_k_tasks",
                "metric_name": metric_friendly,
                "dp_value": noisy_score,
                "dp_estimate": f"Rank #{item.get('rank')} (score: {noisy_score})",
                "statistic_unit": stat_unit,
                "status": "NORMAL",
                "dp_noisy_score": noisy_score,"""

if old_top_k_row in e_content:
    e_content = e_content.replace(old_top_k_row, new_top_k_row)

old_csv_ret = """def build_release_csv_string(release_output: Dict[str, Any]) -> str:
    \"\"\"Serialize DP release output to CSV formatted string.\"\"\"
    df = build_release_dataframe(release_output)
    return df.to_csv(index=False)"""

new_csv_ret = """def build_release_csv_string(release_output: Dict[str, Any]) -> str:
    \"\"\"Serialize DP release output to CSV formatted string with UTF-8 BOM for Windows Excel compatibility.\"\"\"
    df = build_release_dataframe(release_output)
    return "\\ufeff" + df.to_csv(index=False)"""

if old_csv_ret in e_content:
    e_content = e_content.replace(old_csv_ret, new_csv_ret)

old_forbidden = """    forbidden_keys = {
        "raw_dataframe",
        "rng",
        "seed",
        "noise_seed",
        "rng_state",
        "random_state",
        "user_ids",
        "raw_users",
    }"""

new_forbidden = """    forbidden_keys = {
        "raw_dataframe",
        "rng",
        "seed",
        "noise_seed",
        "rng_state",
        "random_state",
        "user_ids",
        "raw_users",
        "exact_ground_truth",
        "exact_data",
        "clean_vector",
        "clipped_vector",
        "raw_prompt",
        "prompt",
    }"""

if old_forbidden in e_content:
    e_content = e_content.replace(old_forbidden, new_forbidden)

with open(exp_path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(e_content.replace('\n', '\r\n'))
print("Updated exporter.py")
