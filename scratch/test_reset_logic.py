import json
from decimal import Decimal
import os
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.privacy.rdp import DEFAULT_RDP_ORDERS, to_decimal, compose_rdp_vectors

adapter = WebAnalyticsAdapter("configs")
print("Initial privacy status:")
ps = adapter.get_privacy_status()
print("  Epoch eps:", ps["epoch"]["effective_epsilon"])
print("  Lifetime eps:", ps["lifetime_exposure"]["lifetime_effective_epsilon"])
print("  Releases count:", len(adapter.list_releases()))

# Now simulate reset_demo_epoch
orchestrator = adapter.service.orchestrator
epoch_mgr = orchestrator.epoch_manager
state_mgr = orchestrator.state_manager

active = epoch_mgr.get_or_create_active_epoch()
active_id = active.epoch_id
start_date = active.epoch_start
days = active.epoch_days
rel_ids = list(active.release_ids)

with state_mgr._connection() as conn:
    cur = conn.cursor()
    if rel_ids:
        placeholders = ",".join("?" for _ in rel_ids)
        cur.execute(f"DELETE FROM privacy_ledger WHERE release_id IN ({placeholders})", rel_ids)
        cur.execute(f"DELETE FROM release_store WHERE release_id IN ({placeholders})", rel_ids)
    cur.execute("DELETE FROM privacy_reservations")
    cur.execute("DELETE FROM privacy_epochs WHERE epoch_id = ?", (active_id,))
    conn.commit()

if hasattr(orchestrator.cache, "_memory_cache"):
    orchestrator.cache._memory_cache.clear()

orders = epoch_mgr.planner.orders
zero_rdp = {alpha: Decimal("0.0") for alpha in orders}
state_mgr.update_lifetime_exposure_record(
    lifetime_rdp_json=json.dumps({str(k): str(v) for k, v in zero_rdp.items()}),
    total_epochs_count=1,
)

fresh_epoch = epoch_mgr.planner.create_epoch(
    epoch_id=active_id,
    start_date=start_date,
    epoch_days=days,
)
state_mgr.save_epoch_record(epoch_mgr._epoch_to_record(fresh_epoch))

print("\nAfter reset_demo_epoch:")
ps2 = adapter.get_privacy_status()
print("  Epoch eps:", ps2["epoch"]["effective_epsilon"])
print("  Lifetime eps:", ps2["lifetime_exposure"]["lifetime_effective_epsilon"])
print("  Workload interaction_volume spent:", ps2["epoch"]["workloads"]["interaction_volume"]["spent_epsilon"])
print("  Reserve spent:", ps2["epoch"]["workloads"]["reserve"]["spent_epsilon"])
print("  Releases count:", len(adapter.list_releases()))
