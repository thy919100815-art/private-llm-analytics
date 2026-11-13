from decimal import Decimal
import json
from private_llm_analytics.web.adapters import WebAnalyticsAdapter

adapter = WebAnalyticsAdapter("configs")
orchestrator = adapter.service.orchestrator
state_mgr = orchestrator.state_manager
ledger = orchestrator.ledger
epoch_mgr = orchestrator.epoch_manager

# Perform reset_all
state_mgr.clear()
with ledger._connection() as conn:
    conn.execute("DELETE FROM privacy_ledger")
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
    epoch_id="epoch_001",
    start_date="2026-10-01",
    epoch_days=epoch_mgr.epoch_days,
)
state_mgr.save_epoch_record(epoch_mgr._epoch_to_record(fresh_epoch))

print("Status after reset_all:")
ps = adapter.get_privacy_status()
print("  Epoch eps:", ps["epoch"]["effective_epsilon"])
print("  Lifetime eps:", ps["lifetime_exposure"]["lifetime_effective_epsilon"])
print("  Workload interaction_volume:", ps["epoch"]["workloads"]["interaction_volume"]["spent_epsilon"])
print("  Releases count:", len(adapter.list_releases()))

# Reload demo data
adapter.load_demo_data()
ds = adapter.get_dataset_status()
print("  Dataset rows:", ds["rows"], "users:", ds["active_users"])
