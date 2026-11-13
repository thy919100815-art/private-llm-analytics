from private_llm_analytics.web.adapters import WebAnalyticsAdapter

adapter = WebAnalyticsAdapter("configs")
status = adapter.get_privacy_status()
print("Current active epoch:", status.get("active_epoch", {}).get("epoch_id"))
print("Current effective eps:", status.get("active_epoch", {}).get("effective_epsilon"))
print("Lifetime exposure:", status.get("lifetime_exposure"))
print("Release history count:", len(adapter.list_releases()))
