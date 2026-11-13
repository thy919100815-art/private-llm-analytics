from private_llm_analytics.web.adapters import WebAnalyticsAdapter
import json

adapter = WebAnalyticsAdapter("configs")
status = adapter.get_privacy_status()
print("Epoch info:")
print(json.dumps(status.get("epoch"), indent=2))
