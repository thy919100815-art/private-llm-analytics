from private_llm_analytics.web.adapters import WebAnalyticsAdapter
import json

adapter = WebAnalyticsAdapter("configs")
ds = adapter.get_dataset_status()
print(json.dumps(ds, indent=2))
