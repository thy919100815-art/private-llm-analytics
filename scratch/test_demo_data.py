from private_llm_analytics.web.adapters import WebAnalyticsAdapter
import json

adapter = WebAnalyticsAdapter("configs")
ds = adapter.load_demo_data()
print("After load_demo_data:")
print(json.dumps(ds, indent=2))
