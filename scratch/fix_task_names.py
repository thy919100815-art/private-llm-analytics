with open('src/private_llm_analytics/web/static/app.js', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find('const TASK_NAMES = {')
end_idx = text.find('};\n\n\n\nconst UNIT_NAMES', idx)
assert idx != -1 and end_idx != -1, "Target boundaries not found"

new_t = """const TASK_NAMES = {
  document: "文档类",
  technical: "技术类",
  analysis: "分析类",
  coding: "代码编写",
  translation: "多语言翻译",
  summarization: "文本摘要",
  customer_support: "客服支持",
  creative_writing: "创意写作",
  data_extraction: "信息抽取",
  qa: "问答咨询",
  code_review: "代码审查",
  brainstorming: "头脑风暴",
};"""

text = text[:idx] + new_t + text[end_idx+2:]
with open('src/private_llm_analytics/web/static/app.js', 'w', encoding='utf-8') as f:
    f.write(text)
print("Successfully replaced TASK_NAMES in app.js")
