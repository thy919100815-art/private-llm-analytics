path = 'src/private_llm_analytics/web/static/app.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read().replace('\r\n', '\n')

func_code = """function setDownloadButtonsEnabled(enabled) {
  const btnCsv = document.getElementById("btn-download-csv");
  const btnJson = document.getElementById("btn-download-json");
  if (btnCsv) {
    btnCsv.disabled = !enabled;
    btnCsv.title = enabled ? "下载 DP 处理后的 CSV 文件" : "请先成功生成 DP 发布结果后下载";
  }
  if (btnJson) {
    btnJson.disabled = !enabled;
    btnJson.title = enabled ? "下载 DP 处理后的 JSON 文件" : "请先成功生成 DP 发布结果后下载";
  }
}
"""

if "function setDownloadButtonsEnabled" not in content:
    target_pos = "let lastReleaseData = null;\n"
    content = content.replace(target_pos, target_pos + func_code, 1)

# In release try block
target_succ = """      lastReleaseData = data;
      renderReleaseResult(data);"""

repl_succ = """      lastReleaseData = data;
      setDownloadButtonsEnabled(true);
      renderReleaseResult(data);"""

if target_succ in content:
    content = content.replace(target_succ, repl_succ, 1)

# In release catch block
target_fail = """    } catch (err) {
      showError(translateError(err) || "差分隐私发布失败。");
    }"""

repl_fail = """    } catch (err) {
      lastReleaseData = null;
      setDownloadButtonsEnabled(false);
      const resContainer = document.getElementById("release-result-card");
      if (resContainer) resContainer.style.display = "none";
      showError(translateError(err) || "差分隐私发布失败。");
    }"""

if target_fail in content:
    content = content.replace(target_fail, repl_fail, 1)

# In load demo
target_demo = """      const data = await res.json();
      await refreshDatasetStatus();"""

repl_demo = """      const data = await res.json();
      lastReleaseData = null;
      setDownloadButtonsEnabled(false);
      const resContainer = document.getElementById("release-result-card");
      if (resContainer) resContainer.style.display = "none";
      await refreshDatasetStatus();"""

if target_demo in content:
    content = content.replace(target_demo, repl_demo, 1)

with open(path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(content.replace('\n', '\r\n'))
print("Successfully updated app.js download button states")
