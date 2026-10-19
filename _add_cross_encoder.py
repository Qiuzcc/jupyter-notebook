"""Add a real Cross-Encoder reranking demo to hybrid_retrieval.ipynb."""
import uuid
import nbformat
from nbclient import NotebookClient

PATH = "hybrid_retrieval.ipynb"
nb = nbformat.read(PATH, as_version=4)


def text_of(cell):
    src = cell.source
    return src if isinstance(src, str) else "".join(src)


def set_source(cell, text):
    cell.source = text.splitlines(keepends=True)
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None


def find_cell(prefix, cell_type):
    for cell in nb.cells:
        if cell.cell_type == cell_type and text_of(cell).startswith(prefix):
            return cell
    raise RuntimeError(f"cell not found: {prefix[:40]}")


def replace_in(cell, old, new):
    src = text_of(cell)
    assert old in src, f"fragment not found: {old[:60]}"
    set_source(cell, src.replace(old, new))


# ---- 1. Intro: pipeline + model dependencies ----
intro = find_cell("# 双路召回 + RRF 融合", "markdown")
replace_in(
    intro,
    "3. **重排序**：对候选用更强的模型（这里用加权打分做教学占位，生产用 Cross-Encoder / LLM）精排。",
    "3. **重排序**：先用加权打分做教学占位（第 4 节），再用真实 Cross-Encoder 对比（第 5 节）。",
)
replace_in(
    intro,
    "首次运行会下载 `all-MiniLM-L6-v2`，约 90MB）；",
    "首次运行会下载 `all-MiniLM-L6-v2` 和 `cross-encoder/ms-marco-MiniLM-L-6-v2`（各约 90MB））；",
)

# ---- 2. Section 4: point to the new comparison section ----
sec4 = find_cell("## 4. 重排序", "markdown")
replace_in(
    sec4,
    "演示「用更丰富信号精排」这一思想；真实 Cross-Encoder 只是把这里的手工权重换成了神经网络学到的交互。",
    "演示「用更丰富信号精排」这一思想；真实 Cross-Encoder 只是把这里的手工权重换成了神经网络学到的交互。\n\n"
    "第 5 节会保留这套公式方案，并额外给出真实 Cross-Encoder 的对比案例。",
)

# ---- 3. Insert section 5 before the end-to-end section, renumber the latter ----
end_to_end = find_cell("## 5. 端到端串联", "markdown")
replace_in(end_to_end, "## 5. 端到端串联 & 小结", "## 6. 端到端串联 & 小结")
insert_at = nb.cells.index(end_to_end)

md_text = """## 5. 真实 Cross-Encoder 重排序（对比案例）

第 4 节的加权公式只是教学占位：BM25 分和稠密余弦各自独立计算，再手工加权。
生产中的重排序通常用 **Cross-Encoder**：把 `(query, doc)` 拼成一对喂进 Transformer，
让模型直接建模两者的交互并输出相关性分数。它逐对计算、比 bi-encoder 慢，所以只用于少量候选。

这里用轻量的 **cross-encoder/ms-marco-MiniLM-L-6-v2**：6 层 MiniLM、约 22.7M 参数、
权重约 90MB，CPU 上即可运行（首次运行会自动下载）。若想更省资源，可以换成
`cross-encoder/ms-marco-TinyBERT-L-2-v2`（约 4M 参数，精度略降）。

注意：Cross-Encoder 输出的是相关性 **logit**（可正可负），只在同一模型内比较大小，
不等于概率；需要 0..1 分数时可以再套一层 sigmoid。
"""

code_text = '''from sentence_transformers import CrossEncoder

# 轻量 Cross-Encoder：6 层 MiniLM，CPU 可跑
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def cross_encoder_rerank(query, candidate_ids, top_n=3):
    pairs = [(query, corpus[i]) for i in candidate_ids]
    scores = cross_encoder.predict(pairs)           # 相关性 logit，越大越相关
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [(i, float(s)) for i, s in ranked][:top_n]

candidate_ids = [i for i, _ in fused[:6]]           # 与端到端流程相同的候选池
formula_top = rerank(query_demo, candidate_ids)     # 第 4 节的加权公式
ce_top = cross_encoder_rerank(query_demo, candidate_ids)

print(f'候选池（RRF Top-{len(candidate_ids)}）:')
for i in candidate_ids:
    print(f"  doc{i}  {corpus[i]}")
print("\\n加权公式重排 Top-3:")
for i, s in formula_top:
    print(f"  doc{i} score={s:.4f}  {corpus[i]}")
print("\\nCross-Encoder 重排 Top-3:")
for i, s in ce_top:
    print(f"  doc{i} logit={s:+.4f}  {corpus[i]}")
'''

md_cell = nbformat.v4.new_markdown_cell(md_text)
code_cell = nbformat.v4.new_code_cell(code_text)
for new_cell in (md_cell, code_cell):
    new_cell.id = uuid.uuid4().hex[:8]
nb.cells[insert_at:insert_at] = [md_cell, code_cell]

# ---- 4. Summary: reflect the Cross-Encoder comparison ----
summary = find_cell("## 小结", "markdown")
replace_in(
    summary,
    "第 2 路已使用真实 embedding（可替换为领域内更强的模型），第 4 路的加权打分在生产中通常换成 bge-reranker / Cohere Rerank 等 Cross-Encoder。",
    "第 2 路已使用真实 embedding（可替换为领域内更强的模型）；第 5 节已用轻量 Cross-Encoder 实测，生产中还可换 bge-reranker / Cohere Rerank 等更强的模型。",
)

# ---- 5. Execute and persist outputs ----
for cell in nb.cells:
    cell.source = text_of(cell)

NotebookClient(nb, timeout=600, kernel_name="python3").execute()

for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    kept = []
    for out in cell.outputs:
        assert out.output_type != "error", "unexpected error output"
        plain = str(out.get("data", {}).get("text/plain", ""))
        if out.output_type == "display_data" and "Loading weights" in plain:
            continue
        kept.append(out)
    cell.outputs = kept

nbformat.write(nb, PATH)

for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        for out in cell.outputs:
            if out.output_type == "stream":
                print(f"--- cell{i} ---")
                print(out.text)
print("=== cross-encoder demo added ===")
