"""Insert a short interpretation note after the Cross-Encoder demo cell."""
import uuid
import nbformat

PATH = "hybrid_retrieval.ipynb"
nb = nbformat.read(PATH, as_version=4)


def text_of(cell):
    src = cell.source
    return src if isinstance(src, str) else "".join(src)


code_cell = next(
    c for c in nb.cells
    if c.cell_type == "code" and text_of(c).startswith("from sentence_transformers import CrossEncoder")
)
idx = nb.cells.index(code_cell)

note = nbformat.v4.new_markdown_cell(
    "> **结果解读**：同一个候选池下，加权公式把仅靠停用词 `a` 命中 BM25 的 `doc1` 排进了 Top-3；\n"
    "> Cross-Encoder 把它挤出了 Top-3，并把语义相关的 `doc6`（Puppies require training...）提了上来。\n"
    "> 注意 `doc0` 的 logit 为负仍排第 2：logit 的绝对值不是概率，只在同一模型内比较相对大小。\n"
    "> 这正是「Cross-Encoder 更准但更慢，只用于少量候选」的直观体现。\n"
)
note.id = uuid.uuid4().hex[:8]
nb.cells.insert(idx + 1, note)

nbformat.write(nb, PATH)
print("=== interpretation note inserted ===")
