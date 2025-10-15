#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
from collections import defaultdict

# ========== 配置区 ==========

OUTPUT_PATH = "merged_output.txt"
LOG_REPORT_PATH = "merge_report.log"  # 用来写警告 / 跳过 / 报表日志

# 要包含的扩展名（小写） —— 包括 R / 生物信息学相关脚本
INCLUDE_EXT = {
    "py", "ipynb", "txt", "md", "rst", "html", "htm",
    "css", "js", "ts", "jsx", "tsx",
    "java", "kt", "c", "cpp", "h", "hpp", "rs",
    "go", "rb", "sh", "ps1", "yaml", "yml", "json",
    "xml", "ini", "cfg", "toml", "sql", "tex", "scala",
    "php", "swift",
    # 生物信息学 / 统计 / 数据分析语言
    "r", "rmd", "bi", "fasta", "fa", "fastq", "fq", "bed", "gtf", "gff", "sam", "bam", "vcf"
}

# 要排除的扩展名（除非你显式希望把它们读进来）
EXCLUDE_EXT = {
    "bin", "exe", "dll", "o", "so", "dylib",
    "class", "jar", "pyc", "pyo", "obj", "apk", "war"
}

# 最大允许合并的字符数（防止过大文件） — 可调
MAX_CHAR_COUNT = 10_000_000  # 超过就跳过

# 最大文件字节数限制（如果觉得有些文件过大在磁盘上就不要读） — 可设 None 表示不限制
MAX_FILE_SIZE = None  

# 警戒线：当文件大小 / 字符数 达到此阈值时发警告（但仍可能合并）
WARNING_BYTE_THRESHOLD = 200 * 1024  # 200 KB
WARNING_CHAR_THRESHOLD = 200 * 1024

# Base64 判断正则（用于判断某行是否极可能是 Base64 嵌入）
_BASE64_RE = re.compile(
    r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$"
)

def is_base64_line(line: str, min_length: int = 200) -> bool:
    s = line.strip()
    if len(s) < min_length:
        return False
    if _BASE64_RE.fullmatch(s):
        return True
    return False

def strip_notebook_base64(nb_obj):
    """递归剥除 `.ipynb` JSON 结构中带 base64 的嵌入字段."""
    if isinstance(nb_obj, dict):
        new = {}
        for k, v in nb_obj.items():
            if k == "data" and isinstance(v, (str, dict)):
                if isinstance(v, str):
                    if "base64" in v:
                        new[k] = ""
                    else:
                        new[k] = v
                else:
                    new[k] = {subk: ("" if (isinstance(subv, str) and "base64" in subv) else strip_notebook_base64(subv))
                              for subk, subv in v.items()}
            else:
                new[k] = strip_notebook_base64(v)
        return new
    elif isinstance(nb_obj, list):
        return [strip_notebook_base64(x) for x in nb_obj]
    else:
        return nb_obj

def count_chars_in_text(text: str) -> int:
    return len(text)

def should_include(filepath: str) -> bool:
    """判断这个文件是否应该被考虑（基础过滤）"""
    parts = filepath.split(os.sep)
    for p in ("node_modules", ".git", "__pycache__", "build", "dist"):
        if p in parts:
            return False
    _, ext = os.path.splitext(filepath)
    ext = ext.lower().lstrip(".")
    # 如果没有扩展名，通常忽略
    if ext == "":
        return False
    # 如果在排除名单，跳过
    if ext in EXCLUDE_EXT:
        return False
    # 如果不在 include 列表，也跳过
    if ext not in INCLUDE_EXT:
        return False
    # 文件大小硬限制
    if MAX_FILE_SIZE is not None:
        try:
            if os.path.getsize(filepath) > MAX_FILE_SIZE:
                return False
        except OSError:
            return False
    return True

def process_file(filepath: str):
    """
    处理单个文件：
    返回 (include_flag, info, content, warning_msg)
    - include_flag: 是否合并
    - info: dict 包含 path, size_bytes, char_count
    - content: 待写入的文本内容（剔除 base64 行等），如果不合并则为 None
    - warning_msg: 若触发警戒线，则返回提示字符串，否则 None
    """
    info = {
        "path": filepath,
        "size_bytes": None,
        "char_count": None,
    }
    try:
        size = os.path.getsize(filepath)
        info["size_bytes"] = size
    except OSError:
        info["size_bytes"] = -1

    _, ext = os.path.splitext(filepath)
    ext = ext.lower().lstrip(".")

    # 专门处理 .ipynb
    if ext == "ipynb":
        try:
            with open(filepath, encoding='utf-8') as f:
                nb = json.load(f)
        except Exception as e:
            # JSON 解析失败，回退为普通文本处理
            ext = "txt"
        else:
            nb_clean = strip_notebook_base64(nb)
            text = json.dumps(nb_clean, indent=2, ensure_ascii=False)
            char_count = count_chars_in_text(text)
            info["char_count"] = char_count
            # 超过最大允许字符数则跳过
            if MAX_CHAR_COUNT is not None and char_count > MAX_CHAR_COUNT:
                return False, info, None, None
            # 警戒线提示
            warning = None
            if ((WARNING_BYTE_THRESHOLD is not None and info["size_bytes"] is not None and info["size_bytes"] >= WARNING_BYTE_THRESHOLD)
                or (WARNING_CHAR_THRESHOLD is not None and char_count >= WARNING_CHAR_THRESHOLD)):
                warning = f"⚠️ WARNING: file near threshold (size={info['size_bytes']} bytes, chars={char_count})"
            return True, info, text + "\n", warning

    # 普通文本 / 代码 / 脚本 文件
    try:
        with open(filepath, encoding='utf-8', errors="ignore") as f:
            raw = f.read()
    except Exception as e:
        info["char_count"] = 0
        return False, info, None, None

    char_count = count_chars_in_text(raw)
    info["char_count"] = char_count
    # 如果字符数超过最大允许，跳过
    if MAX_CHAR_COUNT is not None and char_count > MAX_CHAR_COUNT:
        return False, info, None, None

    # 逐行过滤极可能的 Base64 行
    lines = raw.splitlines(keepends=True)
    filtered = []
    for line in lines:
        if not is_base64_line(line):
            filtered.append(line)
    content = "".join(filtered) + "\n"

    warning = None
    if ((WARNING_BYTE_THRESHOLD is not None and info["size_bytes"] is not None and info["size_bytes"] >= WARNING_BYTE_THRESHOLD)
        or (WARNING_CHAR_THRESHOLD is not None and char_count >= WARNING_CHAR_THRESHOLD)):
        warning = f"⚠️ WARNING: file near threshold (size={info['size_bytes']} bytes, chars={char_count})"

    return True, info, content, warning

def main(base_dir: str):
    file_list = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "__pycache__", "build", "dist")]
        for fn in files:
            full = os.path.join(root, fn)
            if should_include(full):
                file_list.append(full)
    file_list.sort()

    skipped = []
    included = []
    # 用于按扩展名统计合并了多少文件、总字节数、总字符数
    stats_by_ext = defaultdict(lambda: {"count": 0, "bytes": 0, "chars": 0})

    # 打开日志报告文件
    log_fp = open(LOG_REPORT_PATH, "w", encoding='utf-8')

    with open(OUTPUT_PATH, "w", encoding='utf-8') as out_fp:
        for fp in file_list:
            inc, info, content, warning = process_file(fp)
            if inc:
                included.append(info)
                # 写标题
                out_fp.write(f"===== {info['path']} =====\n")
                # 写警告（若有）
                if warning:
                    out_fp.write(f"# {warning}\n")
                    log_fp.write(f"WARNING: {info['path']} — size {info['size_bytes']} bytes, chars {info['char_count']}\n")
                # 写内容
                out_fp.write(content)

                # 更新统计
                _, ext = os.path.splitext(info["path"])
                ext = ext.lower().lstrip(".")
                stats_by_ext[ext]["count"] += 1
                stats_by_ext[ext]["bytes"] += info["size_bytes"] or 0
                stats_by_ext[ext]["chars"] += info["char_count"] or 0
            else:
                skipped.append(info)
                log_fp.write(f"SKIPPED: {info['path']} — size {info['size_bytes']} bytes, chars {info['char_count']}\n")

    # 写统计报告尾部
    log_fp.write("\n=== Summary ===\n")
    log_fp.write(f"Merged {len(included)} files into {OUTPUT_PATH}\n")
    log_fp.write(f"Skipped {len(skipped)} files\n")
    log_fp.write("Stats by extension:\n")
    for ext, d in sorted(stats_by_ext.items()):
        log_fp.write(f"  .{ext}: count={d['count']}, total_bytes={d['bytes']}, total_chars={d['chars']}\n")
    log_fp.close()

    # 同时在 stdout / stderr 输出摘要
    print(f"Merged {len(included)} files into {OUTPUT_PATH}, skipped {len(skipped)} files. See {LOG_REPORT_PATH} for details.")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    main(base)

