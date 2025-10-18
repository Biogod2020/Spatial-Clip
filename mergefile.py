#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
from collections import defaultdict
from typing import Tuple, Optional, Dict, Any

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

# 要排除的目录名
EXCLUDE_DIRS = {"node_modules", ".git", "__pycache__", "build", "dist"}

# 最大允许合并的字符数（防止过大文件） — 可调
MAX_CHAR_COUNT = 1_000_000  # 超过就跳过

# 最大文件字节数限制（如果觉得有些文件过大在磁盘上就不要读） — 可设 None 表示不限制
MAX_FILE_SIZE = 5 * 1024 * 1024  # 示例：5 MB

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
    # 增加一个简单判断，避免纯文本被误判
    if ' ' in s:
        return False
    if _BASE64_RE.fullmatch(s):
        return True
    return False

def strip_notebook_base64(nb_obj):
    """递归剥除 `.ipynb` JSON 结构中带 base64 的嵌入字段."""
    if isinstance(nb_obj, dict):
        new = {}
        for k, v in nb_obj.items():
            if k == "data" and isinstance(v, dict):
                 # 移除包含 base64 的常见 mime 类型
                new_data = {}
                for mime_type, content in v.items():
                    if "image/" in mime_type and isinstance(content, str):
                        new_data[mime_type] = "<base64_image_removed>"
                    else:
                        new_data[mime_type] = strip_notebook_base64(content)
                new[k] = new_data
            else:
                new[k] = strip_notebook_base64(v)
        return new
    elif isinstance(nb_obj, list):
        return [strip_notebook_base64(x) for x in nb_obj]
    else:
        return nb_obj

def should_include(filepath: str) -> Tuple[bool, Optional[str]]:
    """
    判断这个文件是否应该被考虑（基础过滤）。
    返回 (是否包含, 如果不包含则返回原因)
    """
    # 检查是否在排除目录中
    parts = filepath.split(os.sep)
    if any(p in EXCLUDE_DIRS for p in parts):
        return False, f"In excluded directory (e.g., {next(p for p in parts if p in EXCLUDE_DIRS)})"

    # 检查扩展名
    _, ext = os.path.splitext(filepath)
    ext = ext.lower().lstrip(".")
    if not ext:
        return False, "No extension"
    if ext in EXCLUDE_EXT:
        return False, f"Excluded extension (.{ext})"
    if ext not in INCLUDE_EXT:
        return False, f"Extension not in include list (.{ext})"

    # 文件大小硬限制
    if MAX_FILE_SIZE is not None:
        try:
            size = os.path.getsize(filepath)
            if size > MAX_FILE_SIZE:
                return False, f"Exceeds max file size ({size} > {MAX_FILE_SIZE} bytes)"
        except OSError as e:
            return False, f"Cannot access file: {e}"
            
    return True, None

def process_file(filepath: str) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
    """
    处理单个文件。
    返回 (status, reason, info, content)
    - status: "INCLUDED", "SKIPPED"
    - reason: 如果 SKIPPED, 则为原因
    - info: dict 包含 path, size_bytes, char_count
    - content: 待写入的文本内容（剔除 base64 行等）
    """
    info = { "path": filepath, "size_bytes": -1, "char_count": -1 }
    try:
        info["size_bytes"] = os.path.getsize(filepath)
    except OSError:
        pass # size 保持 -1

    _, ext = os.path.splitext(filepath)
    ext = ext.lower().lstrip(".")

    content_to_process = ""
    is_notebook = False

    # 1. 读取文件内容
    try:
        if ext == "ipynb":
            with open(filepath, 'r', encoding='utf-8') as f:
                nb = json.load(f)
                nb_clean = strip_notebook_base64(nb)
                content_to_process = json.dumps(nb_clean, indent=2, ensure_ascii=False)
                is_notebook = True
        else:
            with open(filepath, 'r', encoding='utf-8', errors="ignore") as f:
                content_to_process = f.read()
    except Exception as e:
        return "SKIPPED", f"Failed to read or parse file: {e}", info, None

    # 2. 计算字符数并检查限制
    char_count = len(content_to_process)
    info["char_count"] = char_count
    if MAX_CHAR_COUNT is not None and char_count > MAX_CHAR_COUNT:
        reason = f"Exceeds max character count ({char_count} > {MAX_CHAR_COUNT})"
        return "SKIPPED", reason, info, None

    # 3. 内容后处理（如过滤 base64）
    final_content = ""
    if not is_notebook:
        lines = content_to_process.splitlines(keepends=True)
        filtered_lines = [line for line in lines if not is_base64_line(line)]
        final_content = "".join(filtered_lines)
    else:
        final_content = content_to_process
    
    final_content += "\n"

    # 4. 检查警告阈值
    warning = None
    size_bytes = info.get("size_bytes", 0) or 0
    if ((WARNING_BYTE_THRESHOLD is not None and size_bytes >= WARNING_BYTE_THRESHOLD) or
        (WARNING_CHAR_THRESHOLD is not None and char_count >= WARNING_CHAR_THRESHOLD)):
        warning = f"File near/over threshold (size={size_bytes} bytes, chars={char_count})"

    return "INCLUDED", warning, info, final_content

def main(base_dir: str):
    included_stats = []
    skipped_stats = []
    filtered_stats = []
    
    stats_by_ext = defaultdict(lambda: {"count": 0, "bytes": 0, "chars": 0})

    with open(LOG_REPORT_PATH, "w", encoding='utf-8') as log_fp, \
         open(OUTPUT_PATH, "w", encoding='utf-8') as out_fp:

        for root, dirs, files in os.walk(base_dir):
            # 优化：在 walk 过程中直接排除目录
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for filename in files:
                filepath = os.path.join(root, filename)

                # 第一道关卡：基础过滤
                is_ok, reason = should_include(filepath)
                if not is_ok:
                    filtered_stats.append({"path": filepath, "reason": reason})
                    log_fp.write(f"[FILTERED] {filepath}: {reason}\n")
                    continue

                # 第二道关卡：详细处理和内容过滤
                status, reason_or_warning, info, content = process_file(filepath)
                
                if status == "INCLUDED":
                    included_stats.append(info)
                    out_fp.write(f"===== {info['path']} =====\n")
                    
                    if reason_or_warning: # 这时是 warning
                        warning_msg = f"⚠️ WARNING: {reason_or_warning}"
                        out_fp.write(f"# {warning_msg}\n")
                        log_fp.write(f"[WARNING]  {info['path']}: {reason_or_warning}\n")

                    out_fp.write(content)

                    # 更新统计
                    _, ext = os.path.splitext(info["path"])
                    ext = ext.lower().lstrip(".")
                    stats_by_ext[ext]["count"] += 1
                    stats_by_ext[ext]["bytes"] += info.get("size_bytes", 0) or 0
                    stats_by_ext[ext]["chars"] += info.get("char_count", 0) or 0

                elif status == "SKIPPED":
                    skipped_stats.append(info)
                    log_fp.write(f"[SKIPPED]  {info['path']}: {reason_or_warning} (size={info['size_bytes']}, chars={info['char_count']})\n")

        # --- 生成报告 ---
        log_fp.write("\n" + "="*20 + " Summary " + "="*20 + "\n")
        log_fp.write(f"Total files merged: {len(included_stats)}\n")
        log_fp.write(f"Total files skipped (content reasons): {len(skipped_stats)}\n")
        log_fp.write(f"Total files filtered (path/size reasons): {len(filtered_stats)}\n\n")
        
        log_fp.write("Stats for merged files by extension:\n")
        for ext, d in sorted(stats_by_ext.items()):
            log_fp.write(f"  .{ext:<8} | Count: {d['count']:>4} | Total Bytes: {d['bytes']:>10,} | Total Chars: {d['chars']:>10,}\n")

    # 在 stdout 输出摘要
    summary_msg = (
        f"✅ Done! Merged {len(included_stats)} files into {OUTPUT_PATH}.\n"
        f" Skipped {len(skipped_stats)} files and filtered {len(filtered_stats)} files. "
        f"See {LOG_REPORT_PATH} for full details."
    )
    print(summary_msg)

if __name__ == "__main__":
    # 如果提供了路径，则使用它；否则使用当前目录
    base_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(base_directory):
        print(f"Error: Provided path '{base_directory}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)
    main(base_directory)