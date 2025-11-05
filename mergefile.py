#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import fnmatch
from collections import defaultdict
from typing import Tuple, Optional, Dict, Any

# 确保 pathspec 已安装: pip install pathspec
try:
    import pathspec
except ImportError:
    print("Error: 'pathspec' library not found. Please install it using 'pip install pathspec'", file=sys.stderr)
    sys.exit(1)

# ======================= 配置区 =======================

OUTPUT_PATH = "merged_output.txt"
LOG_REPORT_PATH = "merge_report.log"

# 包含的文件扩展名 (小写)。这是在 .gitignore 过滤后，对文件类型的二次筛选。
INCLUDE_EXT = {
    "py", "ipynb", "txt", "md", "rst", "html", "htm",
    "css", "js", "ts", "jsx", "tsx", "sh", "ps1", 
    "yaml", "yml", "json", "xml", "ini", "cfg", "toml", "sql"
}

# --- 文件大小与内容限制 ---
MAX_CHAR_COUNT = 1_000_000
MAX_FILE_SIZE = 5 * 1024 * 1024
WARNING_BYTE_THRESHOLD = 200 * 1024
WARNING_CHAR_THRESHOLD = 200 * 1024

_BASE64_RE = re.compile(
    r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$"
)

# ======================= 函数区 =======================

def load_gitignore_spec(base_dir: str) -> pathspec.PathSpec:
    """加载 .gitignore 文件并返回一个 pathspec 对象。"""
    gitignore_path = os.path.join(base_dir, '.gitignore')
    patterns = []
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            patterns = f.readlines()
    # 总是排除脚本自身和其输出
    patterns.extend([
        os.path.basename(OUTPUT_PATH),
        os.path.basename(LOG_REPORT_PATH),
        os.path.basename(__file__),
    ])
    return pathspec.PathSpec.from_lines('gitwildmatch', patterns)

def is_base64_line(line: str, min_length: int = 200) -> bool:
    s = line.strip()
    if len(s) < min_length: return False
    if ' ' in s: return False
    return bool(_BASE64_RE.fullmatch(s))

def strip_notebook_base64(nb_obj):
    if isinstance(nb_obj, dict):
        return {k: strip_notebook_base64(v) for k, v in nb_obj.items() if not (k == "data" and isinstance(v, dict) and any("image/" in mt for mt in v))}
    elif isinstance(nb_obj, list):
        return [strip_notebook_base64(item) for item in nb_obj]
    else:
        return nb_obj
    
def format_notebook_content(nb_obj: Dict) -> str:
    """
    将解析后的 Jupyter Notebook JSON 对象转换为一个干净、可读的文本格式。
    仅包含单元格类型、源代码和关键元数据，完全忽略输出。
    """
    formatted_parts = []
    for i, cell in enumerate(nb_obj.get('cells', [])):
        cell_type = cell.get('cell_type', 'unknown')
        source = "".join(cell.get('source', []))
        
        header = f"# ===== Cell {i+1}: {cell_type.capitalize()} =====\n"
        
        # 提取并添加代码单元格的关键元数据
        if cell_type == 'code':
            exec_count = cell.get('execution_count')
            if exec_count is not None:
                header = f"# ===== Cell {i+1}: Code (execution_count: {exec_count}) =====\n"
            
            # 检查 jupyterlab_execute_time 插件的元数据
            exec_time_meta = cell.get('metadata', {}).get('jupyterlab_execute_time')
            if exec_time_meta and exec_time_meta.get('start_time'):
                start = exec_time_meta.get('start_time', 'N/A').split('.')[0]
                end = exec_time_meta.get('end_time', 'N/A').split('.')[0]
                header += f"# Execution Time: {start} to {end}\n"

        formatted_parts.append(header)
        formatted_parts.append(source)
        # 在每个单元格后添加两个换行符以增加间距
        formatted_parts.append("\n\n")

    return "".join(formatted_parts)

def process_file(filepath: str) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
    info = {"path": filepath, "size_bytes": -1, "char_count": -1}
    try:
        info["size_bytes"] = os.path.getsize(filepath)
    except OSError:
        pass

    # 文件大小硬限制 (在主循环中前置检查)
    if MAX_FILE_SIZE is not None and info["size_bytes"] > MAX_FILE_SIZE:
        return "SKIPPED", f"Exceeds max file size ({info['size_bytes']} > {MAX_FILE_SIZE} bytes)", info, None

    _, ext = os.path.splitext(filepath)
    ext = ext.lower().lstrip(".")
    content_to_process = ""
    is_notebook = False

    # 1. 读取文件内容
    try:
        if ext == "ipynb":
            with open(filepath, 'r', encoding='utf-8') as f:
                nb = json.load(f)
                # ===== START: 修改部分 =====
                # 旧逻辑:
                # nb_clean = strip_notebook_base64(nb)
                # content_to_process = json.dumps(nb_clean, indent=2, ensure_ascii=False)
                
                # 新逻辑:
                content_to_process = format_notebook_content(nb)
                # ===== END: 修改部分 =====
            is_notebook = True
        else:
            with open(filepath, 'r', encoding='utf-8', errors="ignore") as f:
                content_to_process = f.read()
    except Exception as e:
        return "SKIPPED", f"Failed to read or parse file: {e}", info, Nonerelo

    char_count = len(content_to_process)
    info["char_count"] = char_count
    if MAX_CHAR_COUNT is not None and char_count > MAX_CHAR_COUNT:
        return "SKIPPED", f"Exceeds max character count ({char_count} > {MAX_CHAR_COUNT})", info, None

    final_content = ""
    if not is_notebook:
        lines = content_to_process.splitlines(keepends=True)
        filtered_lines = [line for line in lines if not is_base64_line(line)]
        final_content = "".join(filtered_lines)
    else:
        final_content = content_to_process
    final_content += "\n"

    warning = None
    size_bytes = info.get("size_bytes", 0) or 0
    if ((WARNING_BYTE_THRESHOLD is not None and size_bytes >= WARNING_BYTE_THRESHOLD) or
        (WARNING_CHAR_THRESHOLD is not None and char_count >= WARNING_CHAR_THRESHOLD)):
        warning = f"File near/over threshold (size={size_bytes} bytes, chars={char_count})"

    return "INCLUDED", warning, info, final_content

# ======================= 主逻辑 =======================

def main(base_dir: str):
    spec = load_gitignore_spec(base_dir)
    
    included_stats = []
    skipped_stats = []
    filtered_stats = []
    stats_by_ext = defaultdict(lambda: {"count": 0, "bytes": 0, "chars": 0})

    with open(LOG_REPORT_PATH, "w", encoding='utf-8') as log_fp, \
         open(OUTPUT_PATH, "w", encoding='utf-8') as out_fp:
        
        log_fp.write("Starting file merge process...\n")
        log_fp.write(f"Scanning base directory: {os.path.abspath(base_dir)}\n")
        log_fp.write("Using .gitignore rules for filtering.\n\n")

        for root, dirs, files in os.walk(base_dir, topdown=True):
            # 高效地修剪 .gitignore 匹配的目录
            rel_root = os.path.relpath(root, base_dir)
            if rel_root == '.': rel_root = ''
            
            # 检查目录是否被忽略
            ignored_dirs = set()
            for d in dirs:
                dir_path = os.path.join(rel_root, d)
                if spec.match_file(dir_path):
                    ignored_dirs.add(d)

            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            
            for filename in files:
                filepath = os.path.join(root, filename)
                rel_filepath = os.path.relpath(filepath, base_dir)

                # 1. 主要过滤：使用 .gitignore 规则
                if spec.match_file(rel_filepath):
                    filtered_stats.append({"path": filepath, "reason": "Matched by .gitignore"})
                    log_fp.write(f"[FILTERED] {filepath}: Matched by .gitignore\n")
                    continue

                # 2. 次要过滤：只包含我们想要的文本文件类型
                ext = (os.path.splitext(filename)[1].lower().lstrip('.') or 'no_ext')
                if ext not in INCLUDE_EXT:
                     filtered_stats.append({"path": filepath, "reason": f"Extension '.{ext}' not in INCLUDE_EXT"})
                     log_fp.write(f"[FILTERED] {filepath}: Extension '.{ext}' not in INCLUDE_EXT\n")
                     continue

                # 3. 处理文件内容
                status, reason_or_warning, info, content = process_file(filepath)
                
                if status == "INCLUDED":
                    included_stats.append(info)
                    out_fp.write(f"===== {rel_filepath} =====\n")
                    
                    if reason_or_warning:
                        out_fp.write(f"# ⚠️ WARNING: {reason_or_warning}\n")
                        log_fp.write(f"[WARNING]  {rel_filepath}: {reason_or_warning}\n")

                    out_fp.write(content)

                    stats_by_ext[ext]["count"] += 1
                    stats_by_ext[ext]["bytes"] += info.get("size_bytes", 0) or 0
                    stats_by_ext[ext]["chars"] += info.get("char_count", 0) or 0

                elif status == "SKIPPED":
                    skipped_stats.append(info)
                    log_fp.write(f"[SKIPPED]  {rel_filepath}: {reason_or_warning} (size={info.get('size_bytes', 'N/A')}, chars={info.get('char_count', 'N/A')})\n")

        # --- 生成报告 ---
        log_fp.write("\n" + "="*20 + " Summary " + "="*20 + "\n")
        log_fp.write(f"Total files merged: {len(included_stats)}\n")
        log_fp.write(f"Total files skipped (content reasons): {len(skipped_stats)}\n")
        log_fp.write(f"Total files filtered out (path/type reasons): {len(filtered_stats)}\n\n")
        
        log_fp.write("Stats for merged files by extension:\n")
        for ext, d in sorted(stats_by_ext.items()):
            log_fp.write(f"  .{ext:<8} | Count: {d['count']:>4} | Total Bytes: {d['bytes']:>10,} | Total Chars: {d['chars']:>10,}\n")

    summary_msg = (
        f"✅ Done! Merged {len(included_stats)} files into {OUTPUT_PATH}.\n"
        f" Skipped {len(skipped_stats)} files and filtered {len(filtered_stats)} files. "
        f"See {LOG_REPORT_PATH} for full details."
    )
    print(summary_msg)

if __name__ == "__main__":
    base_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(base_directory):
        print(f"Error: Provided path '{base_directory}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)
    main(base_directory)