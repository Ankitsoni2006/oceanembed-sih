import os
import re

secret_patterns = [
    re.compile(r"(?i)(api[_-]?key|secret|password|auth[_-]?token|bearer)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{8,}['\"]"),
    re.compile(r"(?i)copernicus.*(password|secret)\s*[:=]\s*['\"][^'\"]+['\"]")
]

hits = []
for root, dirs, files in os.walk("."):
    if any(p in root for p in [".git", "__pycache__", "data", "node_modules"]):
        continue
    for fname in files:
        if fname.endswith((".py", ".json", ".md", ".txt", ".sh", ".ts", ".tsx", ".env")):
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    for line_no, line in enumerate(lines, 1):
                        for p in secret_patterns:
                            if p.search(line):
                                # Mask the actual secret value
                                masked_line = re.sub(r"([:=]\s*['\"]).*?(['\"])", r"\1[REDACTED]\2", line.strip())
                                hits.append((fpath, line_no, masked_line))
            except Exception:
                pass

print(f"Total hits: {len(hits)}")
for h in hits:
    print(f"File: {h[0]}:{h[1]} -> {h[2]}")
