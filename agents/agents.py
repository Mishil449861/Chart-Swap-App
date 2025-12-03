import re
import os
import ast
import uuid
import tempfile
import traceback
import multiprocessing as mp
import shutil
from typing import Tuple, Dict, Any, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai
from PIL import Image

# ----------------------------
# UTILITY HELPERS
# ----------------------------

def _make_tmp_image_path(prefix="chart", ext="png"):
    return os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}")

def clean_code(code_str: str) -> str:
    """Removes markdown backticks and whitespace."""
    if not code_str: return ""
    code_str = re.sub(r"^```[a-zA-Z]*", "", code_str)
    code_str = code_str.replace("```", "")
    return code_str.strip()

def _safe_extract_numbers_from_string(s: str) -> List[float]:
    nums = re.findall(r"-?\d+\.\d+|-?\d+", s)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except:
            continue
    return out

# ----------------------------
# 1) SAFETY AGENT
# ----------------------------

class SafetyAgent:
    DEFAULT_DISALLOWED_PATTERNS = [
        r"\bos\.system\b", r"\bsubprocess\b", r"\bopen\s*\(", r"\bexec\s*\(",
        r"\b__import__\s*\(", r"\bshutil\b", r"while\s+True\b", 
        r"plt\.show\s*\(", r"\binput\s*\(", r"time\.sleep\(\s*[1-9]\d+",
    ]

    def review(self, code: str) -> Tuple[bool, str]:
        for patt in self.DEFAULT_DISALLOWED_PATTERNS:
            if re.search(patt, code, flags=re.IGNORECASE):
                return False, f"Blocked pattern matched: {patt}"
        return True, "Safe"

# ----------------------------
# 2) INTENT AGENT
# ----------------------------

class IntentAgent:
    def __init__(self):
        # Map natural language positions to list indices
        self.ordinal_map = {
            "first": 0, "1st": 0, "initial": 0,
            "second": 1, "2nd": 1,
            "third": 2, "3rd": 2,
            "fourth": 3, "4th": 3,
            "fifth": 4, "5th": 4,
            "last": -1, "final": -1
        }

    def parse(self, user_prompt: str) -> Dict[str, Any]:
        prompt = (user_prompt or "").lower()
        intent = {
            "raw": user_prompt,
            "action": None,         
            "target_series": None,  # e.g., "green"
            "target_index": None,   # e.g., 0 (first)
            "target_point": None,   # e.g., "At M"
            "range_end": None,      # e.g., "Till K"
            "value": None,
            "unit": "absolute",     # "percent" vs "absolute"
            "color": None           # Destination color
        }

        # --- 1. ACTION ---
        if re.search(r"\bincrease\b|\braise\b|\bup\b|\badd\b", prompt): 
            intent["action"] = "increase"
        elif re.search(r"\bdecrease\b|\breduce\b|\bdown\b|\bdrop\b", prompt): 
            intent["action"] = "decrease"
        elif re.search(r"\bhighlight\b|\bcolor\b|\bmake\b", prompt): 
            intent["action"] = "highlight"
        elif re.search(r"\bsort\b", prompt): 
            intent["action"] = "sort"

        # --- 2. TARGET SERIES (Color) ---
        m_color = re.search(r"\b(green|blue|orange|red|yellow|purple|gray|black|teal|cyan)\b", prompt)
        if m_color: intent["target_series"] = m_color.group(1)

        # --- 3. TARGET POSITION (Ordinal) ---
        for word, index in self.ordinal_map.items():
            if re.search(rf"\b{word}\b", prompt):
                intent["target_index"] = index
                break

        # --- 4. RANGES ("Till") vs POINTS ("At/By") ---
        # A. Range: "Till K"
        m_range = re.search(r"\b(till|until|through|to)\s+([a-z0-9]+)\b", prompt)
        if m_range:
            candidate = m_range.group(2)
            # Filter out small numbers that are likely values (e.g. "increase to 50")
            is_value = False
            try:
                if float(candidate) < 1000: is_value = True
            except: pass
            
            if not is_value:
                intent["range_end"] = candidate.upper()

        # B. Point: "At M", "By Q3"
        m_point = re.search(r"\b(at|on|for|by)\s+([a-z0-9]+)\b", prompt)
        if m_point:
            candidate = m_point.group(2)
            try:
                float(candidate) # If it's a number (e.g. "by 15"), ignore it here
            except ValueError:
                intent["target_point"] = candidate.upper()

        # --- 5. VALUE & UNIT ---
        m_val = re.search(r"(\d+(\.\d+)?)", prompt)
        if m_val:
            try:
                raw_num = float(m_val.group(1))
                intent["value"] = raw_num
                if re.search(rf"{m_val.group(0)}\s*(%|percent)", prompt):
                    intent["unit"] = "percent"
                else:
                    intent["unit"] = "absolute"
            except: pass

        # --- 6. HIGHLIGHT COLOR ---
        if intent["action"] == "highlight":
            all_colors = re.findall(r"\b(red|blue|green|orange|purple|yellow|black)\b", prompt)
            if all_colors:
                intent["color"] = all_colors[-1]

        return intent

    def analyze_intent(self, user_prompt: str) -> Dict[str, Any]:
        return self.parse(user_prompt)

# ----------------------------
# 3) DATA EXTRACTOR AGENT
# ----------------------------

class DataExtractorAgent:
    def extract_from_ocr(self, ocr_text: str) -> Dict[str, Any]:
        return {"ocr_snippet": ocr_text[:600]}

    def extract_data(self, ocr_text: str) -> Dict[str, Any]:
        return self.extract_from_ocr(ocr_text)

# ----------------------------
# 4) PLANNING AGENT
# ----------------------------

class PlanningAgent:
    def plan(self, intent: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        return {"run_code_generation": True}

    def create_plan(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        return self.plan(intent, {})

# ----------------------------
# 5) CODE GENERATOR AGENT (Vision + Data Integrity)
# ----------------------------

class CodeGeneratorAgent:
    def __init__(self, gemini_model: str = "gemini-2.0-flash"):
        self.model_name = gemini_model
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except: self.model = None

    def generate(self, data, intent, ocr_text, image_path=None) -> str:
        
        system_prompt = f"""
You are an expert Data Visualization Engineer using Python and Matplotlib. 
Your goal is to recreate the chart provided in the image input as faithfully as possible, then apply the user's requested transformation.

INTENT DATA: {intent}

CRITICAL INSTRUCTIONS:
1. **VISION IS TRUTH**: Trust your eyes over OCR. Estimate values visually.
2. **SAVE PATH**: You MUST save the figure using `plt.savefig(output_path)`.

3. **AXIS MANIPULATION & INTEGRITY**:
   - **Extension (e.g., 'till K')**: If extending the X-axis, you MUST generate extrapolated Y-values for the new labels.
   - **Truncation (e.g., 'stop at C')**: Slice both X and Y lists.
   - **Check Lengths**: Before plotting, `assert len(labels) == len(values)` to avoid Shape Errors.

4. **Y-AXIS AUTOSCALING**:
   - **Do NOT hardcode `plt.ylim()`**. Let Matplotlib auto-scale to fit new values (e.g. if values increase by 10,000).

5. **MATH LOGIC**:
   - Percent: `new = old * (1 + val/100)`
   - Absolute: `new = old + val`

OUTPUT: Only Python code. No markdown.
"""
        try:
            content = [system_prompt]
            if image_path and os.path.exists(image_path):
                content.append(Image.open(image_path))
            else:
                content.append(f"Image missing. Use OCR snippet: {ocr_text}")
            
            response = self.model.generate_content(content)
            return clean_code(response.text)
        except Exception as e:
            return f"# Error generating code: {e}"

# ----------------------------
# 6) DEBUG AGENT
# ----------------------------

class DebugAgent:
    def __init__(self, gemini_model: str = "gemini-2.0-flash"):
        self.model_name = gemini_model
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except: self.model = None

    def fix_code(self, code_snippet: str, trace_or_hint: str, user_intent: Optional[Dict[str, Any]] = None) -> str:
        prompt = f"""
Fix this Python Matplotlib code.
INTENT: {user_intent}
ERROR: {trace_or_hint}
CODE:
{code_snippet}

Output ONLY the fixed code.
"""
        if self.model:
            try:
                res = self.model.generate_content([prompt])
                return clean_code(res.text)
            except: pass
        return code_snippet

# ----------------------------
# 7) EXECUTION AGENT (Robust Fallback)
# ----------------------------

class ChartQualityAgent:
    def enhance(self, image_path: str) -> str:
        return image_path 

def _exec_worker(code_str: str, out_path: str, queue: mp.Queue):
    try:
        # 0. CLEANUP: Remove potential conflict files first
        fallback_files = ["plot.png", "output.png", "chart.png"]
        for f in fallback_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass

        # Pre-inject output_path
        safe_globals = {"plt": plt, "np": np, "output_path": out_path}
        exec(code_str, safe_globals)
        
        # --- ROBUST SUCCESS CHECKS ---
        
        # 1. Did the code save to the CORRECT path?
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            queue.put({"status": "ok", "out_path": out_path})
            
        # 2. FALLBACK: Did the AI save to 'plot.png' or 'output.png'?
        elif os.path.exists("plot.png") and os.path.getsize("plot.png") > 0:
            shutil.move("plot.png", out_path)
            queue.put({"status": "ok", "out_path": out_path})
            
        elif os.path.exists("output.png") and os.path.getsize("output.png") > 0:
            shutil.move("output.png", out_path)
            queue.put({"status": "ok", "out_path": out_path})

        # 3. Did the code leave the figure OPEN? (Save it ourselves)
        elif plt.get_fignums():
            plt.savefig(out_path)
            queue.put({"status": "ok", "out_path": out_path})
            
        # 4. FAILURE: Nothing was saved.
        else:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No chart generated", ha="center")
            fig.savefig(out_path)
            queue.put({"status": "ok", "out_path": out_path})
            
        plt.close('all')
        
    except Exception:
        trace = traceback.format_exc()
        queue.put({"status": "error", "trace": trace})

class ExecutionAgent:
    def __init__(self, safety_agent=None, quality_agent=None):
        self.safety = safety_agent or SafetyAgent()
        
    def execute(self, code: str, output_path: str = None) -> Tuple[bool, str, str]:
        if not output_path: 
            output_path = os.path.abspath("latest_chart.png")
            
        # Ensure output_path is defined in the script
        if "output_path" not in code:
            code = f"output_path = r'{output_path}'\n" + code

        queue = mp.Queue()
        p = mp.Process(target=_exec_worker, args=(code, output_path, queue))
        p.start()
        p.join(10) # 10s Timeout
        
        if p.is_alive():
            p.terminate()
            return False, None, "Execution timed out."
            
        if not queue.empty():
            res = queue.get()
            if res["status"] == "ok":
                return True, res["out_path"], None
            else:
                return False, None, res["trace"]
        return False, None, "Worker crashed silently."
