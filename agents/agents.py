# agents.py
import re
import os
import io
import ast
import time
import uuid
import tempfile
import traceback
import multiprocessing as mp
from typing import Tuple, Dict, Any, Optional, List

import matplotlib.pyplot as plt
import numpy as np

# Optional LLM library — used by CodeGenerator and DebugAgent
import google.generativeai as genai
from PIL import Image

# ----------------------------
# Utility helpers
# ----------------------------

def _make_tmp_image_path(prefix="chart", ext="png"):
    return os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}")

def _make_tmp_log_path(prefix="exec_error", ext="log"):
    return os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}")

def _make_tmp_code_path(prefix="generated_code", ext="py"):
    return os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}")

def _safe_extract_numbers_from_string(s: str) -> List[float]:
    nums = re.findall(r"-?\d+\.\d+|-?\d+", s)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except:
            continue
    return out

def clean_code(code_str: str) -> str:
    """
    Remove Markdown code fences and leading/trailing whitespace.
    """
    if not code_str:
        return ""
    code_str = re.sub(r"^```[a-zA-Z]*", "", code_str)  # remove opening ```python or ```
    code_str = code_str.replace("```", "")  # remove any remaining ```
    return code_str.strip()

# ----------------------------
# 1) SAFETY AGENT
# ----------------------------

class SafetyAgent:
    # extended disallowed patterns: block plt.show(), input() and obvious infinite loop patterns
    DEFAULT_DISALLOWED_PATTERNS = [
        r"\bos\.system\b",
        r"\bsubprocess\b",
        r"\bsubprocess\.",
        r"\bopen\s*\(",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\b__import__\s*\(",
        r"\bshutil\b",
        r"\bsocket\b",
        r"\brequests\b",
        r"\bwget\b",
        r"\bcurl\b",
        r"while\s+True\b",
        r"for\s+\w+\s+in\s+range\(\s*10{4,}",  # huge range pattern
        r"for\s+\w+\s+in\s+range\(\s*\d{6,}",  # very large integer in range()
        r"import\s+sys",
        r"import\s+os",
        r"plt\.show\s*\(",
        r"\binput\s*\(",
        r"time\.sleep\(\s*[1-9]\d+",  # very long sleeps (>=10s)
    ]

    def __init__(self, extra_patterns: Optional[List[str]] = None):
        self.patterns = list(self.DEFAULT_DISALLOWED_PATTERNS)
        if extra_patterns:
            self.patterns += extra_patterns

    def review(self, code: str) -> Tuple[bool, str]:
        """
        Returns (is_safe, message). If suspicious pattern found, returns False + reason.
        """
        try:
            ast.parse(code)
        except SyntaxError:
            # don't fail on snippets, but keep scanning for dangerous tokens
            pass
        for patt in self.patterns:
            if re.search(patt, code, flags=re.IGNORECASE):
                return False, f"Blocked pattern matched: {patt}"
        # Heuristic: too many file writes or os calls
        write_ops = len(re.findall(r"\.save\(|plt\.savefig|open\s*\(", code))
        if write_ops > 10:
            return False, "Too many file write operations detected."
        return True, "Code passed safety review."

# ----------------------------
# 2) INTENT AGENT
# ----------------------------

class IntentAgent:
    def __init__(self):
        pass

    def parse(self, user_prompt: str) -> Dict[str, Any]:
        prompt = (user_prompt or "").lower()
        intent = {"raw": user_prompt, "action": None, "target": None, "value": None, "color": None, "extras": {}}

        if re.search(r"\bincrease\b|\braise\b|\bup by\b", prompt):
            intent["action"] = "increase"
            m = re.search(r"by\s+(-?\d+(\.\d+)?)\s*%?", prompt)
            if m:
                intent["value"] = float(m.group(1))
        elif re.search(r"\bdecrease\b|\breduce\b|\bdown by\b", prompt):
            intent["action"] = "decrease"
            m = re.search(r"by\s+(-?\d+(\.\d+)?)\s*%?", prompt)
            if m:
                intent["value"] = float(m.group(1))
        elif re.search(r"\bhighlight\b|\bmake\b.*\bred\b|\bcolor\b", prompt):
            intent["action"] = "highlight"
            m_color = re.search(r"\b(red|blue|green|orange|purple|yellow|black|grey|gray)\b", prompt)
            if m_color:
                intent["color"] = m_color.group(1)
        elif re.search(r"\bsort\b", prompt):
            intent["action"] = "sort"
        elif re.search(r"\bannotate\b|\blabel\b", prompt):
            intent["action"] = "annotate"

        m_q = re.search(r"\b(q[1-4]|quarter\s*[1-4]|q[1-4]\b)", prompt)
        if m_q:
            intent["target"] = m_q.group(1)

        if intent["value"] is None:
            m2 = re.search(r"(-?\d+(\.\d+)?)\s*%?", prompt)
            if m2:
                try:
                    val = float(m2.group(1))
                    if intent["action"] in ("increase", "decrease") or "%" in prompt:
                        intent["value"] = val
                except:
                    pass

        return intent

    # compatibility wrapper for older name
    def analyze_intent(self, user_prompt: str) -> Dict[str, Any]:
        return self.parse(user_prompt)

# ----------------------------
# 3) DATA EXTRACTOR AGENT
# ----------------------------

class DataExtractorAgent:
    def __init__(self):
        pass

    def extract_from_ocr(self, ocr_text: str) -> Dict[str, Any]:
        lines = [ln.strip() for ln in (ocr_text or "").splitlines() if ln.strip()]
        labels = []
        values = []
        meta = {"title": None, "y_label": None, "x_label": None}

        if lines:
            if not re.search(r"\d", lines[0]) and len(lines[0].split()) > 1:
                meta["title"] = lines[0]
                lines = lines[1:]

        for ln in lines:
            nums = _safe_extract_numbers_from_string(ln)
            if nums:
                label_match = re.match(r"^([^\d]+?)\s*-?\d", ln)
                if label_match:
                    label = label_match.group(1).strip(":- ")
                else:
                    label = ln.split()[0]
                labels.append(label)
                values.append(nums[0])
            else:
                if len(ln.split()) <= 4:
                    labels.append(ln)
        if len(values) and len(labels) != len(values):
            if len(labels) > len(values):
                labels = labels[:len(values)]
            else:
                while len(labels) < len(values):
                    labels.append(f"v{len(labels)+1}")

        return {"labels": labels, "values": values, "meta": meta}

    # compatibility alias used by many callers
    def extract_data(self, ocr_text: str) -> Dict[str, Any]:
        return self.extract_from_ocr(ocr_text)

# ----------------------------
# 4) PLANNING AGENT
# ----------------------------

class PlanningAgent:
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def plan(self, intent: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        plan = {
            "run_data_extraction": True,
            "run_code_generation": True,
            "safety_review_first": True,
            "max_retries": self.max_retries
        }
        if not data.get("values"):
            plan["run_data_extraction"] = True
        return plan

    # compatibility wrapper
    def create_plan(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        return self.plan(intent, data={})

# ----------------------------
# 5) CODE GENERATOR AGENT
# ----------------------------

class CodeGeneratorAgent:
    def __init__(self, gemini_model: str = "gemini-2.5-pro"):
        self.model_name = gemini_model
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except Exception:
            self.model = None

    def _generate_from_structured(self, data: Dict[str, Any], intent: Dict[str, Any], ocr_text: str = "", image_path: str = None) -> str:
        labels = data.get("labels", [])
        values = data.get("values", [])
        meta = data.get("meta", {})
        
        # Updated summary to include visual instructions
        summary = f"Title: {meta.get('title')}\nLabels: {labels}\nValues from OCR (may be unreliable): {values}\nIntent: {intent}\nOCR snippet: {ocr_text[:500]}"
        
        system_prompt = f"""
You are an expert Python matplotlib assistant. Produce runnable Python code that uses matplotlib to recreate the attached chart as faithfully as possible.

CRITICAL INSTRUCTIONS FOR VISUAL ACCURACY:
1. VISION OVER OCR: Look at the provided image. Use it as the PRIMARY SOURCE OF TRUTH for data values, colors, and markers. The OCR data provided is likely inaccurate for 3D charts.
2. COLORS: Match the line colors from the image exactly (e.g., if a line is green in the image, make it green in the code), unless the Intent explicitly changes it.
3. DATA ESTIMATION: Visually estimate the y-values for each point relative to the grid lines in the image.
4. INTENT: Apply the user's intent modification: {intent}

Requirements:
- Use only matplotlib (and numpy if necessary).
- Do not include backticks or markdown.
- Define the data arrays explicitly using your visual estimates.
- Keep code concise and save the result to a PNG using plt.savefig(output_path)
- Do not write files other than saving the image to the provided path.
- If you cannot follow the intent, respond with the single token: ERROR_CANNOT_FULFILL
"""
        user_prompt = f"{system_prompt}\nDATA SUMMARY:\n{summary}\n\nProduce only the python code."
        
        if self.model:
            try:
                # Prepare content payload for multimodal request
                content_payload = [user_prompt]
                
                # If we have a valid image path, open it and attach it to the prompt
                if image_path and os.path.exists(image_path):
                    try:
                        img_obj = Image.open(image_path)
                        content_payload.append(img_obj)
                    except Exception as img_err:
                        print(f"Warning: Could not load image for Vision context: {img_err}")

                response = self.model.generate_content(content_payload)
                code = response.text or ""
                return clean_code(code)
            except Exception as e:
                raise RuntimeError(f"LLM generation failed: {e}")
        else:
            if values:
                code_lines = [
                    "import matplotlib.pyplot as plt",
                    f"labels = {repr(labels)}",
                    f"values = {repr(values)}",
                    "fig, ax = plt.subplots(figsize=(8,5))",
                    "ax.bar(labels, values)",
                    "ax.set_title('Generated Chart')",
                    "plt.tight_layout()",
                    "plt.savefig('output.png')"
                ]
                return "\n".join(code_lines)
            return "ERROR_CANNOT_FULFILL"

    def generate(self, *args, **kwargs) -> str:
        # Support flexible arguments (args or kwargs)
        if args:
            data = args[0] if len(args) > 0 else {}
            intent = args[1] if len(args) > 1 else {}
            ocr_text = args[2] if len(args) > 2 else ""
            # Try to retrieve image_path if it was passed as kwarg alongside args
            image_path = kwargs.get("image_path")
        else:
            data = kwargs.get("structured_data") or kwargs.get("data") or {}
            intent = kwargs.get("intent") or {}
            ocr_text = kwargs.get("ocr_text") or kwargs.get("user_prompt") or ""
            image_path = kwargs.get("image_path")

        return self._generate_from_structured(data or {}, intent or {}, ocr_text or "", image_path=image_path)

# ----------------------------
# 6) DEBUG AGENT
# ----------------------------

class DebugAgent:
    def __init__(self, gemini_model: str = "gemini-2.5-pro"):
        self.model_name = gemini_model
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except Exception:
            self.model = None

    def repair(self, failing_code: str, error_trace: str, user_intent: Dict[str, Any]) -> str:
        prompt = f"""
You are a Python debugging assistant specialized in matplotlib visualizations.

You receive:
1) User intent: {user_intent}
2) The code that failed:
---
{failing_code}
---
3) The Python stacktrace:
---
{error_trace}
---

Task: Identify the cause of the error and output ONLY the corrected Python code that fixes the bug and adheres to the user intent. Do NOT include explanatory text or markdown.

If you cannot fix it, output exactly: ERROR_REPAIR_FAILED
"""
        if self.model:
            try:
                response = self.model.generate_content([prompt])
                out = (response.text or "").strip()
                return clean_code(out)
            except Exception as e:
                raise RuntimeError(f"LLM debug call failed: {e}")
        else:
            if "NameError" in error_trace:
                m = re.search(r"NameError: name '(.+?)' is not defined", error_trace)
                if m:
                    varname = m.group(1)
                    repaired = f"{varname} = []\n" + failing_code
                    return clean_code(repaired)
            return "ERROR_REPAIR_FAILED"

    def fix_code(self, code_snippet: str, trace_or_hint: str, user_intent: Optional[Dict[str, Any]] = None) -> str:
        if user_intent is None:
            user_intent = {}
        return self.repair(code_snippet, trace_or_hint or "", user_intent)

# ----------------------------
# 7) CHART QUALITY AGENT
# ----------------------------

class ChartQualityAgent:
    def __init__(self, dpi: int = 160):
        self.dpi = dpi

    def enhance(self, image_path: str) -> str:
        try:
            if not image_path or not os.path.exists(image_path):
                return image_path
            img = Image.open(image_path)
            try:
                img.save(image_path, dpi=(self.dpi, self.dpi))
            except Exception:
                img.save(image_path)
            return image_path
        except Exception as e:
            print("ChartQualityAgent.enhance() failed:", e)
            return image_path

# ----------------------------
# 8) EXECUTION AGENT
# ----------------------------

def _exec_worker(code_str: str, out_path: str, queue: mp.Queue):
    try:
        safe_globals = {
            "__name__": "__main__",
            "plt": plt,
            "np": np,
        }
        # Ensure the user code knows where to save the chart
        safe_globals["output_path"] = out_path
        exec(code_str, safe_globals)
        # Force a figure save if user code didn’t
        if not plt.get_fignums():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No chart generated by model", ha="center", va="center")
            fig.savefig(out_path)
        else:
            if not os.path.exists(out_path):
                plt.savefig(out_path)
        plt.close('all')
        if os.path.exists(out_path):
            queue.put({"status": "ok", "out_path": out_path})
        else:
            queue.put({"status": "no_output", "message": f"{out_path} not created."})
    except Exception:
        trace = traceback.format_exc()
        # Attempt to print from worker (may or may not show depending on platform), but always send trace back
        try:
            print("\n\n### WORKER ERROR ###")
            print(trace)
        except Exception:
            pass
        queue.put({"status": "error", "trace": trace})

class ExecutionAgent:
    def __init__(self, safety_agent: Optional[SafetyAgent] = None, quality_agent: Optional[ChartQualityAgent] = None):
        self.safety_agent = safety_agent or SafetyAgent()
        self.quality_agent = quality_agent or ChartQualityAgent()
        self.default_timeout = 10

    def _sanitize_and_report_code(self, code_str: str) -> Tuple[bool, str, Optional[str]]:
        """
        Sanitize and optionally block dangerous constructs.
        Returns (ok_to_run, safe_code, code_log_path)
        """
        code = clean_code(code_str)

        # quick safety review (this duplicates SafetyAgent check but gives earlier feedback)
        ok, reason = self.safety_agent.review(code)
        if not ok:
            return False, code, None

        # remove plt.show() if present (non-blocking)
        if re.search(r"plt\.show\s*\(", code):
            code = re.sub(r"plt\.show\s*\(\s*\)", "", code)
            code = code.strip()

        # block input(...) calls by replacing with raising explicit error (safer than letting it hang)
        if re.search(r"\binput\s*\(", code):
            code = re.sub(r"\binput\s*\(", "raise RuntimeError('input() removed in sandbox, would block') # input(", code)

        # reduce foolishly large sleeps (but still let small sleeps)
        def _clamp_sleep(m):
            inner = m.group(1)
            try:
                sec = float(inner)
                if sec > 5:
                    return "time.sleep(0.01)"
                return m.group(0)
            except:
                return m.group(0)
        code = re.sub(r"time\.sleep\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)", _clamp_sleep, code)

        # Save the final code to a temp file so you can inspect it if it fails/hangs
        code_path = None
        try:
            code_path = _make_tmp_code_path()
            with open(code_path, "w", encoding="utf-8") as f:
                f.write("# Generated code saved by ExecutionAgent\n")
                f.write(code)
        except Exception as e:
            code_path = None
            print("Failed to write generated-code log:", e)

        # print short separator and the code to console
        print("\n\n=== GENERATED CODE START ===")
        if len(code) < 2000:
            print(code)
        else:
            # print a truncated preview if very long
            print(code[:2000])
            print("... (truncated, saved to file)")
        print("=== GENERATED CODE END ===\n\n")
        if code_path:
            print("Saved generated code to:", code_path)

        return True, code, code_path

    def execute(self, code: str, output_path: Optional[str] = None, timeout: Optional[int] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        project_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "latest_chart.png")
        safe, msg = self.safety_agent.review(code)
        if not safe:
            return False, None, f"Safety review failed: {msg}"

        if output_path is None:
            output_path = project_output_path
        else:
            if not os.path.isabs(output_path):
                output_path = os.path.abspath(output_path)
        # Clean code before execution
        code = clean_code(code)

        # If code doesn't explicitly save, append a save call (but we will also sanitize first)
        if "plt.savefig" not in code and "savefig(" not in code:
            code_to_run = f"{code}\nplt.savefig(r'{output_path}')"
        else:
            code_to_run = code

        # Sanitize & report
        ok_to_run, safe_code, code_log_path = self._sanitize_and_report_code(code_to_run)
        if not ok_to_run:
            return False, None, f"Sanitizer blocked code. See contents: {code_log_path or 'no log available'}"

        # Run in worker
        queue = mp.Queue()
        p = mp.Process(target=_exec_worker, args=(safe_code, output_path, queue))
        p.start()
        tlimit = timeout or self.default_timeout
        p.join(tlimit)
        if p.is_alive():
            # timed out; attempt to kill and write a helpful message
            p.terminate()
            p.join()
            # write a helpful timeout log
            timeout_log = None
            try:
                timeout_log = _make_tmp_log_path(prefix="exec_timeout")
                with open(timeout_log, "w", encoding="utf-8") as f:
                    f.write("Execution timed out (possible infinite loop or blocking call).\n")
                    f.write("Generated code saved at:\n")
                    f.write(str(code_log_path) + "\n")
            except Exception as e:
                print("Failed to write timeout log:", e)
                timeout_log = None

            err_msg = "Execution timed out (possible infinite loop)."
            if code_log_path:
                err_msg += f"\nGenerated code saved to: {code_log_path}"
            if timeout_log:
                err_msg += f"\nTimeout trace saved to: {timeout_log}"
            return False, None, err_msg

        try:
            result = queue.get_nowait()
        except Exception:
            result = {"status": "no_result", "message": "No result from worker."}

        if result.get("status") == "ok":
            return True, result.get("out_path"), None
        else:
            # Build a helpful trace string
            trace = result.get("trace") or result.get("message") or "Unknown execution error."

            # Print clear separators so logs are obvious in console
            print("\n\n=== EXECUTION AGENT ERROR TRACE ===")
            print(trace)
            print("=== END EXECUTION AGENT ERROR TRACE ===\n\n")

            # Save trace to a temp log file for inspection
            try:
                log_path = _make_tmp_log_path()
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("=== EXECUTION AGENT ERROR TRACE ===\n")
                    f.write(trace)
                    f.write("\n=== END EXECUTION AGENT ERROR TRACE ===\n")
            except Exception as e:
                # If writing log fails, include that fact in returned error
                log_path = None
                print("Failed to write execution trace log:", e)

            # Return error string that includes log path (if available)
            if log_path:
                err_str = f"{trace}\n\nSaved execution trace to: {log_path}"
            else:
                err_str = trace
            return False, None, err_str

# ----------------------------
# Convenience factory
# ----------------------------

class AgentStack:
    def __init__(self, gemini_api_key: Optional[str] = None):
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
            except Exception:
                pass

        self.safety = SafetyAgent()
        self.intent = IntentAgent()
        self.data_extractor = DataExtractorAgent()
        self.planner = PlanningAgent()
        self.codegen = CodeGeneratorAgent()
        self.debugger = DebugAgent()
        self.quality = ChartQualityAgent()
        self.exec = ExecutionAgent(safety_agent=self.safety, quality_agent=self.quality)

    def run_request(self, image_path: str, ocr_text: str, user_prompt: str) -> Dict[str, Any]:
        result = {"success": False, "output_path": None, "error": None, "attempts": 0}
        intent = self.intent.parse(user_prompt)
        data = self.data_extractor.extract_from_ocr(ocr_text)
        plan = self.planner.plan(intent, data)

        attempts = 0
        max_retries = plan.get("max_retries", 2)
        while attempts <= max_retries:
            attempts += 1
            result["attempts"] = attempts
            try:
                # PASSED IMAGE PATH TO ENABLE VISION
                code = self.codegen.generate(data, intent, ocr_text, image_path=image_path)
                code = clean_code(code)
            except Exception as e:
                result["error"] = f"Code generation failed: {e}"
                return result

            if not code or "ERROR_CANNOT_FULFILL" in code:
                result["error"] = "Code generator refused or produced no code."
                return result

            out_path = _make_tmp_image_path()
            success, outp, err = self.exec.execute(code, output_path=out_path, timeout=12)
            if success:
                result["success"] = True
                result["output_path"] = outp
                return result
            else:
                err = err or ""
                result["error"] = err
                try:
                    repaired = self.debugger.repair(code, err, intent)
                    repaired = clean_code(repaired)
                    if not repaired or "ERROR_REPAIR_FAILED" in repaired:
                        continue
                    out_path2 = _make_tmp_image_path()
                    success2, outp2, err2 = self.exec.execute(repaired, output_path=out_path2, timeout=12)
                    if success2:
                        result["success"] = True
                        result["output_path"] = outp2
                        return result
                    else:
                        result["error"] = err2
                        continue
                except Exception as e:
                    result["error"] = f"Debugging failed: {e}"
                    continue

        return result
