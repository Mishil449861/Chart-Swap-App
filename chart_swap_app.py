import tkinter as tk
from tkinter import Entry, Button, Label, Frame, END, messagebox
from PIL import Image, ImageTk, ImageGrab
import os
import io
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Import the updated agents
from agents import (
    SafetyAgent, 
    DebugAgent, 
    IntentAgent, 
    DataExtractorAgent, 
    PlanningAgent, 
    CodeGeneratorAgent, 
    ExecutionAgent,
    ChartQualityAgent
)

# --- CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

# Global paths
ORIGINAL_PATH = "original_screenshot.png"
OUTPUT_PATH = "latest_chart.png"

class ChartSwapApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Chart-Swap v3.0 — Intelligent Vision")
        self.root.geometry("1000x750")

        # --- INITIALIZE AGENTS ---
        self.intent_agent = IntentAgent()
        self.data_agent = DataExtractorAgent()
        self.code_agent = CodeGeneratorAgent() 
        self.safety_agent = SafetyAgent()
        self.debug_agent = DebugAgent()
        self.exec_agent = ExecutionAgent()
        self.quality_agent = ChartQualityAgent()

        # State
        self.current_image_path = None

        # --- UI LAYOUT ---
        
        # 1. Image Display Area
        self.image_frame = Frame(self.root, bg="#f0f0f0", width=800, height=500)
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.image_frame.pack_propagate(False) # Prevent shrinking
        
        self.image_label = Label(self.image_frame, text="No Image Captured", bg="#f0f0f0", font=("Arial", 14, "bold"))
        self.image_label.pack(expand=True)

        # 2. Control Area
        control_frame = Frame(self.root)
        control_frame.pack(pady=10)

        # Prompt Entry
        Label(control_frame, text="Instruction:").pack(side=tk.LEFT, padx=5)
        self.prompt_entry = Entry(control_frame, width=60, font=("Arial", 12))
        self.prompt_entry.pack(side=tk.LEFT, padx=5)
        self.prompt_entry.bind("<Return>", lambda event: self.handle_generate())

        # Generate Button
        self.generate_btn = Button(control_frame, text="Generate Chart", command=self.handle_generate, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.generate_btn.pack(side=tk.LEFT, padx=10)

        # 3. Utility Buttons
        btn_frame = Frame(self.root)
        btn_frame.pack(pady=5)

        self.snap_btn = Button(btn_frame, text="📸 Take Screenshot (3s)", command=self.handle_snapshot)
        self.snap_btn.pack(side=tk.LEFT, padx=10)

        self.revert_btn = Button(btn_frame, text="⏪ Revert to Original", command=self.handle_revert)
        self.revert_btn.pack(side=tk.LEFT, padx=10)

        # 4. Status Bar
        self.status_label = Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # --- UI HELPERS ---
    def update_status(self, message, color="black"):
        self.status_label.config(text=message, fg=color)
        self.root.update()

    def load_image(self, path):
        if not os.path.exists(path):
            self.update_status(f"Image not found: {path}", "red")
            return

        try:
            img = Image.open(path)
            # Resize for display while keeping aspect ratio
            img.thumbnail((900, 550), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo # Keep reference
            self.current_image_path = path
        except Exception as e:
            self.update_status(f"Error loading image: {e}", "red")

    # --- ACTIONS ---
    def handle_snapshot(self):
        self.update_status("Taking screenshot in 3 seconds...", "blue")
        self.root.update()
        time.sleep(3)
        
        try:
            # Capture screen
            screenshot = ImageGrab.grab()
            screenshot.save(ORIGINAL_PATH)
            self.load_image(ORIGINAL_PATH)
            self.update_status("Screenshot captured successfully.", "green")
        except Exception as e:
            self.update_status(f"Snapshot failed: {e}", "red")

    def handle_revert(self):
        if os.path.exists(ORIGINAL_PATH):
            self.load_image(ORIGINAL_PATH)
            self.update_status("Reverted to original screenshot.", "blue")
        else:
            self.update_status("No original screenshot found.", "red")

    def handle_generate(self):
        user_prompt = self.prompt_entry.get()
        
        if not user_prompt:
            self.update_status("Please enter an instruction.", "red")
            return
        if not self.current_image_path:
            self.update_status("Please take a screenshot first.", "red")
            return

        # Disable button to prevent double-click
        self.generate_btn.config(state=tk.DISABLED)
        
        try:
            # 1. INTENT Parsing
            self.update_status("Analyzing intent...", "blue")
            intent = self.intent_agent.parse(user_prompt)
            print(f"DEBUG: Parsed Intent: {intent}")

            # 2. DATA Extraction (Heuristic/OCR)
            self.update_status("Extracting context...", "blue")
            # We pass dummy text since we are relying on Vision mainly now
            data = self.data_agent.extract_data("Image input used.") 

            # 3. CODE GENERATION (Vision)
            self.update_status("Generating Python code (Vision Enabled)...", "blue")
            # CRITICAL: Pass the image path so the LLM can "see" the chart
            code = self.code_agent.generate(data, intent, "", image_path=self.current_image_path)
            
            # 4. SAFETY CHECK
            safe, reason = self.safety_agent.review(code)
            if not safe:
                self.update_status(f"Safety violation: {reason}. Attempting repair...", "orange")
                code = self.debug_agent.fix_code(code, f"Safety violation: {reason}", intent)

            # 5. EXECUTION
            self.update_status("Rendering chart...", "blue")
            # CRITICAL: Pass explicit output_path to ensure file is found
            success, result_path, error_msg = self.exec_agent.execute(code, output_path=OUTPUT_PATH)

            # 6. ERROR RECOVERY LOOP
            if not success:
                self.update_status("Execution failed. Attempting to fix code...", "orange")
                print(f"DEBUG: Execution Error: {error_msg}")
                
                # Ask Debug Agent to fix it
                fixed_code = self.debug_agent.fix_code(code, error_msg, intent)
                success, result_path, error_msg = self.exec_agent.execute(fixed_code, output_path=OUTPUT_PATH)

            # 7. FINAL DISPLAY
            if success:
                self.load_image(result_path)
                self.update_status("Chart generated successfully!", "green")
                self.prompt_entry.delete(0, END)
            else:
                self.update_status("Failed to generate chart. Check console for details.", "red")
                print(f"FINAL ERROR: {error_msg}")

        except Exception as e:
            self.update_status(f"System Error: {e}", "red")
            traceback.print_exc()
        finally:
            self.generate_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    # Ensure cleanup of old temp files (optional)
    if os.path.exists(OUTPUT_PATH):
        try: os.remove(OUTPUT_PATH)
        except: pass

    main_window = tk.Tk()
    app = ChartSwapApp(main_window)
    main_window.mainloop()
