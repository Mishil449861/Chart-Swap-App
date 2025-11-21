# --- STEP 1: IMPORTS ---
from agents import (SafetyAgent, DebugAgent, ChartQualityAgent, IntentAgent, DataExtractorAgent, PlanningAgent, CodeGeneratorAgent, ExecutionAgent)

import tkinter as tk
from tkinter import Entry, Button, Label, Frame, PhotoImage, END
from PIL import Image, ImageTk, ImageGrab
import os
import io
import time
import google.generativeai as genai
from google.cloud import vision
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# --- STEP 2: SETUP ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

original_screenshot_path = "original_screenshot.png"
current_chart_path = ""


# --- MAIN APPLICATION WITH MULTI AGENTS ---
class ChartSwapApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Chart-Swap v2.0 — Multi-Agent Edition")
        self.root.geometry("900x700")

        # AGENTS INITIALIZED HERE
        self.intent_agent = IntentAgent()
        self.data_agent = DataExtractorAgent()
        self.planner_agent = PlanningAgent()
        self.code_agent = CodeGeneratorAgent()
        self.safety_agent = SafetyAgent()
        self.debug_agent = DebugAgent()
        self.quality_agent = ChartQualityAgent()
        self.exec_agent = ExecutionAgent()

        # --- UI WIDGETS ---
        self.image_label = Label(self.root)
        self.image_label.pack(pady=10, expand=True)

        control_frame = Frame(self.root)
        control_frame.pack(pady=10)

        self.prompt_entry = Entry(control_frame, width=70, font=("Arial", 14))
        self.prompt_entry.pack(side=tk.LEFT, padx=10)

        self.submit_button = Button(control_frame, text="Go", command=self.handle_submit_prompt)
        self.submit_button.pack(side=tk.LEFT)

        button_frame = Frame(self.root)
        button_frame.pack(pady=5)

        self.snapshot_button = Button(button_frame, text="Take Screenshot (in 3s)", command=self.handle_snapshot)
        self.snapshot_button.pack(side=tk.LEFT, padx=5)

        self.revert_button = Button(button_frame, text="Revert to Original", command=self.handle_revert)
        self.revert_button.pack(side=tk.LEFT, padx=5)

        self.status_label = Label(self.root, text="Welcome! Take a snapshot to start.", font=("Arial", 12), fg="blue")
        self.status_label.pack(pady=10)


    # ---------------------------- UI HELPERS ----------------------------
    def update_status(self, message, color="blue"):
        self.status_label.config(text=message, fg=color)
        self.root.update_idletasks()

    def load_image_into_ui(self, image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((800, 550))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            self.update_status(f"Error loading image: {e}", "red")


    # ------------------------------ SNAPSHOT ------------------------------
    def handle_snapshot(self):
        global current_chart_path, original_screenshot_path
        self.update_status("Taking screenshot in 3 seconds...")
        time.sleep(3)
        try:
            snapshot = ImageGrab.grab()
            snapshot.save(original_screenshot_path)
            current_chart_path = original_screenshot_path
            self.load_image_into_ui(original_screenshot_path)
            self.update_status("Screenshot captured!", "green")
        except Exception as e:
            self.update_status(f"Snapshot failed: {e}", "red")


    # ------------------------------ REVERT ------------------------------
    def handle_revert(self):
        global current_chart_path
        current_chart_path = original_screenshot_path
        self.load_image_into_ui(current_chart_path)
        self.update_status("Reverted to original.", "blue")


    # ------------------------------ MAIN PIPELINE ------------------------------
    def handle_submit_prompt(self):
        global current_chart_path
        
        user_prompt = self.prompt_entry.get()
        if not user_prompt:
            self.update_status("Prompt empty!", "red")
            return
        if not current_chart_path:
            self.update_status("No screenshot yet!", "red")
            return

        try:
            # ================================
            # 1) OCR
            # ================================
            self.update_status("Step 1/6: Extracting chart text...")
            ocr_text = self.get_text_from_image(current_chart_path)
            if not ocr_text:
                return

            # ================================
            # 2) Intent Detection
            # ================================
            self.update_status("Step 2/6: Analyzing intent...")
            intent = self.intent_agent.parse(user_prompt)

            # ================================
            # 3) Extract Structured Data
            # ================================
            self.update_status("Step 3/6: Extracting chart data...")
            structured_data = self.data_agent.extract_data(ocr_text)

            # ================================
            # 4) Planning (decides which agents)
            # ================================
            self.update_status("Step 4/6: Planning...")
            plan = self.planner_agent.create_plan(intent)

            # ================================
            # 5) Code Generation
            # ================================
            self.update_status("Step 5/6: Generating chart code...")
            generated_code = self.code_agent.generate(structured_data, intent, ocr_text)

            # ================================
            # SAFETY CHECK
            # ================================
            safe, reason = self.safety_agent.review(generated_code)
            if not safe:
                self.update_status("Unsafe code blocked — requesting fix...", "red")
                generated_code = self.debug_agent.fix_code(
                    generated_code,
                    f"SAFETY FAIL: {reason}"
                )

            # ================================
            # EXECUTE CODE
            # ================================
            self.update_status("Executing chart code...")
            success, new_path, error_log = self.exec_agent.execute(generated_code)

            # If failed → send to Debug Agent
            if not success:
                self.update_status("Execution failed — check console.", "red")
                print("=== EXECUTION ERROR LOG ===")
                print(error_log)

                self.update_status("Fixing execution error...", "orange")
                repaired_code = self.debug_agent.fix_code(generated_code, error_log)
                success, new_path, new_error_log = self.exec_agent.execute(repaired_code)
                if not success:
                    self.update_status("Could not fix chart error — check console.", "red")
                    print("=== SECOND EXECUTION ERROR LOG ===")
                    print(new_error_log)
                    return
            # ================================
            # QUALITY ENHANCER
            # ================================
            self.update_status("Enhancing chart visual quality...")
            final_path = self.quality_agent.enhance(new_path)

            # ================================
            # UI SWAP
            # ================================
            current_chart_path = final_path
            self.load_image_into_ui(final_path)
            self.prompt_entry.delete(0, END)
            self.update_status("Success!", "green")

        except Exception as e:
            self.update_status(f"Unexpected error: {e}", "red")



    # ------------------------------ OCR ------------------------------
    def get_text_from_image(self, image_path):
        try:
            client = vision.ImageAnnotatorClient()
            with io.open(image_path, 'rb') as f:
                content = f.read()
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            if response.error.message:
                raise Exception(response.error.message)
            return response.full_text_annotation.text
        except Exception as e:
            self.update_status(f"OCR Error: {e}", "red")
            return None



# ------------------------------ RUN ------------------------------
if __name__ == "__main__":
    main_window = tk.Tk()
    app = ChartSwapApp(main_window)
    main_window.mainloop()