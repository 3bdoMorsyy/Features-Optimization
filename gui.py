import genetic as ge
import tkinter as tk
from tkinter import scrolledtext


class DocumentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Get Best Features For HeartDiseases DataSet")

        self.root.configure(bg="#fff")
        self.root.geometry("600x600")

        self.input_label = tk.Label(
            root,
            text="Enter (File Path, Target Column, Population, Number of generations):",
            bg="#FFFFFF",
            fg="#333333",
            font=("Arial", 12, "bold"),
        )
        self.input_label.pack(pady=(20, 5))
        self.input_text = scrolledtext.ScrolledText(
            root,
            width=60,
            height=10,
            bg="#F0F0F0",
            fg="#333333",
            font=("Arial", 10),
        )
        self.input_text.pack(pady=(0, 10))

        # Buttons
        self.Get_Best_Features_button = tk.Button(
            root,
            text="Get Best Features",
            command=self.Get_Best_Features,
            bg="#4CAF50",
            fg="#FFFFFF",
            font=("Arial", 10, "bold"),
        )
        self.Get_Best_Features_button.pack()

        self.reset_button = tk.Button(
            root,
            text="reset",
            command=self.reset,
            bg="#4CAF50",
            fg="#FFFFFF",
            font=("Arial", 10, "bold"),
        )
        self.reset_button.pack()

        # Output
        self.output_label = tk.Label(
            root, text="Output:", bg="#FFFFFF", fg="#333333", font=("Arial", 12, "bold")
        )
        self.output_label.pack(pady=(20, 5))
        self.output_text = scrolledtext.ScrolledText(
            root, width=60, height=10, bg="#F0F0F0", fg="#000"
        )
        self.output_text.pack(pady=(0, 10))

    def Get_Best_Features(self):
        text = self.input_text.get("1.0", "end-1c")
        text = text.replace("\n", "").replace("'", "").replace('"', "")
        arr = text.replace(" ", "").split(",")
        if len(arr) == 4:
            output_text = ge.get_best_features(arr[0], arr[1], int(arr[2]), int(arr[3]))
        elif len(arr) == 3:
            output_text = ge.get_best_features(arr[0], arr[1], int(arr[2]))
        elif len(arr) == 2:
            output_text = ge.get_best_features(arr[0], arr[1])
        self.output_text.delete("1.0", tk.END)
        for o in output_text:
            self.output_text.insert(tk.END, f"{o}\n")

    def reset(self):
        self.output_text.delete("1.0", tk.END)


root = tk.Tk()
app = DocumentAnalyzerApp(root)
root.mainloop()
