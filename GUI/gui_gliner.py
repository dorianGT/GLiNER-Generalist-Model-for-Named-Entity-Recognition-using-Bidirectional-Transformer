import tkinter as tk
import customtkinter as ctk
from action import submit_action 

ctk.set_appearance_mode("light")

# Main window setup
root = ctk.CTk()
root.title("GLInER")
root.geometry("800x800")

# Header Section
header_label = ctk.CTkLabel(root, text="GLInER", font=("Arial", 20, "bold"), text_color="#f53d99")
header_label.pack(pady=10)

# Content Frame
content = ctk.CTkFrame(root, fg_color="white", corner_radius=15)
content.pack(pady=20, padx=20, fill="both", expand=True)

# Configure grid columns and rows to control the layout better
content.grid_columnconfigure(0, weight=0, minsize=10)  # First column (labels) does not grow
content.grid_columnconfigure(1, weight=1)  # Second column (inputs and outputs) should grow

# Input Fields and Labels (position labels on top of the input fields)
input_label = ctk.CTkLabel(content, text="Input Text", font=("Arial", 16), text_color="#f53d99")
input_label.grid(row=0, column=1, padx=(10, 20), pady=10, sticky="w")

# Replace CTkEntry with CTkTextbox for multi-line input
input_textbox = ctk.CTkTextbox(content, width=300, height=100, corner_radius=10, border_width=1, border_color="lightgray")
input_textbox.grid(row=1, column=1, padx=(10, 20), pady=10, sticky="nsew")

label_label = ctk.CTkLabel(content, text="Label Text", font=("Arial", 16), text_color="#f53d99")
label_label.grid(row=2, column=1, padx=(10, 20), pady=10, sticky="w")

label_entry = ctk.CTkEntry(content, width=300, height=40, corner_radius=10, border_width=1, border_color="lightgray")
label_entry.grid(row=3, column=1, padx=(10, 20), pady=10, sticky="nsew")

# Output Label with hover effect
output_label = ctk.CTkLabel(content, text="Output", font=("Arial", 16), text_color="#f53d99")
output_label.grid(row=4, column=1, padx=(10, 20), pady=10, sticky="w")

output_frame = tk.Canvas(content, bg="#ffffff", bd=0, highlightthickness=0)
output_frame.grid(row=5, column=1, padx=(10, 20), pady=20, sticky="nsew")

output_frame.create_oval(0, 0, 20, 20, fill="#ffffff", outline="#ffffff")
output_frame.create_oval(300, 0, 320, 20, fill="#ffffff", outline="#ffffff")
output_frame.create_oval(0, 100, 20, 120, fill="#ffffff", outline="#ffffff")
output_frame.create_oval(300, 100, 320, 120, fill="#ffffff", outline="#ffffff")
output_frame.create_rectangle(20, 0, 300, 100, outline="#ffffff", fill="#ffffff")

output_textbox = tk.Text(output_frame, width=40, height=10, wrap=tk.WORD, bg="white", fg="black", bd=0, highlightthickness=0, relief="flat", font=("Arial", 15))
output_textbox.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
output_textbox.configure(state="disabled")

# Submit Button
submit_button = ctk.CTkButton(
    content, 
    text="Submit", 
    command=lambda: submit_action(input_textbox, label_entry, output_textbox),  # Call the function with parameters
    width=200, 
    height=40, 
    corner_radius=10, 
    fg_color="#f53d99", 
    hover_color="#d31f75"
)
submit_button.grid(row=6, column=0, columnspan=2, pady=10, padx=20)

def on_enter_pressed(event):
    submit_action(input_textbox, label_entry, output_textbox)

# Bind the Enter key to the on_enter_pressed function
root.bind('<Return>', on_enter_pressed)

footer_label = ctk.CTkLabel(root, text="GLInER | Created by Bertrand Noureddine, Grouteau Dorian, Oliver Jiang | Version 1.0", font=("Arial", 10), text_color="black")
footer_label.pack(side="bottom", pady=0)

# Run the main loop
root.mainloop()
