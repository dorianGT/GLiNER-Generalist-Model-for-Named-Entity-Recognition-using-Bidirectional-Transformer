import tkinter as tk
import customtkinter as ctk
from GUI.action import submit_action

from importlib import reload

from model import gliner_model
reload(gliner_model)
from model import gliner_model
from model.gliner_model import GLiNER

from model import model_link
reload(model_link)
from model.model_link import EntityDetectionModel

entityDetectionModel = EntityDetectionModel()

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
content.grid_columnconfigure(2, weight=0, minsize=10)  # Third column for placing the threshold and toggle next to each other

# Input Fields and Labels (position labels on top of the input fields)
input_label = ctk.CTkLabel(content, text="Input Text", font=("Arial", 16), text_color="#f53d99")
input_label.grid(row=0, column=1, padx=(10, 20), pady=10, sticky="w")

# Replace CTkEntry with CTkTextbox for multi-line input
input_textbox = ctk.CTkTextbox(content, width=300, height=100, corner_radius=10, border_width=1, border_color="lightgray")
input_textbox.grid(row=1, column=1, padx=(10, 20), pady=10, sticky="nsew")

# Label Text (now placed to the right of the input_textbox)
label_label = ctk.CTkLabel(content, text="Label Text", font=("Arial", 16), text_color="#f53d99")
label_label.grid(row=0, column=2, padx=(10, 20), pady=10, sticky="w")

label_entry = ctk.CTkTextbox(content, width=300, height=40, corner_radius=10, border_width=1, border_color="lightgray")
label_entry.grid(row=1, column=2, padx=(10, 20), pady=10, sticky="nsew")

# Nested NER Title
parameters_title = ctk.CTkLabel(content, text="Parameters", font=("Arial", 16), text_color="#f53d99")
parameters_title.grid(row=3, column=1, padx=(10, 20), pady=10, sticky="w")

nested_ner_label = ctk.CTkLabel(content, text="Nested NER", font=("Arial", 16), text_color="#f53d99")
nested_ner_label.grid(row=4, column=1, padx=(10, 20), pady=10, sticky="w")

# Toggle for Nested NER (moved to row 4 and now on the left side)
nested_ner_switch = ctk.CTkSwitch(
    content,
    text="Allow for nested NER?",
    font=("Arial", 14),
    text_color="#f53d99",
    onvalue=True,
    offvalue=False
)
nested_ner_switch.grid(row=5, column=1, padx=(10, 20), pady=10, sticky="w")

# Slider for Threshold with Entry for exact value placed to the right of toggle
threshold_label = ctk.CTkLabel(content, text="Threshold", font=("Arial", 16), text_color="#f53d99")
threshold_label.grid(row=4, column=2, padx=(10, 20), pady=10, sticky="w")

threshold_frame = ctk.CTkFrame(content)
threshold_frame.grid(row=5, column=2, padx=(10, 20), pady=10, sticky="w")

threshold_slider = ctk.CTkSlider(
    threshold_frame,
    from_=0,
    to=1,
    number_of_steps=100,
    width=250,
    height=10,
    button_color="#f53d99",
    progress_color="#d31f75"
)
threshold_slider.set(0.5)  # Default value
threshold_slider.grid(row=0, column=0, padx=10)

# Entry to display and set exact threshold value
threshold_value_entry = ctk.CTkEntry(
    threshold_frame, 
    width=50, 
    height=40, 
    corner_radius=10, 
    border_width=1, 
    border_color="lightgray"
)
threshold_value_entry.grid(row=0, column=1, padx=10)
threshold_value_entry.insert(0, "0.5")  # Default value as text

# Function to update the slider when user types in the entry
def update_slider_from_entry(*args):
    try:
        value = float(threshold_value_entry.get())
        if 0 <= value <= 1:
            threshold_slider.set(value)
    except ValueError:
        pass  # Ignore invalid input

# Function to update the entry when slider value changes
def update_entry_from_slider(value):
    threshold_value_entry.delete(0, tk.END)
    threshold_value_entry.insert(0, f"{value:.2f}")

# Bind slider to the update function
threshold_slider.configure(command=update_entry_from_slider)

# Bind entry to the update function
threshold_value_entry.bind("<FocusOut>", update_slider_from_entry)
threshold_value_entry.bind("<Return>", update_slider_from_entry)

# Output Label with hover effect
output_label = ctk.CTkLabel(content, text="Output", font=("Arial", 16), text_color="#f53d99")
output_label.grid(row=6, column=1, padx=(10, 20), pady=10, sticky="w")

# Content frame with the same style as input blocks
output_frame = ctk.CTkFrame(content, fg_color="white", corner_radius=15)
output_frame.grid(row=7, column=1, columnspan=2, padx=(10, 20), pady=20, sticky="nsew")

# Output Textbox (use CTkTextbox for consistent styling)
output_textbox = tk.Text(output_frame, width=40, height=8, wrap=tk.WORD, bg="lightgray", fg="black", bd=0, highlightthickness=0, relief="flat", font=("Arial", 15))
output_textbox.pack(side="left", expand=True, fill=tk.BOTH, padx=10, pady=10)

# Adding scrollbar for vertical scrolling
scrollbar = tk.Scrollbar(output_frame, orient="vertical", command=output_textbox.yview)
scrollbar.pack(side="right", fill="y", padx=10)

# Link the scrollbar with the output_textbox
output_textbox.config(yscrollcommand=scrollbar.set)

# Disable the Textbox to make it read-only
output_textbox.configure(state="disabled")

# Submit Button centered between the two columns
submit_button = ctk.CTkButton(
    content, 
    text="Submit", 
    command=lambda: submit_action(
        input_textbox, 
        label_entry, 
        output_textbox, 
        entityDetectionModel, 
        nested_ner_switch.get(),  # Pass the state of the toggle
        threshold_slider.get()   # Pass the value of the slider
    ),
    width=200, 
    height=40, 
    corner_radius=10, 
    fg_color="#f53d99", 
    hover_color="#d31f75"
)
submit_button.grid(row=8, column=1, columnspan=2, pady=10, padx=20, sticky="nsew")

def on_enter_pressed(event):
    submit_action(
        input_textbox, 
        label_entry, 
        output_textbox, 
        entityDetectionModel, 
        nested_ner_switch.get(), 
        threshold_slider.get()
    )

# Bind the Enter key to the on_enter_pressed function
#root.bind('<Return>', on_enter_pressed)
input_textbox.bind('<Return>', lambda e: 'break')
label_entry.bind('<Return>', lambda e: 'break')

footer_label = ctk.CTkLabel(root, text="GLInER | Created by Bertrand Noureddine, Grouteau Dorian, Oliver Jiang | Version 1.0", font=("Arial", 10), text_color="black")
footer_label.pack(side="bottom", pady=0)

# Run the main loop
root.mainloop()
