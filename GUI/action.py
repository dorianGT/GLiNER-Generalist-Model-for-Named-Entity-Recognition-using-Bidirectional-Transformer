import random

def generate_color():
    """Generate a random darker color in hex format."""
    # The RGB components are chosen to be in the lower range (0 to 100)
    return f"#{random.randint(10, 200):02x}{random.randint(10, 200):02x}{random.randint(10, 200):02x}"

def submit_action(input_textbox, label_entry, output_textbox, entityDetectionModel, nested_ner, threshold):
    input_text = input_textbox.get("1.0", "end-1c")  # Get the multi-line text
    label_text = label_entry.get("1.0", "end-1c")
    label_set = set(label_text.split(", "))
    label_set = {item.lower() for item in label_set}

    if not label_text or not input_text:  
        output_textbox.configure(state="normal")  
        output_textbox.delete("1.0", "end")  
        output_textbox.insert("1.0", "Error: Empty input text or label.")  
        output_textbox.configure(state="disabled")
        return

    output_textbox.configure(state="normal")  
    output_textbox.delete("1.0", "end")  

    # Call the model's process function with the nested_ner and threshold parameters
    full_text = entityDetectionModel.process(input_text, label_text, threshold, nested_ner)

    output_textbox.insert("1.0", full_text)
    
    color_map = {}

    for label in label_set:
        if label not in color_map:
            color_map[label] = generate_color()  # Assign a unique color to each label

        # Search for the label in the text and apply color
        start_idx = output_textbox.search(f"[{label}]", "1.0", stopindex="end")
        while start_idx:
            end_idx = f"{start_idx}+{len(label) + 2}c"  # Include square brackets in the length

            output_textbox.tag_add(label, start_idx, end_idx)
            output_textbox.tag_configure(label, foreground=color_map[label])

            # Handle possible nested NER by finding the full entity span
            if True:
                preceding_word_end_idx = start_idx
                preceding_word_start_idx = output_textbox.search("{", preceding_word_end_idx, backwards=True, stopindex="1.0")
                if preceding_word_start_idx:
                    closing_brace_idx = output_textbox.search("}", preceding_word_start_idx, stopindex=preceding_word_end_idx)
                    if closing_brace_idx:
                        word_start = preceding_word_start_idx
                        word_end = f"{closing_brace_idx}+1c"  # Include the closing '}'
                        
                        output_textbox.tag_add(f"{label}_underline", word_start, word_end)
                        output_textbox.tag_configure(f"{label}_underline", foreground=color_map[label], underline=True)

            start_idx = output_textbox.search(f"[{label}]", end_idx, stopindex="end")

    output_textbox.configure(state="disabled")
