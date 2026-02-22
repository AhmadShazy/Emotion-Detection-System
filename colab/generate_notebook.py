import json
import os

def generate_notebook():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(script_dir, 'notebook_source.md')
    output_file = os.path.join(script_dir, 'Emotion_Detection_SOTA_Colab.ipynb')
    
    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found.")
        return

    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []
    
    # Simple parser for the markdown structure I used
    parts = content.split('---')
    for part in parts:
        part = part.strip()
        if not part: continue
        
        # Check if it has a code block
        if '```python' in part:
            md_part, code_part = part.split('```python', 1)
            code_part, _ = code_part.split('```', 1)
            
            # Add markdown cell
            if md_part.strip():
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [line + '\n' for line in md_part.strip().split('\n')]
                })
            
            # Add code cell
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + '\n' for line in code_part.strip().split('\n')]
            })
        else:
            # Just markdown
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + '\n' for line in part.split('\n')]
            })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Successfully generated {output_file}")

if __name__ == "__main__":
    generate_notebook()
