#!/bin/bash
python plot_heatmap.py --model gpt-4.1 --strategy code_form --title "GPT-4.1 Code-Form Planning" --yaxis

python plot_heatmap.py --model gemini-2.0-flash-001 --strategy code_form --title "Gemini-2.0-Flash-001 Code-Form Planning"

python plot_heatmap.py --model gemini-2.0-flash-lite-001 --strategy code_form --title "Gemini-2.0-Flash-Lite-001 Code-Form Planning" --legend


python plot_heatmap.py --model gpt-4.1 --strategy text_feedback --title "GPT-4.1 Text Planning" --yaxis

python plot_heatmap.py --model gemini-2.0-flash-001 --strategy text_feedback --title "Gemini-2.0-Flash-001 Text Planning"

python plot_heatmap.py --model gemini-2.0-flash-lite-001 --strategy text_feedback --title "Gemini-2.0-Flash-Lite-001 Text Planning" --legend
