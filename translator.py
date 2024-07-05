import time
from deep_translator import GoogleTranslator
import concurrent.futures
import threading

def translate_caption(line, semaphore):
    translator = GoogleTranslator(source='en', target='sr')
    try:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            image_name = parts[0].strip()
            english_caption = ','.join(parts[1:]).strip()
            with semaphore:
                translated_caption = translator.translate(english_caption)
                time.sleep(0.2) 
            return f"{image_name},{translated_caption}\n"
        else:
            return line
    except Exception as e:
        print(f"Error translating line: {line}\nError: {e}")
        return line

def translate_captions(input_file, output_file, max_workers=10):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    translated_lines = []
    semaphore = threading.Semaphore(5) 
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_line = {executor.submit(translate_caption, line, semaphore): line for line in lines}
        for future in concurrent.futures.as_completed(future_to_line):
            translated_lines.append(future.result())
            if len(translated_lines) % 100 == 0:
                print(f"line: { len(translated_lines) }/n")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(translated_lines)
        
input_file = 'captions.txt'
output_file = 'captions_serbian.txt'
translate_captions(input_file, output_file)
