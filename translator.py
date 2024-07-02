from translate import Translator

def translate_captions(input_file, output_file):
    translator = Translator(to_lang="sr")

    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    translated_lines = []
    for line in lines:
        try:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                image_name = parts[0].strip()
                english_caption = ','.join(parts[1:]).strip()
                translated_caption = translator.translate(english_caption)
                translated_line = f"{image_name},{translated_caption}\n"
                translated_lines.append(translated_line)
            else:
                translated_lines.append(line)
        except Exception as e:
            print(f"Error translating line: {line}\nError: {e}")
            translated_lines.append(line)
        if len(translated_lines) == 10:
            break

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(translated_lines)
        
input_file = 'captions.txt'
output_file = 'captions_serbian.txt'
translate_captions(input_file, output_file)
