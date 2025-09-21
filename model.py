import google.generativeai as genai
import json, os
import time

# API ключ Gemini
genai.configure(api_key="AIzaSyDZKichaVvgH5pUOplwyPSoWqZUrwWAIsg")

PROMPT_TEMPLATE = """
Ты — помощник, который анализирует комментарии клиентов на русском и казахском языках, относительно услуг мобильного оператора Altel/Tele2 в Казахстане.

Твоя задача — проанализировать комментарий и вернуть результат СТРОГО в формате JSON.

Структура JSON должна быть следующей:


{{
  "comment": comment
  "type": ["<question | complaint | review | thanks | spam>"],  # список типов, если комментарий достаточно сильно подходит к нескольким типам
  "sentiment": "<very_negative | negative | neutral | positive | very_positive>",
  "key_idea": "<краткое содержание/основная мысль комментария (1–2 предложения)>"
  "swear": []  # Изначально пустой список; если есть нецензурная брань — добавляй её сюда
  "lang": <основной язык комментария>
}}

Правила:
- Поле "type" всегда должно содержать хотя бы один из: question, complaint, review, thanks, spam.
- В поле "type" лишние типы не добавляй, неподходящие к комменту
- Если сообщение содержит контент, не соответствующий бренду Altel/Tele2 (например, рекламу конкурентов, мошенничество или нерелевантную информацию), то "type" = "spam".
- Поле "sentiment" всегда выбирается из: very_negative, negative, neutral, positive, very_positive.
- "key_idea" должно быть кратким, но отражать суть сообщения.
- Учитывай, что комментарии могут быть на русском и казахском языках, и могут относиться к тарифам, интернету, звонкам, услугам операторов и работе сети в Казахстане.
- Не добавляй никакого текста вне JSON.
"""

def analyze_comment_gemini(comment_text):
    # Instantiate the model
    model = genai.GenerativeModel('gemini-2.5-flash') # Using 'gemini-1.5-pro' as a good choice

    prompt = PROMPT_TEMPLATE + f'\nКомментарий:\n"{comment_text}"\nОтвет:'

    
    try:
        response = model.generate_content(prompt)
        raw_output = response.text

        # Step 1: Check if the response is wrapped in a Markdown code block
        if raw_output.startswith('```json') and raw_output.endswith('```'):
            # Remove the Markdown code block syntax
            json_string = raw_output.removeprefix('```json\n').removesuffix('```')
        else:
            # Assume the output is a plain JSON string
            json_string = raw_output

        # Step 2: Parse the cleaned JSON string
        parsed = json.loads(json_string)

    except (json.JSONDecodeError, AttributeError) as e:
        # Fallback for any parsing errors, including the original issue
        print(f"Error parsing JSON from Gemini: {e}")
        parsed = {
            "type": None,
            "sentiment": None,
            "key_idea": raw_output
        }
    return parsed

folder_path = "data/raw"  # замените на путь к вашей папке
all_results = []

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        # Loop through each post object in the list
        for post in data_list:
            # Get the comments list for the current post
            comments = post.get("comments", [])
            
            # Loop through each comment in the comments list
            for comment in comments:
                text = comment.get("text", "")
                if text:
                    analysis = analyze_comment_gemini(text)
                    result = {
                        "filename": filename,
                        "original_text": text,
                        "analysis": analysis
                    }
                    all_results.append(result)
                    time.sleep(3) 

# -----------------------------
# 5️⃣ Сохраняем результаты
# -----------------------------
with open("analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("Анализ завершён. Результаты сохранены в analysis_results.json")