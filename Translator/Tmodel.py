from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\moame\OneDrive\Desktop\ITI\NLP\project\Translator", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(r"C:\Users\moame\OneDrive\Desktop\ITI\NLP\project\Translator", local_files_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


lang_map = {
    "pt": "por_Latn",
    "bg": "bul_Cyrl",
    "zh": "zho_Hans",
    "th": "tha_Thai",
    "ru": "rus_Cyrl",
    "pl": "pol_Latn",
    "ur": "urd_Arab",
    "sw": "swh_Latn",
    "tr": "tur_Latn",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "it": "ita_Latn",
    "hi": "hin_Deva",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "nl": "nld_Latn",
    "fr": "fra_Latn",
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "ja": "jpn_Jpan"
}

def translate(text, src_lang, target_lang):
    tokenizer.src_lang = lang_map[src_lang]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(lang_map[target_lang])

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(translate("Hello, how are you?", "en"))
    print(translate("مرحبا كيف حالك؟", "ar"))
    print(translate("Bonjour, comment ça va?", "fr"))
    print(translate("Hola, ¿cómo estás?", "es"))