# -*- coding: utf-8 -*-
import os
import re
import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------- Settings --------
DEFAULT_MODEL = os.environ.get("MODEL_REPO", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

def generate_response(prompt: str, model_repo: str = DEFAULT_MODEL, max_new_tokens: int = 256, temperature: float = 0.7):
    if not prompt or not prompt.strip():
        return "", 0.0, "الـPrompt فارغ."
    token = HF_TOKEN
    if token is None:
        return "", 0.0, "⚠️ لم يتم ضبط مفتاح HF_TOKEN في إعدادات الـSpace."

    client = InferenceClient(model=model_repo, token=token)
    try:
        # ✅ استخدام واجهة chat بدلاً من text_generation
        chat_completion = client.chat.completions.create(
            model=model_repo,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        text = chat_completion.choices[0].message["content"]
        return text, 1.0, "تم بنجاح."
    except Exception as e:
        return "", 0.0, f"حدث خطأ أثناء توليد النص من النموذج: {e}"

# ---------- Evaluation ----------
INSTRUCTION_HINTS = {
    "list_style": ["list", "قائمة", "عدّد", "عد", "bullet", "نقاط", "•", "–", "١.", "1."],
    "code_style": ["code", "كود", "python", "جافا", "سي", "++c", "javascript", "js", "go", "rust"],
    "translate": ["ترجم", "translate", "ترجمة", "to english", "إلى العربية", "to arabic"],
    "summarize": ["خصّص", "اختصر", "لخّص", "summarize", "ملخص", "ملخّص"],
}

def detect_expected_format(prompt: str):
    p = prompt.lower()
    found = set()
    for k, kws in INSTRUCTION_HINTS.items():
        for kw in kws:
            if kw in p:
                found.add(k)
                break
    return found

def format_score(prompt: str, response: str):
    expected = detect_expected_format(prompt)
    if not expected:
        return 0.5, ["لا يوجد نمط محدد مطلوب في الـPrompt."]

    reasons = []
    score = 0.0

    if "list_style" in expected:
        has_bullets = bool(re.search(r"(^|\n)\s*(?:[-*•–]|\d+\.)\s+\S", response))
        score += 1.0 if has_bullets else 0.0
        reasons.append("قائمة/نقاط: " + ("✅" if has_bullets else "❌"))

    if "code_style" in expected:
        has_codeblock = "```" in response or bool(re.search(r"(^|\n)\s{4}\S", response))
        score += 1.0 if has_codeblock else 0.0
        reasons.append("تنسيق كود: " + ("✅" if has_codeblock else "❌"))

    if "translate" in expected:
        arabic_chars = re.findall(r"[\u0600-\u06FF]", response)
        latin_chars = re.findall(r"[A-Za-z]", response)
        has_mix = bool(arabic_chars) and bool(latin_chars)
        score += 0.8 if has_mix or len(latin_chars) > len(arabic_chars) else 0.0
        reasons.append("إشارات ترجمة: " + ("✅" if has_mix or len(latin_chars) > len(arabic_chars) else "❌"))

    if "summarize" in expected:
        score += 0.8 if len(response.split()) < max(40, len(prompt.split())) else 0.0
        reasons.append("تلخيص: " + ("✅" if len(response.split()) < max(40, len(prompt.split())) else "❌"))

    max_possible = 0.0
    max_possible += 1.0 if "list_style" in expected else 0.0
    max_possible += 1.0 if "code_style" in expected else 0.0
    max_possible += 0.8 if "translate" in expected else 0.0
    max_possible += 0.8 if "summarize" in expected else 0.0
    max_possible = max(max_possible, 1e-6)
    return float(score / max_possible), reasons

def similarity_score(prompt: str, response: str):
    emb = get_embedder()
    vecs = emb.encode([prompt, response])
    sim = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
    return float(sim)

def length_score(response: str):
    w = len(response.split())
    return float(np.clip((w - 10) / (60 - 10), 0, 1))

def overall_evaluate(prompt: str, response: str):
    try:
        sim = similarity_score(prompt, response)
        sim01 = (sim + 1) / 2.0
    except Exception:
        sim01 = 0.0

    fmt, fmt_reasons = format_score(prompt, response)
    lng = length_score(response)

    score01 = 0.6 * sim01 + 0.2 * fmt + 0.2 * lng
    score100 = round(100 * score01, 1)

    if score100 >= 70:
        verdict = "✔✔ النموذج فَهِم الـPrompt بشكل جيد"
    elif score100 >= 50:
        verdict = "⚠️ الفهم متوسط – يحتاج تحسين"
    else:
        verdict = "❌ النموذج لم يفهم المطلوب بشكل كافٍ"

    details = {
        "تشابه الدلالة (0-100)": round(sim01 * 100, 1),
        "مطابقة التنسيق (0-100)": round(fmt * 100, 1),
        "ثراء الإجابة (0-100)": round(lng * 100, 1),
        "الدرجة الكلية (0-100)": score100,
        "ملاحظات التنسيق": " | ".join(fmt_reasons) if fmt_reasons else "لا يوجد نمط مطلوب",
    }
    return score100, verdict, details

def run_test(prompt, model_repo, max_new_tokens, temperature):
    response, ok, status = generate_response(prompt, model_repo, max_new_tokens, temperature)
    if not response:
        return "", status, {}, ""
    score, verdict, details = overall_evaluate(prompt, response)
    details_txt = "\n".join([f"- {k}: {v}" for k, v in details.items()])
    return response, verdict, details, details_txt

# ------------- UI -------------
with gr.Blocks(title="اختبار فهم النماذج للنصوص") as demo:
    gr.Markdown("""
# 🧪 اختبار فهم النماذج للنصوص (Prompt Understanding Test)
اكتب أي Prompt وشاهد ردّ النموذج وتقييم الفهم تلقائيًا.
> **مهم:** يجب ضبط سرّ `HF_TOKEN` من إعدادات الـSpace.
    """)

    with gr.Row():
        model_repo = gr.Dropdown(
            choices=[
                "HuggingFaceH4/zephyr-7b-beta",
                "google/gemma-2-2b-it",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "tiiuae/falcon-7b-instruct",
            ],
            value=DEFAULT_MODEL,
            label="HF Model Repo",
            info="يمكنك تغيير الموديل من هنا",
        )
    prompt = gr.Textbox(lines=6, label="اكتب الـPrompt هنا")
    with gr.Row():
        max_new_tokens = gr.Slider(32, 512, value=256, step=1, label="Max New Tokens")
        temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")

    btn = gr.Button("اختبر الآن ✨")

    with gr.Row():
        response = gr.Textbox(label="ردّ النموذج", lines=10)
    verdict = gr.Label(label="تقييم الفهم")
    details_dict = gr.JSON(label="تفاصيل الدرجات", value={})
    details_txt = gr.Textbox(label="تفاصيل نصية", lines=6)

    btn.click(
        fn=run_test,
        inputs=[prompt, model_repo, max_new_tokens, temperature],
        outputs=[response, verdict, details_dict, details_txt]
    )

if __name__ == "__main__":
    demo.launch()
