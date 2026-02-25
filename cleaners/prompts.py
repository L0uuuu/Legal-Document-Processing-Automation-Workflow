"""Prompt templates for LLM-based OCR correction."""


FRENCH_CORRECTION_PROMPT = """Tu es un correcteur d'erreurs OCR spécialisé dans les documents juridiques tunisiens en français.

Le texte suivant a été extrait par OCR et contient des erreurs:
- Mots coupés ou fusionnés (ex: "collectr e" → "collectives", "déQurent" → "demeurent")
- Caractères manquants ou ajoutés (ex: "ci-aprè ais" → "ci-après relatifs")
- Bruit aléatoire (symboles isolés, lettres parasites)

RÈGLES STRICTES:
1. Corrige UNIQUEMENT les erreurs OCR évidentes
2. Ne change JAMAIS le sens juridique du texte
3. Ne traduis rien
4. N'ajoute pas de contenu nouveau
5. Ne supprime pas de contenu intentionnel
6. Si tu n'es pas sûr d'une correction, garde le texte original
7. Conserve la ponctuation juridique (tirets, points, numérotation)
8. Retourne UNIQUEMENT le texte corrigé, sans explication

TEXTE OCR:
\"\"\"
{chunk}
\"\"\"

TEXTE CORRIGÉ:"""


ARABIC_CORRECTION_PROMPT = """أنت مصحح أخطاء التعرف الضوئي على الحروف (OCR) متخصص في الوثائق القانونية التونسية باللغة العربية.

النص التالي تم استخراجه بواسطة OCR ويحتوي على أخطاء:
- كلمات مقطعة أو مدمجة
- أحرف ناقصة أو مضافة
- ضوضاء عشوائية (رموز معزولة، أحرف طفيلية)

القواعد الصارمة:
1. صحح فقط أخطاء OCR الواضحة
2. لا تغير المعنى القانوني للنص أبداً
3. لا تترجم شيئاً
4. لا تضف محتوى جديداً
5. إذا لم تكن متأكداً من التصحيح، احتفظ بالنص الأصلي
6. أعد النص المصحح فقط، بدون شرح

نص OCR:
\"\"\"
{chunk}
\"\"\"

النص المصحح:"""


CONFIDENCE_CHECK_PROMPT = """Compare these two texts and rate your confidence that the corrections are accurate.
Return ONLY a number between 0.0 and 1.0.

ORIGINAL: \"\"\"{original}\"\"\"
CORRECTED: \"\"\"{corrected}\"\"\"

CONFIDENCE (0.0-1.0):"""