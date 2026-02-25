"""Prompt templates for Phase 3: AI-powered document parsing."""


HEADER_EXTRACTION_PROMPT = """Tu es un expert en droit tunisien. Analyse l'en-tête de ce document juridique et extrais les métadonnées.

TEXTE DE L'EN-TÊTE:
\"\"\"
{header_text}
\"\"\"

Retourne UNIQUEMENT un objet JSON valide avec ces champs (null si non trouvé):

{{
  "law_type": "Loi | Décret | Décret-loi | Arrêté | Circulaire | Ordonnance | Loi organique | Décret gouvernemental",
  "law_number": "ex: 66-27, 97-47, 2017-51",
  "year": 1966,
  "title_french": "titre complet de la loi en français",
  "title_arabic": "عنوان القانون بالعربية أو null",
  "institution": "Ministère ou institution principale, ou plusieurs séparés par virgule",
  "institutions": ["Ministère 1", "Ministère 2"],
  "publication_date": "YYYY-MM-DD",
  "effective_date": "YYYY-MM-DD ou null",
  "gazette_name": "JORT ou autre",
  "gazette_number": "numéro ou null",
  "gazette_date": "YYYY-MM-DD ou null",
  "gazette_page": 716
}}

IMPORTANT:
- Retourne UNIQUEMENT le JSON, pas d'explication
- Les dates en format YYYY-MM-DD
- Si un champ n'est pas trouvable, mets null
- Pour les institutions, liste TOUTES celles mentionnées"""


ARTICLE_EXTRACTION_PROMPT = """Tu es un expert en droit tunisien. Analyse cet article de loi et extrais les métadonnées structurées.

CONTEXTE DU DOCUMENT:
- Type: {law_type}
- Numéro: {law_number}
- Année: {year}
- Chapitre détecté: {chapter}
- Section détectée: {section}

TEXTE DE L'ARTICLE:
\"\"\"
{article_text}
\"\"\"

Retourne UNIQUEMENT un objet JSON valide avec ces champs:

{{
  "article_number": "premier | 2 | 3bis | etc.",
  "chapter": "CHAPITRE X - TITRE ou null",
  "chapter_normalized": "chapitre_x_titre ou null",
  "section": "SECTION X - TITRE ou null",
  "content_french": "texte nettoyé de l'article en français (sans numéro d'article, sans bruit OCR)",
  "content_arabic": "نص المادة بالعربية ou vide si pas disponible",
  "summary_french": "résumé concis de l'article en 1-2 phrases",
  "summary_arabic": "ملخص المادة في جملة أو جملتين",
  "keywords": ["mot-clé 1", "mot-clé 2", "max 5-7 mots-clés"],
  "legal_domains": ["domaine juridique 1", "domaine juridique 2"],
  "business_impact": "LOW | MEDIUM | HIGH",
  "target_audience": ["audience 1", "audience 2"],
  "related_laws": ["lois référencées dans l'article"],
  "entity_names": ["Ministère X", "Organisation Y"],
  "entity_types": ["ORGANIZATION", "PERSON", "INSTITUTION"],
  "entity_ids": ["tn-org-nom-normalise"],
  "relation_target_ids": ["tn-loi-xx-xx"],
  "relation_types": ["REFERENCES", "AMENDS", "REPEALS"],
  "has_obligations": false,
  "has_penalties": false,
  "has_deadlines": false,
  "has_exceptions": false,
  "is_abrogation": false,
  "is_transitional": false,
  "ambiguity_level": "LOW | MEDIUM | HIGH"
}}

RÈGLES:
- Retourne UNIQUEMENT le JSON, pas d'explication
- content_french: nettoie le texte OCR, corrige les mots cassés
- summary: résume le SENS juridique, pas juste le texte
- keywords: termes juridiques pertinents pour la recherche
- legal_domains: ex: "droit du travail", "droit administratif", "droit pénal"
- business_impact: HIGH si sanctions/obligations directes, MEDIUM si réglementaire, LOW si informatif
- entity_ids: format "tn-org-nom-en-minuscules-sans-accents"
- has_obligations: true si l'article impose une obligation ("doit", "est tenu", "obligatoire")
- has_penalties: true si l'article mentionne des sanctions/amendes/peines
- has_deadlines: true si l'article mentionne des délais
- is_abrogation: true si l'article abroge d'autres textes
- is_transitional: true si l'article contient des dispositions transitoires"""