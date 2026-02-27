"""Prompt templates for Phase 3: AI-powered document parsing."""


HEADER_EXTRACTION_PROMPT = """Tu es un expert en droit tunisien. Analyse l'en-tête de ce document juridique et extrais les métadonnées.

TEXTE DE L'EN-TÊTE:
\"\"\"
{header_text}
\"\"\"

Retourne UNIQUEMENT un objet JSON valide avec ces champs (null si non trouvé):

{{
  "law_type": "Loi | Décret | Décret-loi | Arrêté | Circulaire | Ordonnance | Loi organique | Décret gouvernemental",
  "law_number": "ex: 66-27, 97-47, 2017-51, 14",
  "year": 1966,
  "title_french": "titre complet de la loi en français",
  "title_arabic": "عنوان القانون بالعربية (traduis du français si non disponible)",
  "institution": "Ministère ou institution principale, ou plusieurs séparés par virgule",
  "institutions": ["Ministère 1", "Ministère 2"],
  "publication_date": "YYYY-MM-DD",
  "effective_date": "YYYY-MM-DD ou null",
  "source_name": "Journal Officiel de la République Tunisienne ou JORT",
  "source_number": "numéro ou null",
  "source_date": "YYYY-MM-DD ou null"
}}

IMPORTANT:
- Retourne UNIQUEMENT le JSON, pas d'explication
- Les dates en format YYYY-MM-DD
- Si un champ n'est pas trouvable, mets null
- Pour title_arabic: si le titre arabe n'est pas dans le texte, TRADUIS le titre français en arabe
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
  "article_number": "premier | 2 | 3bis | 98 | etc.",
  "chapter": "CHAPITRE X - TITRE ou Unique si pas de chapitre ou null",
  "chapter_normalized": "chapitre_x_titre ou unique ou null",
  "section": "SECTION X - TITRE ou null",

  "content_french": "texte complet nettoyé de l'article en français (sans numéro d'article au début, sans bruit OCR). Si le texte original est en arabe UNIQUEMENT, TRADUIS en français.",
  "content_arabic": "النص الكامل للمادة بالعربية. إذا كان النص الأصلي بالفرنسية فقط، ترجم إلى العربية.",

  "summary": "résumé concis et neutre de l'article en 1 phrase",
  "summary_french": "résumé détaillé en français (1-2 phrases)",
  "summary_arabic": "ملخص مفصل بالعربية (جملة أو جملتين)",

  "keywords": ["mot-clé 1", "mot-clé 2", "max 5-7 mots-clés pertinents pour la recherche"],
  "search_content": "termes de recherche en français et arabe séparés par virgules",

  "article_type": "PENAL | CIVIL | COMMERCIAL | ADMINISTRATIF | FISCAL | SOCIAL | TRAVAIL | FONCIER | CONSTITUTIONNEL | ENVIRONNEMENTAL | AUTRE",

  "legal_domains": ["domaine juridique 1", "domaine juridique 2"],
  "business_impact": "LOW | MEDIUM | HIGH",
  "target_audience": ["audience 1", "audience 2"],
  "related_laws": ["lois ou articles référencés dans le texte"],

  "community_label": "nom court du cluster thématique (ex: Droit pénal tunisien révisé, Réglementation du travail)",
  "community_summary": "description courte du cluster thématique en 1 phrase",
  "community_id": "tn-cluster-xxx (identifiant normalisé du cluster)",

  "entity_names": ["Ministère X", "Organisation Y", "tribunal"],
  "entity_types": ["INSTITUTION", "ORGANIZATION", "COURT", "PERSON", "COMMISSION"],
  "entity_ids": ["tn-inst-nom-normalise", "tn-org-nom", "tn-ct-tribunal"],

  "relation_target_ids": ["tn-code-penal-art-96", "tn-loi-xx-xx"],
  "relation_types": ["REFERENCES", "AMENDS", "REPEALS", "IMPLEMENTS", "COMPLEMENTS"],

  "has_obligations": false,
  "has_penalties": false,
  "has_deadlines": false,
  "has_exceptions": false,
  "is_abrogation": false,
  "is_transitional": false,
  "ambiguity_level": "LOW | MEDIUM | HIGH"
}}

RÈGLES CRITIQUES:
1. Retourne UNIQUEMENT le JSON, pas d'explication
2. content_french: OBLIGATOIRE. Si le texte est uniquement en arabe, TRADUIS en français
3. content_arabic: OBLIGATOIRE. Si le texte est uniquement en français, TRADUIS en arabe
4. summary_french et summary_arabic: TOUJOURS remplis (traduis si nécessaire)
5. Le premier caractère de ta réponse DOIT être {{
6. article_type: choisis parmi la liste fournie selon le domaine de l'article
7. keywords: termes juridiques pertinents pour la recherche
8. legal_domains: ex "droit du travail", "droit pénal", "droit administratif"
9. business_impact: HIGH si sanctions/obligations directes, MEDIUM si réglementaire, LOW si informatif
10. entity_ids: format "tn-org-nom-en-minuscules-sans-accents" ou "tn-inst-nom" ou "tn-ct-nom"
11. has_obligations: true si l'article impose une obligation ("doit", "est tenu", "obligatoire")
12. has_penalties: true si sanctions/amendes/peines mentionnées
13. has_deadlines: true si délais mentionnés
14. has_exceptions: true si exceptions/dérogations mentionnées
15. is_abrogation: true si l'article abroge d'autres textes
16. is_transitional: true si dispositions transitoires
17. community_id: format "tn-cluster-xxx" (identifiant court, normalisé)
18. search_content: mélange de termes français ET arabes pour la recherche full-text"""