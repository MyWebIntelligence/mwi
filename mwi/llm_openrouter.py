import json
from typing import Optional, List

import requests

from . import model
import settings


_call_count = 0


def _get_land_terms(land: model.Land) -> List[str]:
    """Retrieve all dictionary terms associated with a land.

    Args:
        land: Land object to retrieve terms for.

    Returns:
        List of term strings from the land's dictionary.
    """
    rows = (
        model.Word.select(model.Word.term)
        .join(model.LandDictionary)
        .where(model.LandDictionary.land == land)
    )
    return [r.term for r in rows]


def build_relevance_prompt(land: model.Land, expression: model.Expression, readable_text: str) -> str:
    """Build a French-language LLM prompt for relevance evaluation.

    Args:
        land: Land object containing project context.
        expression: Expression object to evaluate.
        readable_text: Extracted readable content from the expression.

    Returns:
        Formatted prompt string requesting yes/no relevance judgment.

    Note:
        Prompt is in French and instructs the LLM to respond only with
        "oui" or "non" without additional commentary.
    """
    terms = ", ".join(_get_land_terms(land))
    title = str(getattr(expression, "title", "") or "")
    desc = str(getattr(expression, "description", "") or "")
    url = str(getattr(expression, "url", "") or "")
    land_desc = str(getattr(land, "description", "") or "")

    prompt = (
        "Dans le cadre de la constitution d'un corpus de pages Web à des fins d'analyse de contenu, "
        "nous voulons savoir si la page crawlée est pertinente pour le projet ou non.\n"
        "Le projet a les caractéristiques suivantes :\n"
        f"- Nom du projet : {land.name}\n"
        f"- Description : {land_desc}\n"
        f"- Mots clés : {terms}\n"
        "La page suivante :\n"
        f"- URL = {url}\n"
        f"- Titre : {title}\n"
        f"- Description : {desc}\n"
        f"- Readable (extrait) : {readable_text}\n"
        "Tu répondras ABSOLUMENT et uniquement par \"oui\" ou \"non\" sans aucun commentaire."
    )
    return prompt


def _normalize_yesno(text: str) -> str:
    """Normalize LLM response to standard yes/no format.

    Args:
        text: Raw response text from LLM.

    Returns:
        Normalized response: "oui", "non", or "?" for unclear responses.

    Note:
        Handles both French (oui/non) and English (yes/no) responses.
    """
    t = (text or "").strip().lower()
    if t.startswith("non") or t == "no":
        return "non"
    if t.startswith("oui") or t == "yes":
        return "oui"
    return "?"


def ask_openrouter_yesno(prompt: str) -> str:
    """Send a yes/no question to OpenRouter API.

    Args:
        prompt: Formatted prompt string for the LLM.

    Returns:
        Raw response content from the LLM.

    Raises:
        requests.HTTPError: If the API request fails.

    Note:
        Uses temperature=0 for deterministic responses.
    """
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": settings.openrouter_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(body),
        timeout=settings.openrouter_timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def is_relevant_via_openrouter(land: model.Land, expression: model.Expression) -> Optional[bool]:
    """Evaluate expression relevance using OpenRouter LLM.

    Args:
        land: Land object providing research context.
        expression: Expression object to evaluate for relevance.

    Returns:
        True if relevant, False if not relevant, None if inconclusive or
        disabled/error.

    Note:
        Respects settings for enabled status, API credentials, and call
        budget. Increments global call counter. Prints verdict to stdout.
    """
    global _call_count

    # Preconditions
    if not getattr(settings, "openrouter_enabled", False):
        return None
    if not settings.openrouter_api_key or not settings.openrouter_model:
        print("OpenRouter disabled: missing API key or model")
        return None
    if _call_count >= settings.openrouter_max_calls_per_run:
        print("OpenRouter budget reached for this run; skipping gate")
        return None

    readable_text = str(getattr(expression, "readable", "") or "")
    if readable_text:
        readable_text = readable_text[: settings.openrouter_readable_max_chars]
    else:
        # Fallback to a minimal context if readable is missing
        readable_text = ""

    prompt = build_relevance_prompt(land, expression, readable_text)

    try:
        _call_count += 1
        content = ask_openrouter_yesno(prompt)
        verdict = _normalize_yesno(content)
        if verdict == "non":
            print(f"OpenRouter gate verdict=NON for {expression.url}")
            return False
        if verdict == "oui":
            print(f"OpenRouter gate verdict=OUI for {expression.url}")
            return True
        print(f"OpenRouter gate verdict=INCONNU for {expression.url}: '{content[:50]}...'")
        return None
    except Exception as e:
        print(f"OpenRouter gate error for {expression.url}: {e}")
        return None

