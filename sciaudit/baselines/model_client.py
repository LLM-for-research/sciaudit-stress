#!/usr/bin/env python3
"""Клиент к OpenAI-совместимому API для бейзлайнов B1-B4 (решение issue #12).

Модель дёргается по HTTP с ключом, а не грузится весами. К конкретному бэкенду
клиент не привязан: локальный vLLM, университетский шлюз и сторонний провайдер
говорят на одном протоколе и отличаются только адресом. Поэтому адрес, модель и
ключ приходят из ``.env``, а ``configs/models.yaml`` держит лишь параметры
запроса и список одобренных моделей.

**Ключ никогда не попадает ни в репозиторий, ни в командную строку, ни в файл
предсказаний, ни в лог.** Он читается из переменной окружения, названной в
профиле (``api_key_env``), либо из ``.env`` в корне репозитория — файл в
``.gitignore``. Строка команды запуска для этого не годится: она видна в ``ps``
и остаётся в истории оболочки.

Настройка целиком в окружении или в ``.env``:

===========================  ==================================================
``SCIAUDIT_BASE_URL``        базовый URL, ``http://localhost:8000/v1`` для vLLM
``SCIAUDIT_MODEL``           идентификатор модели, он же в ``system_info.model``
``SCIAUDIT_API_KEY``         ключ (имя переменной задаётся в ``api_key_env``)
===========================  ==================================================

Новых зависимостей клиент не приносит: HTTP идёт через ``urllib`` из
стандартной библиотеки, а не через ``requests`` или SDK провайдера (§6.4).
Единственное, что нужно сверх stdlib, — PyYAML для чтения
``configs/models.yaml``; он уже в dev-группе и ставится вместе с ``uv sync``.

Запуск бейзлайна через API::

    python -m sciaudit.baselines.b1_bm25_llm \\
        --input inputs.jsonl --output preds.jsonl --model-api

Проверить настройку, ничего не запрашивая (годится для CI без ключа)::

    python -m sciaudit.baselines.model_client --check

Проверить настройку и достучаться до эндпоинта одним пробным запросом::

    python -m sciaudit.baselines.model_client --ping
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"
DOTENV_PATH = REPO_ROOT / ".env"

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 3
RETRIABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class ModelClientError(RuntimeError):
    """Запрос к модели не удался. Текст ошибки никогда не содержит ключа."""


# --- конфигурация ---------------------------------------------------------------

def load_dotenv(path=DOTENV_PATH, environ=None):
    """Подгрузить ``KEY=VALUE`` из .env, не затирая уже заданные переменные.

    Реальное окружение сильнее файла: в CI секреты приходят переменными, и
    случайно оставленный .env не должен их перебивать.
    """
    environ = os.environ if environ is None else environ
    path = Path(path)
    if not path.is_file():
        return {}

    loaded = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in environ:
            environ[key] = value
            loaded[key] = value
    return loaded


def load_config(config_path=CONFIG_PATH, environ=None):
    """Собрать настройки запроса: файл конфига плюс адрес, модель и ключ из окружения.

    Файл описывает только то, что не зависит от бэкенда: параметры запроса, имя
    переменной с ключом и список одобренных моделей. Всё, что различает
    эндпоинты, приходит из ``.env``.
    """
    environ = os.environ if environ is None else environ

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - зависимость есть в dev-группе
        raise ModelClientError("для чтения configs/models.yaml нужен PyYAML") from exc

    try:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ModelClientError(f"не читается {config_path}: {exc}") from exc

    defaults = config.get("request_defaults") or {}
    api_key_env = config.get("api_key_env") or "SCIAUDIT_API_KEY"
    approved = config.get("approved_models") or []
    model = environ.get("SCIAUDIT_MODEL")

    return {
        "base_url": (environ.get("SCIAUDIT_BASE_URL") or "").rstrip("/"),
        "model": model,
        "api_key_env": api_key_env,
        "api_key": environ.get(api_key_env),
        "max_tokens": int(defaults.get("max_tokens") or DEFAULT_MAX_TOKENS),
        "temperature": float(defaults.get("temperature", DEFAULT_TEMPERATURE)),
        "seed": None if defaults.get("seed") is None else int(defaults["seed"]),
        "timeout_seconds": int(defaults.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS),
        "max_retries": int(defaults.get("max_retries") or DEFAULT_MAX_RETRIES),
        "approved_models": list(approved),
        "approved": bool(model) and model in approved,
    }


def check_ready(config):
    """Вернуть список того, чего не хватает для запроса. Значение ключа не печатается."""
    missing = []
    if not config["base_url"]:
        missing.append("адрес эндпоинта в SCIAUDIT_BASE_URL")
    if not config["model"]:
        missing.append("идентификатор модели в SCIAUDIT_MODEL")
    if not config["api_key"]:
        missing.append(f"ключ в переменной окружения {config['api_key_env']}")
    return missing


# --- запрос ----------------------------------------------------------------------

def _post(url, payload, api_key, timeout_seconds):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def complete(prompt, profile=None, sleep=time.sleep):
    """Отправить промпт и вернуть текст ответа.

    Повторяет запрос с экспоненциальной задержкой на 429 и 5xx — по ним нельзя
    судить о модели, это состояние транспорта.
    """
    profile = load_config() if profile is None else profile

    missing = check_ready(profile)
    if missing:
        raise ModelClientError("профиль не готов к запросу: " + "; ".join(missing))

    payload = {
        "model": profile["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": profile["max_tokens"],
        "temperature": profile["temperature"],
    }
    if profile.get("seed") is not None:
        payload["seed"] = profile["seed"]
    url = f"{profile['base_url']}/chat/completions"

    last_error = "запрос не выполнялся"
    for attempt in range(1, profile["max_retries"] + 1):
        try:
            body = _post(url, payload, profile["api_key"], profile["timeout_seconds"])
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code not in RETRIABLE_STATUS or attempt == profile["max_retries"]:
                raise ModelClientError(f"эндпоинт ответил {exc.code}") from None
            sleep(2 ** (attempt - 1))
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"транспорт: {exc.__class__.__name__}"
            if attempt == profile["max_retries"]:
                raise ModelClientError(f"эндпоинт недоступен ({last_error})") from None
            sleep(2 ** (attempt - 1))
            continue
        except json.JSONDecodeError:
            raise ModelClientError("эндпоинт вернул не JSON") from None

        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise ModelClientError("в ответе нет choices[0].message") from None

        content = message.get("content") or ""
        if content.strip():
            return content

        # Пустой content — почти всегда не поломка эндпоинта, а исчерпанный бюджет.
        # У reasoning-моделей рассуждение списывается из тех же max_tokens, что и
        # ответ: модель думает до упора и до JSON не доходит. Без этой проверки
        # ошибка доезжает наверх как «model returned empty output» и уводит
        # разбирательство в сторону эндпоинта.
        if choice.get("finish_reason") == "length":
            raise ModelClientError(
                f"ответ обрезан лимитом max_tokens={profile['max_tokens']}: бюджет "
                "израсходован до того, как модель выдала ответ. У reasoning-моделей "
                "рассуждение считается в том же лимите — поднимите "
                "request_defaults.max_tokens в configs/models.yaml"
            )
        if message.get("reasoning") or message.get("reasoning_content"):
            raise ModelClientError(
                "модель вернула только рассуждение, без content — поднимите "
                "request_defaults.max_tokens в configs/models.yaml"
            )
        raise ModelClientError("в ответе пустой choices[0].message.content")

    raise ModelClientError(f"не удалось получить ответ: {last_error}")


def make_model_fn():
    """Замыкание для ``model_audit.run(model_fn=...)``: конфиг читается один раз."""
    load_dotenv()
    config = load_config()

    def model_fn(prompt):
        return complete(prompt, config)

    model_fn.config = config
    return model_fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Запрос к OpenAI-совместимому эндпоинту. Промпт читается из stdin.")
    parser.add_argument("--check", action="store_true",
                        help="Только проверить настройку, без обращения к сети.")
    parser.add_argument("--ping", action="store_true",
                        help="Проверить настройку и сделать один пробный запрос.")
    args = parser.parse_args(argv)

    load_dotenv()

    try:
        config = load_config()
    except ModelClientError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.check or args.ping:
        missing = check_ready(config)
        print(f"эндпоинт: {config['base_url'] or '—'}", file=sys.stderr)
        print(f"модель: {config['model'] or '—'}", file=sys.stderr)
        print(f"ключ в {config['api_key_env']}: "
              f"{'задан' if config['api_key'] else 'НЕ ЗАДАН'}", file=sys.stderr)
        print(f"температура: {config['temperature']}", file=sys.stderr)
        approved = config["approved_models"]
        if not approved:
            print("approved_models пуст — список одобренных моделей ещё не утверждён, "
                  "прогон не считается лидербордным", file=sys.stderr)
        elif not config["approved"]:
            print(f"ВНИМАНИЕ: модель {config['model']} не входит в approved_models "
                  f"{approved} — прогон годится для разработки, но не для лидерборда",
                  file=sys.stderr)
        if missing:
            print("НЕ ГОТОВ: " + "; ".join(missing), file=sys.stderr)
            return 1
        if not args.ping:
            print("настройка заполнена (сеть не проверялась, см. --ping)",
                  file=sys.stderr)
            return 0

        # Заполненный .env ещё не значит, что эндпоинт отвечает и что бюджета
        # токенов хватает на ответ: у reasoning-моделей рассуждение съедает тот
        # же лимит. Пробный запрос ловит и то, и другое до прогона бейзлайна.
        try:
            reply = complete('Ответь ровно одним словом: ready', config)
        except ModelClientError as exc:
            print(f"НЕ ГОТОВ: {exc}", file=sys.stderr)
            return 1
        print(f"эндпоинт ответил: {reply.strip()[:60]!r}", file=sys.stderr)
        print("готов", file=sys.stderr)
        return 0

    try:
        print(complete(sys.stdin.read(), config))
    except ModelClientError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
