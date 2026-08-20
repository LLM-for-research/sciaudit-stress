# Воспроизводимый образ для финальной оценки, собирается через uv (мануал §6.4).
FROM python:3.11-slim

# Версия uv закреплена: с :latest один и тот же Dockerfile через месяц соберёт
# другое окружение, а §6.4 требует воспроизводимости.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

# Берём интерпретатор из образа; молча скачивать другой нельзя.
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Зависимости ставим до кода — так слой кэшируется. `uv.lock*` со звёздочкой:
# файла может ещё не быть, и COPY по несуществующему пути упал бы.
COPY pyproject.toml uv.lock* ./

# С локфайлом сборка воспроизводима, без него uv резолвит на месте. Второй
# случай допустим только на этапе разработки, поэтому он громко предупреждает.
#
# Режим записывается в образ. Проверять его по логу сборки нельзя: BuildKit
# печатает в лог текст самой команды RUN, поэтому любой grep по предупреждению
# находит его и на успешной сборке. Файл — свойство образа, а не журнала.
RUN if [ -f uv.lock ]; then \
        uv sync --no-dev --locked && echo locked > /etc/sciaudit_build_mode; \
    else \
        echo "WARNING: в контексте сборки нет uv.lock — образ НЕ воспроизводим." >&2; \
        echo "         Выполните 'uv lock' в system_template/ и закоммитьте результат." >&2; \
        uv sync --no-dev && echo resolved > /etc/sciaudit_build_mode; \
    fi

# Кладём venv проекта в PATH, чтобы контракт запуска остался `python run.py ...`.
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

# Контракт: читаем /input, пишем /output.
#   docker run --rm -v $PWD/input:/input:ro -v $PWD/output:/output \
#     <image> python run.py --input /input/inputs.jsonl --output /output/predictions.jsonl
