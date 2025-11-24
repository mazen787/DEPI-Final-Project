# 1. Use an official Python runtime
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install Poetry
RUN pip install poetry

# 4. Copy poetry files first (to cache dependencies)
COPY pyproject.toml poetry.lock ./

# 5. Install dependencies (without creating a virtualenv inside Docker)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# 6. Copy the rest of the code
COPY . .

# 7. Expose port
EXPOSE 8501

# 8. Run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--browser.serverAddress=localhost"]