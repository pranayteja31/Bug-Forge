FROM python:3.11-slim

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project into /app/bugforge/ so Python can resolve `bugforge.*` imports
COPY . /app/bugforge/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "bugforge.server.app:app", "--host", "0.0.0.0", "--port", "8000"]