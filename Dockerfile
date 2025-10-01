FROM python:3.11-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    zlib1g-dev \
    libxml2-dev \
    libxslt1-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for mercury-parser
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install mercury-parser globally
RUN npm install -g @postlight/mercury-parser

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
ARG WITH_ML=0
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$WITH_ML" = "1" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi

# Pre-download NLTK data to avoid repeated downloads
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .

# Optionally install Playwright browsers inside the image
ARG WITH_PLAYWRIGHT_BROWSERS=0
RUN if [ "$WITH_PLAYWRIGHT_BROWSERS" = "1" ]; then python install_playwright.py; fi

CMD ["tail", "-f", "/dev/null"]
