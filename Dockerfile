FROM python:3.12-slim
LABEL org.opencontainers.image.title="RionMASS" \
      org.opencontainers.image.source="https://github.com/GSI-Nuclear-Astrophysics/RionMASS" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"
WORKDIR /app
COPY pyproject.toml MANIFEST.in README.md LICENSE CITATION.cff ./
COPY src/ src/
COPY tests/ tests/
COPY examples/ examples/
COPY docs/ docs/
RUN python -m pip install --no-cache-dir '.[data,plot]' 'pytest>=8'
ENTRYPOINT ["rionmass"]
CMD ["--help"]
