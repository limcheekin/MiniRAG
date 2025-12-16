# File: Dockerfile
FROM postgres:17.7-trixie AS builder

# Switch to root to install dependencies
USER root

# Install compilation tools
# flex/bison required for AGE
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    postgresql-server-dev-17 \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# 1. Build pgvector (Pinned to v0.8.1 for stability)
RUN git clone --depth 1 --branch v0.8.1 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install

# 2. Build Apache AGE (Pinned to PG17 RC branch)
RUN git clone --depth 1 --branch PG17/v1.6.0-rc0 https://github.com/apache/age.git \
    && cd age \
    && make install

# --- Final Stage ---
FROM postgres:17.7-trixie

# Copy compiled shared objects (.so)
COPY --from=builder /usr/lib/postgresql/17/lib/vector.so /usr/lib/postgresql/17/lib/
COPY --from=builder /usr/lib/postgresql/17/lib/age.so /usr/lib/postgresql/17/lib/

# Copy extension control files (.control and .sql)
COPY --from=builder /usr/share/postgresql/17/extension/vector* /usr/share/postgresql/17/extension/
COPY --from=builder /usr/share/postgresql/17/extension/age* /usr/share/postgresql/17/extension/

# Reset permissions
USER postgres

# Default command includes shared_preload_libraries
CMD ["postgres", "-c", "shared_preload_libraries=age,vector"]