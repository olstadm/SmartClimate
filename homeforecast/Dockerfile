ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.11-alpine3.18
FROM $BUILD_FROM

# Install system packages and Python dependencies
RUN \
    apk add --no-cache \
        python3-dev \
        gcc \
        musl-dev \
        linux-headers \
        g++ \
        gfortran \
        openblas-dev \
        lapack-dev \
        freetype-dev \
        libpng-dev \
        nginx \
        py3-numpy \
        py3-scipy \
        py3-pandas \
        py3-pip && \
    python3 -m pip install --upgrade pip setuptools wheel

# Install Python packages with better compatibility
COPY requirements.txt /
RUN pip3 install --no-cache-dir --break-system-packages \
    --prefer-binary --no-build-isolation -r /requirements.txt

# Copy addon code
COPY rootfs /
RUN chmod a+x /run.sh

# Configure nginx
RUN rm -f /etc/nginx/http.d/default.conf
COPY nginx.conf /etc/nginx/http.d/

# Create necessary directories
RUN mkdir -p /data/homeforecast /share/homeforecast /var/log/homeforecast

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/

# Expose ports
EXPOSE 8099

# Start the addon
CMD [ "/run.sh" ]