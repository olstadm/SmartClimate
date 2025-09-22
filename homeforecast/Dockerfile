ARG BUILD_FROM
FROM $BUILD_FROM

# Install Python and required system packages
RUN \
    apk add --no-cache \
        python3 \
        py3-pip \
        python3-dev \
        gcc \
        musl-dev \
        linux-headers \
        g++ \
        gfortran \
        openblas-dev \
        freetype-dev \
        libpng-dev \
        nginx

# Install Python packages
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r /requirements.txt

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