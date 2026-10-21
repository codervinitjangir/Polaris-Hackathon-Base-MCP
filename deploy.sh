#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Polaris MCP Server: 1-Click AWS Deployment & Healthcheck Script
# ==============================================================================

echo "===================================================================="
echo "🚀 Starting 1-Click Polaris MCP Server Deployment"
echo "===================================================================="

# 1. Determine Public IP
PUBLIC_IP="${1:-}"
if [ -z "$PUBLIC_IP" ]; then
    echo "🔍 Auto-detecting public IP..."
    # Try AWS IMDSv2 first
    TOKEN=$(curl -s -S -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
    if [ -n "$TOKEN" ]; then
        PUBLIC_IP=$(curl -s -S -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/public-ipv4" 2>/dev/null || true)
    fi
    # Fallback to IMDSv1
    if [ -z "$PUBLIC_IP" ]; then
        PUBLIC_IP=$(curl -s -S "http://169.254.169.254/latest/meta-data/public-ipv4" 2>/dev/null || true)
    fi
    # Fallback to public IP resolvers
    if [ -z "$PUBLIC_IP" ]; then
        PUBLIC_IP=$(curl -s -S https://checkip.amazonaws.com 2>/dev/null || curl -s -S https://ifconfig.me 2>/dev/null || true)
    fi
fi

if [ -z "$PUBLIC_IP" ]; then
    echo "⚠️  Could not auto-detect public IP. Using wildcard for allowed hosts."
    ALLOWED_HOSTS="localhost:*,127.0.0.1:*,dummy-mcp.01m5erq7wn81nf6mhfpqkjb0ka.apps.springwinter.dev,dummy-mcp.01m5erq7wn81nf6mhfpqkjb0ka.apps.springwinter.dev:*,*"
else
    echo "✅ Target Public IP: $PUBLIC_IP"
    ALLOWED_HOSTS="localhost:*,127.0.0.1:*,dummy-mcp.01m5erq7wn81nf6mhfpqkjb0ka.apps.springwinter.dev,dummy-mcp.01m5erq7wn81nf6mhfpqkjb0ka.apps.springwinter.dev:*,${PUBLIC_IP}:*,${PUBLIC_IP},*"
fi

# 2. Check and Install Docker if missing
if ! command -v docker >/dev/null 2>&1; then
    echo "📦 Docker not found. Installing Docker..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y
        sudo apt-get install -y docker.io git git-lfs curl
        sudo systemctl enable --now docker || true
        sudo usermod -aG docker "$USER" || true
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y docker git git-lfs curl
        sudo systemctl enable --now docker || true
        sudo usermod -aG docker "$USER" || true
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y docker git git-lfs curl
        sudo systemctl enable --now docker || true
        sudo usermod -aG docker "$USER" || true
    fi
fi

# 3. Verify Git LFS Data
echo "📦 Verifying catalog data..."
if [ -f "data/products.json" ]; then
    FILE_SIZE=$(wc -c < "data/products.json")
    if [ "$FILE_SIZE" -lt 50000000 ]; then
        echo "⚠️  products.json appears to be an LFS pointer ($FILE_SIZE bytes). Pulling full dataset..."
        git lfs install
        git lfs pull
    else
        echo "✅ products.json verified (~$(( FILE_SIZE / 1024 / 1024 )) MB)"
    fi
else
    echo "⚠️  products.json missing. Running git lfs pull..."
    git lfs install
    git lfs pull
fi

# 4. Build Docker Container
echo "🔨 Building Docker image: polaris-search:prod ..."
docker build -t polaris-search:prod .

# 5. Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -Eq "^polaris-search$"; then
    echo "🛑 Stopping existing polaris-search container..."
    docker stop polaris-search >/dev/null 2>&1 || true
    docker rm polaris-search >/dev/null 2>&1 || true
fi

# 6. Run Production Container
echo "🚀 Starting Polaris MCP Server on port 8000..."
docker run -d \
  --name polaris-search \
  --restart=always \
  -p 8000:8000 \
  -e HOST="0.0.0.0" \
  -e PORT="8000" \
  -e MCP_ALLOWED_HOSTS="$ALLOWED_HOSTS" \
  polaris-search:prod

# 7. Wait for Healthcheck Readiness
echo "⏳ Waiting for server startup & indexing..."
READY=false
for i in {1..30}; do
    if curl -s -f "http://127.0.0.1:8000/healthz" >/dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 1
done

if [ "$READY" = true ]; then
    HEALTH=$(curl -s "http://127.0.0.1:8000/healthz")
    echo "✅ Server Healthy! Response: $HEALTH"
else
    echo "❌ Server failed to respond to /healthz within 30 seconds."
    docker logs polaris-search
    exit 1
fi

# 8. Run Smoke Test inside container or locally
echo "🧪 Running protocol smoke test..."
if command -v python3 >/dev/null 2>&1 && python3 -c "import mcp" >/dev/null 2>&1; then
    python3 smoke_test.py http://127.0.0.1:8000/mcp
else
    docker exec polaris-search python smoke_test.py http://127.0.0.1:8000/mcp
fi

# 9. Output Final Submission URL
echo ""
echo "===================================================================="
echo "🎉 DEPLOYMENT SUCCESSFUL!"
echo "===================================================================="
if [ -n "$PUBLIC_IP" ]; then
    echo "👉 Submit this URL on the SpringWinter portal:"
    echo "   http://${PUBLIC_IP}:8000/mcp"
    echo ""
    echo "👉 Health check URL:"
    echo "   http://${PUBLIC_IP}:8000/healthz"
else
    echo "👉 Submit this URL (replace YOUR_IP with your public EC2 IP):"
    echo "   http://<YOUR_EC2_PUBLIC_IP>:8000/mcp"
fi
echo "===================================================================="
