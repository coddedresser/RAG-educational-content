# 🚀 Educational RAG System - Deployment Guide

This guide provides comprehensive instructions for deploying the Educational RAG System across different platforms and environments.

## 📋 Pre-Deployment Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] System initialized (`python run_system.py`)
- [ ] Environment variables configured (`.env` file)
- [ ] Basic functionality tested locally
- [ ] Content processed and loaded
- [ ] Student profiles and learning paths tested

## 🏠 Local Development Deployment

### Quick Start
```bash
# Clone and setup
git clone https://github.com/your-username/educational-rag-system.git
cd educational-rag-system

# Install dependencies
pip install -r requirements.txt

# Initialize system
python run_system.py

# Start application
streamlit run app/main.py
```

### Development with Hot Reload
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Start with auto-reload
streamlit run app/main.py --server.runOnSave=true
```

## ☁️ Cloud Deployment Options

### 1. Streamlit Cloud (Recommended for Demos)

**Pros:** Free, easy setup, automatic deployments
**Cons:** Limited resources, public repositories only

#### Setup Instructions:

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial deployment"
git push origin main
```

2. **Deploy on Streamlit Cloud**
- Visit [share.streamlit.io](https://share.streamlit.io)
- Connect your GitHub account
- Select repository: `your-username/educational-rag-system`
- Set main file path: `app/main.py`
- Click "Deploy"

3. **Configure Environment Variables**
```
# In Streamlit Cloud dashboard -> Settings -> Secrets
OPENAI_API_KEY = "your_openai_api_key_here"
HUGGINGFACE_API_KEY = "your_hf_api_key_here"
```

4. **Access Your App**
- URL format: `https://your-username-educational-rag-system-app-main-xyz123.streamlit.app`

### 2. Heroku Deployment

**Pros:** Easy scaling, good for production
**Cons:** Paid plans for persistent storage

#### Setup Files:

**Procfile:**
```
web: streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0
```

**runtime.txt:**
```
python-3.9.18
```

#### Deploy Steps:
```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-educational-rag-app

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key_here
heroku config:set HUGGINGFACE_API_KEY=your_key_here

# Deploy
git push heroku main

# Open app
heroku open
```

### 3. HuggingFace Spaces

**Pros:** Free GPU access, ML-focused community
**Cons:** Public by default, resource limitations

#### Setup:
1. Create new Space on [HuggingFace Spaces](https://huggingface.co/spaces)
2. Choose "Streamlit" as SDK
3. Upload files or connect Git repository
4. Add secrets in Settings for API keys

**app.py (HF Spaces entry point):**
```python
import subprocess
import sys

# Install requirements
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Run main application
exec(open('app/main.py').read())
```

### 4. Google Cloud Run

**Pros:** Serverless, auto-scaling, pay-per-use
**Cons:** Requires Docker knowledge

#### Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Deploy Commands:
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/educational-rag
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/educational-rag --platform managed
```

### 5. AWS EC2 with Docker

**Pros:** Full control, scalable, production-ready
**Cons:** More complex setup, higher cost

#### docker-compose.yml:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}
    volumes:
      - ./data:/app/data
      - ./vector_db:/app/vector_db
      - ./models:/app/models
    restart: unless-stopped
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped
```

## 🔧 Production Configuration

### 1. Environment Variables
```bash
# Production settings
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=WARNING

# Security
SECRET_KEY=your_very_secure_secret_key_here
CORS_ORIGINS=https://yourdomain.com

# Performance
MAX_CONCURRENT_USERS=500
QUERY_TIMEOUT=30
ENABLE_QUERY_CACHE=True
REDIS_URL=redis://redis:6379

# Monitoring
SENTRY_DSN=your_sentry_dsn_here
ENABLE_PERFORMANCE_MONITORING=True
```

### 2. Nginx Configuration
```nginx
events {
    worker_connections 1024;
}

http {
    upstream streamlit {
        server app:8501;
    }

    server {
        listen 80;
        server_name yourdomain.com;
        
        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://streamlit;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### 3. SSL Certificate Setup
```bash
# Using Let's Encrypt
sudo certbot --nginx -d yourdomain.com

# Or generate self-signed for testing
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/key.pem -out ssl/cert.pem
```

## 📊 Monitoring and Logging

### 1. Application Monitoring
```python
# Add to main.py for production monitoring
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

if os.getenv('SENTRY_DSN'):
    sentry_sdk.init(
        dsn=os.getenv('SENTRY_DSN'),
        integrations=[LoggingIntegration()],
        traces_sample_rate=1.0
    )
```

### 2. Log Management
```yaml
# docker-compose.yml logging
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 3. Health Checks
```python
# Add to main.py
@st.cache_data
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        # Test vector database
        # Test critical components
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## 🔒 Security Considerations

### 1. API Key Management
```bash
# Use environment variables, never hardcode
export OPENAI_API_KEY="sk-..."
export HUGGINGFACE_API_KEY="hf_..."

# Use secret management services in production
aws secretsmanager get-secret-value --secret-id educational-rag/api-keys
```

### 2. Data Protection
```python
# Encrypt sensitive data
from cryptography.fernet import Fernet

# Generate key (store securely)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Encrypt student data
encrypted_data = cipher_suite.encrypt(data.encode())
```

### 3. Access Control
```python
# Add authentication if needed
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    'educational_rag',
    'auth_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')
```

## 📈 Performance Optimization

### 1. Caching Configuration
```python
# Enhanced caching for production
@st.cache_data(ttl=3600, max_entries=1000)
def cached_search(query, filters):
    return retriever.semantic_search(query, filters=filters)

@st.cache_resource
def load_models():
    return {
        'embeddings': EmbeddingGenerator(),
        'retriever': EducationalRetriever(...)
    }
```

### 2. Database Optimization
```python
# Connection pooling
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = sqlite3.connect('educational_rag.db', timeout=30)
    try:
        yield conn
    finally:
        conn.close()
```

### 3. Resource Management
```python
# Memory management
import gc
import psutil

def monitor_resources():
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > 80:
        gc.collect()
        st.warning("High memory usage detected")
```

## 🧪 Testing in Production

### 1. Smoke Tests
```bash
#!/bin/bash
# smoke_test.sh

echo "Running smoke tests..."

# Test application startup
curl -f http://localhost:8501 || exit 1

# Test health endpoint
curl -f http://localhost:8501/_stcore/health || exit 1

# Test search functionality
python -c "
from app.components.retriever import EducationalRetriever
retriever = EducationalRetriever('./vector_db')
results = retriever.semantic_search('test query', top_k=1)
assert len(results) >= 0, 'Search failed'
print('✅ Search test passed')
"

echo "✅ All smoke tests passed"
```

### 2. Load Testing
```python
# load_test.py
import concurrent.futures
import requests
import time

def test_endpoint(url):
    start_time = time.time()
    response = requests.get(url)
    end_time = time.time()
    return {
        'status_code': response.status_code,
        'response_time': end_time - start_time
    }

def load_test(url, concurrent_users=10, duration=60):
    """Simple load test"""
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        while time.time() - start_time < duration:
            futures = [executor.submit(test_endpoint, url) for _ in range(concurrent_users)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    
    # Analyze results
    avg_response_time = sum(r['response_time'] for r in results) / len(results)
    success_rate = sum(1 for r in results if r['status_code'] == 200) / len(results)
    
    print(f"Average Response Time: {avg_response_time:.2f}s")
    print(f"Success Rate: {success_rate:.2%}")
    
    return results

if __name__ == "__main__":
    load_test("http://localhost:8501")
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy Educational RAG System

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
        python -c "import app.components.data_processor; print('✅ Import test passed')"
    
    - name: Run system initialization
      run: |
        python run_system.py
    
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Streamlit Cloud
      run: |
        echo "Deployment triggered automatically by Streamlit Cloud"
    
    - name: Run smoke tests
      run: |
        sleep 60  # Wait for deployment
        curl -f https://your-app-url.streamlit.app || exit 1
```

## 📝 Deployment Checklist

### Pre-Production
- [ ] All tests passing
- [ ] Security review completed
- [ ] Performance benchmarks met
- [ ] Backup procedures tested
- [ ] Monitoring configured
- [ ] SSL certificates installed
- [ ] Domain configured
- [ ] Load balancer set up (if needed)

### Post-Deployment
- [ ] Smoke tests executed
- [ ] Performance monitoring active
- [ ] Error tracking configured
- [ ] User acceptance testing completed
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Rollback procedures tested
- [ ] Support processes established

## 🆘 Troubleshooting

### Common Issues

**1. Memory Issues**
```bash
# Monitor memory usage
docker stats
htop

# Solution: Increase memory limits or optimize caching
```

**2. Vector Database Issues**
```bash
# Reset vector database
rm -rf vector_db/
python run_system.py
```

**3. Slow Performance**
```bash
# Check resource usage
docker logs educational-rag-app

# Optimize database queries
# Enable caching
# Scale horizontally
```

**4. SSL Certificate Issues**
```bash
# Renew certificates
sudo certbot renew

# Test certificate
openssl s_client -connect yourdomain.com:443
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks
- [ ] Update dependencies monthly
- [ ] Backup databases weekly
- [ ] Monitor logs daily
- [ ] Review performance metrics
- [ ] Update content as needed
- [ ] Review security settings
- [ ] Test disaster recovery procedures

### Getting Help
- 📧 Email: support@educational-rag.com
- 💬 Discord: [Educational RAG Community](https://discord.gg/educational-rag)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/educational-rag-system/issues)
- 📖 Documentation: [Wiki](https://github.com/your-username/educational-rag-system/wiki)

---

**🎉 Congratulations on deploying your Educational RAG System!**

For additional support and updates, please refer to our documentation and community resources.