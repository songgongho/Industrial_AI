# Deployment Guide

This guide explains how to deploy the PCB Press Defect Prediction system in different environments.

## Deployment Options

| Option | Use Case | Effort | Cost | Auto-Updates |
|--------|----------|--------|------|--------------|
| GitHub Pages (Static) | Demo dashboard | ⭐ Easy | Free | ✅ Yes |
| Streamlit Cloud | Interactive dashboard | ⭐⭐ Medium | Free (with limitations) | ✅ Yes |
| Heroku | Simple app hosting | ⭐⭐ Medium | $ (≥$7/month) | ✅ Yes |
| AWS EC2 | Production serving | ⭐⭐⭐ Hard | $$ (pay-as-you-go) | ❌ Manual |
| Docker (Self-hosted) | On-premise | ⭐⭐⭐ Hard | $ (infrastructure) | ✅ (with orchestration) |
| Azure ML | Enterprise ML | ⭐⭐⭐⭐ Very Hard | $$ | ✅ Yes |

We recommend **GitHub Pages** for initial demo, then **Streamlit Cloud** for interactive features.

---

## Option 1: GitHub Pages (Recommended for Static Demo)

### What You Get
- Free static website hosting
- Auto-deploy from GitHub
- No server costs
- supports HTML/CSS/JS

### What You Can't Do
- Run Python on the server
- Real-time predictions
- Dynamic data updates
- User logins

### Setup Steps

#### 1a. Generate Static Dashboard

```bash
# Train model first
python scripts/train.py --epochs 5 --batch-size 32

# Generate static report
python scripts/generate_html_report.py \
  --predictions outputs/predictions.json \
  --output app/index.html
```

#### 1b. Configure GitHub Pages

1. Go to your GitHub repository Settings
2. Scroll to "Pages" section
3. Select:
   - Source: "Deploy from a branch"
   - Branch: `main`
   - Folder: `/app` (where index.html lives)
4. Click "Save"

#### 1c. Push to GitHub

```bash
git add app/
git commit -m "docs: initial dashboard"
git push origin main
```

#### 1d. Access Your Site

GitHub Pages will deploy automatically. Visit:
```
https://your-username.github.io/pcb-lamination-press-defect-prediction
```

### Updating the Dashboard

```bash
# 1. Re-train or generate new predictions
python scripts/predict.py --data new_data.parquet --output outputs/new_predictions.json

# 2. Regenerate static report
python scripts/generate_html_report.py \
  --predictions outputs/new_predictions.json \
  --output app/index.html

# 3. Push changes
git add app/index.html
git commit -m "docs: update dashboard with latest results"
git push origin main

# GitHub Pages automatically redeploys!
```

> **Note**: GitHub Pages updates within 30-60 seconds of push

---

## Option 2: Streamlit Cloud (Recommended for Interactive Demo)

### What You Get
- Interactive Streamlit app (python -based)
- File uploads, real-time predictions
- Auto-deploy from GitHub
- Free tier: 3 apps, 1 core, 1GB memory

### Limitations
- 1GB memory limit (suitable for demo, not production)
- 12-hour app timeout
- No GPU support in free tier

### Prerequisites
- GitHub account with this repository
- Streamlit account (free at streamlit.io)

### Setup Steps

#### 2a. Verify scripts/ui.py Works Locally

```bash
streamlit run scripts/ui.py
# Should open at http://localhost:8501
```

#### 2b. Create Streamlit Cloud Account

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign up with GitHub
3. Grant repository access

#### 2c. Deploy App

1. Click "New app"
2. Select:
   - Repository: your-username/pcb-lamination-press-defect-prediction
   - Branch: `main`
   - File: `scripts/ui.py`
3. Click "Deploy"

Streamlit automatically:
- Installs requirements.txt
- Runs the app
- Assigns a public URL

#### 2d. Access Your App

Your app is live at:
```
https://your-username-pcb-press-demo.streamlit.app
```

### Updating the App

```bash
# 1. Make changes locally
# 2. Test: streamlit run scripts/ui.py
# 3. Push to GitHub
git push origin main
# Streamlit auto-redeploys!
```

---

## Option 3: Docker (Production-Grade)

### What You Get
- Reproducible environment
- Easy scaling
- Works anywhere (local, cloud, on-prem)
- GPU support

### Prerequisites
- Docker installed ([docker.com/get-started](https://docs.docker.com/get-started/))
- (Optional) Docker Hub account for image sharing

### Setup Steps

#### 3a. Create Dockerfile

Already exists in repo: `Dockerfile`

#### 3b. Build Image

```bash
docker build -t pcb-press:latest .
```

#### 3c. Run Container

**For Streamlit app:**
```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  pcb-press:latest \
  streamlit run scripts/ui.py --server.address 0.0.0.0
```

**For model training:**
```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  pcb-press:latest \
  python scripts/train.py --epochs 10
```

**For predictions:**
```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  pcb-press:latest \
  python scripts/predict.py --data /app/data/demo/sample.parquet
```

#### 3d. Push to Docker Hub (Optional)

```bash
# Tag image
docker tag pcb-press:latest your-username/pcb-press:latest

# Push
docker push your-username/pcb-press:latest

# Others can now pull:
# docker pull your-username/pcb-press:latest
```

### Docker Compose (Multiple Services)

For production with separate services:

```yaml
# docker-compose.yml
version: '3.9'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    command: python scripts/api.py
    
  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    command: streamlit run scripts/ui.py --server.address 0.0.0.0
```

Then run:
```bash
docker-compose up -d
```

---

## Option 4: AWS EC2 (Production Cloud)

### What You Get
- Full control
- Scalability
- Enterprise-grade infrastructure
- Available worldwide

### Rough Timeline: 20-30 minutes

#### 4a. Launch EC2 Instance

1. AWS Console → EC2 → "Launch Instance"
2. Select: **Ubuntu Server 22.04 LTS**
3. Instance type: `t3.medium` (2 vCPU, 4GB RAM, ~$0.04/hour)
4. Configure storage: 50 GB EBS
5. Create/select security group:
   - HTTP (port 80) from anywhere
   - HTTPS (port 443) from anywhere
   - SSH (port 22) from your IP
6. Review & Launch
7. Create/select key pair (download .pem file)

#### 4b. Connect to Instance

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@your-instance-public-ip
```

#### 4c. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python & Git
sudo apt install -y python3.11 python3.11-venv git curl

# Clone repository
git clone https://github.com/your-username/pcb-lamination-press-defect-prediction.git
cd pcb-lamination-press-defect-prediction

# Setup Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 4d. Run Service

**Option A: Streamlit (Simple)**
```bash
streamlit run scripts/ui.py \
  --server.port 80 \
  --server.address 0.0.0.0
```

**Option B: API Server (FastAPI)**
```bash
# Create scripts/api.py with FastAPI endpoints
uvicorn scripts.api:app --host 0.0.0.0 --port 80
```

**Option C: Background Service (Systemd)**

Create `/etc/systemd/system/pcb-press.service`:
```ini
[Unit]
Description=PCB Press Defect Prediction Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/pcb-lamination-press-defect-prediction
Environment="PATH=/home/ubuntu/pcb-lamination-press-defect-prediction/.venv/bin"
ExecStart=/home/ubuntu/pcb-lamination-press-defect-prediction/.venv/bin/streamlit run scripts/ui.py --server.port 80 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then start:
```bash
sudo systemctl enable pcb-press
sudo systemctl start pcb-press
sudo systemctl status pcb-press
```

#### 4e. Access Your App

Visit:
```
http://your-instance-public-ip
```

### Monitoring & Maintenance

```bash
# View logs
sudo journalctl -u pcb-press -f

# Restart service
sudo systemctl restart pcb-press

# Update code
cd ~/pcb-lamination-press-defect-prediction
git pull origin main
# Restart service to apply updates
sudo systemctl restart pcb-press
```

---

## Comparison: Which Option to Choose?

### Demo Phase (Now)
→ **GitHub Pages** (static dashboard)
- No cost, minimal effort
- Good for showcasing results to advisors

### Thesis Defense & Demo
→ **GitHub Pages + Streamlit Cloud**
- Interactive demo for examiners
- No infrastructure costs
- Auto-deploys from GitHub

### Production (Post-Thesis)
→ **Docker + AWS EC2** or **Azure ML**
- Proper infrastructure
- Monitoring and scaling
- Team collaboration

---

## Security Checklist

When deploying to production:

- [ ] Never commit credentials (use environment variables)
- [ ] Use HTTPS (Let's Encrypt is free)
- [ ] Restrict data file access
- [ ] Enable authentication if needed
- [ ] Monitor resource usage
- [ ] Keep dependencies updated
- [ ] Regular backups
- [ ] Audit logs for model predictions

---

## Troubleshooting

### GitHub Pages - Site Not Updating

```bash
# Clear cache and force rebuild
git add -A
git commit --allow-empty -m "rebuild"
git push origin main
```

### Streamlit Cloud - App Crashes

Check logs in Streamlit Cloud dashboard. Common issues:
- Missing dependencies → Update `requirements.txt`
- Memory limit exceeded → Reduce data size
- Missing data files → Place in `data/` folder

### Docker - Build Fails

```bash
# Check Docker daemon is running
docker version

# Build with verbose logging
docker build --progress=plain -t pcb-press:latest .
```

### AWS - High Costs

- Use scheduled startup/shutdown (Lambda + EventBridge)
- Reduce instance size if memory unutilized
- Use spot instances (70% discount, caveat: interruption)

---

## Support

- **Deployment Issues**: [GitHub Issues](https://github.com/your-username/pcb-lamination-press-defect-prediction/issues)
- **Questions**: [GitHub Discussions](https://github.com/your-username/pcb-lamination-press-defect-prediction/discussions)

---

**Last Updated**: May 2026

